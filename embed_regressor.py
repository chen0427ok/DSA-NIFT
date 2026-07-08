"""
E14 — Frozen Embedding + SVR / Ridge 回歸（VA Regression Ensemble，借鑑 TCU@ROCLING-2025）

不再訓 transformer，把 encoder 當 frozen feature extractor：
    feature = [mask-mean-pooled embedding] + [L1 10 維] + [simple stats 6 維]
分別對 valence / arousal 訓 SVR(RBF) 與 Ridge，輸出統一預測檔給 ensemble.py 當補位成員
（補一條非 neural-head 的分數分布，讓 arousal 不被單一 head 決定）。

embedding 會存到 outputs/emb_{model短名}_{split}.npy 快取，換回歸器不用重抽。

用法：
    python embed_regressor.py --model hfl/chinese-macbert-base
    # 之後把 svr_chinese-macbert-base / ridge_chinese-macbert-base 加進 ensemble.py 的 runs

產出：outputs/preds/{svr|ridge}_{model短名}_dev.csv / _val.csv
"""
import os
import csv
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from train import read_csv, pick_device
from lexicon import LexiconFeaturizer

HERE = os.path.dirname(__file__)
DIMS = ["valence", "arousal"]

NEG_WORDS = ("不", "沒", "別", "無法", "不能")
AROUSAL_PUNCT = "！!？?…"


def simple_stats(text):
    t = text or ""
    n = max(len(t), 1)
    return [
        min(len(t), 500) / 500.0,                       # 長度
        min(sum(t.count(c) for c in "！!"), 5) / 5.0,    # 驚嘆
        min(sum(t.count(c) for c in "？?"), 5) / 5.0,    # 疑問
        min(t.count("我"), 10) / 10.0,                   # 第一人稱
        min(sum(t.count(w) for w in NEG_WORDS), 10) / 10.0,  # 否定
        sum(t.count(c) for c in AROUSAL_PUNCT) / n,      # 高喚醒標點密度
    ]


@torch.no_grad()
def embed(rows, model_name, split, device, max_len=256, batch_size=32):
    short = model_name.rstrip("/").split("/")[-1]
    cache = os.path.join(HERE, "outputs", f"emb_{short}_{split}.npy")
    if os.path.exists(cache):
        E = np.load(cache)
        if len(E) == len(rows):
            print(f"  [{split}] 用快取 {cache}")
            return E
    tok = AutoTokenizer.from_pretrained(model_name)
    enc_model = AutoModel.from_pretrained(model_name).to(device).eval()
    E = []
    for i in range(0, len(rows), batch_size):
        texts = [r["text"] for r in rows[i:i + batch_size]]
        b = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out = enc_model(**b)
        mask = b["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        E.append(pooled.cpu().numpy())
    E = np.vstack(E)
    np.save(cache, E)
    del enc_model
    print(f"  [{split}] embedding {E.shape} -> {cache}")
    return E


def build_features(rows, emb, featurizer):
    lex = np.array([featurizer.featurize(r["text"]) for r in rows])
    stats = np.array([simple_stats(r["text"]) for r in rows])
    return np.hstack([emb, lex, stats])


def write_preds(name, dev_rows, val_rows, gold, dev_pred, val_pred, pred_dir):
    with open(os.path.join(pred_dir, f"{name}_dev.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "valence_true", "arousal_true", "valence_pred", "arousal_pred", "model_name", "split"])
        for r, g, p in zip(dev_rows, gold, dev_pred):
            w.writerow([r["id"], f"{g[0]:.4f}", f"{g[1]:.4f}", f"{p[0]:.4f}", f"{p[1]:.4f}", name, "dev"])
    with open(os.path.join(pred_dir, f"{name}_val.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "valence_pred", "arousal_pred", "model_name", "split"])
        for r, p in zip(val_rows, val_pred):
            w.writerow([r["id"], f"{p[0]:.4f}", f"{p[1]:.4f}", name, "val"])


def main():
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--regressors", default="svr,ridge")
    ap.add_argument("--svr_c", type=float, default=10.0)
    ap.add_argument("--svr_eps", type=float, default=0.1)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    args = ap.parse_args()

    device = pick_device()
    short = args.model.rstrip("/").split("/")[-1]
    pred_dir = os.path.join(HERE, "outputs", "preds")
    os.makedirs(pred_dir, exist_ok=True)

    train_rows = read_csv(os.path.join(args.data_dir, "train.csv"))
    dev_rows = read_csv(os.path.join(args.data_dir, "dev.csv"))
    val_rows = read_csv(os.path.join(args.data_dir, "val_unlabeled.csv"))
    print(f"device={device} | model={args.model} | train={len(train_rows)} dev={len(dev_rows)} val={len(val_rows)}")

    print("抽 frozen embedding ...")
    E_tr = embed(train_rows, args.model, "train", device)
    E_de = embed(dev_rows, args.model, "dev", device)
    E_va = embed(val_rows, args.model, "val", device)

    print("組特徵 (embedding + L1 + stats) ...")
    fz = LexiconFeaturizer()
    X_tr = build_features(train_rows, E_tr, fz)
    X_de = build_features(dev_rows, E_de, fz)
    X_va = build_features(val_rows, E_va, fz)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_de, X_va = scaler.transform(X_tr), scaler.transform(X_de), scaler.transform(X_va)

    y_tr = np.array([[float(r["valence"]), float(r["arousal"])] for r in train_rows])
    gold = np.array([[float(r["valence"]), float(r["arousal"])] for r in dev_rows])

    for reg_name in args.regressors.split(","):
        reg_name = reg_name.strip()
        dev_pred = np.zeros_like(gold)
        val_pred = np.zeros((len(val_rows), 2))
        for j, d in enumerate(DIMS):
            if reg_name == "svr":
                m = SVR(kernel="rbf", C=args.svr_c, epsilon=args.svr_eps)
            elif reg_name == "ridge":
                m = Ridge(alpha=args.ridge_alpha)
            else:
                raise ValueError(f"未知回歸器 {reg_name}")
            m.fit(X_tr, y_tr[:, j])
            dev_pred[:, j] = m.predict(X_de)
            val_pred[:, j] = m.predict(X_va)
        dev_pred = np.clip(dev_pred, 1.0, 9.0)
        val_pred = np.clip(val_pred, 1.0, 9.0)
        name = f"{reg_name}_{short}"
        msg = []
        for j, d in enumerate(DIMS):
            mae = np.mean(np.abs(dev_pred[:, j] - gold[:, j]))
            pcc = np.corrcoef(dev_pred[:, j], gold[:, j])[0, 1]
            msg.append(f"{d}_MAE={mae:.4f} {d}_PCC={pcc:.4f}")
        print(f"[{name}] dev  " + "  ".join(msg))
        write_preds(name, dev_rows, val_rows, gold, dev_pred, val_pred, pred_dir)
        print(f"  preds -> {pred_dir}/{name}_dev.csv / _val.csv")


if __name__ == "__main__":
    main()
