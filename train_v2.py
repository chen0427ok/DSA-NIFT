"""
train_v2.py — 實驗 E10+ 的擴充訓練器。

train.py 凍結不動（實驗 1–9b 復現用）；本檔重用其元件並加上新實驗需要的旋鈕。
**所有新參數的預設值 = 完全復現 train.py（實驗 4）的行為**，方便公平對照。

新增能力：
- --run_name        每個 run 獨立輸出檔名（多 seed / 多 encoder 不互相覆蓋）
- 統一預測輸出      outputs/preds/{run}_dev.csv / {run}_val.csv（給 ensemble.py / calibrate.py）
- --train_file      指定訓練檔（不改動 data/train.csv）
- --extra_train     追加訓練資料（可重複；如偽標增強 train_aug_pseudo.csv），原檔不動
- --arousal_weight  arousal loss 加權（E15，建議先試 1.2）
- --pcc_weight      batch 內 (1 - PCC) arousal 排序 loss（E15，建議先試 0.1）
- --mae_weight      early-stop score = mean(PCC) - mae_weight * mean(MAE)（預設 1.0 = train.py）
- --lex_mode        none / l1 / l1l2（E16：l1l2 = L1 + L2 OOV 詞 VA 特徵）

用法範例：
    # E10 multi-seed（跑 5 次，之後 ensemble.py --mode mean 融合）
    python train_v2.py --seed 42 --run_name macbert_s42 --epochs 4 --batch_size 32
    # E11 RoBERTa-wwm-ext-large（T4 建議 batch 8 / lr 1e-5）
    python train_v2.py --model hfl/chinese-roberta-wwm-ext-large --batch_size 8 --lr 1e-5 --run_name robertaL_s42
    # E13 偽標增強
    python train_v2.py --extra_train data/train_aug_pseudo.csv --run_name macbert_pseudo_s42

產出（{run} = --run_name）：
    outputs/{run}_best.pt              最佳權重
    outputs/{run}_submission.csv       官方格式 ID,Valence,Arousal
    outputs/preds/{run}_dev.csv        ID,valence_true,arousal_true,valence_pred,arousal_pred,model_name,split
    outputs/preds/{run}_val.csv        ID,valence_pred,arousal_pred,model_name,split
"""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from train import (VADataset, VARegressor, pick_device, read_csv, denorm,
                   LABEL_MIN, LABEL_MAX)
from lexicon import LexiconFeaturizer, FEATURE_DIM

HERE = os.path.dirname(__file__)


def pearson_loss(pred, gold):
    """batch 內 1 - PCC，直接優化排序（僅在 batch >= 4 時使用）。"""
    p = pred - pred.mean()
    g = gold - gold.mean()
    denom = torch.sqrt((p * p).sum() * (g * g).sum()).clamp(min=1e-8)
    return 1.0 - (p * g).sum() / denom


@torch.no_grad()
def predict(model, loader, device):
    """回傳 [N,2] 的 1-9 尺度預測（不含 clip）。"""
    model.eval()
    P = []
    for batch in loader:
        batch.pop("labels", None)
        out = model(**{k: v.to(device) for k, v in batch.items()})
        P.append(out.cpu().numpy())
    return denorm(np.vstack(P))


def metrics(pred, gold):
    res = {}
    for j, name in enumerate(["valence", "arousal"]):
        res[f"{name}_MAE"] = float(np.mean(np.abs(pred[:, j] - gold[:, j])))
        p, g = pred[:, j], gold[:, j]
        res[f"{name}_PCC"] = 0.0 if p.std() < 1e-8 else float(np.corrcoef(p, g)[0, 1])
    return res


def build_featurizer(args):
    if args.lex_mode == "none":
        return None, 0
    if args.lex_mode == "l1":
        return LexiconFeaturizer(), FEATURE_DIM
    from lexicon_l2 import L1L2Featurizer, FEATURE_DIM_L2
    return L1L2Featurizer(args.l2_pkl), FEATURE_DIM_L2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--train_file", default=None, help="預設 data_dir/train.csv")
    ap.add_argument("--extra_train", action="append", default=[],
                    help="追加訓練資料 csv（可重複），需含 text/valence/arousal 欄")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default=None, help="預設 {model名}_s{seed}")
    ap.add_argument("--arousal_weight", type=float, default=1.0, help="E15：arousal loss 加權")
    ap.add_argument("--pcc_weight", type=float, default=0.0, help="E15：batch 1-PCC arousal loss 權重")
    ap.add_argument("--mae_weight", type=float, default=1.0,
                    help="early-stop score = mean(PCC) - mae_weight*mean(MAE)")
    ap.add_argument("--lex_mode", choices=["none", "l1", "l1l2"], default="l1")
    ap.add_argument("--l2_pkl", default=os.path.join(HERE, "outputs", "l2_word_va.pkl"))
    args = ap.parse_args()

    model_short = args.model.rstrip("/").split("/")[-1]
    run = args.run_name or f"{model_short}_s{args.seed}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"run={run} | device={device} | model={args.model} | lex={args.lex_mode} "
          f"| aw={args.arousal_weight} pcc_w={args.pcc_weight}")
    preds_dir = os.path.join(args.out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    train_rows = read_csv(args.train_file or os.path.join(args.data_dir, "train.csv"))
    for extra in args.extra_train:
        extra_rows = read_csv(extra)
        print(f"extra_train += {len(extra_rows)}  ({extra})")
        train_rows += extra_rows
    dev_rows = read_csv(os.path.join(args.data_dir, "dev.csv"))
    val_rows = read_csv(os.path.join(args.data_dir, "val_unlabeled.csv"))
    print(f"train={len(train_rows)} dev={len(dev_rows)} val={len(val_rows)}")

    featurizer, lex_dim = build_featurizer(args)
    print(f"lexicon dim = {lex_dim}")

    train_loader = DataLoader(VADataset(train_rows, tok, args.max_len, featurizer=featurizer),
                              batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(VADataset(dev_rows, tok, args.max_len, featurizer=featurizer),
                            batch_size=args.batch_size)

    model = VARegressor(args.model, lex_dim=lex_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    dev_gold = np.array([[float(r["valence"]), float(r["arousal"])] for r in dev_rows])
    best_score, best_state = -1e9, None
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            out = model(**{k: v.to(device) for k, v in batch.items()})
            el = loss_fn(out, labels)  # [B,2]
            # aw=1, pcc_w=0 時 = SmoothL1 全元素平均，與 train.py 完全一致
            loss = (el[:, 0].mean() + args.arousal_weight * el[:, 1].mean()) / (1.0 + args.arousal_weight)
            if args.pcc_weight > 0 and labels.size(0) >= 4:
                loss = loss + args.pcc_weight * pearson_loss(out[:, 1], labels[:, 1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"  ep{ep} step{step}/{len(train_loader)} loss={running/step:.4f}")
        dev_pred = predict(model, dev_loader, device)
        res = metrics(dev_pred, dev_gold)
        score = (res["valence_PCC"] + res["arousal_PCC"]) / 2 \
            - args.mae_weight * (res["valence_MAE"] + res["arousal_MAE"]) / 2
        print(f"[epoch {ep}] " + " ".join(f"{k}={v:.4f}" for k, v in res.items()) + f"  score={score:.4f}")
        if score > best_score:
            best_score, best_state = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
        torch.save(best_state, os.path.join(args.out_dir, f"{run}_best.pt"))
    print(f"best dev score = {best_score:.4f}")

    # ---- 統一預測輸出（dev 含 gold，供 ensemble.py 算 dimension-wise 權重）----
    dev_pred = predict(model, dev_loader, device)
    res = metrics(dev_pred, dev_gold)
    print("[best] " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))
    with open(os.path.join(preds_dir, f"{run}_dev.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "valence_true", "arousal_true", "valence_pred", "arousal_pred",
                    "model_name", "split"])
        for r, g, p in zip(dev_rows, dev_gold, dev_pred):
            w.writerow([r["id"], f"{g[0]:.4f}", f"{g[1]:.4f}", f"{p[0]:.4f}", f"{p[1]:.4f}", run, "dev"])

    val_loader = DataLoader(VADataset(val_rows, tok, args.max_len, has_label=False, featurizer=featurizer),
                            batch_size=args.batch_size)
    val_pred = np.clip(predict(model, val_loader, device), LABEL_MIN, LABEL_MAX)
    with open(os.path.join(preds_dir, f"{run}_val.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "valence_pred", "arousal_pred", "model_name", "split"])
        for r, p in zip(val_rows, val_pred):
            w.writerow([r["id"], f"{p[0]:.4f}", f"{p[1]:.4f}", run, "val"])

    sub_path = os.path.join(args.out_dir, f"{run}_submission.csv")
    with open(sub_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Valence", "Arousal"])
        for r, (v, a) in zip(val_rows, val_pred):
            w.writerow([r["id"], f"{v:.4f}", f"{a:.4f}"])
    print(f"submission -> {sub_path}  ({len(val_pred)} rows)")


if __name__ == "__main__":
    main()
