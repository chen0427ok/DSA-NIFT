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
- --lex_mode        none / l1 / l1l2 / l1_intensity（E16：l1l2；E18：l1_intensity = L1++ 31 維）
- --source_aware    E19：依 granularity 來源對 V/A loss 加權（CVAS arousal 降權等）
- --source_weights  覆寫 E19 權重，格式 "sentence=1:0.25,text=1:0.5,..."（V:A）
- --rank_aug        E20：合成資料 ranking-only（檔案只提供 pairwise 排序約束，不進 SmoothL1）
- --rank_lambda_a / --rank_lambda_v / --rank_margin / --rank_min_gap / --rank_batch  E20 超參
- --pooling         E21：mean（=train.py）/ mean_cls_dim_attention（V/A 各自 attention pooling）

用法範例：
    # E10 multi-seed（跑 5 次，之後 ensemble.py --mode mean 融合）
    python train_v2.py --seed 42 --run_name macbert_s42 --epochs 4 --batch_size 32
    # E11 RoBERTa-wwm-ext-large（T4 建議 batch 8 / lr 1e-5）
    python train_v2.py --model hfl/chinese-roberta-wwm-ext-large --batch_size 8 --lr 1e-5 --run_name robertaL_s42
    # E13 偽標增強
    python train_v2.py --extra_train data/train_aug_pseudo.csv --run_name macbert_pseudo_s42
    # E18 L1++ intensity
    python train_v2.py --lex_mode l1_intensity --run_name e18_l1_intensity
    # E19 source-aware arousal loss
    python train_v2.py --lex_mode l1_intensity --source_aware --run_name e19_source_aware
    # E20 synthetic ranking-only augmentation
    python train_v2.py --lex_mode l1_intensity --source_aware --rank_aug data/train_aug.csv \
        --rank_lambda_a 0.1 --rank_lambda_v 0.05 --run_name e20_rank_aug
    # E21 dimension-specific attention
    python train_v2.py --pooling mean_cls_dim_attention --lex_mode l1_intensity --source_aware \
        --run_name e21_dim_attention

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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train import (VADataset, VARegressor, pick_device, read_csv, denorm,
                   LABEL_MIN, LABEL_MAX)
from lexicon import LexiconFeaturizer, FEATURE_DIM

HERE = os.path.dirname(__file__)

# ---------------- E19：source-aware loss weights ----------------
# granularity -> (valence_weight, arousal_weight)。依 next_experiments_plan §E19 第一版。
# train_aug 的 augment 預設 (0.25, 0.0)：raw 合成標籤 arousal 不進 SmoothL1（排序交給 E20）。
DEFAULT_SOURCE_WEIGHTS = {
    "sentence":   (1.0, 0.25),   # CVAS：句子級通用情緒，arousal domain mismatch 最大
    "text":       (1.0, 0.50),   # CVAT：篇章級，較接近但仍非反思
    "reflection": (1.0, 1.00),   # DSA-MST：形式最接近目標域
    "edu2021":    (1.0, 0.75),   # ROCLING-2021 教育反思
    "augment":    (0.25, 0.0),   # 合成資料 raw 標籤（若經 --extra_train 混入）
}


def parse_source_weights(spec):
    """解析 --source_weights "sentence=1:0.25,text=1:0.5" -> dict。"""
    weights = dict(DEFAULT_SOURCE_WEIGHTS)
    if spec:
        for part in spec.split(","):
            src, vw_aw = part.split("=")
            vw, aw = vw_aw.split(":")
            weights[src.strip()] = (float(vw), float(aw))
    return weights


class WeightedVADataset(VADataset):
    """VADataset + 每筆 (w_v, w_a) loss 權重（依 granularity 查表；缺欄位 = 1:1）。"""

    def __init__(self, rows, tokenizer, max_len, featurizer=None, source_weights=None):
        super().__init__(rows, tokenizer, max_len, has_label=True, featurizer=featurizer)
        sw = source_weights or {}
        self.weights = [sw.get((r.get("granularity") or "").strip(), (1.0, 1.0)) for r in rows]

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["loss_w"] = torch.tensor(self.weights[i], dtype=torch.float)
        return item


# ---------------- E21：dimension-specific attention pooling ----------------
class AttnPool(nn.Module):
    """單 query 加性 attention pooling（masked softmax）。"""

    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, hidden_states, attention_mask):
        s = self.score(torch.tanh(self.proj(hidden_states))).squeeze(-1)  # [B,T]
        s = s.masked_fill(attention_mask == 0, float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1)                          # [B,T,1]
        return (w * hidden_states).sum(1)                                  # [B,h]


class VARegressorDimAttn(nn.Module):
    """E21：mean + CLS + V/A 各自 attention pooling，V/A 分頭回歸。

    介面與 VARegressor 相同（forward kwargs、sigmoid 輸出 [B,2]），
    可直接沿用既有 train/predict 迴圈。head dropout 0.2 抑制 overfit。
    """

    def __init__(self, model_name, lex_dim=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.lex_dim = lex_dim
        h = self.encoder.config.hidden_size
        self.pool_v = AttnPool(h)
        self.pool_a = AttnPool(h)
        in_dim = 3 * h + lex_dim  # [mean, cls, dim-specific attn] + lexicon
        self.head_v = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_dim, 1))
        self.head_a = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_dim, 1))

    def forward(self, input_ids, attention_mask, token_type_ids=None, lex_feats=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        mean_pool = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        cls_pool = hs[:, 0]
        v_in = [mean_pool, cls_pool, self.pool_v(hs, attention_mask)]
        a_in = [mean_pool, cls_pool, self.pool_a(hs, attention_mask)]
        if self.lex_dim and lex_feats is not None:
            v_in.append(lex_feats)
            a_in.append(lex_feats)
        v = self.head_v(torch.cat(v_in, dim=-1))
        a = self.head_a(torch.cat(a_in, dim=-1))
        return torch.sigmoid(torch.cat([v, a], dim=-1))  # [B,2] in (0,1)


# ---------------- E20：synthetic ranking-only loss ----------------
def pairwise_rank_loss(pred, gold, margin, min_gap):
    """batch 內 pairwise hinge：gold 差距 >= min_gap 的 pair 要求 pred 同向拉開 margin。

    pred / gold 皆為 [B]（正規化 0-1 尺度）。無有效 pair 時回傳 0。
    """
    dg = gold.unsqueeze(1) - gold.unsqueeze(0)   # [B,B] gold_i - gold_j
    dp = pred.unsqueeze(1) - pred.unsqueeze(0)
    pair_mask = dg >= min_gap                     # i 應該 > j
    if not pair_mask.any():
        return pred.new_zeros(())
    return torch.relu(margin - dp[pair_mask]).mean()


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
    if args.lex_mode == "l1_intensity":
        from lexicon_intensity import L1IntensityFeaturizer, FEATURE_DIM_INTENSITY
        return L1IntensityFeaturizer(), FEATURE_DIM_INTENSITY
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
    ap.add_argument("--lex_mode", choices=["none", "l1", "l1l2", "l1_intensity"], default="l1")
    ap.add_argument("--l2_pkl", default=os.path.join(HERE, "outputs", "l2_word_va.pkl"))
    # E19：source-aware loss
    ap.add_argument("--source_aware", action="store_true",
                    help="依 granularity 對 V/A loss 加權（見 DEFAULT_SOURCE_WEIGHTS）")
    ap.add_argument("--source_weights", default=None,
                    help='覆寫來源權重，如 "sentence=1:0.25,text=1:0.5"（V:A）')
    # E20：synthetic ranking-only augmentation
    ap.add_argument("--rank_aug", default=None,
                    help="合成資料 csv（如 data/train_aug.csv），只做 ranking 約束，不進 SmoothL1")
    ap.add_argument("--rank_lambda_a", type=float, default=0.1)
    ap.add_argument("--rank_lambda_v", type=float, default=0.05)
    ap.add_argument("--rank_margin", type=float, default=0.125,
                    help="hinge margin（正規化 0-1 尺度；0.125 = 1 分）")
    ap.add_argument("--rank_min_gap", type=float, default=0.1875,
                    help="建 pair 的最小 gold 差距（0.1875 = 1.5 分，可分開高低 bin）")
    ap.add_argument("--rank_batch", type=int, default=16)
    # E21：pooling 架構
    ap.add_argument("--pooling", choices=["mean", "mean_cls_dim_attention"], default="mean")
    args = ap.parse_args()

    model_short = args.model.rstrip("/").split("/")[-1]
    run = args.run_name or f"{model_short}_s{args.seed}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"run={run} | device={device} | model={args.model} | lex={args.lex_mode} "
          f"| aw={args.arousal_weight} pcc_w={args.pcc_weight} | src_aware={args.source_aware} "
          f"| rank_aug={args.rank_aug} | pooling={args.pooling}")
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

    if args.source_aware:
        src_w = parse_source_weights(args.source_weights)
        print("source weights (V:A):", {k: v for k, v in src_w.items()})
        train_ds = WeightedVADataset(train_rows, tok, args.max_len,
                                     featurizer=featurizer, source_weights=src_w)
    else:
        train_ds = VADataset(train_rows, tok, args.max_len, featurizer=featurizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(VADataset(dev_rows, tok, args.max_len, featurizer=featurizer),
                            batch_size=args.batch_size)

    # E20：ranking-only 資料（獨立 loader，每步抽一小批建 pairwise 約束）
    rank_iter = None
    if args.rank_aug:
        rank_rows = read_csv(args.rank_aug)
        print(f"rank_aug: {len(rank_rows)} rows ({args.rank_aug}) "
              f"| λ_A={args.rank_lambda_a} λ_V={args.rank_lambda_v} "
              f"margin={args.rank_margin} min_gap={args.rank_min_gap}")
        rank_loader = DataLoader(
            VADataset(rank_rows, tok, args.max_len, featurizer=featurizer),
            batch_size=args.rank_batch, shuffle=True, drop_last=True)

        def _cycle(loader):
            while True:
                for b in loader:
                    yield b
        rank_iter = _cycle(rank_loader)

    model_cls = VARegressorDimAttn if args.pooling == "mean_cls_dim_attention" else VARegressor
    model = model_cls(args.model, lex_dim=lex_dim).to(device)
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
            loss_w = batch.pop("loss_w", None)  # E19：[B,2] 來源權重（無 --source_aware 時不存在）
            out = model(**{k: v.to(device) for k, v in batch.items()})
            el = loss_fn(out, labels)  # [B,2]
            if loss_w is not None:
                el = el * loss_w.to(device)
            # aw=1, pcc_w=0, 無 loss_w 時 = SmoothL1 全元素平均，與 train.py 完全一致
            loss = (el[:, 0].mean() + args.arousal_weight * el[:, 1].mean()) / (1.0 + args.arousal_weight)
            if args.pcc_weight > 0 and labels.size(0) >= 4:
                loss = loss + args.pcc_weight * pearson_loss(out[:, 1], labels[:, 1])
            if rank_iter is not None:
                rb = next(rank_iter)
                r_labels = rb.pop("labels").to(device)
                rb.pop("loss_w", None)
                r_out = model(**{k: v.to(device) for k, v in rb.items()})
                loss = loss \
                    + args.rank_lambda_a * pairwise_rank_loss(
                        r_out[:, 1], r_labels[:, 1], args.rank_margin, args.rank_min_gap) \
                    + args.rank_lambda_v * pairwise_rank_loss(
                        r_out[:, 0], r_labels[:, 0], args.rank_margin, args.rank_min_gap)
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
