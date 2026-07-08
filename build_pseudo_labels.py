"""
E13 — Multi-teacher 偽標精修（CYUT 冠軍流程）

實驗 9/9b 的診斷：合成文本有效（A_PCC 0.426→0.46），爛的是 bin 中心弱標籤 + 校準偏移，
且「整批收縮」救不回。本檔改成**逐篇**用訓好的 teacher 重標，再做兩層清理：

1. teacher disagreement 過濾：多 teacher 預測 std 過大（V_sd > thr_v 或 A_sd > thr_a）的樣本丟掉。
2. 每 bin 離群移除：按原始弱標籤歸回 5 個生成 bin，teacher 重標值落在該 bin
   mean ± sd_k * SD 之外的丟掉（V、A 都檢查）。

新標籤 = teacher 預測平均（真實尺度、逐篇），可用 --blend 與原 bin 中心混合
（new = blend*bin中心 + (1-blend)*teacher平均；預設 0 = 純 teacher）。

用法（teacher 格式 model_name=ckpt，可重複；ckpt 是 train_v2.py 存的 {run}_best.pt
或 train.py 的 best_model.pt，皆為含 L1 的 VARegressor state_dict）：
    python build_pseudo_labels.py \
        --teacher hfl/chinese-macbert-base=outputs/macbert_s42_best.pt \
        --teacher hfl/chinese-roberta-wwm-ext=outputs/roberta_s42_best.pt \
        --out data/train_aug_pseudo.csv

之後：
    python train_v2.py --extra_train data/train_aug_pseudo.csv --run_name macbert_pseudo_s42

⚠️ 實驗 9 的教訓：增強資料與 dev 同風格，**dev 分數對增強實驗不可信**，最終要看官方提交。
"""
import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train import VADataset, VARegressor, pick_device, read_csv, denorm
from lexicon import LexiconFeaturizer, FEATURE_DIM

HERE = os.path.dirname(__file__)
# augment_generate.py 的 5 個生成 bin（V 中心, A 中心）
BIN_CENTERS = [(3.0, 7.5), (5.0, 7.5), (7.0, 7.5), (3.0, 2.5), (7.0, 2.5)]


@torch.no_grad()
def teacher_predict(model_name, ckpt, rows, max_len, batch_size, device, featurizer):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = VARegressor(model_name, lex_dim=FEATURE_DIM if featurizer else 0)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    loader = DataLoader(VADataset(rows, tok, max_len, has_label=False, featurizer=featurizer),
                        batch_size=batch_size)
    P = []
    for batch in loader:
        out = model(**{k: v.to(device) for k, v in batch.items()})
        P.append(out.cpu().numpy())
    del model
    return denorm(np.vstack(P))  # [N,2] 1-9 尺度


def assign_bin(v, a):
    d = [(v - bv) ** 2 + (a - ba) ** 2 for bv, ba in BIN_CENTERS]
    return int(np.argmin(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default=os.path.join(HERE, "data", "train_aug.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "train_aug_pseudo.csv"))
    ap.add_argument("--teacher", action="append", required=True,
                    help="model_name=ckpt_path，可重複指定多個 teacher")
    ap.add_argument("--thr_v", type=float, default=0.60, help="teacher V 分歧上限（僅多 teacher 時檢查）")
    ap.add_argument("--thr_a", type=float, default=0.75, help="teacher A 分歧上限")
    ap.add_argument("--sd_k", type=float, default=1.5, help="每 bin mean ± k*SD 離群移除")
    ap.add_argument("--blend", type=float, default=0.0,
                    help="new = blend*bin中心 + (1-blend)*teacher平均")
    ap.add_argument("--no_lexicon", action="store_true", help="teacher 訓練時未用 L1 才開")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = pick_device()
    rows = read_csv(args.aug)
    print(f"device={device} | 增強資料 {len(rows)} 篇 | teachers={len(args.teacher)}")
    featurizer = None if args.no_lexicon else LexiconFeaturizer()

    all_preds = []  # [T, N, 2]
    for spec in args.teacher:
        model_name, ckpt = spec.split("=", 1)
        print(f"teacher 推論中: {model_name}  ({ckpt})")
        all_preds.append(teacher_predict(model_name, ckpt, rows, args.max_len,
                                         args.batch_size, device, featurizer))
    all_preds = np.stack(all_preds)               # [T,N,2]
    mean = all_preds.mean(axis=0)                 # [N,2]
    sd = all_preds.std(axis=0)                    # [N,2]

    orig = np.array([[float(r["valence"]), float(r["arousal"])] for r in rows])
    bins = np.array([assign_bin(v, a) for v, a in orig])

    keep = np.ones(len(rows), dtype=bool)
    # 1) teacher disagreement
    if len(args.teacher) >= 2:
        dis = (sd[:, 0] > args.thr_v) | (sd[:, 1] > args.thr_a)
        keep &= ~dis
        print(f"disagreement 過濾: 移除 {int(dis.sum())} 篇 (V_sd>{args.thr_v} 或 A_sd>{args.thr_a})")
    # 2) 每 bin mean ± k*SD 離群移除（用 teacher 重標值算）
    for b in range(len(BIN_CENTERS)):
        idx = np.where((bins == b) & keep)[0]
        if len(idx) < 5:
            continue
        for j in range(2):
            m, s = mean[idx, j].mean(), mean[idx, j].std()
            out_mask = np.abs(mean[idx, j] - m) > args.sd_k * s
            keep[idx[out_mask]] = False
        print(f"bin{b+1} (V={BIN_CENTERS[b][0]},A={BIN_CENTERS[b][1]}): "
              f"{len(idx)} -> {int(keep[idx].sum())} 篇")

    new_labels = mean.copy()
    if args.blend > 0:
        centers = np.array([BIN_CENTERS[b] for b in bins])
        new_labels = args.blend * centers + (1 - args.blend) * mean
    new_labels = np.clip(new_labels, 1.0, 9.0)

    kept = int(keep.sum())
    print(f"\n共保留 {kept}/{len(rows)} 篇")
    print(f"原標籤     V mean/std={orig[:,0].mean():.2f}/{orig[:,0].std():.2f} "
          f"A mean/std={orig[:,1].mean():.2f}/{orig[:,1].std():.2f}")
    ok = new_labels[keep]
    print(f"偽標(保留) V mean/std={ok[:,0].mean():.2f}/{ok[:,0].std():.2f} "
          f"A mean/std={ok[:,1].mean():.2f}/{ok[:,1].std():.2f}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "granularity", "text", "valence", "arousal"])
        for i, r in enumerate(rows):
            if keep[i]:
                w.writerow([r["id"], r.get("granularity", "augment"), r["text"],
                            f"{new_labels[i,0]:.2f}", f"{new_labels[i,1]:.2f}"])
    print(f"偽標資料 -> {args.out}")


if __name__ == "__main__":
    main()
