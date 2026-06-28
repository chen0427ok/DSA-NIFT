"""
DAPT — Domain-Adaptive Pretraining (領域適應續訓)

在「新住民文本」上做 MLM 續訓，縮短 EmoBank 訓練域 ↔ 新住民目標域的落差，
主要目的是把 arousal 的跨域表現拉起來。

流程：
    1) python dapt.py                      # 在 val(+test) 文本上做 MLM 續訓
       -> 產出 outputs/dapt_macbert/       (HF 格式，含 encoder + tokenizer)
    2) python train.py --model outputs/dapt_macbert   # 從續訓後權重接著微調

語料來源：
    - data/val_unlabeled.csv  (官方 validation 200 篇)
    - 若官方釋出 test，放成 csv (含 text 欄) 用 --extra 加進來，語料越多越好

注意：目標域語料偏小(200篇)，所以用較多 epoch、較高 mask 比例讓它「讀熟」這個語域。
"""
import os
import csv
import argparse
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, get_linear_schedule_with_warmup)

HERE = os.path.dirname(__file__)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_texts(path, col_candidates=("text", "Text")):
    texts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        col = next((c for c in col_candidates if c in reader.fieldnames), None)
        if col is None:
            raise ValueError(f"{path} 找不到文本欄位 {col_candidates}，實際欄位={reader.fieldnames}")
        for r in reader:
            t = (r[col] or "").strip()
            if t:
                texts.append(t)
    return texts


class MLMDataset(Dataset):
    """把每篇文本 tokenize（截斷到 max_len）；masking 交給 collator 動態處理，
    每個 epoch 看到的 mask 位置不同，等於資料增強，對小語料特別有幫助。"""
    def __init__(self, texts, tokenizer, max_len):
        self.examples = [tokenizer(t, truncation=True, max_length=max_len)["input_ids"]
                         for t in texts]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {"input_ids": self.examples[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "dapt_macbert"))
    ap.add_argument("--extra", default=None, help="額外語料 csv（如官方 test，含 text 欄）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device = {device} | base model = {args.model}")

    # 蒐集目標域語料
    texts = read_texts(os.path.join(args.data_dir, "val_unlabeled.csv"))
    if args.extra and os.path.exists(args.extra):
        extra = read_texts(args.extra)
        texts += extra
        print(f"額外語料 +{len(extra)} 篇")
    print(f"DAPT 語料合計 {len(texts)} 篇")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=args.mlm_prob)

    ds = MLMDataset(texts, tok, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    model.train()
    for ep in range(1, args.epochs + 1):
        running, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item(); n += 1
        avg = running / max(n, 1)
        if ep % 5 == 0 or ep == 1:
            print(f"[dapt epoch {ep:2d}] mlm_loss={avg:.4f}  ppl={math.exp(min(avg,20)):.2f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\n續訓完成 -> {args.out}")
    print(f"下一步:  python train.py --model {args.out}")


if __name__ == "__main__":
    main()
