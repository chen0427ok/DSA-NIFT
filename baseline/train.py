"""
ROCLING 2026 DSA baseline — 中文 encoder + 雙回歸頭 (valence / arousal) 端到端微調。

設計重點：
- 標籤 1-9 先正規化到 [0,1]，模型輸出 sigmoid，推論時再反轉回 1-9（穩定、好收斂）。
- Loss = SmoothL1（對離群值較穩），同時優化 valence + arousal。
- 評估同時印 MAE 與 PCC，對齊官方四個指標。
- 自動偵測裝置：Colab 用 cuda、MacBook 用 mps、否則 cpu。

用法：
    # 訓練 + 在 dev 評估 + 對官方驗證集產生 submission.csv
    python train.py --epochs 4 --batch_size 16 --model hfl/chinese-macbert-base

產出：
    outputs/submission.csv      欄位: ID, valence_rating, arousal_rating  (官方格式)
    outputs/best_model/         最佳權重
"""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from lexicon import LexiconFeaturizer, FEATURE_DIM

HERE = os.path.dirname(__file__)
LABEL_MIN, LABEL_MAX = 1.0, 9.0


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def norm(y):    # 1-9 -> 0-1
    return (y - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)


def denorm(y):  # 0-1 -> 1-9
    return y * (LABEL_MAX - LABEL_MIN) + LABEL_MIN


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class VADataset(Dataset):
    def __init__(self, rows, tokenizer, max_len, has_label=True, featurizer=None):
        self.rows, self.tok, self.max_len, self.has_label = rows, tokenizer, max_len, has_label
        self.featurizer = featurizer  # L1: 提供時加入詞典特徵

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.featurizer is not None:
            item["lex_feats"] = torch.tensor(self.featurizer.featurize(r["text"]), dtype=torch.float)
        if self.has_label:
            item["labels"] = torch.tensor(
                [norm(float(r["valence"])), norm(float(r["arousal"]))], dtype=torch.float)
        return item


class VARegressor(nn.Module):
    def __init__(self, model_name, lex_dim=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.lex_dim = lex_dim  # L1: >0 表示把詞典特徵 concat 進回歸頭
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h + lex_dim, 2))

    def forward(self, input_ids, attention_mask, token_type_ids=None, lex_feats=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        # mean pooling over tokens (用 attention mask 加權)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.lex_dim and lex_feats is not None:
            pooled = torch.cat([pooled, lex_feats], dim=-1)  # [B, h+lex_dim]
        return torch.sigmoid(self.head(pooled))  # [B,2] in (0,1)


def pcc(pred, gold):
    pred, gold = np.asarray(pred), np.asarray(gold)
    if pred.std() < 1e-8 or gold.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, gold)[0, 1])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, G = [], []
    for batch in loader:
        labels = batch.pop("labels")
        out = model(**{k: v.to(device) for k, v in batch.items()})
        P.append(out.cpu().numpy())
        G.append(labels.numpy())
    P, G = denorm(np.vstack(P)), denorm(np.vstack(G))
    res = {}
    for j, name in enumerate(["valence", "arousal"]):
        res[f"{name}_MAE"] = float(np.mean(np.abs(P[:, j] - G[:, j])))
        res[f"{name}_PCC"] = pcc(P[:, j], G[:, j])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_lexicon", action="store_true", help="關閉 L1 詞典特徵融合（消融用）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"device = {device} | model = {args.model}")
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    train_rows = read_csv(os.path.join(args.data_dir, "train.csv"))
    dev_rows = read_csv(os.path.join(args.data_dir, "dev.csv"))
    val_rows = read_csv(os.path.join(args.data_dir, "val_unlabeled.csv"))

    # L1: 詞典特徵融合（--no_lexicon 可關閉）
    featurizer = None if args.no_lexicon else LexiconFeaturizer()
    lex_dim = 0 if featurizer is None else FEATURE_DIM
    print(f"lexicon fusion = {not args.no_lexicon} (lex_dim={lex_dim})")

    train_loader = DataLoader(VADataset(train_rows, tok, args.max_len, featurizer=featurizer),
                              batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(VADataset(dev_rows, tok, args.max_len, featurizer=featurizer),
                            batch_size=args.batch_size)

    model = VARegressor(args.model, lex_dim=lex_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    loss_fn = nn.SmoothL1Loss()

    best_score, best_state = -1e9, None
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            out = model(**{k: v.to(device) for k, v in batch.items()})
            loss = loss_fn(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"  ep{ep} step{step}/{len(train_loader)} loss={running/step:.4f}")
        res = evaluate(model, dev_loader, device)
        # 綜合分數：PCC 高、MAE 低 -> 用 (PCC平均 - MAE平均) 當 early-stop 依據
        score = (res["valence_PCC"] + res["arousal_PCC"]) / 2 - (res["valence_MAE"] + res["arousal_MAE"]) / 2
        print(f"[epoch {ep}] " + " ".join(f"{k}={v:.4f}" for k, v in res.items()) + f"  score={score:.4f}")
        if score > best_score:
            best_score, best_state = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(args.out_dir, "best_model.pt"))
    print(f"best dev score = {best_score:.4f}")

    # 推論官方驗證集 -> submission.csv
    val_loader = DataLoader(VADataset(val_rows, tok, args.max_len, has_label=False, featurizer=featurizer),
                            batch_size=args.batch_size)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            out = model(**{k: v.to(device) for k, v in batch.items()})
            preds.append(out.cpu().numpy())
    preds = denorm(np.vstack(preds))
    preds = np.clip(preds, LABEL_MIN, LABEL_MAX)
    sub_path = os.path.join(args.out_dir, "submission.csv")
    with open(sub_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Valence", "Arousal"])
        for r, (v, a) in zip(val_rows, preds):
            w.writerow([r["id"], f"{v:.4f}", f"{a:.4f}"])
    print(f"submission -> {sub_path}  ({len(preds)} rows)")


if __name__ == "__main__":
    main()
