"""
ROCLING 2026 Dimensional Sentiment Analysis — 資料準備腳本

把 Chinese EmoBank + ROCLING-2025 DSA-MST 整理成統一的 (text, valence, arousal)
regression 格式，並切出 train / dev。

資料來源：
- Chinese EmoBank：CVAS(句) + CVAT(篇)，--use_all 可加 CVAW/CVAP 字詞層級。
- DSA-MST（去年同型醫療反思任務，2535 篇含 VA 標籤）：register / 文長 / 繁中 / VA尺度
  都和今年的新住民文本一致，是最貼近目標域的監督資料。預設啟用，--no_dsamst 可關。

dev 策略（重要）：預設 dev 從 DSA-MST 切（--dev_from reflection），因為它最接近目標域，
dev 分數才能真正預測新住民驗證分數（避免拿 EmoBank 當 dev 高估表現）。

在 MacBook 本機跑即可（純 CPU、很快）：
    python prepare_data.py
產出：
    data/train.csv  data/dev.csv   欄位: id, granularity, text, valence, arousal
    data/val_unlabeled.csv         官方驗證集 (id, text)，給推論用
"""
import csv
import os
import random
import argparse

ROOT = os.path.dirname(__file__)
EMOBANK = os.path.join(ROOT, "..", "ChineseEmoBank")
VAL_CSV = os.path.join(ROOT, "..", "DSANIDF_ValidationSet.csv")
DSAMST_DIR = os.path.join(ROOT, "external", "DSA-MST")
EDU2021_CSV = os.path.join(ROOT, "external", "ROCLING-2021", "edu_va.csv")
OUT_DIR = os.path.join(ROOT, "data")


def read_tsv(path, text_field):
    """EmoBank 檔案是 tab 分隔；CVAT 有 1 行壞 byte，用 errors=replace 容錯。"""
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            text = (r.get(text_field) or "").strip()
            if not text or "�" in text:  # 跳過空白或解碼壞掉的列
                continue
            try:
                v = float(r["Valence_Mean"])
                a = float(r["Arousal_Mean"])
            except (KeyError, ValueError):
                continue
            rows.append((text, v, a))
    return rows


def read_dsamst():
    """DSA-MST 是逗號 CSV，欄位 ID,Text,Valence,Arousal（已是 1-9 文檔級標籤）。"""
    rows = []
    for fn in ["DSAMST-ValidationSet_ans.csv", "DSAMST-TestSet_ans.csv"]:
        path = os.path.join(DSAMST_DIR, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for r in csv.DictReader(f):
                text = (r.get("Text") or "").strip()
                if not text or "�" in text:
                    continue
                try:
                    rows.append((text, float(r["Valence"]), float(r["Arousal"])))
                except (KeyError, ValueError):
                    continue
    return rows


def read_edu2021():
    """ROCLING-2021 教育反思短文，欄位 text,valence,arousal（已是 1-9）。"""
    rows = []
    if not os.path.exists(EDU2021_CSV):
        return rows
    with open(EDU2021_CSV, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            text = (r.get("text") or "").strip()
            if not text:
                continue
            try:
                rows.append((text, float(r["valence"]), float(r["arousal"])))
            except (KeyError, ValueError):
                continue
    return rows


def load_all(use_all, use_dsamst, use_edu2021):
    sources = {
        "sentence": (os.path.join(EMOBANK, "CVAS_SD", "CVAS_all.csv"), "Text"),
        "text":     (os.path.join(EMOBANK, "CVAT_SD", "CVAT_all_SD.csv"), "Text"),
    }
    if use_all:
        sources["word"]   = (os.path.join(EMOBANK, "CVAW_SD", "CVAW_all_SD.csv"), "Word")
        sources["phrase"] = (os.path.join(EMOBANK, "CVAP_SD", "CVAP_all_SD.csv"), "Phrase")

    data = []
    for gran, (path, field) in sources.items():
        rows = read_tsv(path, field)
        print(f"  {gran:8s} {len(rows):5d}  <- {os.path.basename(path)}")
        for i, (t, v, a) in enumerate(rows):
            data.append({"id": f"{gran}_{i:05d}", "granularity": gran,
                         "text": t, "valence": v, "arousal": a})

    if use_dsamst:
        refl = read_dsamst()
        if refl:
            print(f"  reflection {len(refl):5d}  <- DSA-MST (val+test ans)")
            for i, (t, v, a) in enumerate(refl):
                data.append({"id": f"reflection_{i:05d}", "granularity": "reflection",
                             "text": t, "valence": v, "arousal": a})
        else:
            print("  [警告] 找不到 DSA-MST 資料，請先下載到 external/DSA-MST/（見 download_external.sh）")

    if use_edu2021:
        edu = read_edu2021()
        if edu:
            print(f"  edu2021    {len(edu):5d}  <- ROCLING-2021 教育反思短文")
            for i, (t, v, a) in enumerate(edu):
                data.append({"id": f"edu2021_{i:05d}", "granularity": "edu2021",
                             "text": t, "valence": v, "arousal": a})
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_all", action="store_true",
                    help="連 CVAW/CVAP 字詞層級也一起用")
    ap.add_argument("--no_dsamst", action="store_true",
                    help="不要使用 DSA-MST 反思語料")
    ap.add_argument("--no_edu2021", action="store_true",
                    help="不要使用 ROCLING-2021 教育反思語料")
    ap.add_argument("--dev_from", choices=["reflection", "stratified"], default="reflection",
                    help="dev 來源：reflection=從 DSA-MST 切(最貼近目標域,推薦)；stratified=各來源等比例")
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("讀取資料來源:")
    data = load_all(args.use_all, not args.no_dsamst, not args.no_edu2021)
    print(f"合計 {len(data)} 筆")

    random.seed(args.seed)
    by_gran = {}
    for d in data:
        by_gran.setdefault(d["granularity"], []).append(d)

    train, dev = [], []
    if args.dev_from == "reflection" and "reflection" in by_gran:
        # dev 全部從 DSA-MST 反思文本切，當作目標域(新住民)的代理驗證集
        refl = by_gran["reflection"][:]
        random.shuffle(refl)
        k = int(len(refl) * args.dev_ratio)
        dev = refl[:k]
        train = refl[k:]
        for gran, rows in by_gran.items():
            if gran != "reflection":
                train += rows
    else:
        # 各來源等比例切 dev
        for gran, rows in by_gran.items():
            random.shuffle(rows)
            k = int(len(rows) * args.dev_ratio)
            dev += rows[:k]
            train += rows[k:]
    random.shuffle(train)
    random.shuffle(dev)

    def write(path, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "granularity", "text", "valence", "arousal"])
            w.writeheader()
            w.writerows(rows)

    write(os.path.join(OUT_DIR, "train.csv"), train)
    write(os.path.join(OUT_DIR, "dev.csv"), dev)
    print(f"train={len(train)}  dev={len(dev)}")

    # 官方驗證集 -> 只留 id, text
    with open(VAL_CSV, encoding="utf-8", errors="replace") as f:
        val = [{"id": r["ID"], "text": r["Text"].strip()} for r in csv.DictReader(f)]
    with open(os.path.join(OUT_DIR, "val_unlabeled.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text"])
        w.writeheader()
        w.writerows(val)
    print(f"val_unlabeled={len(val)}")
    print(f"\n完成，輸出在 {OUT_DIR}/")


if __name__ == "__main__":
    main()
