"""export_l1pp_keywords.py — 把 L1++ 的 14 份 cue list 匯出成公開的 JSON。

ROCLING-2026 reviewer 2 要求「完整 keyword list 要放在 supplementary 或公開 repo」。
本檔直接從 `lexicon_intensity.py` 讀清單（不是手抄），所以 JSON 永遠跟實際跑實驗
的程式一致；論文引用的每類條數也由這裡印出來。

用法：
    python export_l1pp_keywords.py            # 寫入 resources/l1_plus_plus_keywords.json
    python export_l1pp_keywords.py --check    # 只檢查檔案是否與程式碼同步（CI 用）
"""
import argparse
import json
import os

import lexicon_intensity as li

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "resources", "l1_plus_plus_keywords.json")

# 論文 Table 2 的類別名 -> lexicon_intensity.py 裡的清單
CATEGORIES = [
    ("degree_adverb", "DEGREE_ADVERBS"),
    ("bodily_reaction", "BODY_REACTION"),
    ("sleep_disturbance", "SLEEP_DISTURBANCE"),
    ("anxiety_fear", "ANXIETY_FEAR"),
    ("pressure_event", "PRESSURE_EVENT"),
    ("low_activation", "LOW_AROUSAL"),
    ("negation", "NEGATION"),
    ("contrast", "CONTRAST"),
    ("language", "CUE_LANGUAGE"),
    ("documentation", "CUE_DOCUMENT"),
    ("work", "CUE_WORK"),
    ("family_separation", "CUE_FAMILY"),
    ("cultural_adaptation", "CUE_CULTURE"),
    ("financial", "CUE_FINANCIAL"),
]


def build():
    return {name: list(getattr(li, attr)) for name, attr in CATEGORIES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只比對，不覆寫")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    data = build()
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n"

    if args.check:
        current = open(args.out, encoding="utf-8").read() if os.path.exists(args.out) else ""
        if current != text:
            raise SystemExit(f"{args.out} 與 lexicon_intensity.py 不同步，請重跑本腳本")
        print(f"{args.out} 與程式碼一致")
    else:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")

    total = sum(len(v) for v in data.values())
    width = max(len(k) for k in data)
    for name, words in data.items():
        print(f"  {name:<{width}}  n={len(words):3d}  e.g. {'、'.join(words[:3])}")
    print(f"  {'TOTAL':<{width}}  n={total:3d}  ({len(data)} categories, "
          f"{min(len(v) for v in data.values())}-{max(len(v) for v in data.values())} per category)")


if __name__ == "__main__":
    main()
