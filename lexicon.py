"""
L1 — 情感詞典特徵 (Lexicon Features)

用 Chinese EmoBank 的 CVAW(字) + CVAP(詞) 當現成 VA 詞典，對一段文本抽出
「情緒詞 VA 聚合特徵」，之後 concat 進 BERT embedding 一起回歸（架構方案四：
lexicon + contextual 融合）。重點是給模型一個顯性的「這篇有哪些高/低喚醒詞」訊號，
特別針對 arousal。

可獨立執行驗證：
    python lexicon.py
會印出詞典大小，以及幾個範例句子的特徵向量。
"""
import os
import csv
import jieba

HERE = os.path.dirname(__file__)
LABEL_MIN, LABEL_MAX = 1.0, 9.0


def emobank_file(fname, nested_subdir):
    """優先讀內建 external/emobank/（Colab 自包含），找不到再退回 ../ChineseEmoBank。"""
    for cand in (os.path.join(HERE, "external", "emobank", fname),
                 os.path.join(HERE, "..", "ChineseEmoBank", nested_subdir, fname)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"找不到 {fname}（external/emobank/ 或 ../ChineseEmoBank/{nested_subdir}/）")

# 特徵維度（順序固定，train.py 會依賴）
FEATURE_NAMES = [
    "coverage",      # 命中情緒詞數 / 斷詞數
    "count",         # 命中數 (capped 20 後正規化)
    "v_mean", "v_max", "v_min", "v_std",
    "a_mean", "a_max", "a_min", "a_std",
]
FEATURE_DIM = len(FEATURE_NAMES)


def _norm(x):
    return (x - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)


def _read_tsv(path, term_field):
    rows = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            term = (r.get(term_field) or "").strip()
            if not term or "�" in term:
                continue
            try:
                rows[term] = (_norm(float(r["Valence_Mean"])), _norm(float(r["Arousal_Mean"])))
            except (KeyError, ValueError):
                continue
    return rows


class LexiconFeaturizer:
    """載入 CVAW+CVAP -> {term: (v_norm, a_norm)}，對文本抽 FEATURE_DIM 維特徵。"""

    def __init__(self, use_cvaw=True, use_cvap=True):
        self.lex = {}
        if use_cvaw:
            self.lex.update(_read_tsv(emobank_file("CVAW_all_SD.csv", "CVAW_SD"), "Word"))
        if use_cvap:
            self.lex.update(_read_tsv(emobank_file("CVAP_all_SD.csv", "CVAP_SD"), "Phrase"))
        # 把詞典詞加進 jieba，提升多字情緒詞的斷詞命中率
        for term in self.lex:
            if len(term) > 1:
                jieba.add_word(term, freq=100)
        self._max_term_len = max((len(t) for t in self.lex), default=1)

    def __len__(self):
        return len(self.lex)

    def _match(self, text):
        """斷詞命中 + 額外做 n-gram 子字串掃描，盡量抓到詞典裡的詞。"""
        hits = []
        tokens = list(jieba.cut(text))
        for tok in tokens:
            if tok in self.lex:
                hits.append(self.lex[tok])
        return hits, len(tokens)

    def featurize(self, text):
        """回傳 FEATURE_DIM 維 list[float]，皆在 [0,1]（沒命中時 stat 設中性 0.5）。"""
        hits, n_tok = self._match(text or "")
        n = len(hits)
        if n == 0:
            # 無情緒詞：coverage/count=0，VA 統計給中性 0.5、std=0
            return [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0]
        vs = [h[0] for h in hits]
        as_ = [h[1] for h in hits]

        def stats(xs):
            m = sum(xs) / len(xs)
            var = sum((x - m) ** 2 for x in xs) / len(xs)
            return m, max(xs), min(xs), var ** 0.5

        v_mean, v_max, v_min, v_std = stats(vs)
        a_mean, a_max, a_min, a_std = stats(as_)
        coverage = n / max(n_tok, 1)
        count = min(n, 20) / 20.0
        return [coverage, count, v_mean, v_max, v_min, v_std,
                a_mean, a_max, a_min, a_std]


def _demo():
    fz = LexiconFeaturizer()
    print(f"詞典大小: {len(fz)} 個情緒詞 (CVAW+CVAP)")
    print(f"特徵維度: {FEATURE_DIM}  {FEATURE_NAMES}\n")
    examples = [
        "今天上課很有趣，心情很好",
        "我覺得非常焦慮又緊張，快崩潰了",
        "目前還在公司正職上班",
        "他先在家庭醫生的協助下就診",
    ]
    for t in examples:
        f = fz.featurize(t)
        print(f"「{t}」")
        print("  " + "  ".join(f"{n}={v:.2f}" for n, v in zip(FEATURE_NAMES, f)))


if __name__ == "__main__":
    _demo()
