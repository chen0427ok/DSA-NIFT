"""
E16 — L1 + L2 融合特徵 (L1L2Featurizer)

L1 詞典（CVAW+CVAP 7,761 詞）覆蓋有限；L2（outputs/l2_word_va.pkl，FastText char n-gram
+ SVR）能對**任何 OOV 詞**估 VA。本檔在 L1 的 10 維之外，對「jieba 斷詞後不在 L1 詞典、
長度 >= 2 的內容詞」用 L2 估 VA，再加 6 維聚合特徵 → 共 16 維，餵 train_v2.py --lex_mode l1l2。

注意：
- 需在 .venv 跑（anaconda base 載 pkl 會炸 scipy/gensim 版本衝突，見 handover.md）。
- pkl ~814MB，載入需十幾秒；SVR 預測結果按詞快取，整個 train set 只算一次。

可獨立驗證：
    .venv/bin/python lexicon_l2.py
"""
import os
import re
import pickle
import numpy as np
import jieba
from lexicon import LexiconFeaturizer, FEATURE_NAMES, LABEL_MIN, LABEL_MAX

HERE = os.path.dirname(__file__)
DEFAULT_L2_PKL = os.path.join(HERE, "outputs", "l2_word_va.pkl")

EXTRA_NAMES = ["l2_coverage", "l2_v_mean", "l2_a_mean", "l2_a_max", "l2_a_min", "l2_a_std"]
FEATURE_NAMES_L2 = FEATURE_NAMES + EXTRA_NAMES
FEATURE_DIM_L2 = len(FEATURE_NAMES_L2)

_HAS_CJK = re.compile(r"[一-鿿]")


def _norm(x):
    return (x - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)


class L1L2Featurizer(LexiconFeaturizer):
    """L1 的 10 維 + L2 對 OOV 內容詞估 VA 的 6 維聚合。"""

    def __init__(self, l2_pkl=DEFAULT_L2_PKL, **kw):
        super().__init__(**kw)
        with open(l2_pkl, "rb") as f:
            d = pickle.load(f)
        self.ft, self.svr_v, self.svr_a = d["fasttext"], d["svr_v"], d["svr_a"]
        self._cache = {}  # word -> (v_norm, a_norm)
        self.dim = FEATURE_DIM_L2

    def _oov_tokens(self, text):
        """L1 沒命中、含中文、長度 >= 2 的詞才交給 L2（避免功能詞噪音）。"""
        return [t for t in jieba.cut(text or "")
                if len(t) >= 2 and t not in self.lex and _HAS_CJK.search(t)]

    def _l2_va(self, words):
        todo = [w for w in words if w not in self._cache]
        if todo:
            X = np.stack([self.ft.wv[w] for w in todo])
            vs = np.clip(self.svr_v.predict(X), LABEL_MIN, LABEL_MAX)
            as_ = np.clip(self.svr_a.predict(X), LABEL_MIN, LABEL_MAX)
            for w, v, a in zip(todo, vs, as_):
                self._cache[w] = (_norm(float(v)), _norm(float(a)))
        return [self._cache[w] for w in words]

    def featurize(self, text):
        base = super().featurize(text)
        toks = self._oov_tokens(text)
        if not toks:
            return base + [0.0, 0.5, 0.5, 0.5, 0.5, 0.0]
        va = self._l2_va(toks)
        vs = [x[0] for x in va]
        as_ = [x[1] for x in va]
        a_mean = sum(as_) / len(as_)
        a_std = (sum((x - a_mean) ** 2 for x in as_) / len(as_)) ** 0.5
        coverage = min(len(toks), 20) / 20.0
        return base + [coverage, sum(vs) / len(vs), a_mean, max(as_), min(as_), a_std]


def _demo():
    fz = L1L2Featurizer()
    print(f"L1 詞典 {len(fz)} 詞 + L2 OOV 擴充；特徵 {FEATURE_DIM_L2} 維\n")
    for t in ["我覺得非常焦慮又緊張，快崩潰了",
              "整個人心灰意冷，做什麼都提不起勁",
              "目前還在公司正職上班"]:
        f = fz.featurize(t)
        print(f"「{t}」")
        print("  L2: " + "  ".join(f"{n}={v:.2f}" for n, v in zip(EXTRA_NAMES, f[len(FEATURE_NAMES):])))


if __name__ == "__main__":
    _demo()
