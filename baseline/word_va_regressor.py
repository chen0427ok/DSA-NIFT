"""
L2 — VA-aware 詞向量迴歸 (借鑑 IALP2016 / IJCNLP2017)

早期 shared task 的標準做法：用詞向量(Word2Vec/FastText) + 迴歸模型，學「詞 -> VA」。
本檔用 FastText（char n-gram，能對 CVAW 以外的 OOV 詞也給向量）在我們的語料上訓練詞向量，
再用 SVR 學 詞向量 -> (V,A)，於是**任何詞**都能估 VA → 把 L1 的詞典從 5512 字擴充到無限覆蓋。

可獨立執行驗證：
    python word_va_regressor.py
會印出：held-out CVAW 詞的 MAE/PCC（證明學到 詞->VA），以及對幾個「不在 CVAW 裡」的
情緒詞的 VA 預測（證明覆蓋率擴充）。並存出 outputs/l2_word_va.pkl。
"""
import os
import csv
import pickle
import numpy as np
import jieba
from gensim.models import FastText
from sklearn.svm import SVR
from scipy.stats import pearsonr
from lexicon import emobank_file

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
LABEL_MIN, LABEL_MAX = 1.0, 9.0


def read_cvaw():
    """CVAW: 單字 + VA(1-9)。回傳 [(word, v, a), ...]"""
    rows = []
    path = emobank_file("CVAW_all_SD.csv", "CVAW_SD")
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            w = (r.get("Word") or "").strip()
            if not w or "�" in w:
                continue
            try:
                rows.append((w, float(r["Valence_Mean"]), float(r["Arousal_Mean"])))
            except (KeyError, ValueError):
                continue
    return rows


def load_corpus():
    """拿 train.csv 的文本當 FastText 訓練語料，jieba 斷詞。"""
    sents = []
    path = os.path.join(DATA, "train.csv")
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            toks = [t for t in jieba.cut(r["text"]) if t.strip()]
            if toks:
                sents.append(toks)
    return sents


def main():
    os.makedirs(OUT, exist_ok=True)
    print("載入語料訓練 FastText ...")
    sents = load_corpus()
    # 把 CVAW 的字也加進語料，確保它們有出現
    cvaw = read_cvaw()
    sents += [[w] for w, _, _ in cvaw]
    ft = FastText(sentences=sents, vector_size=100, window=5, min_count=1,
                  sg=1, epochs=20, min_n=1, max_n=3, seed=42, workers=4)
    print(f"  FastText 詞表={len(ft.wv)}  語料句數={len(sents)}")

    # 建 詞向量 -> VA 訓練資料
    X = np.array([ft.wv[w] for w, _, _ in cvaw])
    yv = np.array([v for _, v, _ in cvaw])
    ya = np.array([a for _, _, a in cvaw])

    # 90/10 split 驗證「詞->VA」確實學得起來
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(cvaw))
    k = int(len(idx) * 0.1)
    te, tr = idx[:k], idx[k:]

    def fit_eval(y, name):
        m = SVR(kernel="rbf", C=10, epsilon=0.2)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        mae = float(np.mean(np.abs(pred - y[te])))
        pcc = float(pearsonr(pred, y[te])[0])
        print(f"  [{name}] held-out  MAE={mae:.3f}  PCC={pcc:.3f}")
        # 用全部資料重訓當最終模型
        full = SVR(kernel="rbf", C=10, epsilon=0.2); full.fit(X, y)
        return full

    print("訓練 SVR: 詞向量 -> VA ...")
    mv = fit_eval(yv, "valence")
    ma = fit_eval(ya, "arousal")

    # 覆蓋率擴充驗證：對「不在 CVAW」的情緒詞估 VA
    cvaw_set = {w for w, _, _ in cvaw}
    oov = ["雀躍", "崩潰", "惆悵", "亢奮", "心灰意冷", "煎熬"]
    print("\n覆蓋率擴充 — CVAW 沒有的詞也能估 VA:")
    for w in oov:
        vec = ft.wv[w].reshape(1, -1)
        v, a = float(mv.predict(vec)[0]), float(ma.predict(vec)[0])
        tag = "(已在CVAW)" if w in cvaw_set else "(OOV,新覆蓋)"
        print(f"  {w:6s} V={v:.2f} A={a:.2f}  {tag}")

    with open(os.path.join(OUT, "l2_word_va.pkl"), "wb") as f:
        pickle.dump({"fasttext": ft, "svr_v": mv, "svr_a": ma}, f)
    print(f"\n已存模型 -> {os.path.join(OUT, 'l2_word_va.pkl')}")


if __name__ == "__main__":
    main()
