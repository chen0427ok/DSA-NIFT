"""
L3 — 情感知識圖譜 (Affective Knowledge Graph)

把 CVAW+CVAP 情緒詞當節點（帶 VA），用 L2 的 FastText 向量做 kNN 連邊（語義相關），
建成 networkx 圖。兩個用途：
  (1) VA 標籤傳播：用鄰居平滑/補全 VA（半監督）。
  (2) 可控生成種子：指定目標 VA 區間（如「高 arousal」）→ 從圖上取一群種子詞，
      餵給 LLM 生成「特定情緒強度」的新住民文本 → 直接補我們 arousal 分布壓縮的洞。

依賴 L2 的 outputs/l2_word_va.pkl（FastText）。可獨立執行驗證：
    python word_va_regressor.py   # 先產生 L2 模型
    python affective_graph.py     # 建圖 + 印統計 + 鄰居/種子/prompt 範例
"""
import os
import csv
import pickle
import random
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from lexicon import emobank_file

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
LABEL_MIN, LABEL_MAX = 1.0, 9.0


def load_terms():
    """讀 CVAW+CVAP -> {term: (V, A)}（原始 1-9）。"""
    terms = {}
    for fname, sub, field in [("CVAW_all_SD.csv", "CVAW_SD", "Word"),
                              ("CVAP_all_SD.csv", "CVAP_SD", "Phrase")]:
        with open(emobank_file(fname, sub), encoding="utf-8", errors="replace", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                t = (r.get(field) or "").strip()
                if not t or "�" in t:
                    continue
                try:
                    terms[t] = (float(r["Valence_Mean"]), float(r["Arousal_Mean"]))
                except (KeyError, ValueError):
                    continue
    return terms


def build_graph(k=8):
    """建情感知識圖譜：節點=情緒詞(帶VA)，邊=FastText 向量 kNN（語義相關）。"""
    with open(os.path.join(OUT, "l2_word_va.pkl"), "rb") as f:
        ft = pickle.load(f)["fasttext"]
    terms = load_terms()
    words = list(terms)
    emb = np.array([ft.wv[w] for w in words])

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    dist, idx = nn.kneighbors(emb)

    G = nx.Graph()
    for i, w in enumerate(words):
        v, a = terms[w]
        G.add_node(w, valence=v, arousal=a)
    for i, w in enumerate(words):
        for j, d in zip(idx[i][1:], dist[i][1:]):  # 跳過自己
            G.add_edge(w, words[j], weight=float(1 - d))
    return G, terms


def neighbors(G, word, n=6):
    if word not in G:
        return []
    nbrs = sorted(G[word].items(), key=lambda kv: -kv[1]["weight"])[:n]
    return [(w, G.nodes[w]["valence"], G.nodes[w]["arousal"]) for w, _ in nbrs]


def seeds_for_target(G, v_range=None, a_range=None, n=15):
    """取 VA 落在指定區間的種子詞（給可控生成用）。v_range/a_range=(lo,hi)。"""
    out = []
    for w, d in G.nodes(data=True):
        if v_range and not (v_range[0] <= d["valence"] <= v_range[1]):
            continue
        if a_range and not (a_range[0] <= d["arousal"] <= a_range[1]):
            continue
        out.append((w, d["valence"], d["arousal"]))
    out.sort(key=lambda x: -x[2])  # 依 arousal 高到低
    return out[:n]


# ============================================================================
# 種子詞選取策略（消融用）。上面的 seeds_for_target 是實驗 5 的原始行為，
# 已凍結不得更動（條件 C 必須可復現）。以下為新增策略：
#   seeds_random           條件 A：隨機抽詞（不看 VA）
#   seeds_by_expansion     條件 E/F：G1 圖擴散 + G3 多樣性取樣（真正用到圖的邊）
# 規劃見 docs/paper/kg_experiment_plan.md
# ============================================================================

def seeds_random(G, n=15, rng=None):
    """條件 A：完全隨機抽 n 個情緒詞，不看 VA 區間。

    對照組的意義：如果 A 與 C（VA 查表）打平，代表「VA 區間過濾」沒有作用，
    增益只是「prompt 裡有一些情緒詞」而已。
    """
    rng = rng or random.Random(0)
    words = list(G.nodes())
    picked = rng.sample(words, min(n, len(words)))
    return [(w, G.nodes[w]["valence"], G.nodes[w]["arousal"]) for w in picked]


def seeds_by_expansion(G, v_range=None, a_range=None, n=15, rng=None,
                       n_anchors=4, hops=2, va_slack=1.0):
    """條件 E/F：G1 圖擴散 + G3 多樣性取樣。**這是唯一真正使用圖的邊的種子策略。**

    與 seeds_for_target（查表）的差別：
      查表 = 「全詞典裡 arousal 最高的前 n 個詞」，語意上是東拼西湊的集合，
             而且是 sort 後取前 n，決定性 → 同一 bin 的每篇文章看到同一組詞。
      本函式 = 從目標 VA 區間【隨機抽少數 anchor】(G3)，沿 kNN 邊擴散 hops 跳 (G1)，
             收集語意鄰居後再用放寬的 VA 區間軟過濾。
             → 種子集【語意連貫】（來自同一批 anchor 的鄰域）
             → 每次呼叫【都不同】（anchor 隨機），解決 80 篇共用一組詞的問題
             → 鄰居可能是詞典裡 VA 沒那麼極端、但語意相關的詞，擴大了覆蓋

    Args:
        n_anchors: 抽幾個 anchor（越少越聚焦、越多越發散）
        hops:      沿邊擴散幾跳
        va_slack:  擴散得到的鄰居，VA 可以超出目標區間多少（給圖結構發揮空間）
    Returns:
        [(word, valence, arousal)]，最多 n 個
    """
    rng = rng or random.Random(0)

    def in_range(d, slack=0.0):
        if v_range and not (v_range[0] - slack <= d["valence"] <= v_range[1] + slack):
            return False
        if a_range and not (a_range[0] - slack <= d["arousal"] <= a_range[1] + slack):
            return False
        return True

    # 1) 候選 anchor = 嚴格落在目標 VA 區間的節點
    candidates = [w for w, d in G.nodes(data=True) if in_range(d)]
    if not candidates:
        return []
    anchors = rng.sample(candidates, min(n_anchors, len(candidates)))

    # 2) 沿 kNN 邊擴散 hops 跳，記錄每個詞的最佳連結強度（供排序用）
    scores = {}
    frontier = {a: 1.0 for a in anchors}
    for w in anchors:
        scores[w] = 1.0
    for _ in range(hops):
        nxt = {}
        for w, s in frontier.items():
            for u, e in G[w].items():
                cand = s * e["weight"]
                if cand > scores.get(u, 0.0):
                    scores[u] = cand
                    nxt[u] = cand
        frontier = nxt
        if not frontier:
            break

    # 3) 用放寬的 VA 區間軟過濾（保留語意相關但 VA 稍偏的詞 = 圖結構的貢獻）
    kept = [(w, G.nodes[w]["valence"], G.nodes[w]["arousal"], s)
            for w, s in scores.items() if in_range(G.nodes[w], va_slack)]
    # 4) 依連結強度排序（anchor 本身最高），取前 n
    kept.sort(key=lambda x: -x[3])
    return [(w, v, a) for w, v, a, _ in kept[:n]]


def seed_coherence(G, words):
    """診斷用：種子集的語意連貫度 = 詞對之間在圖上直接相連的比例（0–1）。

    圖擴散（條件 E）應該顯著高於 VA 查表（條件 C），因為 E 的詞來自同一批 anchor 的鄰域。
    這是「圖結構有在做事」最直接的量化證據，論文可直接引用。
    """
    ws = [w for w in words if w in G]
    if len(ws) < 2:
        return 0.0
    linked = total = 0
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            total += 1
            if G.has_edge(ws[i], ws[j]):
                linked += 1
    return linked / total if total else 0.0


def propagate_va(G, iters=2):
    """簡單標籤傳播：用鄰居加權平均平滑 VA（示範半監督補全）。"""
    va = {w: np.array([G.nodes[w]["valence"], G.nodes[w]["arousal"]]) for w in G}
    for _ in range(iters):
        new = {}
        for w in G:
            nb = list(G[w].items())
            if not nb:
                new[w] = va[w]; continue
            ws = np.array([e["weight"] for _, e in nb])
            vals = np.array([va[u] for u, _ in nb])
            smoothed = (ws[:, None] * vals).sum(0) / ws.sum()
            new[w] = 0.5 * va[w] + 0.5 * smoothed  # 保留一半自身
        va = new
    return va


def build_generation_prompt(seed_words, retrieved_examples, target_desc):
    """可控生成的 prompt（實際呼叫 LLM 在 Colab/API；此函式只組 prompt）。"""
    seeds = "、".join(w for w, _, _ in seed_words)
    examples = "\n".join(f"- {t}" for t in retrieved_examples)
    return (
        f"你是新住民，請寫一段 60-120 字的第一人稱生活反思短文。\n"
        f"情緒要求：{target_desc}。可自然帶入這些情緒詞：{seeds}。\n"
        f"風格參考（語氣口吻相近，但內容要不同、不要照抄）：\n{examples}\n"
        f"只輸出短文本身。"
    )


def _demo():
    print("建情感知識圖譜 ...")
    G, terms = build_graph(k=8)
    print(f"  節點={G.number_of_nodes()}  邊={G.number_of_edges()}  "
          f"平均度={2*G.number_of_edges()/G.number_of_nodes():.1f}\n")

    for w in ["焦慮", "開心"]:
        nb = neighbors(G, w)
        if nb:
            print(f"「{w}」語義鄰居:", "  ".join(f"{x[0]}(A={x[2]:.1f})" for x in nb))

    print("\n高 arousal 種子詞 (A≥7，給可控生成補 arousal 用):")
    seeds = seeds_for_target(G, a_range=(7.0, 9.0), n=12)
    print("  " + "、".join(f"{w}" for w, _, _ in seeds))

    print("\n--- 可控生成 prompt 範例 (dry-run，實際生成在 Colab/API) ---")
    examples = ["今天心情很複雜，既期待又有點不安。", "回想剛來台灣的日子，真的很不容易。"]
    print(build_generation_prompt(seeds[:8], examples, "高喚醒度(arousal≥7)、效價中性偏負"))


if __name__ == "__main__":
    _demo()
