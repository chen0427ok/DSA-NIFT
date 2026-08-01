# DSA-NIFT @ ROCLING 2026 — 新住民文本維度情緒分析

ROCLING 2026 Shared Task：**Chinese Dimensional Sentiment Analysis for New Immigrants' Feeling Texts**。
對新住民的自我反思文本預測 **valence（效價）/ arousal（喚醒度）**，兩者皆為 1–9 實數。

- 官方任務頁：<https://rocling-sigai.github.io/task2026/>
- 提交格式：`ID, Valence, Arousal`
- 評分：V / A 各算 **MAE（↓）+ PCC（↑）**，共 4 指標取 mean rank
- 規模：validation 200 篇、test 1,100 篇（**兩者皆未釋出 gold 標籤**）

---

## 🏁 最終成績（Test Set）

提交模型：**E19**（MacBERT + L1++ 31 維詞典特徵 + source-aware arousal loss）

| | MAE ↓ | PCC ↑ |
|---|---|---|
| Valence | **0.6200** | **0.8663** |
| Arousal | **0.9259** | **0.3566** |

**關鍵觀察**：validation（200 篇）上量到的 A_PCC 0.452，到 test（1,100 篇）只剩 0.357，
而 valence 幾乎不動（0.869 → 0.866）。這指向小型 leaderboard 上的 **selection overfitting**——
完整分析見 [docs/experiments.md](docs/experiments.md) §0 與 §5.4，這也是論文的核心 claim 之一。

---

## 📄 文件地圖

| 文件 | 內容 |
|---|---|
| **[docs/experiments.md](docs/experiments.md)** | **實驗全記錄（權威）**：13 次官方提交的分數、每個實驗的設計與診斷、五大結論 |
| [docs/dataset.md](docs/dataset.md) | 資料集：訓練來源、dev 的兩個失真、合成資料、外部詞典 |
| [docs/method.md](docs/method.md) | 架構圖、L1/L2/L3 三層資源、`train_v2.py` 所有旋鈕 |
| [docs/reproduce.md](docs/reproduce.md) | 各實驗復現指令、Colab 流程、**環境雷點**、提交格式 |
| [docs/remaining_experiments.md](docs/remaining_experiments.md) | 論文所需補跑的實驗清單（依可行性分級） |
| [docs/paper/outline.md](docs/paper/outline.md) | ROCLING-2026 system paper 英文骨架與各節要點 |
| `docs/archive/` | 歷史交接與舊規劃文件，**僅供追溯、不再更新** |

---

## 🧠 方法一句話

```
文本 ─┬─→ MacBERT encoder → mask 加權 mean pooling → (768,) ─┐
      └─→ CVAW+CVAP 詞典聚合 (L1) ───────────────→ (10,) ─┴─→ concat → 雙回歸頭 → (V, A)
```

再加上兩個關鍵設計：
1. **L3 情感知識圖譜引導的可控生成**——用圖譜撈特定 arousal 區間的種子詞，
   請 LLM 生成該區間的新住民反思文本，補訓練集缺失的 arousal 分布。
2. **Source-aware loss**——訓練資料來自 4 個不同語域，
   對 arousal 依來源加權（通用情緒庫降權、反思語料全權重），處理 arousal 的 domain shift。

---

## 📊 官方 validation 結果摘要

| 實驗 | V_MAE | V_PCC | A_MAE | A_PCC |
|---|---|---|---|---|
| 1. Baseline | 0.654 | 0.867 | 0.985 | 0.412 |
| 3. + 反思語料 | 0.627 | 0.870 | 0.944 | 0.388 |
| **4. + L1 詞典融合** | **0.600** | **0.880** | 0.882 | 0.426 |
| 5. + L3 生成增強 | 0.611 | 0.870 | 1.100 | **0.461** |
| 12. Ensemble | 0.613 | 0.882 | 0.906 | 0.418 |
| 13. Teacher 偽標 | 0.617 | 0.878 | 0.898 | 0.423 |
| **19. + Source-aware loss** ⭐ | 0.666 | 0.869 | **0.870** | **0.452** |

完整 13 列表格與每一列的診斷見 [docs/experiments.md](docs/experiments.md) §3。

**三個可寫成論文的發現**：
1. ✅ 顯性詞典訊號與 contextual embedding **互補**（實驗 4 是唯一 4 指標全勝的版本）。
2. ❌ **Ensemble 對壓縮型 arousal 無效**——多模型平均把已壓縮的 arousal 再壓一次（與常識相反）。
3. ⚖️ arousal 增強存在**「校準 ↔ 排序」根本 trade-off**（實驗 5 → 5b → E13 的完整證據鏈）。

---

## 🚀 快速開始

```bash
bash download_external.sh          # 下載 DSA-MST（第一次）
python prepare_data.py             # 產生 data/train.csv, dev.csv, val_unlabeled.csv
python train_v2.py --run_name e4_repro                            # 復現實驗 4
python train_v2.py --lex_mode l1_intensity --source_aware \
    --run_name e19_source_aware                                   # 復現最終提交模型
```

> ⚠️ **正式訓練請上 Colab A100**。M2 本機的 `mps` 在 `--batch_size 32 --max_len 256`
> 下會 hang（process 存活但不前進），細節見 [docs/reproduce.md](docs/reproduce.md) §4。

---

## 📁 目錄結構

```
baseline/
├── docs/                 # 所有文件（archive/ 為歷史文件）
├── notebooks/            # Colab notebooks
├── data/                 # 訓練 / dev / 官方無標籤資料
├── external/             # CVAW/CVAP 詞典、DSA-MST、ROCLING-2021
├── outputs/              # 權重、預測、submission（gitignore）
├── train.py              # 實驗 1–5b 專用，已凍結
├── train_v2.py           # 主訓練器（預設 = 實驗 4）
├── lexicon.py            # L1 詞典特徵（10 維）
├── lexicon_intensity.py  # L1++ 強度特徵（31 維，E18/E19）
├── lexicon_l2.py         # L1+L2 OOV 特徵（16 維，未跑）
├── word_va_regressor.py  # L2：詞 → VA 回歸
├── affective_graph.py    # L3：情感知識圖譜
├── augment_generate.py   # L3 引導的可控生成
├── build_pseudo_labels.py# multi-teacher 偽標
├── ensemble.py           # dimension-wise 融合
├── calibrate.py          # arousal 後校準（未跑）
├── embed_regressor.py    # frozen embedding + SVR（未跑）
├── build_silver_pairs.py # LLM pairwise 排序標註
├── eval_silver_ranking.py# silver ranking 評估
├── prepare_data.py / dapt.py / fetch_results.py
└── download_external.sh
```

---

## 專案約束

- **一律用繁體中文回覆。**
- **commit message 不得提及 Claude 共同作者。**
- 已跑過的實驗程式碼**不可覆蓋**、必須可復現。
- 功能開在各自 branch；remote：`https://github.com/chen0427ok/DSA-NIFT.git`（private）。
