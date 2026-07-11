# 實驗記錄 (Experiments) — DSA-NIDF ROCLING 2026

任務：對新住民自我反思文本預測 valence / arousal（1–9 實數）。
評分：Valence / Arousal 各算 MAE（↓）+ PCC（↑），共 4 指標取 mean rank。

> **指標一致性**：本專案 `train.py` 的評估與官方 `scoring.py` 數學上等價——
> MAE = `sklearn.metrics.mean_absolute_error`、PCC = `scipy.stats.pearsonr[0]`，
> 與我們用的 `np.mean(|pred-gold|)`、`np.corrcoef[0,1]` 同公式。

---

## 資料集（Dataset）

**關鍵前提：官方 DSA-NIDF 未提供「有標註」的訓練資料**，只給 `val_unlabeled.csv`（200 篇新住民文本，
**無標籤**，即提交目標）。因此**全部訓練標註都來自相鄰域的借用語料**——沒有任何目標域（新住民反思）的監督樣本。
這是本任務 low-resource / zero in-domain label 的本質，也是 arousal 難的根源（目標域 arousal 分布無監督訊號）。

### 訓練集 `data/train.csv`（9,435 筆，`prepare_data.py` 合併）
| 來源 | 筆數 | granularity | 說明 |
|---|---|---|---|
| Chinese EmoBank — CVAS | 2,583 | `sentence` | 中文情感庫，**句子級** VA（1–9） |
| Chinese EmoBank — CVAT | 2,970 | `text` | 中文情感庫，**篇章級** VA |
| DSA-MST（ROCLING-2025） | 2,282 | `reflection` | 去年同型任務的**醫療自我反思**（全 2,535，另 253 切為 dev） |
| ROCLING-2021 | 1,600 | `edu2021` | **教育反思**短文 |

- **dev `data/dev.csv`（253 筆）**：從 DSA-MST 反思切出——最接近目標域，避免拿 EmoBank 當 dev 高估。
  ⚠️ 但**與合成資料同風格**，故增強實驗的 dev 分數失真、不可信（見實驗 9 / E13）。
- **EmoBank-only 舊版** `data/orign_train_data.csv`（4,998 筆）：實驗 1–2 用，現保留供復現。

### 合成資料（獨立檔，非預設併入 train.csv）
| 檔案 | 筆數 | 說明 |
|---|---|---|
| `data/train_aug.csv` | 400 | **Opus 4.8 生成**的新住民第一人稱反思（L3 圖譜引導、5 個 VA 象限各 80），gitignore、僅本機 |
| `data/train_aug_pseudo.csv` | 318 | 上者經 3 顆 teacher 逐篇重標 + 每 bin ±1.5SD 離群移除（E13 用，已入 branch） |

> **`train_aug.csv` 是唯一「目標域風格 × 目標 arousal 區間」的監督樣本**，故實驗 9 能提 A_PCC（補了別處都沒有的洞）。

### 外部詞典（特徵/資源用，非訓練樣本）
- `external/emobank/`：CVAW（字，5,512）+ CVAP（詞）= **7,761 VA 詞典**，餵 L1（`lexicon.py`）與 L2/L3。

---

## 共同設定（四個實驗都一樣）

- **Encoder**：`hfl/chinese-macbert-base`
- **架構**：encoder → attention-mask 加權 mean pooling → 2 個回歸頭（valence, arousal）
- **標籤處理**：1–9 正規化到 [0,1]，sigmoid 輸出，推論時反轉回 1–9
- **微調超參**：4 epochs、batch 32、lr 2e-5、max_len 256、AdamW(wd 0.01)、
  warmup 10%、SmoothL1 loss、grad clip 1.0
- **Early-stop 依據**：`mean(PCC) − mean(MAE)` 取最佳 epoch
- **Submission set（三者相同）**：官方 `DSANIDF_ValidationSet.csv`（200 篇，只有 ID+Text，無標籤）

---

## 四個實驗的差異

| | 實驗 1：Baseline | 實驗 2：DAPT | 實驗 3：反思語料版 | 實驗 4：L1 詞典融合 |
|---|---|---|---|---|
| **DAPT 續訓** | 無 | ✅ 新住民 200 篇 MLM | 無 | 無 |
| **詞典特徵** | 無 | 無 | 無 | ✅ CVAW+CVAP 10 維 |
| **Train set** | EmoBank CVAS+CVAT | EmoBank CVAS+CVAT | EmoBank + DSA-MST + ROCLING-2021 | 同實驗 3 |
| **Train 筆數** | 4,998 | 4,998 | 9,435 | 9,435 |
| **Val set (dev)** | EmoBank 切出（stratified）| EmoBank 切出（stratified）| DSA-MST 反思切出 | DSA-MST 反思切出 |
| **Dev 筆數** | ~555 | ~555 | 253 | 253 |
| **每 epoch steps** | 157 | 157 | 295 | 295 |
| **Encoder 來源** | macbert-base 原權重 | DAPT 續訓後權重 | macbert-base 原權重 | macbert-base 原權重 |

### 實驗 1：Baseline
- **Train**：Chinese EmoBank 的 CVAS（句）+ CVAT（篇），即 `data/orign_train_data.csv`。
- **Val(dev)**：從 EmoBank 等比例切 10%。
- **模型/訓練**：macbert-base 直接微調，無任何續訓。
- **目的**：建立可跑的完整 pipeline 與基準分數。

### 實驗 2：DAPT（領域適應續訓）
- **唯一差異**：微調前先用 `dapt.py` 在**新住民 200 篇文本**上做 MLM 續訓
  （30 epochs、batch 16、lr 5e-5、mlm_prob 0.15），再用續訓後的 encoder 微調。
- **Train / Val**：與實驗 1 完全相同（EmoBank 4,998 / dev ~555）。
- **目的**：測試「先讓 encoder 讀熟目標語域」能否補領域落差。
- **註**：此次 train set 仍是原始 EmoBank（steps=157 可佐證），未含新反思語料。

### 實驗 3：反思語料版
- **Train**：EmoBank CVAS+CVAT(5,553) + **DSA-MST 醫療反思(2,535)** +
  **ROCLING-2021 教育反思(1,600)** = 9,435 筆，即 `data/train.csv`。
- **Val(dev)**：改從 **DSA-MST 反思文本**切 253 篇——當作新住民目標域的代理驗證集，
  讓 dev 分數更能預測真實表現。
- **模型/訓練**：macbert-base 直接微調（無 DAPT），超參同上。
- **目的**：把訓練的「監督語域」換成第一人稱情緒反思，補 label 層級的領域落差。

### 實驗 4：L1 情感詞典特徵融合 ✅ 目前最佳
- **Train / Val**：與實驗 3 相同（train.csv 9,435 / DSA-MST dev 253）。
- **唯一新增**：`lexicon.py` 用 Chinese EmoBank 的 **CVAW(字)+CVAP(詞) 共 7,761 個 VA 詞典**，
  對每篇文本抽 **10 維情緒詞聚合特徵**（coverage / count / V·A 的 mean·max·min·std），
  concat 進 BERT pooled embedding 後再進回歸頭（`nn.Linear(768+10, 2)`）。
- **目的**：給模型一個顯性的「這篇有哪些高/低喚醒詞」訊號，直攻 arousal。
- **結果**：**4 指標全面贏過實驗 3**——Valence MAE 0.627→0.600、Valence PCC 0.870→0.880、
  Arousal MAE 0.944→0.882、**Arousal PCC 0.388→0.426**。詞典的顯性 VA 訊號同時改善了校準（MAE）
  與排序（PCC），arousal 預測 std 從 0.755 更接近真實分布。
- **dev**：epoch 2 最佳（V_PCC 0.815 / A_PCC 0.609），與實驗 3 一致在第 2–3 epoch 後過擬合。

### 實驗 5：L3 圖譜可控生成增強（進行中，= `current_pipeline.md` 待跑清單 #9）
- **基礎**：實驗 4（L1 詞典融合）不變——encoder、架構、dev（DSA-MST 反思 253）、超參全部沿用，
  **唯一變數是 train set 多了合成資料**，故可與實驗 4 公平對照（純粹量測「資料增強」的效果）。
- **增強流程**（`augment_generate.py`，本機 Claude `claude-opus-4-8` 生成）：
  1. 讀 L3 情感知識圖譜（`outputs/l3_graph.pkl`，7,761 節點 / FastText kNN 連邊）。
  2. 對 **5 個代表性不足的 VA 目標區間**用 `seeds_for_target()` 撈情緒種子詞，條件化生成
     新住民第一人稱反思短文（50–120 字、題材多元、prompt 內過濾暴力/粗俗詞）。
  3. 標上**目標 bin 中心 ± jitter(0.4)** 的弱標籤（粗但一致），輸出 `data/train_aug.csv`。
  4. **近似去重**：jieba 詞集 Jaccard > 0.5 視為重複丟棄（跨 bin 去重 + 每區間呼叫上限保護）。
- **5 個目標區間（各 80 篇，共 400 篇）**：

  | bin | V 中心 | A 中心 | 情緒象限 | 補的洞 |
  |---|---|---|---|---|
  | 1 | 3.0 | 7.5 | 負效價·**高喚醒**（焦慮/憤怒/恐慌/崩潰）| arousal 高端 |
  | 2 | 5.0 | 7.5 | 中效價·**高喚醒**（緊張/激動/坐立難安）| arousal 高端 × 中 valence（最稀缺）|
  | 3 | 7.0 | 7.5 | 正效價·**高喚醒**（興奮/狂喜/喜極而泣）| arousal 高端 |
  | 4 | 3.0 | 2.5 | 負效價·**低喚醒**（疲憊/無力/麻木/沮喪）| arousal 低端 |
  | 5 | 7.0 | 2.5 | 正效價·**低喚醒**（平靜/安心/滿足/安逸）| arousal 低端 |

- **資料分布變化**（400 篇全部 `--append` 後）：
  - 筆數 **9,435 → 9,835**（+400，+4.2%）。
  - Arousal **std 1.27 → 1.35**（+6%），直接把被壓縮的 A 分布拉開；
    A≥6 比例 16.6%、A≤3.5 比例 19.6% 同步上升（高/低兩端各補 160 篇，中間帶補 80 篇）。
  - 增強資料本身 V mean/std = 5.01/1.83、A mean/std = 5.50/2.46，刻意比真實分布更分散。
- **目的**：直接補 arousal 分布壓縮——這是四個瓶頸觀察裡**唯一「補缺目標標籤」**的招
  （L1 是補特徵、DAPT 是補表徵，都沒補到「對的 VA 區間的監督樣本」）。期望提升 A_PCC。
- **（可選）偽標精修**：依 CYUT 冠軍流程，可用訓好的 L1 當 teacher 重標 + 每 bin mean±1.5SD 離群移除
  （`Rocling2026_Colab_aug.ipynb` cell 8 已備），本批先用 bin 中心弱標籤直接訓。
- **結果（bin 中心弱標籤版）**：**4 指標輸 3 贏 1，整體比實驗 4 差，不提交**。
  - Valence MAE 0.600→**0.611**、Valence PCC 0.880→**0.870**（皆略退）。
  - Arousal MAE 0.882→**1.100**（🔴 爆掉 +0.218）、**Arousal PCC 0.426→0.461（🟢 首次突破 0.43）**。
  - **診斷**：submission 預測 arousal std 僅 0.755→0.821（幾乎沒變寬），MAE 卻暴增 → 不是「攤開預測」
    而是「整體被帶偏」。極端弱標籤（A=7.5/2.5，超出真實 arousal 尺度）讓模型學到**虛假的
    「題材→極端喚醒」關聯**，驗證集同題材但情緒中等者被預測過高/過低：**排序變好(PCC↑)、校準變糟(MAE↑)**。
  - **dev 失真**：best epoch 2 dev A_PCC 0.600 / A_MAE 0.84 看似漂亮，official A_MAE 卻 1.10——
    增強資料與反思 dev 同風格，dev 不再是可靠代理。
  - **收穫**：文本本身有效（PCC 證明它教會模型分辨 arousal），**爛的是標籤不是文本**。
- **實驗 9b（標籤收縮 k=0.6）結果**：`new = μ + 0.6·(bin中心 − μ)`（aug A std 2.47→1.48、合併後 A std 1.29≈真實）。
  - Valence MAE 0.611→**0.632**、Valence PCC 0.870→**0.874**；Arousal MAE 1.100→**1.058**（仍🔴）、**Arousal PCC 0.461→0.460（守住）**。
  - **關鍵**：預測 arousal std 0.821→**0.743**（收回到 ≈實驗 4 的 0.755），**A_PCC 仍保 0.460、A_MAE 卻沒回到 0.88**。
  - **結論**：這證實 **A_PCC 增益來自「文本教的特徵」不是「攤開分布」**（spread 收回了、PCC 沒掉）；但
    **A_MAE 的偏移不是單純標籤太極端造成的，收縮救不回** → 問題在合成文本本身帶入的**校準/風格偏移**，
    對官方驗證集分布不合。**實驗 9/9b 兩版都輸實驗 4（3/4 指標），不提交。**
  - **下一步（擇一）**：① teacher 偽標精修（cell 8，逐篇實際尺度重標，比整批收縮更根治）；
    ② 減量到 100–200 篇；③ **放棄增強、以實驗 4 為提交**，改攻 arousal 的其他槓桿（見觀察 6）。

---

## 實驗 10–12：Multi-seed + 多 encoder Ensemble（A100，`train_v2.py` / `ensemble.py`）

- **基礎**：實驗 4（L1 詞典融合）架構不變；`train_v2.py` 預設參數復現實驗 4，僅 batch 32→64（A100，dev 與 32 一致）。
- **E10**：MacBERT+L1 跑 5 個 seed（42/1/2/3/4），`ensemble.py --mode mean` 等權平均。
- **E11**：RoBERTa-wwm-ext（base）與 **-large**（batch 16, lr 1e-5）各一顆 + L1。
- **E12**：把 E10 五顆 + E11 兩顆做 dimension-wise weighted / mean 融合（V/A 分開權重）。
- **dev（DSA-MST 253）**：各單顆 A_PCC 0.60–0.615；E10 融合 A_PCC 0.616、E12 A_PCC 0.615；
  roberta-large 最弱（dev A_PCC 0.584、epoch 3–4 過擬合）。dev 全部落在 A_PCC≈0.61，與實驗 4 dev 一致。
- **結果（E12 官方）：4 指標輸實驗 4 三項，只在 V_PCC 微贏 0.002 → 不採用。**
  - Valence MAE 0.600→**0.613**、Valence PCC 0.880→**0.882**（微升）。
  - Arousal MAE 0.882→**0.906**（🔴）、**Arousal PCC 0.426→0.418（🔴 不升反降）**。
  - **診斷**：multi-encoder ensemble 只穩定方差、**沒突破 arousal 天花板**；平均把 arousal 預測壓縮/糊化，
    連 A_MAE 都變差；roberta-large（最弱）進 ensemble 稀釋 arousal 訊號。
  - **結論**：**堆模型 / 融合對 arousal PCC 是死路**（官方驗證）。唯一破過 0.43 的仍是實驗 9 的合成資料（0.46）
    → 回到 E13（teacher 偽標精修）。
  - **E10 官方（純 macbert 5-seed）= A_PCC 0.415，與 E12 幾乎相同、同樣輸實驗 4** → **證實傷害不是 roberta-large，
    而是「多模型平均」本身把已壓縮的 arousal 再壓一次**（校準與排序雙輸）；batch 64（vs 實驗 4 的 32）可能讓每顆
    seed 也略弱。**ensemble 路線 E10/E11/E12 全數確認為死路，實驗 4 維持最佳提交。**
  - 交付物：`outputs/{macbert_s*,roberta_s42,robertaL_s42}_best.pt`（7 顆，E13 的 teacher）、
    `outputs/preds/*`（統一預測，供 ensemble/校準）、各 `*_submission.csv`。

---

## 實驗 18–21：L1++ 強度特徵 / Source-aware loss / Ranking-only 增強 / Dim-attention（A100，`train_v2.py` 新旋鈕）

規劃見 `dsa_nift_next_experiments_plan.md`。全部疊在實驗 4 架構上，嚴格 batch 32 / lr 2e-5 / 4 epochs 對照。

- **E18（`--lex_mode l1_intensity`）**：`lexicon_intensity.py` 把 L1 十維擴成 31 維
  （+15 維 arousal intensity：標點密度/程度副詞/身體反應/睡眠/焦慮/壓力事件/低喚醒詞/疊字/句長節奏/否定轉折；
  +6 維新住民 domain cues：語言/證件/工作/家庭分離/文化/經濟）。
- **E19（+`--source_aware`）**：V/A loss 依 granularity 來源加權——CVAS `1:0.25`、CVAT `1:0.5`、
  DSA-MST `1:1`、edu2021 `1:0.75`（arousal 對 domain shift 敏感，通用情緒庫降權；V 全部 1.0）。
- **E20（+`--rank_aug data/train_aug.csv`）**：合成 400 篇**不進 SmoothL1**，每 step 抽 16 篇建
  batch 內 pairwise hinge（gold 差 ≥1.5 分才成 pair、margin 1 分、λ_A=0.1 λ_V=0.05）。
- **E21a/b（`--pooling mean_cls_dim_attention`）**：mean + CLS + V/A 各自 attention pooling 分頭回歸
  （head dropout 0.2）；b 版再疊 rank_aug。
- **dev（253）**：E19 最佳（A_PCC 0.618 / A_MAE 0.823，唯一超出 5-seed 雜訊帶 0.604–0.616 的 run）；
  E20 dev 墊底（A_PCC 0.611）——但 dev 對增強實驗失真（實驗 9 教訓）。

### Silver ranking benchmark（`build_silver_pairs.py` / `eval_silver_ranking.py`，新工具）

因 dev（DSA-MST）與目標域偏移，對官方 200 篇無標籤文本抽 500 pairs，請 LLM（claude-opus-4-8）
做 pairwise 判斷「哪篇 arousal / valence 較高」（只信排序、不信絕對分數，避開 E13 的校準陷阱），
評各 run val 預測的排序一致性（A_rank_acc，0.5 = 隨機）：

- e20 **0.766**（第一）＞ e18 0.762 ＞ … ＞ E4 復現 0.748 = e19 0.748 ＞ … ＞ e21a/b **0.726–0.728（墊底）**。
- **對 e21 的否決是準的**（官方 A_PCC 0.394 確實最差）；**但 e19/e20 的相對排序判斷錯了**
  （官方 A_PCC：e19 0.452 ＞ e20 0.407）→ silver 量的是 pairwise 排序一致性，與 PCC（受分布形狀影響）
  不完全等價，且目前只有單一 judge。**當否決訊號（篩掉明顯壞的 run）比當「選第一名」可靠。**

### 官方結果（e19 / e20 / e21b 已提交）

- **E19 = arousal 首次「雙贏」**：A_MAE **0.882→0.870**（優於實驗 4）且 A_PCC **0.426→0.452**
  （所有 A_MAE 未爆的模型中最高；實驗 9 的 0.461 伴隨 A_MAE 1.10）。**代價在 valence**：
  V_MAE 0.600→0.666、V_PCC 0.880→0.869。4 指標 2 勝 2 敗，與實驗 4 互補。
  - **診斷**：CVAS/CVAT 的 arousal 降權讓模型少學通用情緒庫的 arousal 慣性、多信反思語料 →
    arousal 排序與校準同升。V 權重全為 1.0 卻變差，可能是 loss 重新配比的間接影響，值得解耦實驗。
- **E20 / E21b 官方皆輸實驗 4**（A_PCC 0.407 / 0.394）：ranking-only 增強把 val arousal 分布撐開
  （std 0.74–0.83）但沒轉成 PCC；dim attention 屬目標域過擬合（dev 好看、官方最差）。

---

## 結果（官方 validation 實際分數）

| 實驗 | Valence MAE ↓ | Valence PCC ↑ | Arousal MAE ↓ | Arousal PCC ↑ |
|---|---|---|---|---|
| 1. Baseline | 0.654 | 0.867 | 0.985 | 0.412 |
| 2. DAPT | 0.649 | 0.866 | 1.009 | 0.395 |
| 3. 反思語料 | 0.627 | 0.870 | 0.944 | 0.388 |
| **4. L1 詞典融合** | **0.600** | **0.880** | **0.882** | **0.426** |
| 5. L3 生成增強（#9，bin 中心標籤） | 0.611 | 0.870 | 1.100 🔴 | **0.461** 🟢 |
| 5b. L3 增強（標籤收縮 k=0.6） | 0.632 | 0.874 | 1.058 🔴 | 0.460 🟢 |
| 12. 多 encoder Ensemble（E12） | 0.613 | 0.882 | 0.906 🔴 | 0.418 🔴 |
| 10. 純 macbert 5-seed Ensemble（E10） | 0.614 | 0.881 | 0.907 🔴 | 0.415 🔴 |
| 13. Teacher 偽標增強（E13，blend=0） | 0.617 | 0.878 | 0.898 | 0.423 |
| **19. Source-aware arousal loss（E19）** | 0.666 🔴 | 0.869 | **0.870** 🟢 | **0.452** 🟢 |
| 20. Ranking-only 增強（E20） | 0.631 | 0.877 | 0.923 🔴 | 0.407 🔴 |
| 21. Dim-attention + rank（E21b） | 0.638 | 0.865 | 0.923 🔴 | 0.394 🔴 |

（實驗 2 dev 最佳 epoch 4：V_PCC 0.859 / A_PCC 0.620；實驗 3 dev 最佳 epoch 3：V_PCC 0.822 / A_PCC 0.599；
實驗 4 dev 最佳 epoch 2：V_PCC 0.815 / A_PCC 0.609）

## 重點觀察
1. **Valence 已接近上限**（PCC ~0.88），**Arousal PCC 一直是瓶頸**（~0.39–0.43）。
2. **DAPT（小語料 + 舊資料）沒幫助**，arousal 反而略退 → 缺的是「對的語域的監督標籤」，不是 encoder 表徵。
3. **實驗 3 修好了校準**（雙 MAE 下降、4 指標贏 3 項），但**沒修好排序**（arousal PCC 仍 0.388）。
4. **實驗 4（L1 詞典融合）是第一個 4 指標全勝的版本**：顯性 VA 詞典特徵同時抬升了 arousal 的
   校準與排序（PCC 0.388→0.426），證明「contextual embedding + lexical VA 訊號」互補有效。
5. **實驗 9/9b（L3 生成增強）證明「合成資料能提升 arousal 排序」但代價是校準**：A_PCC 0.426→0.46 首次破 0.43
   且在標籤收縮後仍守住；然而 A_MAE 從 0.882 惡化到 1.06–1.10 且**收縮救不回**（pred std 已收回正常）。
   → 增益來自文本特徵、傷害來自合成文本的校準/風格偏移。**兩版皆輸實驗 4，暫不採用增強。**
6. 下一步仍應**專攻 arousal PCC**：先把實驗 9b 的校準修好（最有機會 4 指標全勝）、
   multi-seed ensemble（兩篇得獎論文共識的最大槓桿）、更強 encoder（roberta-wwm-ext-large）、
   把 L2 word→VA 回歸器接進 L1 當額外特徵。
7. **實驗 12（多 encoder ensemble）已用官方分數證明是死路**：A_PCC 0.426→0.418（不升反降），4 指標輸實驗 4 三項。
   ensemble 只穩方差、平均壓縮 arousal 反傷校準；roberta-large 最弱還拖累。
   **⇒ 堆模型/融合無法突破 arousal 天花板，唯一破 0.43 的仍是合成資料（實驗 9 = 0.46）。**
   下一步 **E13：用實驗 10–12 的 7 顆 teacher 對合成文本逐篇偽標精修 + 每 bin ±1.5SD 離群移除**
   （`build_pseudo_labels.py`），目標留住實驗 9 的 A_PCC 增益、修回被 bin 弱標籤搞爆的 A_MAE。
8. **E13（teacher 偽標精修，blend=0）官方結果：校準修好了，但排序增益丟了。**
   - A_MAE **1.100（實驗9）→ 0.898**（teacher 校準把爆掉的 MAE 修回 0.20，證明偽標精修對「校準」有效）；
     但 A_PCC **0.461（實驗9）→ 0.423**（**arousal 排序增益一起消失**）。4 指標仍全數略輸實驗 4。
   - **診斷（自我參照陷阱）**：teacher 本身 arousal 壓縮（偽標把低喚醒 bin 標成 ~5.0、認不出低端），
     用它重標等於「用壓縮的老師教壓縮的問題」，把撐開 arousal 排序的訊號一起壓平 → PCC 掉回。
   - **⇒ arousal 增強存在「校準 ↔ 排序」根本 trade-off**：raw 極端標籤（spread 大）給 PCC、傷 MAE；
     teacher 標籤（校準準）救 MAE、失 PCC。**下一步兩條**：① 找中間點（`--blend` 0.3–0.5，
     或高喚醒 bin 用 teacher、低喚醒 bin 用 moderate 標籤的「分端策略」）；
     ② **改精進實驗 4 架構本身（Enhanced L1 加 arousal 強度表面特徵）——不動資料分布，避開此 trade-off。**
9. **E19（source-aware arousal loss）官方結果：arousal 首次雙贏（A_MAE 0.870 + A_PCC 0.452），但 valence 付出代價。**
   - 這是第一個**不靠合成資料就把 A_PCC 推過 0.45** 的模型，且 A_MAE 同時優於實驗 4 →
     證明「來源加權」直接處理了 arousal 的 domain shift，繞開了校準↔排序 trade-off。
   - 與實驗 4 形成互補：實驗 4 贏 valence 雙指標、E19 贏 arousal 雙指標（官方計分 = 4 指標 mean rank，兩者相當）。
   - **下一步方向**：解耦 valence 退化——(a) 調 `--source_weights` 讓 V 少受影響（如 CVAS/CVAT 的 V 權重 >1 補償）；
     (b) V/A 分頭訓練或分頭 early-stop；(c) E26 校準只修 E19 的 V_MAE。
   - **E20/E21 確認淘汰**：ranking loss 與 dim attention 在官方分數上皆全面輸實驗 4。
   - **Silver ranking benchmark 首次實戰**：否決 e21 準確、e19/e20 排序誤判 → 定位為「篩壞的」工具，
     花提交額度前先過濾，但不能取代官方分數選最佳。
