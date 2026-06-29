# 實驗記錄 (Experiments) — DSA-NIDF ROCLING 2026

任務：對新住民自我反思文本預測 valence / arousal（1–9 實數）。
評分：Valence / Arousal 各算 MAE（↓）+ PCC（↑），共 4 指標取 mean rank。

> **指標一致性**：本專案 `train.py` 的評估與官方 `scoring.py` 數學上等價——
> MAE = `sklearn.metrics.mean_absolute_error`、PCC = `scipy.stats.pearsonr[0]`，
> 與我們用的 `np.mean(|pred-gold|)`、`np.corrcoef[0,1]` 同公式。

---

## 共同設定（三個實驗都一樣）

- **Encoder**：`hfl/chinese-macbert-base`
- **架構**：encoder → attention-mask 加權 mean pooling → 2 個回歸頭（valence, arousal）
- **標籤處理**：1–9 正規化到 [0,1]，sigmoid 輸出，推論時反轉回 1–9
- **微調超參**：4 epochs、batch 32、lr 2e-5、max_len 256、AdamW(wd 0.01)、
  warmup 10%、SmoothL1 loss、grad clip 1.0
- **Early-stop 依據**：`mean(PCC) − mean(MAE)` 取最佳 epoch
- **Submission set（三者相同）**：官方 `DSANIDF_ValidationSet.csv`（200 篇，只有 ID+Text，無標籤）

---

## 三個實驗的差異

| | 實驗 1：Baseline | 實驗 2：DAPT | 實驗 3：反思語料版 |
|---|---|---|---|
| **DAPT 續訓** | 無 | ✅ 新住民 200 篇 MLM | 無 |
| **Train set** | EmoBank CVAS+CVAT | EmoBank CVAS+CVAT | EmoBank + DSA-MST + ROCLING-2021 |
| **Train 筆數** | 4,998 | 4,998 | 9,435 |
| **Val set (dev)** | EmoBank 切出（stratified）| EmoBank 切出（stratified）| DSA-MST 反思切出 |
| **Dev 筆數** | ~555 | ~555 | 253 |
| **每 epoch steps** | 157 | 157 | 295 |
| **Encoder 來源** | macbert-base 原權重 | DAPT 續訓後權重 | macbert-base 原權重 |

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

---

## 結果（官方 validation 實際分數）

| 實驗 | Valence MAE ↓ | Valence PCC ↑ | Arousal MAE ↓ | Arousal PCC ↑ |
|---|---|---|---|---|
| 1. Baseline | 0.654 | 0.867 | 0.985 | 0.412 |
| 2. DAPT | 0.649 | 0.866 | 1.009 | 0.395 |
| 3. 反思語料 | 0.627 | 0.870 | 0.944 | 0.388 |
| **4. L1 詞典融合** | **0.600** | **0.880** | **0.882** | **0.426** |

（實驗 2 dev 最佳 epoch 4：V_PCC 0.859 / A_PCC 0.620；實驗 3 dev 最佳 epoch 3：V_PCC 0.822 / A_PCC 0.599；
實驗 4 dev 最佳 epoch 2：V_PCC 0.815 / A_PCC 0.609）

## 重點觀察
1. **Valence 已接近上限**（PCC ~0.88），**Arousal PCC 一直是瓶頸**（~0.39–0.43）。
2. **DAPT（小語料 + 舊資料）沒幫助**，arousal 反而略退 → 缺的是「對的語域的監督標籤」，不是 encoder 表徵。
3. **實驗 3 修好了校準**（雙 MAE 下降、4 指標贏 3 項），但**沒修好排序**（arousal PCC 仍 0.388）。
4. **實驗 4（L1 詞典融合）是第一個 4 指標全勝的版本**：顯性 VA 詞典特徵同時抬升了 arousal 的
   校準與排序（PCC 0.388→0.426），證明「contextual embedding + lexical VA 訊號」互補有效。
5. 下一步仍應**專攻 arousal PCC**：multi-seed ensemble（兩篇得獎論文共識的最大槓桿）、
   更強 encoder（roberta-wwm-ext-large）、把 L2 word→VA 回歸器接進 L1 當額外特徵。
