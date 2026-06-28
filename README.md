# DSA-NIDF — ROCLING 2026 新住民文本維度情緒分析

ROCLING 2026 Shared Task：對**新住民 (new immigrants) 的自我反思文本**預測
**valence（效價）/ arousal（喚醒度）**，兩者皆為 1–9 實數。

- 官方任務頁：<https://rocling-sigai.github.io/task2026/>
- 提交格式：`ID, Valence, Arousal`（皆 1–9 實數）
- 評分：Valence / Arousal 各算 **MAE（越低越好）+ PCC（越高越好）**，共 4 指標取平均排名
- 規模：validation 200 篇、test 1100 篇

---

## 1. 方法 (How it works)

整條 pipeline 是「**中文 encoder + 雙回歸頭**」的端到端微調：

```
文本 → MacBERT encoder → mask 加權 mean pooling → 兩個回歸頭 → (valence, arousal)
```

設計重點：
- **標籤正規化**：1–9 → [0,1]，模型 sigmoid 輸出，推論時反轉回 1–9（收斂穩定）。
- **Pooling**：用 attention mask 加權的 mean pooling 取得 document embedding。
- **Loss**：SmoothL1（對離群值穩健），同時優化 valence + arousal。
- **Early-stop 依據**：`mean(PCC) − mean(MAE)`，直接對齊官方四指標。
- **DAPT（可選）**：先在目標域文本上做 MLM 續訓，讓 encoder 讀熟新住民反思語感，再微調。

預設 encoder：`hfl/chinese-macbert-base`。

### 檔案

| 檔案 | 說明 |
|------|------|
| `prepare_data.py` | 整理所有來源成統一 `(text, valence, arousal)`，切 train / dev |
| `download_external.sh` | 下載官方推薦外部資源（DSA-MST 反思語料，含 VA 標籤） |
| `dapt.py` | 領域適應續訓 (MLM)，在新住民文本上續訓 encoder |
| `train.py` | encoder + 雙回歸頭微調，產生 `outputs/submission.csv` |
| `Rocling2026_Colab.ipynb` | Colab GPU 訓練 notebook |
| `data/` | 處理後的訓練/驗證資料（見下） |
| `external/` | 下載/轉換後的外部語料 |

---

## 2. 原始資料 (Datasets)

### 主要訓練資源：Chinese EmoBank
官方指定的訓練資源 (Lee et al., 2022)，含四種粒度的 VA 標註（1–9）：

| 子集 | 內容 | 數量 |
|------|------|------|
| CVAW | 單字 | 5,512 |
| CVAP | 片語 | 2,998 |
| CVAS | 單句 | 2,582 |
| CVAT | 多句篇章 | 2,969 |

本專案預設使用 **CVAS（句）+ CVAT（篇）**，因為目標文本是篇章級。

### 目標 / 提交資料：DSANIDF（新住民文本）
- `DSANIDF_ValidationSet.csv`：官方 validation 200 篇（只有 `ID, Text`，**無標籤**）。
- 這就是 **submission 的輸入**——模型對它輸出預測，產生 `submission.csv`。

### 額外引入的反思語料（提升用，見第 5 節）
| 來源 | 篩數 | 平均文長 | Arousal std | 角色 |
|------|------|---------|-------------|------|
| EmoBank CVAS+CVAT | 5,553 | 57 | 1.05 | 主要訓練（量大，但域偏書評/新聞）|
| **DSA-MST 醫療反思** | 2,535 | 76 | 1.31 | 訓練 + **dev**（最貼近目標域）|
| **ROCLING-2021 教育反思** | 1,600 | 19 | 1.32 | 訓練（arousal 訊號最強，但偏短）|

> `data/train.csv` 已 append 上述三個來源（共 9,435 筆）。
> `data/orign_train_data.csv` 保留**最初只用 EmoBank** 的版本（4,998 筆），方便做 A/B 對照。

---

## 3. 在 Colab 上訓練 (How to train)

MacBook Air M2 不建議本機訓練，走 Colab 免費 GPU：

**步驟 1（本機）準備資料：**
```bash
cd baseline
bash download_external.sh   # 下載 DSA-MST（第一次才要）
python prepare_data.py      # 產生 data/train.csv, dev.csv, val_unlabeled.csv
```

**步驟 2：** 把整個 `baseline/` 壓成 `baseline.zip`。

**步驟 3（Colab）：** 開 `Rocling2026_Colab.ipynb`，執行階段選 **T4 GPU**，依序執行：
1. 安裝套件
2. 上傳並解壓 `baseline.zip`
3. （可選）`dapt.py` 領域適應續訓
4. `train.py` 微調 → 產生 `outputs/submission.csv`
5. 下載 submission

本機 Mac MPS 也可跑（較慢）：`python train.py --epochs 4 --batch_size 8`。

---

## 4. 目前的兩個實驗 (Experiments)

**三個資料集的角色固定：**
- **Train set**：模型學習用（EmoBank / 或加反思語料）。
- **Valid set (dev)**：調參與 early-stop 用，**不參與訓練**。
- **Submission set**：官方 DSANIDF validation 200 篇——**兩個實驗都用同一份**，輸出上傳評分。

| | 實驗 1：Baseline | 實驗 2：DAPT | 實驗 3：反思語料版 |
|---|---|---|---|
| Encoder | macbert-base | macbert-base | macbert-base |
| DAPT | 無 | 新住民 200 篇 MLM (30 ep) | _(待定)_ |
| **Train set** | EmoBank CVAS+CVAT (4,998) | EmoBank CVAS+CVAT (4,998) | EmoBank + DSA-MST + ROCLING-2021 (9,435) |
| **Valid set (dev)** | EmoBank 切出 (~555) | EmoBank 切出 (~555) | DSA-MST 反思切出 (253) |
| **Submission set** | DSANIDF validation 200 | DSANIDF validation 200 | DSANIDF validation 200 |

### 結果

**實驗 1（Baseline）** — 在官方 validation 的實際分數：

| | MAE ↓ | PCC ↑ |
|---|---|---|
| Valence | 0.654 | **0.867** |
| Arousal | 0.985 | **0.412** |

**實驗 2（DAPT）** — 官方 validation 實際分數：

| | MAE ↓ | PCC ↑ |
|---|---|---|
| Valence | 0.649 | 0.866 |
| Arousal | 1.009 | 0.395 |

> 與實驗 1 為乾淨 ablation（同 train/dev，只差有無 DAPT）：**DAPT 沒幫助，arousal 反而略退**
> （PCC 0.412→0.395、MAE 0.985→1.009）。推論：DAPT 只改 encoder 無監督表徵，但回歸頭學的
> 仍是 EmoBank 域的 arousal 分布，補不到「label 層級」的領域落差；加上語料僅 200 篇太小。
> → 真正的解法是換訓練語域的**監督訊號**（實驗 3）。

**實驗 3（反思語料版）** — _(待跑)_

| | MAE ↓ | PCC ↑ |
|---|---|---|
| Valence | _待填_ | _待填_ |
| Arousal | _待填_ | _待填_ |

> 注意：實驗 2 這次跑的 train set 仍是**原始 EmoBank**（157 steps ≈ 5,000 筆），
> 尚未吃到新加的反思語料；實驗 3 才是真正換語域的版本。

---

## 5. 觀察與分析 (Observations)

1. **Valence 已經很強、Arousal 是瓶頸。** Valence PCC 0.867（實測）幾乎到頂；
   Arousal PCC 只有 0.412，是排名的主要失分點。

2. **Arousal 崩在「領域落差」。** dev（EmoBank 域）arousal PCC 有 0.64，一到新住民文本就掉到
   0.41。原因是 EmoBank 是書評/新聞，學到的 arousal 線索搬不到第一人稱反思文本。

3. **Arousal 預測被壓縮。** submission 的 arousal 預測 std 只有 ~0.68（valence ~1.28），
   模型傾向往平均值靠 → 直接拉低 PCC。

4. **DAPT（舊資料）實測沒幫助、arousal 反而略退**（PCC 0.412→0.395）。原因是 DAPT 只改
   encoder 無監督表徵，回歸頭仍學 EmoBank 域的 arousal 分布，補不到 label 層級的落差；
   且語料僅 200 篇太小。→ 缺的是「對的語域的**監督標籤**」，不是 encoder 表徵。
   （DAPT 之後可在 test 1100 篇釋出、語料變大後，疊在實驗 3 上再評估。）

5. **最有希望的一步：換訓練資料的「語域」。** 與其只靠 EmoBank，引入 **DSA-MST（醫療反思）**
   與 **ROCLING-2021（教育反思）**——同為第一人稱情緒反思、VA 1–9、繁中，且 arousal 變異度
   更大——直接用「對的語域」的監督訊號教模型。`train.csv` 已備好此版本待跑。

---

## 6. 官方推薦的相關資源 (Recommended Resources)

主辦單位推薦了以下相關情感資源，本專案逐一評估：

| 資源 | 內容 | 對本任務 | 是否採用 |
|------|------|---------|---------|
| **ROCLING-2025 DSA-MST** | 醫師 ICU 反思文本，文檔級 VA 1–9，繁中，2,535 篇含答案 | 同型任務、文長/語域/尺度全對齊，**最貼近目標域** | ✅ 訓練 + dev |
| **ROCLING-2021 教育 DSA** | 學生課堂回饋短文，VA 1–9，繁中，1,600 篇 | 反思語域、arousal 變異最大，但文本偏短 | ✅ 訓練（補 arousal）|
| **DimABSA2026** | 6 語言 aspect-based VA（餐廳/筆電/旅館/財經…），`V#A` 格式 | aspect 級、評論域，需轉換、域不合 | ❌ 暫不採用 |
| **SIGHAN2024-dimABSA** | 中文餐廳評論 aspect-based VA，繁簡皆有 | aspect 級、評論域 | ❌ 暫不採用 |

採用原則：**文檔級 + 反思語域 + VA 1–9 直接可用**的優先（DSA-MST、ROCLING-2021）；
aspect 級/評論域的（DimABSA2026、SIGHAN2024）因 domain 與格式落差大，暫不納入。

---

## 7. 後續方向 (Next steps)

1. 跑「加入反思語料」的訓練版（`train.csv` 9,435 筆）與 baseline 對照。
2. test 釋出後，把 1,100 篇併入 DAPT 語料重做續訓。
3. Arousal 專項：loss 加權、抗壓縮校正。
4. 換更強 encoder（roberta-wwm-ext / macbert-large）、multi-seed ensemble。
