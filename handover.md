# Handover — DSA-NIDF ROCLING 2026 情緒維度分析

> 交接文件。涵蓋任務、目前最佳模型、本階段（L2/L3/生成增強）的完整過程與結論、如何復現、環境雷點、下一步。

---

## 1. 任務

對**新住民自我反思文本**預測 valence / arousal（1–9 實數，維度情緒）。

- **輸出格式**：`ID, Valence, Arousal`
- **評分**：Valence / Arousal 各算 **MAE（↓）+ PCC（↑）**，共 4 指標取 **mean rank**。
- **官方提交集**：`data/val_unlabeled.csv`（= `DSANIDF_ValidationSet`，200 篇，只有 ID+Text，無標籤）。
- **目前瓶頸**：**Arousal PCC**（長期卡在 ~0.39–0.46）。Valence 已近上限（PCC ~0.88）。

環境：使用者 MacBook Air M2 本機開發、**Colab T4 GPU 訓練**。

---

## 2. 目前最佳模型 = 實驗 4（L1 詞典融合）✅ 提交用這個

| 實驗 | Valence MAE ↓ | Valence PCC ↑ | Arousal MAE ↓ | Arousal PCC ↑ |
|---|---|---|---|---|
| 1. Baseline | 0.654 | 0.867 | 0.985 | 0.412 |
| 2. DAPT | 0.649 | 0.866 | 1.009 | 0.395 |
| 3. 反思語料 | 0.627 | 0.870 | 0.944 | 0.388 |
| **4. L1 詞典融合** ⭐ | **0.600** | **0.880** | **0.882** | **0.426** |
| 5. L3 生成增強（k=1.0，bin中心標籤） | 0.611 | 0.870 | 1.100 🔴 | 0.461 🟢 |
| 5b. L3 增強（標籤收縮 k=0.6） | 0.632 | 0.874 | 1.058 🔴 | 0.460 🟢 |

**實驗 4 是唯一 4 指標全勝的版本**，是目前該提交的模型。完整細節見 `experiment.md`。

- 架構：`hfl/chinese-macbert-base` → attention-mask 加權 mean pooling → 2 個回歸頭（V/A）。
- L1 創新：`lexicon.py` 用 Chinese EmoBank 的 **CVAW(字)+CVAP(詞) 共 7,761 詞典**抽 **10 維情緒特徵**，
  concat 進 BERT pooled embedding（`nn.Linear(768+10, 2)`）再進回歸頭 → 顯性 VA 訊號直攻 arousal。
- 超參：4 epochs、batch 32、lr 2e-5、max_len 256、AdamW(wd 0.01)、warmup 10%、SmoothL1、grad clip 1.0。
- 標籤：1–9 正規化到 [0,1]、sigmoid 輸出、推論反轉回 1–9。
- Early-stop：`score = mean(PCC) − mean(MAE)` 取最佳 epoch（**這只是內部挑 epoch 用，不是官方指標；
  PCC~0.7 與 MAE~0.7 相減本來就在 0 附近，score 接近 0 或小負屬正常**）。

---

## 3. 本階段做了什麼（L1 → L2 → L3 → 生成增強）

三層情緒資源，L1 已上線且最佳；L2/L3 是為「生成增強」鋪路的基礎設施。

| 層 | 檔案 | 產出 | 角色 |
|---|---|---|---|
| **L1** | `lexicon.py` + `train.py` | `outputs/best_model.pt` | 詞典特徵融合（**實驗 4，最佳**）|
| **L2** | `word_va_regressor.py` | `outputs/l2_word_va.pkl`（~814MB）| FastText(char n-gram)+SVR 學「詞→VA」，無限覆蓋，餵 L3 |
| **L3** | `affective_graph.py` | `outputs/l3_graph.pkl`（~1.8MB）| 情感知識圖譜（節點=CVAW/CVAP+VA，邊=FastText kNN k=8）|
| **增強** | `augment_generate.py` | `data/train_aug.csv`（400 篇）| 讀 L3 圖 → LLM 生成特定 arousal 的新住民文本 → 補分布 |

### 生成增強（實驗 9 / 9b）—— 這是本階段主戰場

`augment_generate.py` 流程：讀 `l3_graph.pkl` → 對 5 個代表性不足的 VA 區間用 `seeds_for_target()` 撈種子詞
→ 請 `claude-opus-4-8`（或 `--provider openai`）生成新住民第一人稱反思短文 → 標 bin 中心 ± jitter(0.4)
→ jieba 詞集 Jaccard>0.5 近似去重 → 輸出 `data/train_aug.csv`。

**已生成 400 篇**（5 象限各 80，完美平衡）：負/中/正×高喚醒 + 負/正×低喚醒。品質經人工抽檢：完全在域內、
情緒象限正確、低喚醒兩端與中性高喚醒（原訓練集最缺）補得好。**這批文本是好的。**

### ⚠️ 關鍵結論：增強能提 A_PCC，但會傷 A_MAE，且收縮救不回

- **實驗 9（k=1.0，直接用 bin 中心標籤 7.5/2.5）**：A_PCC **0.426→0.461（首次破 0.43）**，
  但 A_MAE **0.882→1.100（爆掉）**。4 指標輸 3 贏 1。
- **實驗 9b（k=0.6，標籤向真實均值收縮）**：想修 MAE。結果 A_PCC 仍守 **0.460**、但 A_MAE 仍 **1.058**（沒回 0.88）。
  即使預測 arousal std 已收回 0.743（≈實驗 4 的 0.755），MAE 依舊高。
- **診斷**：A_PCC 增益來自**文本教的特徵**（不是攤開分布，因 pred std 收回後 PCC 沒掉）；
  A_MAE 的傷害來自**合成文本本身的校準/風格偏移**（LLM 敘事對官方驗證集分布不合），**單純調標籤救不回**。
- **dev 失真**：dev（DSA-MST 反思切出）在增強後 A_MAE 0.84 漂亮，但 official 1.06–1.10 →
  增強資料與反思 dev 同風格，dev 不再是可靠代理，別只看 dev。

**⇒ 實驗 9/9b 兩版都不提交，維持實驗 4。** 但「合成資料能穩定拉高 arousal 排序」是有價值的發現。

---

## 4. 交付物與檔案

### git 追蹤（`baseline/` 是獨立 git repo，root = `/Users/brian/Rocling2026/baseline`）
- `train.py`（實驗 1–5 共用；`--no_lexicon` 消融、`--model` 指定 encoder）
- `lexicon.py` / `word_va_regressor.py` / `affective_graph.py` / `augment_generate.py`
- `dapt.py`（實驗 2 領域續訓）、`prepare_data.py`（合併 EmoBank+DSA-MST+ROCLING-2021）
- `experiment.md`（**實驗全記錄，權威**）、`current_pipeline.md`（架構 + 待跑清單，L3=清單#9）、`pipeline.md`
- `Rocling2026_Colab.ipynb`（L2→L3→L1 全流程）
- `Rocling2026_Colab_aug.ipynb`（**專跑增強實驗**；cell 3 有 `SHRINK` 旋鈕；cell 8 偽標精修草稿）
- `data/train.csv`（9,435 筆）、`data/dev.csv`、`data/val_unlabeled.csv`、`data/orign_train_data.csv`
- `external/emobank/`（CVAW/CVAP 詞典）、`external/DSA-MST/`、`external/ROCLING-2021/`

### **未追蹤（gitignore）**——交接時要注意
- `data/train_aug.csv`（400 篇增強，**本機生成、不在 git**；打包時要手動 `zip -g` 加進去）
- `data/train_base.csv`（`--append` 或 notebook cell 3 首次跑時的原始 train.csv 備份）
- `outputs/`（含 `l2_word_va.pkl` 814MB、`l3_graph.pkl`、`best_model.pt`、`submission.csv`）、`*.pt`、`.venv/`

### zip 打包（給 Colab）
```bash
cd /Users/brian/Rocling2026/baseline
git archive --format=zip -o baseline_aug.zip HEAD           # 追蹤檔（1.5MB，不含大 pkl/pt）
zip -gq baseline_aug.zip data/train_aug.csv Rocling2026_Colab_aug.ipynb experiment.md  # 補未追蹤/最新檔
```
`baseline_aug.zip` 已含生成好的 `train_aug.csv`，Colab 端**不需重呼叫 API**即可跑增強實驗。

---

## 5. 如何復現（Colab T4 GPU）

- **實驗 4（最佳，提交用）**：上傳 `baseline.zip`（或 aug 版）→ `Rocling2026_Colab.ipynb` 跑 L2→L3→cell 7 訓練，
  或直接 `python train.py --epochs 4 --batch_size 32 --model hfl/chinese-macbert-base`。
- **實驗 9 / 9b（增強）**：`Rocling2026_Colab_aug.ipynb`，上傳 `baseline_aug.zip` →
  cell 3 設 `SHRINK`（1.0=實驗9、0.6=實驗9b）→ 訓練。產出 `outputs/{best_model.pt, submission.csv}`，最後 cell 下載回本機。
- **消融/其他**：實驗 1 baseline 加 `--no_lexicon`；實驗 2 先 `python dapt.py` 再 `--model outputs/dapt_macbert`；
  資料重建 `python prepare_data.py`。

---

## 6. 環境雷點（踩過的坑）

- **`.venv` 是 `uv venv`，沒有 pip**：裝套件用 `uv pip install ...`，別用 `.venv/bin/python -m pip`（會報 No module named pip）。
- **本機系統 `python3` 沒有 pandas**：跑分析腳本要用 `.venv/bin/python`。
- **anaconda base 載 `l2_word_va.pkl` 會炸**（`cannot import name 'triu' from 'scipy.linalg'`，gensim/scipy 版本衝突）→
  必須用 `.venv`；Colab install 已 pin `gensim>=4.3.3`。
- **`git checkout` 報 "Unable to read current working directory"**：shell 抓到被刪 inode → 重新 `cd` 進 baseline。
- **API key**：`augment_generate.py` 需 `ANTHROPIC_API_KEY`（或 `OPENAI_API_KEY` + `--provider openai`）。
  **Claude Code 的 Bash sandbox 沒有這些 key**，生成要在使用者自己終端機或 Colab（用 🔑 Secrets）跑。生成會計費。
- 改 `.ipynb`：Claude Code 的 Edit 工具不能改 .ipynb，用 `python -c "import json..."` 直接改 JSON（改完 `json.load` 驗證）。

---

## 7. 下一步（優先序）

專攻 **Arousal PCC**（唯一瓶頸）。增強實驗暫告一段落（淨負，但證明合成資料能提排序）。

1. **提交實驗 4**（目前最佳、4 指標全勝）——先確保有一份可提交的好成績。
2. **若要續攻增強**：teacher 偽標精修（`Rocling2026_Colab_aug.ipynb` cell 8 草稿，依 CYUT 冠軍流程）——
   用實驗 4 當 teacher **逐篇**重標合成文本（非整批收縮）+ 每 bin mean±1.5SD 離群移除，目標留住 A_PCC 增益、修回 A_MAE。
   或先把增強量從 400 減到 100–200 試。
3. **其他 arousal 槓桿**（兩篇得獎論文共識）：**multi-seed ensemble**（最大槓桿）、
   更強 encoder（`roberta-wwm-ext-large`）、把 L2 word→VA 回歸器接進 L1 當額外特徵。
4. 想讓 early-stop 更偏排序，可把 `train.py:173` 的 `score` 改成加權（如 `mean_PCC − 0.5*mean_MAE`）——**只影響挑 epoch，不影響官方分數**。

---

## 8. 專案約束（務必遵守）

- **一律用繁體中文回覆。**
- **commit message 不得提及 Claude 共同作者。**
- 功能開在各自 branch；git remote：`https://github.com/chen0427ok/DSA-NIFT.git`。
- 已跑過的實驗程式碼**不可覆蓋**、必須可復現（`train.py` 的 `--append`/`SHRINK` 都設計成可還原原始 train.csv）。
