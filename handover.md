# Handover — DSA-NIDF ROCLING 2026 情緒維度分析

> 交接文件。涵蓋任務、目前最佳模型、本階段（L2/L3/生成增強）的完整過程與結論、如何復現、環境雷點、下一步。

---

## 1. 任務

對**新住民自我反思文本**預測 valence / arousal（1–9 實數，維度情緒）。

- **輸出格式**：`ID, Valence, Arousal`
- **評分**：Valence / Arousal 各算 **MAE（↓）+ PCC（↑）**，共 4 指標取 **mean rank**。
- **官方提交集**：`data/val_unlabeled.csv`（= `DSANIDF_ValidationSet`，200 篇，只有 ID+Text，無標籤）。
- **目前瓶頸**：**Arousal PCC**（長期卡在 ~0.39–0.46）。Valence 已近上限（PCC ~0.88）。

環境：使用者 MacBook Air M2 本機開發（`.venv`）、**Colab 訓練**。
本 session 起改用 **VS Code + 官方 Google Colab 擴充套件 + Colab Pro（A100-40GB）**：
本機 notebook、遠端 Colab 運算 → 資料靠 **git clone**（非本機檔案）、產出在 VM `outputs/`（斷線即失，要主動拉回）。
詳見 §5、`session_handover.md`。

---

## 1.5 資料集（Dataset）

**關鍵前提：官方 DSA-NIDF 沒給「有標註」的訓練資料**，只給 `val_unlabeled.csv`（200 篇新住民文本，**無標籤**=提交目標）。
所以**全部訓練標註都是相鄰域的借用語料，零目標域監督樣本** → 這是 low-resource 本質、也是 arousal 難的根源。

**訓練集 `data/train.csv`（9,435）= `prepare_data.py` 合併：**
| 來源 | 筆數 | granularity | 說明 |
|---|---|---|---|
| Chinese EmoBank CVAS | 2,583 | `sentence` | 句子級 VA |
| Chinese EmoBank CVAT | 2,970 | `text` | 篇章級 VA |
| DSA-MST（ROCLING-2025 醫療反思） | 2,282 | `reflection` | 全 2,535，另 253 → dev |
| ROCLING-2021 教育反思 | 1,600 | `edu2021` | 教育反思短文 |

- **dev（253）** 從 DSA-MST 切（最接近目標域，但與合成資料同風格 → 增強實驗 dev 失真）。
- **合成**：`data/train_aug.csv`（400，Opus 4.8 生成、L3 引導、gitignore）、`data/train_aug_pseudo.csv`（318，teacher 重標）。
  `train_aug.csv` 是**唯一「目標域風格 × 目標 arousal 區間」的監督樣本**（實驗 9 提 A_PCC 的來源）。
- **詞典**：`external/emobank/` CVAW+CVAP = 7,761 VA 詞典（餵 L1/L2/L3，非訓練樣本）。
- `data/orign_train_data.csv`（4,998，EmoBank-only）：實驗 1–2 用，保留供復現。

詳細 dataset 章節見 `experiment.md`。

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
| 10. 純 macbert 5-seed Ensemble | 0.614 | 0.881 | 0.907 🔴 | 0.415 🔴 |
| 12. 多 encoder Ensemble | 0.613 | 0.882 | 0.906 🔴 | 0.418 🔴 |
| 13. Teacher 偽標增強（blend=0） | 0.617 | 0.878 | 0.898 | 0.423 |

**實驗 4 仍是唯一 4 指標全勝、目前該提交的模型**（本 session 跑的 E10/E12/E13 官方分數皆未超越它）。完整細節見 `experiment.md`。

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

## 3.5 本 session 新增：E10–E13（Ensemble + Teacher-Student）與結論

本 session 在 A100 上把 handover 原「下一步」清單實跑到底，並用**官方分數**得出明確結論。
新程式全部獨立於凍結的 `train.py`（實驗 1–9b 仍可復現）。

### 新增程式（`train.py` 不動）
| 檔案 | 用途 |
|---|---|
| `train_v2.py` | 擴充訓練器：統一預測輸出 + multi-seed / 多 encoder / arousal-weighted / PCC-loss / `--lex_mode` 旋鈕；**預設參數復現實驗 4** |
| `ensemble.py` | dimension-wise weighted / mean 融合（V、A 分開權重） |
| `build_pseudo_labels.py` | multi-teacher 逐篇偽標 + 分歧過濾 + 每 bin ±1.5SD 離群移除（`--blend` 可混 bin/teacher 標籤） |
| `embed_regressor.py` | frozen embedding + SVR/Ridge（E14，未跑） |
| `lexicon_l2.py` | L1+L2 OOV 詞 VA 特徵 16 維（E16，未跑） |
| `calibrate.py` | arousal 後校準（只改 MAE，PCC 不變；E17，未跑） |
| `fetch_results.py` | 把 Colab 下載的 `*_results.zip` 解進 `outputs/`，並 `--pack <run>` 打包成 `submission.csv.zip` |

統一預測格式：`outputs/preds/{run}_{dev,val}.csv`；規劃見 `future_experiments.md`（E10–E17）。

### E10/E11/E12（Ensemble）— **官方確認：對 arousal 是死路**
- E10（純 macbert 5-seed，mean）A_PCC **0.415**、E12（+roberta base/large，weighted/mean）A_PCC **0.418**，
  **都輸實驗 4（0.426）**，A_MAE 也更差（~0.907）。
- 診斷：**多模型平均把已壓縮的 arousal 再壓一次**（校準與排序雙輸）；拿掉 roberta 也沒救（E10≈E12）→
  **傷害不是 roberta-large，而是「平均」本身**。roberta-large 最弱（dev A_PCC 0.584、epoch 3–4 過擬合）。
- **⇒ 堆模型/融合 E10–E12 全數確認死路，別再花提交額度。**

### E13（Teacher 偽標精修）— **官方確認：修好校準、丟了排序**
- 本機用 3 顆 teacher（macbert_s42/s1 + roberta_s42）對 `train_aug.csv` 逐篇重標 + 每 bin ±1.5SD 離群 →
  `data/train_aug_pseudo.csv`（**400→318 篇**，已 push）。上 Colab `train_v2.py --extra_train` 訓練。
- 官方：A_MAE **1.100（實驗9）→ 0.898**（✅ teacher 校準修回 0.20）；但 A_PCC **0.461 → 0.423**（❌ 排序增益一起丟）。
- **關鍵診斷（自我參照陷阱）**：teacher 本身 arousal 壓縮（偽標把**低喚醒 bin 標成 ~5.0、認不出低端**），
  用它重標把撐開 arousal 排序的訊號一起壓平 → PCC 掉回。
- **⇒ arousal 增強有「校準 ↔ 排序」根本 trade-off**：raw 極端標籤給 PCC/傷 MAE；teacher 標籤救 MAE/失 PCC。

### 論文故事線（workshop）
1. L1 詞典融合（正面，實驗 4）→ 2. ensemble 救不了 arousal（負面，E10–E12）→
3. 增強能提 PCC 但有校準↔排序 trade-off（實驗 9 / E13，核心分析）→ 4. **Enhanced L1 從架構面繞開 trade-off**（收尾）。
需補的消融：**L3 引導 vs 隨機種子**（證明知識圖譜必要）、`--no_lexicon`/L1/L1+強度、raw/teacher/blend 標籤曲線。

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

### 本 session 新增（在 branch `feat/ensemble-teacher-student-experiments`，未合回 main）
- 程式：`train_v2.py`、`ensemble.py`、`build_pseudo_labels.py`、`embed_regressor.py`、`lexicon_l2.py`、`calibrate.py`、`fetch_results.py`
- 規劃/交接：`future_experiments.md`（E10–E17）、`session_handover.md`
- notebook：`Rocling2026_Colab_ensemble.ipynb`、`Rocling2026_Colab_e13.ipynb`
- 資料：`data/train_aug_pseudo.csv`（318 篇偽標，**已 `git add -f` 進 branch**，供 Colab clone）
- `.gitignore` 新增 `*.zip`、`data/train_aug_pseudo.csv`（後者靠 `-f` 強制追蹤）

### **未追蹤（gitignore）**——交接時要注意
- `data/train_aug.csv`（400 篇增強，**本機生成、不在 git**）、`data/train_base.csv`
- `outputs/`（含 `l2_word_va.pkl` 814MB、`l3_graph.pkl`、**7 顆 teacher 權重 `*_best.pt`**、各 `*_submission.csv`）、`*.pt`、`*.zip`、`.venv/`
- 本機 `outputs/` 已備妥全部權重（E10–E13），E13 偽標/後續實驗不用重訓 teacher。

### zip 打包（給 Colab）
```bash
cd /Users/brian/Rocling2026/baseline
git archive --format=zip -o baseline_aug.zip HEAD           # 追蹤檔（1.5MB，不含大 pkl/pt）
zip -gq baseline_aug.zip data/train_aug.csv Rocling2026_Colab_aug.ipynb experiment.md  # 補未追蹤/最新檔
```
`baseline_aug.zip` 已含生成好的 `train_aug.csv`，Colab 端**不需重呼叫 API**即可跑增強實驗。

---

## 5. 如何復現

### 舊法（zip 上傳，Colab 網頁版）
- **實驗 4（最佳，提交用）**：上傳 `baseline.zip` → `Rocling2026_Colab.ipynb` 跑 L2→L3→cell 7，
  或 `python train.py --epochs 4 --batch_size 32 --model hfl/chinese-macbert-base`。
- **實驗 9 / 9b（增強）**：`Rocling2026_Colab_aug.ipynb`，cell 3 設 `SHRINK`（1.0=實驗9、0.6=實驗9b）。
- **消融/其他**：實驗 1 加 `--no_lexicon`；實驗 2 先 `python dapt.py`；資料重建 `python prepare_data.py`。

### 新法（本 session 採用：VS Code + Colab 擴充 + A100）
- notebook：`Rocling2026_Colab_ensemble.ipynb`（E10–E12）、`Rocling2026_Colab_e13.ipynb`（E13）。
- **cell 2 用 `git clone` 拉 branch**（不是上傳 zip；private repo 用 getpass 輸入 token，**別把 token 寫進 notebook**）。
  若 `repo` 已存在會跳過 clone → 缺新檔時要 `git pull` 或砍掉重 clone。
- 產出在 VM `outputs/`，**斷線即失**：用 cell 打包 → Drive/下載拉回；submission 小檔可 `git add -f ... && git push`。
- 本機收尾：`python fetch_results.py`（解 `~/Downloads/*_results.zip` 進 `outputs/`）、
  `python fetch_results.py --pack <run>`（打包 `submission.csv.zip` 上傳評分站）。
- **偽標（E13）在本機跑**（純推論、免 GPU）：`build_pseudo_labels.py --teacher name=ckpt ...`（7 顆 teacher 權重已在本機 `outputs/`）。

### 提交格式雷點（評分站 = CodaLab/Codabench）
- 要傳的是 **zip**，內部檔名必須是 **`submission.csv`**（欄位 `ID,Valence,Arousal`，200 列）。
  裸 csv 或錯檔名 → 報「Could not find scores file」。用 `fetch_results.py --pack` 會自動正名 + 去 macOS 雜項。

---

## 6. 環境雷點（踩過的坑）

- **`.venv` 是 `uv venv`，沒有 pip**：裝套件用 `uv pip install ...`，別用 `.venv/bin/python -m pip`（會報 No module named pip）。
- **本機系統 `python3` 沒有 pandas**：跑分析腳本要用 `.venv/bin/python`。
- **anaconda base 載 `l2_word_va.pkl` 會炸**（`cannot import name 'triu' from 'scipy.linalg'`，gensim/scipy 版本衝突）→
  必須用 `.venv`；Colab install 已 pin `gensim>=4.3.3`。
- **`git checkout` 報 "Unable to read current working directory"**：shell 抓到被刪 inode → 重新 `cd` 進 baseline。
- **API key**：`augment_generate.py` 需 `ANTHROPIC_API_KEY`（或 `OPENAI_API_KEY` + `--provider openai`）。
  **Claude Code 的 Bash sandbox 沒有這些 key**，生成要在使用者自己終端機或 Colab（用 🔑 Secrets）跑。生成會計費。
- 改 `.ipynb`：用 `NotebookEdit` 工具或 `python -c "import json..."` 直接改 JSON（改完 `json.load` 驗證）。
  **VS Code 開著且在跑時會自動存檔覆蓋外部修改** → 要改正在跑的 notebook 的 cell，請在 VS Code 介面改。
- **VS Code + Colab 是遠端運算**：本機 `data/` 不在 VM 上，一律 git clone；產出要主動拉回，斷線即失。
- **git push 常被擋**（你從 Colab VM push 過 submission）：本地 push 前先 `git fetch` → rebase；
  遠端 commit 常加 `outputs/*.csv`（tracked），與本機同名未追蹤檔衝突時，先 `rm` 那些未追蹤 csv（保留 `.pt`/`.pkl`）再 rebase。
- **偽標的 `--blend`/離群移除是本機生成**：teacher 權重只在本機，別上傳 3.5GB 到 Colab；偽標產出小檔再帶上去。

---

## 7. 下一步（優先序）

專攻 **Arousal PCC**（唯一瓶頸）。**本 session 已排除 ensemble（死路）與純 teacher 偽標（失 PCC）**。

1. **🏆 首選：Enhanced L1（精進實驗 4 架構本身，尚未實作）**——在 `lexicon.py` 加 arousal **強度表面特徵**
   （驚嘆/問號密度、程度副詞 超/非常/完全、身體反應詞 發抖/心跳/喘不過氣、字元/詞重複、句長節奏），
   原 10 維 → ~16–18 維，`train_v2.py --lex_mode l1_intensity` 切換（不覆蓋原檔）。
   **理由**：不動資料分布 → 繞開 E13 的「校準↔排序」trade-off，只加分不傷 valence/MAE，直攻 arousal 本質（intensity）。
   驗證：本機看特徵合理 → Colab **嚴格 batch 32** 對照實驗 4。
2. **平行（為論文非為分數）：E13 blend 變體**——`build_pseudo_labels.py --blend 0.3/0.5` 生成
   bin(排序)×teacher(校準) 中間點，測 trade-off 曲線；或「高喚醒 bin 用 teacher、低喚醒 bin 用 moderate 標籤」分端策略。
3. **論文必做消融**：L3 引導 vs 隨機種子（證明知識圖譜）、`--no_lexicon`/L1/L1+強度、raw/teacher/blend 標籤。
4. 已備未跑：E14（`embed_regressor.py` frozen emb+SVR）、E16（`lexicon_l2.py` L1+L2）、E17（`calibrate.py` 只修 MAE）。
5. `train_v2.py` 已支援 `--mae_weight`/`--pcc_weight`/`--arousal_weight`，要讓 early-stop 或 loss 更偏排序可直接調。

---

## 8. 專案約束（務必遵守）

- **一律用繁體中文回覆。**
- **commit message 不得提及 Claude 共同作者。**
- 功能開在各自 branch；git remote：`https://github.com/chen0427ok/DSA-NIFT.git`。
- 已跑過的實驗程式碼**不可覆蓋**、必須可復現（`train.py` 的 `--append`/`SHRINK` 都設計成可還原原始 train.csv）。
