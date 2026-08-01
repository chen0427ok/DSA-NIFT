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

## 2. 目前最佳模型 = 實驗 4（valence）＋ E19（arousal）雙雄並立

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
| 18. L1++ intensity 31 維（E18） | 0.641 🔴 | 0.865 | 0.909 🔴 | 0.408 🔴 |
| **19. Source-aware arousal loss** ⭐ | 0.666 🔴 | 0.869 | **0.870** 🟢 | **0.452** 🟢 |
| 20. Ranking-only 增強（E20） | 0.631 | 0.877 | 0.923 🔴 | 0.407 🔴 |
| 21. Dim-attention + rank（E21b） | 0.638 | 0.865 | 0.923 🔴 | 0.394 🔴 |
| 11b. RoBERTa-large 單顆 | 0.691 🔴 | 0.869 | 0.914 🔴 | 0.407 🔴 |

**E19 是第一個不靠合成資料把 A_PCC 推過 0.45 的模型，且 A_MAE 同時優於實驗 4（arousal 雙贏）；
代價是 valence 退化（V_MAE 0.666）→ 與實驗 4 各贏兩指標，官方 mean rank 相當。**
E18/E20/E21/robertaL 官方確認淘汰。**E18 官方分數揭露：E19 的增益全來自 source-aware 加權、
31 維強度特徵反而有害 → 下一發首選 `--lex_mode l1 --source_aware`（10 維 L1 + 來源加權）。**
完整細節見 `experiment.md` 實驗 18–21 章。

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

## 3.6 E18–E21 + Silver Ranking benchmark + E22 現況（2026-07-21 更新）

「Enhanced L1」（上面 §7 舊版下一步 #1）已經跑完並提交，官方結果見 §2 表格與 `experiment.md` 實驗 18–21 章。
這裡記本階段新增的**工具**、**silver ranking 的實戰經驗**、以及**目前卡住的 E22**。

### 新增程式（`feat/ensemble-teacher-student-experiments` branch，已 push）
| 檔案 | 用途 |
|---|---|
| `lexicon_intensity.py` | L1 十維 → 31 維（+15 arousal intensity + 6 新住民 domain cues），`--lex_mode l1_intensity` |
| `train_v2.py` 新旋鈕 | `--source_aware`（依 granularity 對 V/A loss 加權）、`--rank_aug`（合成資料 ranking-only pairwise hinge）、`--pooling mean_cls_dim_attention` |
| `Rocling2026_Colab_e18_e21.ipynb` | E18→E21 Colab 流程；cell 2 已修成「repo 存在時自動 checkout+pull」 |
| `build_silver_pairs.py` | 對官方 200 篇無標籤文本抽 pair，LLM（claude-opus-4-8）pairwise 判斷 arousal/valence 相對高低 → `data/silver_pairs_{judge}.csv` |
| `eval_silver_ranking.py` | 讀 silver pairs + `outputs/preds/{run}_val.csv`，算各 run 的 pairwise ranking accuracy |

### Silver ranking benchmark 的真實戰績（重要，別高估它）
動機：dev（DSA-MST）對增強類實驗會失真（實驗 9 教訓），想找一個更貼目標域的模型選擇依據。
跑了一輪（單一 judge = claude-opus-4-8，500 pairs），結果：
- **對 e21 的否決是準的**（silver 排最後，官方 A_PCC 也確實最差 0.394）。
- **但對 e19/e20 的相對排序判斷錯了**：silver 押 e20（0.766）> e18（0.762）> e19（0.748，與 E4 打平）；
  官方分數卻是 e19（A_PCC 0.452，本階段最佳）> e20（0.407）> e18（0.408）。
- **結論／定位**：silver ranking 的 pairwise accuracy 對「排序方向」判斷不夠可靠（可能與 PCC 依賴的分布形狀無關、
  且只有單一 judge），**只適合當「否決過濾器」（篩掉明顯壞的，如 e21），不能拿來當「選第一名」的依據**。
  之前建議的「多 judge 一致性過濾」（`build_silver_pairs.py --provider openai` 再標一輪）**還沒做**，
  若要讓這個工具更可信，這是下一步；但即使做了，也該把它當輔助訊號，決策仍以官方分數為準。

### 關鍵消融發現：E19 的增益 100% 來自 source-aware，不是強度特徵
把三個官方分數排起來看：
```
E18（31 維強度特徵，無 source_aware）   A_PCC 0.408  ← 比 E4 還差
E4（10 維 L1，baseline）                A_PCC 0.426
E19（31 維 + source_aware）             A_PCC 0.452  ← 目前最佳
```
E19 = E18 + `--source_aware`。E18 單獨用反而傷 arousal，代表 31 維強度特徵本身可能是雜訊／對訓練域過擬合，
**E19 的增益完全是 source-aware 加權的功勞**。這指出一個乾淨的下一發：

### E22（下一發，尚未成功跑出結果）
```bash
python train_v2.py --lex_mode l1 --source_aware --run_name e22_l1_source_aware \
    --epochs 4 --batch_size 32 --lr 2e-5
```
拿掉 31 維強度特徵，只留 10 維原始 L1 + source-aware 加權。假說：這樣既能保留 E19 的 arousal 增益，
又能減少 E19 造成的 valence 退化（V_MAE 0.600→0.666）——因為強度特徵被懷疑是拖累 valence 的來源之一
（未證實，是本次假說，跑出來才知道）。

**現況：本機（M2 Air `mps`）跑了三次都失敗，尚未產出任何結果。**
- 第 1、2 次：訓練用 `nohup` 背景啟動，但 Claude Code session 中途重啟（非使用者主動終止），
  process 被一併殺掉，`outputs/e22_train.log` 只停在 HF 模型載入階段（`BertModel LOAD REPORT`），
  連第一個 training step 都沒開始印。
- 第 3 次：process 存活了較久，但**診斷出是真的卡住、不是在慢慢跑**——用 `ps -o etime,time` 查：
  跑了 **5 小時 19 分鐘 wall time，只累積 9 分 58 秒 CPU time**，卡在第一個 batch 的 forward/backward。
  判斷：**M2 Air 的 `mps` backend 跑 batch_size=32 / max_len=256 這個設定會 hang**（之前的小 smoke test
  用的是 batch 8 / max_len 64，撐得住；正式設定沒試過）。使用者發現後下令 `幫我停掉`，已用 `kill` 清乾淨
  （`ps aux | grep train_v2.py` 確認為 0）。
- **結論：本機 MPS 不適合跑這個設定的正式訓練，之後 E18–E21 之後的新實驗都應該直接上 Colab（A100），
  別再嘗試本機 batch 32 訓練。**（本機只適合先前用過的小 batch smoke test，驗證程式碼邏輯用。）
- **`Rocling2026_Colab_e18_e21.ipynb` 已加了 cell 18（未 commit）**：內容就是上面那行 e22 指令，
  可以直接在 Colab 上執行；跑完記得執行打包 cell（cell 10/12）把 submission 拉回或 push 回 branch。

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
- **E18–E21 新增**：`lexicon_intensity.py`（L1++ 31 維）、`build_silver_pairs.py` / `eval_silver_ranking.py`
  （silver ranking benchmark，見 §3.6）、`Rocling2026_Colab_e18_e21.ipynb`
- 規劃/交接：`future_experiments.md`（E10–E17）、`session_handover.md`、`dsa_nift_next_experiments_plan.md`（E18–E27 規劃）
- notebook：`Rocling2026_Colab_ensemble.ipynb`、`Rocling2026_Colab_e13.ipynb`
- 資料：`data/train_aug_pseudo.csv`（318 篇偽標，**已 `git add -f` 進 branch**，供 Colab clone）、
  `data/silver_pairs_claude-opus-4-8.csv`（silver ranking 500 pairs 標註，已 commit）
- `.gitignore` 新增 `*.zip`、`data/train_aug_pseudo.csv`（後者靠 `-f` 強制追蹤）

### **未追蹤（gitignore）**——交接時要注意
- `data/train_aug.csv`（400 篇增強，**本機生成、不在 git**）、`data/train_base.csv`
- `outputs/`（含 `l2_word_va.pkl` 814MB、`l3_graph.pkl`、**7+ 顆 teacher/實驗權重 `*_best.pt`**、各 `*_submission.csv`、
  `e22_train.log` 本機失敗訓練的殘留 log）、`*.pt`、`*.zip`、`.venv/`
- 本機 `outputs/` 已備妥全部權重（E10–E13、E18–E21），**沒有 e22 的權重**（三次本機訓練皆未完成，見 §3.6）。

### ⚠️ 目前工作目錄狀態（2026-07-21，交接時待處理）
- **`git status` 顯示 3 個 notebook dirty 但未 commit**：`Rocling2026_Colab_e13.ipynb`（+312/-0 行）、
  `Rocling2026_Colab_e18_e21.ipynb`（+674/-92，含未 commit 的 e22 cell 18）、
  `Rocling2026_Colab_ensemble.ipynb`（+787/-66，**且仍嵌著未撤銷的 GitHub token**，見 §6）。
  這些多半是 Colab/VS Code 執行後同步回來的 cell outputs，commit 前建議先看過 diff、
  尤其 ensemble 版**千萬別把 token 一起 commit 上去**。
- **未追蹤且來源不明**：`CLAUDE.md`、`docs/agents/{domain,issue-tracker,triage-labels}.md`——
  內容與本專案無關（像是別的 agent skill 樣板），先跟使用者確認再處理。
- **上層目錄 `../` 有多個 submission zip**（`submission_e18_l1_intensity.csv.zip`、`submission_e19.csv.zip`、
  `submission_e20_rank_aug.csv.zip`、`submission.csv.zip`、`submission_v0/v1/v2/v3.csv.zip`）——
  e18/e19/e20/e21b/robertaL 的官方分數都已經拿到並記錄，這些 zip 多半已完成任務；
  `v0`–`v3` 命名不符合目前慣例（`submission_{run}.csv.zip`），來源不明，可能是更早的手動打包，
  清理前先確認不是還沒上傳的東西。

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
- **本機 `mps` 訓練會在正式 batch size 下 hang，且不易發現**：M2 Air 用 `--batch_size 32 --max_len 256`
  跑 `train_v2.py` 時，process 存活但卡在第一個 training step 不動——`ps aux` 看得到 process、
  但 `outputs/{run}_train.log` 永遠停在 HF `BertModel LOAD REPORT` 之後，不會印出 `[epoch 1] step50/...`。
  **判斷方法**：`ps -o etime,time -p <pid>` 比對 wall time 與 CPU time，如果 wall time 遠大於 CPU time
  （例：跑了 5 小時只累積 10 分鐘 CPU），代表卡住而非慢跑，直接 `kill` 別等。
  **教訓：本機只拿來跑小 batch（如 8）+ 短 max_len（如 64）的 smoke test 驗證程式碼，
  正式訓練（batch 32 對照組）一律上 Colab A100，不要在本機嘗試。**
- **Claude Code session 中斷會殺掉背景訓練**：用 `run_in_background: true` 啟動的訓練，若 session 重啟
  （非使用者主動關閉），process 會被一併終止、且不留下錯誤訊息，只會在下次啟動時看到
  「No completion record was found」。若需要長跑訓練撐過可能的 session 中斷，
  改在使用者自己的終端機（不透過 Claude Code）用 `nohup ... &` 啟動，與 Claude Code 脫鉤。
- **⚠️ 安全**：`Rocling2026_Colab_ensemble.ipynb`（未 commit 的工作目錄版本）**目前仍嵌著一個 GitHub token**
  （`ghp_qljV...`，cell 2）。已提醒過使用者去 GitHub 撤銷重發，**尚未確認是否已撤銷**。
  這個檔案目前是 dirty（未 commit），千萬別把它原樣 commit 上去；新版 notebook
  （`Rocling2026_Colab_e18_e21.ipynb`）已改用 `getpass` 輸入 token，沒有這個問題。
- **未追蹤的 `CLAUDE.md` / `docs/agents/*.md`（issue-tracker、triage-labels、domain）來源不明**：
  這些檔案內容看起來像某個通用 agent skill 的樣板（issue tracker、triage 標籤、domain doc 佈局），
  與 DSA-NIFT 情感分析任務本身無關，也不在 git 歷史裡。**不確定是誰、何時建立的**——
  交接時看到不要假設它們是這個專案需要的東西，先跟使用者確認再決定保留或刪除。

---

## 7. 下一步（優先序，2026-07-21 更新）

專攻 **Arousal PCC 同時不犧牲 Valence**（E19 已破 arousal 瓶頸，但 valence 退化）。
**已排除**：ensemble（死路）、純 teacher 偽標（失 PCC）、E18 單獨強度特徵（拖累）、E20 ranking-only、
E21 dim-attention、roberta-large 單顆（皆四指標輸實驗 4）。

1. **🏆 立即待辦：E22（`--lex_mode l1 --source_aware`，10 維 L1 + 來源加權，拿掉強度特徵）——三次本機嘗試皆失敗，
   尚未有結果，需上 Colab 跑**：
   ```bash
   python train_v2.py --lex_mode l1 --source_aware --run_name e22_l1_source_aware \
       --epochs 4 --batch_size 32 --lr 2e-5
   ```
   `Rocling2026_Colab_e18_e21.ipynb` cell 18 已有這行（未 commit），直接在 Colab A100 執行即可
   （**別在本機 M2 跑，`mps` 在這個 batch size 下會 hang**，見 §6 環境雷點）。
   跑完 `fetch_results.py --zip <下載的 zip>` 解壓、`--pack e22_l1_source_aware` 打包提交。
2. **若 E22 修好 valence（同時保留 arousal 增益）→ 新王，優先解耦 E19 的 valence 退化**：
   (a) `--source_weights` 讓 CVAS/CVAT 的 V 權重 >1 補償；(b) V/A 分開 early-stop；
   (c) `calibrate.py`（E26）只後校準 E19 的 V_MAE，不動排序。
3. **Silver ranking 補第二個 judge**（`build_silver_pairs.py --provider openai`，同一批 500 pairs，同 seed）→
   `eval_silver_ranking.py` 用雙 judge 一致性過濾，看能否修正上一輪「押錯 e19 排名」的問題；
   即使修正了也只當否決過濾器用，不取代官方分數。
4. **平行（為論文非為分數）：E13 blend 變體**——`build_pseudo_labels.py --blend 0.3/0.5` 生成
   bin(排序)×teacher(校準) 中間點，測 trade-off 曲線；或「高喚醒 bin 用 teacher、低喚醒 bin 用 moderate 標籤」分端策略。
5. **論文必做消融**：L3 引導 vs 隨機種子（證明知識圖譜）、`--no_lexicon`/L1/L1+強度/L1+source_aware 對照表
   （E4/E18/E19/E22 剛好是這張表的四個格子）、raw/teacher/blend 標籤曲線。
6. 已備未跑：E14（`embed_regressor.py` frozen emb+SVR）、E16（`lexicon_l2.py` L1+L2）、E17（`calibrate.py` 只修 MAE）。
7. `train_v2.py` 已支援 `--mae_weight`/`--pcc_weight`/`--arousal_weight`，要讓 early-stop 或 loss 更偏排序可直接調。

---

## 8. 專案約束（務必遵守）

- **一律用繁體中文回覆。**
- **commit message 不得提及 Claude 共同作者。**
- 功能開在各自 branch；git remote：`https://github.com/chen0427ok/DSA-NIFT.git`。
- 已跑過的實驗程式碼**不可覆蓋**、必須可復現（`train.py` 的 `--append`/`SHRINK` 都設計成可還原原始 train.csv）。
