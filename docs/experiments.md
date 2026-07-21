# 實驗全記錄 — DSA-NIFT @ ROCLING 2026

> **權威文件。** 所有實驗設定、官方分數與結論以本檔為準。
> 歷史文件（handover、逐次 session 交接、舊規劃）已移入 `docs/archive/`，僅供追溯，不再更新。

任務：對新住民自我反思文本預測 valence / arousal（1–9 實數）。
評分：V / A 各算 MAE（↓）與 PCC（↑），共 4 指標取 mean rank。

---

## 0. 最終官方成績（Test Set，shared task 結束）

**提交內容：E19（`e19_source_aware`）對 `DSANIDF_TestSet.csv`（1,100 篇）的推論結果。**

| | MAE ↓ | PCC ↑ |
|---|---|---|
| **Valence** | **0.6200** | **0.8663** |
| **Arousal** | **0.9259** | **0.3566** |

### ⚠️ 最重要的發現：validation 上的 arousal 增益沒有轉移到 test

| 指標 | E19 @ official val（200 篇） | E19 @ official test（1,100 篇） | 變化 |
|---|---|---|---|
| V_MAE | 0.666 | 0.620 | −0.046 🟢 |
| V_PCC | 0.869 | 0.866 | −0.003 ≈ |
| A_MAE | 0.870 | 0.926 | +0.056 🔴 |
| **A_PCC** | **0.452** | **0.357** | **−0.095 🔴🔴** |

- **Valence 完全穩定**（PCC 0.869 → 0.866），與整個專案的觀察一致：valence 早已飽和、且跨集合可靠。
- **Arousal PCC 崩了 0.095**。0.357 這個水準已經低於本專案在 val 上量到的**所有** run（最差的 E21b 都有 0.394）。
- **成因判讀（這是論文的核心分析，見 §5.4）**：我們整個模型選擇流程是對著**一個 200 篇的 leaderboard 標量**做的，
  用 13 次提交在雜訊上挑第一名 → **典型的 selection overfitting**。

### 量化證據（`analyze_variance.py`，已跑完）

在有 gold 標籤的內部 dev（253 篇）上 bootstrap 抽 n=200 × 5,000 次：

| | 95% CI 寬度 @ n=200 |
|---|---|
| **V_PCC** | **0.083** |
| **A_PCC** | **0.193** |

- **arousal 的評估不確定性是 valence 的 2.31 倍。**
- 我們據以選 E19 而非 E4 的 A_PCC 差距只有 **0.026**——比抽樣噪聲寬度**小了 7.4 倍**。
  ⇒ **validation 上 E4 / E18 / E19 的 arousal 排名在統計上根本無法區分。**
- 對照 seed 變異（5 顆同設定不同 seed，dev 全量）：A_PCC std 僅 **0.0055**、V_PCC std 0.0033。
  ⇒ 問題**不是訓練隨機性**，而是**評估集太小**。

### 補充證據：模型行為沒有漂移，漂移的是評估

用 `predict.py` 以 `e19_source_aware_best.pt` 重跑 test set 推論，預測分布幾乎與 val 完全一致：

| | valence mean / std | arousal mean / std |
|---|---|---|
| val（200） | 5.601 / 1.375 | 5.020 / **0.729** |
| test（1,100） | 5.720 / 1.366 | 5.004 / **0.730** |

**arousal 預測 std 一模一樣（0.729 vs 0.730）** → A_PCC 從 0.452 掉到 0.357
**不能歸因於模型在 test 上表現不同**，只能歸因於 val 那個 0.452 本身是雜訊帶上緣的一次抽樣。

> **論文立場**：據實報告。這是一個誠實且有價值的 negative result——
> low-resource 維度情緒任務 + 小型 validation leaderboard，模型選擇極易過擬合。

---

## 1. 資料集

見 `docs/dataset.md`。關鍵前提：**官方未提供任何有標註的訓練資料**，
所有監督訊號都來自相鄰域借用語料（zero in-domain label）。這是 arousal 難的根源。

---

## 2. 共同設定（除非另外註明，所有實驗都相同）

- **Encoder**：`hfl/chinese-macbert-base`
- **架構**：encoder → attention-mask 加權 mean pooling → concat 詞典特徵 → 2 個回歸頭（V, A）
- **標籤**：1–9 正規化到 [0,1]，sigmoid 輸出，推論反轉回 1–9
- **超參**：4 epochs、batch 32、lr 2e-5、max_len 256、AdamW(wd 0.01)、warmup 10%、SmoothL1、grad clip 1.0
- **Early-stop**：`score = mean(PCC) − mean(MAE)` 取最佳 epoch
  （**僅供內部挑 epoch，不是官方指標**；PCC≈0.7 減 MAE≈0.7 本來就在 0 附近）
- **內部 dev**：DSA-MST 反思切出的 253 篇（分數系統性高於官方，**不可當官方分數看**）
- **官方 val**：`DSANIDF_ValidationSet.csv` 200 篇（無標籤，靠提交取得分數）

> **指標一致性**：本專案評估與官方 `scoring.py` 數學等價
> （MAE = `sklearn.mean_absolute_error`、PCC = `scipy.stats.pearsonr[0]`）。

---

## 3. 官方 validation 結果總表（全部 13 次提交）

| # | 實驗 | V_MAE ↓ | V_PCC ↑ | A_MAE ↓ | A_PCC ↑ | 結論 |
|---|---|---|---|---|---|---|
| 1 | Baseline（EmoBank only） | 0.654 | 0.867 | 0.985 | 0.412 | 起點 |
| 2 | + DAPT（新住民 200 篇 MLM） | 0.649 | 0.866 | 1.009 | 0.395 | 無效 |
| 3 | + 反思語料（9,435） | 0.627 | 0.870 | 0.944 | 0.388 | 修校準、未修排序 |
| **4** | **+ L1 詞典融合（10 維）** | **0.600** | **0.880** | 0.882 | 0.426 | **valence 最佳** |
| 5 | + L3 生成增強（bin 中心標籤 k=1.0） | 0.611 | 0.870 | 1.100 🔴 | **0.461** 🟢 | PCC↑ MAE 爆 |
| 5b | + L3 增強（標籤收縮 k=0.6） | 0.632 | 0.874 | 1.058 🔴 | 0.460 🟢 | 收縮救不回 MAE |
| 10 | 純 MacBERT 5-seed ensemble | 0.614 | 0.881 | 0.907 🔴 | 0.415 🔴 | ensemble 死路 |
| 12 | 多 encoder ensemble | 0.613 | **0.882** | 0.906 🔴 | 0.418 🔴 | ensemble 死路 |
| 13 | Teacher 偽標增強（blend=0） | 0.617 | 0.878 | 0.898 | 0.423 | 修校準、失排序 |
| 18 | L1++ intensity 31 維 | 0.641 🔴 | 0.865 | 0.909 🔴 | 0.408 🔴 | 強度特徵有害 |
| **19** | **+ Source-aware arousal loss** ⭐ | 0.666 🔴 | 0.869 | **0.870** 🟢 | **0.452** 🟢 | **最終提交** |
| 11b | RoBERTa-large 單顆 | 0.691 🔴 | 0.869 | 0.914 🔴 | 0.407 🔴 | 淘汰 |
| 20 | Ranking-only 增強 | 0.631 | 0.877 | 0.923 🔴 | 0.407 🔴 | 淘汰 |
| 21 | Dim-attention + rank | 0.638 | 0.865 | 0.923 🔴 | 0.394 🔴 | 淘汰 |

> 🟢/🔴 是相對實驗 4 的優劣。**E4 與 E19 各贏兩個維度**（E4 贏 valence 雙指標、E19 贏 arousal 雙指標），
> 官方 mean rank 相當——最終選 E19 是押注 arousal 是差異化來源。**test 結果顯示這個押注輸了。**

---

## 4. 各實驗細節

### 實驗 1–4：資料與特徵的三步走

| | 1. Baseline | 2. DAPT | 3. 反思語料 | 4. L1 詞典融合 |
|---|---|---|---|---|
| DAPT 續訓 | 無 | ✅ 新住民 200 篇 MLM | 無 | 無 |
| 詞典特徵 | 無 | 無 | 無 | ✅ CVAW+CVAP 10 維 |
| Train | EmoBank 4,998 | EmoBank 4,998 | 全部 9,435 | 全部 9,435 |
| dev | EmoBank 切 ~555 | EmoBank 切 ~555 | DSA-MST 253 | DSA-MST 253 |

- **實驗 2（DAPT）無效**：arousal 反而退（PCC 0.412→0.395）。DAPT 只改 encoder 的無監督表徵，
  回歸頭仍學 EmoBank 域的 arousal 分布 → **缺的是「對的語域的監督標籤」，不是表徵**。且語料僅 200 篇太小。
- **實驗 3 修好校準、沒修好排序**：雙 MAE 下降（4 指標贏 3），A_PCC 仍 0.388。
- **實驗 4 = 第一個 4 指標全勝**：`lexicon.py` 用 CVAW(字)+CVAP(詞) 共 7,761 詞典，
  對每篇文本抽 10 維聚合特徵（coverage / count / V·A 的 mean·max·min·std），
  concat 進 pooled embedding（`nn.Linear(768+10, 2)`）。
  A_PCC 0.388→0.426、A_MAE 0.944→0.882 → **contextual embedding 與 lexical VA 訊號互補有效**。

### 實驗 5 / 5b：L3 圖譜引導的可控生成增強

`augment_generate.py`：讀 L3 情感知識圖譜 → 對 5 個代表性不足的 VA 區間用 `seeds_for_target()`
撈情緒種子詞 → 請 `claude-opus-4-8` 生成新住民第一人稱反思短文（50–120 字）→ 標 bin 中心 ± jitter(0.4)
→ jieba 詞集 Jaccard > 0.5 近似去重 → `data/train_aug.csv`（400 篇）。

| bin | V 中心 | A 中心 | 象限 | 補的洞 |
|---|---|---|---|---|
| 1 | 3.0 | 7.5 | 負·高喚醒（焦慮/憤怒/崩潰） | arousal 高端 |
| 2 | 5.0 | 7.5 | 中·高喚醒（緊張/坐立難安） | 高端 × 中 valence（最稀缺） |
| 3 | 7.0 | 7.5 | 正·高喚醒（興奮/喜極而泣） | arousal 高端 |
| 4 | 3.0 | 2.5 | 負·低喚醒（疲憊/麻木/沮喪） | arousal 低端 |
| 5 | 7.0 | 2.5 | 正·低喚醒（平靜/安心/滿足） | arousal 低端 |

資料分布變化：9,435 → 9,835（+4.2%）；arousal **std 1.27 → 1.35**。

**結果與診斷**：
- 實驗 5（k=1.0）：**A_PCC 0.426→0.461（首破 0.43）**，但 A_MAE 0.882→**1.100**（爆掉）。
- 實驗 5b（k=0.6 標籤向真實均值收縮）：預測 arousal std 已收回 0.743（≈實驗 4 的 0.755），
  **A_PCC 仍守 0.460、A_MAE 仍 1.058**。
- **關鍵推論**：A_PCC 增益來自**文本教的特徵**（分布收回了 PCC 沒掉），
  A_MAE 的傷害來自**合成文本本身的校準/風格偏移**，**單純調標籤救不回**。
- **dev 失真警訊**：dev A_MAE 0.84 很漂亮，official 卻 1.06–1.10 →
  增強資料與 dev（DSA-MST 反思）同風格，**dev 不再是可靠代理**。

### 實驗 10–12：Ensemble — 官方確認為死路

- E10：MacBERT+L1 × 5 seeds（42/1/2/3/4）等權平均。E11：RoBERTa-wwm-ext base / large。
  E12：7 顆做 dimension-wise weighted / mean 融合。
- dev 上一切正常（各單顆 A_PCC 0.60–0.615，融合 0.615–0.616），**官方卻全輸實驗 4**。
- **診斷**：多模型平均把**已經被壓縮的 arousal 再壓一次**（校準與排序雙輸）。
  拿掉 roberta 也沒救（E10 ≈ E12）→ **傷害不是某顆爛模型，是「平均」這件事本身**。
- ⇒ 堆模型 / 融合無法突破 arousal 天花板。

### 實驗 13：Multi-teacher 偽標精修 — 修好校準、丟了排序

- 3 顆 teacher（macbert_s42 / macbert_s1 / roberta_s42）對 400 篇合成文本逐篇重標
  + 每 bin ±1.5SD 離群移除 → `data/train_aug_pseudo.csv`（400 → 318 篇）。
- 官方：A_MAE **1.100 → 0.898**（✅ 校準修回 0.20），但 A_PCC **0.461 → 0.423**（❌ 排序增益一起丟）。
- **診斷（自我參照陷阱）**：teacher 本身 arousal 就是壓縮的（把低喚醒 bin 標成 ~5.0、認不出低端），
  用它重標等於「用壓縮的老師教壓縮的問題」，把撐開排序的訊號一起壓平。
- ⇒ **arousal 增強存在「校準 ↔ 排序」根本 trade-off**：
  raw 極端標籤給 PCC / 傷 MAE；teacher 標籤救 MAE / 失 PCC。

### 實驗 18–21：架構與 loss 層面的嘗試

- **E18（`--lex_mode l1_intensity`）**：`lexicon_intensity.py` 把 L1 十維擴成 31 維
  （+15 維 arousal intensity：標點密度 / 程度副詞 / 身體反應 / 睡眠 / 焦慮 / 壓力事件 / 低喚醒詞 /
  疊字 / 句長節奏 / 否定轉折；+6 維新住民 domain cues：語言 / 證件 / 工作 / 家庭分離 / 文化 / 經濟）。
- **E19（= E18 + `--source_aware`）**：V/A loss 依 granularity 來源加權——
  CVAS `1:0.25`、CVAT `1:0.5`、DSA-MST `1:1`、edu2021 `1:0.75`（arousal 對 domain shift 敏感，
  通用情緒庫降權；valence 權重全部 1.0）。
- **E20（`--rank_aug`）**：合成 400 篇不進 SmoothL1，每 step 抽 16 篇建 batch 內 pairwise hinge
  （gold 差 ≥1.5 才成 pair、margin 1、λ_A=0.1 λ_V=0.05）。
- **E21a/b（`--pooling mean_cls_dim_attention`）**：mean + CLS + V/A 各自 attention pooling 分頭回歸。

**最關鍵的消融（三個官方分數並排看）**：

```
E18（31 維強度特徵，無 source_aware）   A_PCC 0.408  ← 比 E4 還差
E4 （10 維 L1，baseline）              A_PCC 0.426
E19（31 維 + source_aware）            A_PCC 0.452  ← val 上最佳
```

E19 = E18 + `--source_aware`，而 E18 單獨用**低於** E4 →
**E19 在 val 上的增益全部來自 source-aware 加權，31 維強度特徵本身是拖累**（疑似對訓練域表面 cue 過擬合）。
這直接指向從未跑成的 **E22（`--lex_mode l1 --source_aware`）**：拿掉強度特徵、只留 10 維 L1 + 來源加權。
（E22 三次本機 MPS 嘗試皆 hang，見 `docs/reproduce.md`。）

> ⚠️ **test 結果出來後，這條推論需要重新評估**：E19 的 A_PCC 在 test 上掉到 0.357，
> 代表 val 上 E18/E4/E19 這 0.408 / 0.426 / 0.452 的排序**本身可能就在雜訊帶內**，
> 「source-aware 有效」的證據強度遠比當時以為的弱。論文必須以此為準來下結論。

### Silver ranking benchmark（`build_silver_pairs.py` / `eval_silver_ranking.py`）

動機：dev 對增強實驗失真，想找更貼目標域的模型選擇依據。做法：對官方 200 篇無標籤文本抽 500 pairs，
請 LLM（claude-opus-4-8）pairwise 判斷「哪篇 arousal / valence 較高」，
只信排序、不信絕對分數（避開 E13 的校準陷阱），評各 run val 預測的 pairwise ranking accuracy。

實戰戰績：
- ✅ **對 e21 的否決是準的**（silver 排最後 0.726–0.728，官方 A_PCC 也確實最差 0.394）。
- ❌ **正向排序判斷錯了**：silver 押 e20（0.766）> e18（0.762）> e19（0.748），
  官方卻是 e19（0.452）> e20（0.407）≈ e18（0.408）。
- **定位**：只適合當**否決過濾器**（花提交額度前篩掉明顯壞的 run），
  **不能拿來選第一名**。原因推測：pairwise accuracy 與 PCC（受分布形狀影響）不等價、且只有單一 judge。
- 未做：多 judge 一致性過濾（`--provider openai` 再標一輪）。

---

## 5. 核心結論（論文的四個 claim）

### 5.1 ✅ 正面結果：顯性詞典訊號與 contextual embedding 互補
實驗 4 是唯一 4 指標全勝的版本。把 7,761 詞的 CVAW+CVAP 聚合成 10 維 concat 進 pooled embedding，
**同時**改善了 arousal 的校準（MAE 0.944→0.882）與排序（PCC 0.388→0.426）。
這是本專案最穩、也最可能跨集合成立的貢獻。

### 5.2 ❌ 負面結果：ensemble 對壓縮型 arousal 無效
E10 / E12 官方分數證明多模型平均把已壓縮的 arousal 再壓一次，校準與排序雙輸。
這與多數 shared task 論文「ensemble 必漲」的常識相反，值得寫。

### 5.3 ⚖️ 核心分析：arousal 增強的「校準 ↔ 排序」trade-off，以及它為何無法用標籤修
實驗 5 → 5b → E13 是一條完整的證據鏈：
- 5：極端 bin 標籤 → PCC↑ MAE 爆
- 5b：整批收縮標籤 → 預測 std 收回了，PCC 守住、**MAE 還是沒回來** ⇒ 傷害在文本不在標籤
- E13：teacher 逐篇重標 → MAE 修回、**PCC 一起丟** ⇒ teacher 自身壓縮造成自我參照陷阱

### 5.4 🔬 方法論貢獻：小型 leaderboard 上的 selection overfitting
這是 test 結果逼出來的、也是最有價值的一節。完整數據見 §0。

三個環環相扣的證據：
1. **落差本身**：A_PCC val 0.452 → test 0.357（−0.095），而 V_PCC 0.869 → 0.866（−0.003）。
2. **噪聲量級**：bootstrap 顯示 n=200 時 A_PCC 的 95% CI 寬達 **0.193**（valence 只有 0.083），
   而我們做決策依據的差距只有 0.026。**決策差距比噪聲小 7.4 倍。**
3. **排除替代解釋**：模型在 test 上的預測分布與 val 幾乎相同（arousal std 0.729 vs 0.730），
   所以不是模型漂移；seed 變異只有 0.0055，所以不是訓練隨機性。**只剩「評估集太小」這一個解釋。**

⚠️ **誠實的但書（論文必須寫）**：bootstrap 是在 **dev（DSA-MST，253 篇）**上做的，
不是在官方 validation 上（官方沒給 gold 標籤，我們做不到）。
所以 0.193 是**代理估計**而非官方集合上的真實 CI。但方向與量級足以支撐論證。

**建議（寫進 Limitations / Lessons）**：此類任務應報告 seed 變異與信賴區間，
並避免用單一小型 leaderboard 標量做多輪模型選擇；
當某個維度的評估變異是另一個維度的 2 倍以上時，該維度的排名差距需要更高的門檻才值得相信。

---

## 6. 已排除的方向（別再花時間）

| 方向 | 證據 |
|---|---|
| DAPT（小語料） | 實驗 2，arousal 反退 |
| Multi-seed / 多 encoder ensemble | E10 / E12，四指標輸 E4 |
| 純 teacher 偽標增強 | E13，失 A_PCC |
| 31 維 arousal intensity 特徵 | E18，四指標輸 E4 |
| Ranking-only 增強 | E20，A_PCC 0.407 |
| Dim-attention pooling | E21，A_PCC 0.394（最差） |
| RoBERTa-large 單顆 | 11b，V_MAE 0.691 最差 |
| 通用 GraphRAG / 事實型 KG | 與「風格/情緒」檢索錯位（未跑，評估排除） |
| 翻譯 English EmoBank | scale 不一致 + 翻譯漂移 + 語域不合（未跑，評估排除） |
