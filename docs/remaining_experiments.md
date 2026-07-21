# 待跑實驗清單（論文導向）

> **前提已經改變。** Shared task 評測已結束，且官方 **validation 與 test 都沒有釋出 gold 標籤**。
> 這代表：**任何新模型都無法再取得官方分數**。
> 因此以下實驗的排序依據不再是「能不能拿分」，而是**「對論文的證據強度貢獻多少」**。
>
> 可用的評估管道只剩三條：
> 1. **內部 dev**（DSA-MST 253 篇，有標籤）— 唯一能算真實 MAE/PCC 的集合
> 2. **Silver ranking**（LLM pairwise，官方 200 篇）— 只能當否決過濾器
> 3. **預測分布統計**（官方 200 + 1,100 篇，無標籤）— 可算 std / 模型間相關性 / 漂移

投稿截止：**2026-08-10**。以下依「投入 vs 論文價值」排序。

---

## 🔴 P0 — 必做。零訓練成本，直接支撐論文核心 claim

### P0-1. 補一支 `predict.py`（推論專用）
目前 `train_v2.py` 沒有 inference-only 路徑，而**我們沒有留下任何 test set 的預測檔**
（最終提交是在 Colab 上跑完直接上傳的）。但 `outputs/e19_source_aware_best.pt` 還在本機。

需求：吃 `--ckpt` + `--input <csv>` + `--lex_mode` → 輸出 `outputs/preds/{run}_test.csv`。
**這是 P0-2 / P0-3 的前置。**

### P0-2. 用全部 13 顆 checkpoint 對 `DSANIDF_TestSet.csv` 重跑推論
```bash
for ck in e19_source_aware e18_l1_intensity macbert_s42 macbert_s1 ... ; do
    python predict.py --ckpt outputs/${ck}_best.pt --input ../DSANIDF_TestSet.csv --run_name $ck
done
```
沒有 gold 標籤，**不能算分數**，但可以得到論文需要的三組數字：
- 各模型在 test 上的**預測 std**（驗證 arousal 壓縮是否在 1,100 篇上依然成立）
- **模型間預測相關性**（支撐「ensemble 為何無效」——若各模型 arousal 預測高度相關，平均自然只會壓縮）
- **val（200）與 test（1,100）預測分布的漂移**（支撐 §5.4）

> 已知本機 val 預測分布：arousal std 落在 0.67–0.83，valence std 1.23–1.60。
> 對照 test 的同一組數字是論文 Analysis 節的關鍵圖表。

### P0-3. 量化「n=200 上 A_PCC 的抽樣誤差」——**這是最重要的一個實驗**
支撐論文最強的 claim（selection overfitting）。純統計，幾分鐘跑完：

1. 在內部 dev（253 篇，有 gold）上，對每個 run **bootstrap 抽 200 篇** × 2,000 次，
   算 A_PCC 與 V_PCC 的分布 → 得到 **n=200 時的 95% 信賴區間寬度**。
2. 假說：**A_PCC 的 CI 寬度會遠大於 V_PCC**，且寬到足以涵蓋 E4（0.426）與 E19（0.452）的差距。
3. 若成立，就能直接寫：「我們在 validation 上觀察到的 arousal 排名差異在統計上不可區分，
   test 結果（0.452 → 0.357）正是這個效應的實現。」

同時算：5 顆 seed（macbert_s42/s1/s2/s3/s4）在 dev 上的 A_PCC **seed 變異**（已知落在 0.604–0.616）。

### P0-4. 整理 dev 全表
把 `outputs/preds/*_dev.csv` 全部算成 4 指標表，與官方 val 表並排 →
量化「dev 排名與 official 排名的 Spearman 相關」。
若相關性很低，就是「代理驗證集不可靠」這個主張的直接證據，同時解釋了為何 silver ranking 也失敗。

---

## 🟠 P1 — 強烈建議。這是讓所有 ablation 站得住腳的唯一方法

### P1-1. **關鍵 ablation 每格跑 3–5 個 seed**（Colab A100）
**目前所有結論都是單 seed 單次跑出來的**——這是本專案方法論上最大的漏洞，
而 test 結果已經證明這個漏洞是真的會咬人的。

要跑的 2×2 ablation 表（`lex_mode` × `source_aware`）：

| | 無 `--source_aware` | 有 `--source_aware` |
|---|---|---|
| `--lex_mode l1`（10 維） | **E4**（已有 1 seed） | **E22（從未跑成）** |
| `--lex_mode l1_intensity`（31 維） | **E18**（已有 1 seed） | **E19**（已有 1 seed） |

```bash
for s in 42 1 2; do
  python train_v2.py --lex_mode l1            --seed $s --run_name e4_s$s
  python train_v2.py --lex_mode l1 --source_aware --seed $s --run_name e22_s$s
  python train_v2.py --lex_mode l1_intensity  --seed $s --run_name e18_s$s
  python train_v2.py --lex_mode l1_intensity --source_aware --seed $s --run_name e19_s$s
done
```
- 成本：12 次訓練 × 約 15 分鐘（A100）≈ **3 小時**，一個 Colab session 內可完成。
- 產出：dev 上的 **mean ± std** 四格表。這才是論文能放的 ablation。
- **E22 順帶補齊**（`--lex_mode l1 --source_aware`）——它是這張表唯一的空格，
  也是原本被判定為「最有希望的下一發」，論文若不補這格，「source-aware 有效」的宣稱會有明顯缺口。
- ⚠️ **一律上 Colab**，本機 M2 的 `mps` 在此設定下會 hang（見 `docs/reproduce.md` §4）。

### P1-2. 把 P1-1 的模型也對 test 推論
接上 P0-2 的分析管線，看多 seed 的預測在 test 上的變異有多大。

---

## 🟡 P2 — 加分。有明確論文價值但成本較高

### P2-1. L3 引導 vs 隨機種子的消融（**證明知識圖譜有必要**）
論文的方法創新之一是「用 L3 情感知識圖譜挑種子詞來引導生成」。
**目前完全沒有對照組**——沒有任何證據顯示圖譜比隨機挑情緒詞好。

做法：改 `augment_generate.py`，用「從 CVAW/CVAP 隨機抽詞」取代 `seeds_for_target()`，
生成同樣 400 篇 → 訓練 → 在 dev 上比較，並比較兩批合成文本的
**arousal 分布覆蓋度**與**去重後保留率**。

- ⚠️ 成本：**要重新呼叫 LLM API 生成 400 篇，會計費**，且需在使用者自己的終端機跑（sandbox 無 key）。
- 若時間/預算不夠：退而求其次，只比較**生成文本的統計特性**（不重新訓練），
  也足以寫成一小節。

### P2-2. Silver ranking 補第二個 judge
```bash
python build_silver_pairs.py --provider openai   # 同一批 500 pairs、同 seed
python eval_silver_ranking.py                    # 雙 judge 一致性過濾
```
價值：silver ranking 的失敗（押錯 e19 排名）本身就是論文可寫的一段。
補第二個 judge 能判斷「失敗是因為單一 judge 偏誤，還是 pairwise accuracy 與 PCC 本質不等價」。
成本：一次 API 呼叫，數美元。

---

## ⚪ P3 — 除非有餘力，否則略過

| 實驗 | 程式 | 為何可略 |
|---|---|---|
| E14 frozen embedding + SVR | `embed_regressor.py` | 原本是為了進 ensemble 補位，而 ensemble 已證明是死路 |
| E16 L1+L2 特徵（16 維） | `lexicon_l2.py` | E18 已顯示「加更多詞典衍生特徵」會傷 arousal，先驗不佳 |
| E17 arousal 後校準 | `calibrate.py` | 只改 MAE、PCC 不變，且沒有官方分數可驗證校準是否過擬合 dev |
| E13 `--blend 0.3/0.5` trade-off 曲線 | `build_pseudo_labels.py` | 概念上漂亮（畫出校準↔排序曲線），但每個點都要一次訓練且只能看 dev |

> E13 blend 曲線如果**只在 dev 上畫**其實還是有價值的（trade-off 是內部一致的現象，
> 不依賴官方分數）。若 P0/P1 提前完成，這是 P2 之後最值得補的一項。

---

## 建議執行順序（以 8/10 截止回推）

```
第 1 天   P0-1 predict.py → P0-2 test 推論 → P0-3 bootstrap CI → P0-4 dev 全表
          （全部本機可跑，零 GPU 成本，直接產出論文 Analysis 節的所有數字）
第 2–3 天 P1-1 上 Colab 跑 12 個 run（3 小時）→ P1-2 推論 → 產出 ablation 表
第 4 天起 開始寫論文（骨架見 docs/paper/outline.md）
有餘力    P2-1 L3 消融 / P2-2 第二個 judge
```

**最低限度**：只做完 P0（一天），論文就已經有完整的結果表 + 一個誠實有力的
selection-overfitting 分析。P1 讓 ablation 從「單次跑分」升級成「有誤差棒的結論」，
是審稿人最可能質疑的地方，強烈建議補。
