# 實驗總表：方法配置 × val/test 分數 × 待提交清單

> ROCLING-2026 DSA-NIFT。本檔一次彙整：①每個實驗的方法配置、②官方 val 與 test 分數、
> ③哪些 `.csv.zip` 還沒交。權威資料源：`docs/experiments.md`（val）、`docs/test_results.md`（test）。
> 指標：Valence/Arousal 各算 MAE↓、PCC↑。val n=200、test n=1,100，**兩者皆無 gold 標籤**（靠提交取分）。

---

## 0. 共同設定（除非該列另註）

| 項目 | 值 |
|---|---|
| Encoder | `hfl/chinese-macbert-base` |
| 架構 | encoder → attention-mask mean pooling → concat 詞典特徵 → 2 回歸頭 (V/A) |
| 標籤 | 1–9 → [0,1]、sigmoid、推論反轉 |
| 超參 | 4 epochs、batch 32、lr 2e-5、max_len 256、AdamW(wd 0.01)、warmup 10%、SmoothL1、grad clip 1.0 |
| Early-stop | mean(PCC) − mean(MAE)（僅挑 epoch，非官方指標） |
| 訓練集 | `data/train.csv` 9,435 筆（EmoBank CVAS/CVAT + DSA-MST + ROCLING-2021） |
| dev | DSA-MST 反思 253 篇（分數系統性偏高，**≠官方**） |

---

## 1. 主線實驗（val + test 官方分數）

| # | 系統 | 方法配置（相對共同設定的差異） | V-MAE | V-PCC | A-MAE | A-PCC | 集合 |
|---|---|---|---|---|---|---|---|
| 1 | Baseline | EmoBank only（4,998）、無詞典 | 0.654 | 0.867 | 0.985 | 0.412 | val |
| 2 | DAPT | 先 `dapt.py` 新住民 200 篇 MLM 續訓 | 0.649 | 0.866 | 1.009 | 0.395 | val |
| 3 | 反思語料 | train.csv 9,435、無詞典（`--no_lexicon`） | 0.627 | 0.870 | 0.944 | 0.388 | val |
| **4** | **E4：L1 詞典融合** | **+ 10 維 CVAW/CVAP 特徵** | **0.600** | **0.880** | 0.882 | 0.426 | val |
| 5 | L3 生成增強 | + `train_aug` 400 篇、bin 中心標籤(k=1.0) | 0.611 | 0.870 | 1.100 | 0.461 | val |
| 5b | L3 增強(收縮) | 同上，標籤收縮 k=0.6 | 0.632 | 0.874 | 1.058 | 0.460 | val |
| 10 | 5-seed ensemble | E4 × 5 seed 等權平均 | 0.614 | 0.881 | 0.907 | 0.415 | val |
| 12 | 多 encoder ensemble | + roberta base/large，dim-wise 加權 | 0.613 | 0.882 | 0.906 | 0.418 | val |
| 13 | Teacher 偽標 | + `train_aug_pseudo` 318 篇（3 teacher 重標） | 0.617 | 0.878 | 0.898 | 0.423 | val |
| 18 | E18：L1++ intensity | `--lex_mode l1_intensity`（31 維） | 0.641 | 0.865 | 0.909 | 0.408 | val |
| **19** | **E19：source-aware（提交）** | **E18 + `--source_aware`（CVAS/CVAT arousal 降權）** | 0.666 | 0.869 | 0.870 | 0.452 | val |
| 20 | E20：ranking-only | E19 + `--rank_aug`（合成資料只進 pairwise hinge） | 0.631 | 0.877 | 0.923 | 0.407 | val |
| 21b | E21b：dim-attention + ranking | E19 的 31 維 intensity/source-aware + `--pooling mean_cls_dim_attention` + ranking | 0.638 | 0.865 | 0.923 | 0.394 | val |

### 同一批系統的官方 **test** 分數（n=1,100）

| 系統 | 方法 | V-MAE | V-PCC | A-MAE | A-PCC |
|---|---|---|---|---|---|
| **E4**（=macbert_s42, 10 維 L1） | 詞典融合 | 0.61 | 0.87 | 0.94 | **0.372** |
| E18 | 31 維 intensity | 0.61 | 0.87 | 0.93 | 0.370 |
| **E19**（正式提交） | + source-aware | 0.6200 | 0.8663 | 0.9259 | **0.3566** |
| E20 | ranking-only | 0.63 | 0.87 | 0.93 | 0.370 |
| E13 teacher pseudo（3-seed mean） | 318 篇 teacher 偽標增強 | 0.620 | 0.870 | 0.927 | 0.370 |
| roberta_s42 | RoBERTa-base + L1 | 0.62 | 0.86 | 0.92 | 0.38 |
| robertaL_s42 | RoBERTa-large + L1 | 0.67 | 0.86 | 0.91 | **0.39** |

> **關鍵**：val 上 E19 最高(0.452)、robertaL 最低(0.407)；test 上完全**翻轉**
> （E19→0.357 最低、robertaL→0.39 最高）。Spearman(val,test A_PCC)=**−0.55**。
> valence 全部 val≈test≈0.87。→ 論文核心圖 `paper/figures/val_vs_test_pcc.pdf`。

---

## 2. 無增強 baseline：5-seed test 變異

同 E4 設定（10 維 L1），5 個 seed 全部提交官方 test：

| seed | V-MAE | V-PCC | A-MAE | A-PCC |
|---|---|---|---|---|
| 42 | 0.61 | 0.87 | 0.94 | 0.37 |
| 1 | 0.62 | 0.87 | 0.91 | 0.37 |
| 2 | 0.61 | 0.87 | 0.92 | 0.37 |
| 3 | 0.61 | 0.87 | 0.92 | 0.38 |
| 4 | 0.62 | 0.87 | 0.95 | 0.37 |
| **mean±std** | 0.614±0.006 | **0.870±0.000** | 0.928±0.016 | **0.372±0.004** |

→ **官方 test 上的真實 seed 變異**（arousal PCC std 僅 0.004）。取代 dev bootstrap 代理值。

---

## 3. Batch-32 controlled ablation（L1 與 augmentation）

四個條件均使用 MacBERT、batch 32、lr 2e-5、4 epochs、max_len 256、相同 checkpoint criterion，
並跑 seeds 42/1/2。這組實驗同時提供純 L1 對照（No L1 vs L1）與純 augmentation 對照
（L1 vs E vs F2）。

| Condition | Seed | V-MAE | V-PCC | A-MAE | A-PCC |
|---|---:|---:|---:|---:|---:|
| No L1 | 42 | 0.61 | 0.87 | 0.95 | 0.38 |
| No L1 | 1 | 0.62 | 0.87 | 0.91 | 0.34 |
| No L1 | 2 | 0.62 | 0.87 | 0.91 | 0.37 |
| L1 | 42 | 0.61 | 0.87 | 0.90 | 0.37 |
| L1 | 1 | 0.61 | 0.87 | 0.90 | 0.37 |
| L1 | 2 | 0.62 | 0.87 | 0.97 | 0.37 |
| E | 42 | 0.62 | 0.87 | 1.08 | 0.38 |
| E | 1 | 0.62 | 0.87 | 1.03 | 0.36 |
| E | 2 | 0.65 | 0.86 | 1.06 | 0.38 |
| F2 | 42 | 0.63 | 0.87 | 1.19 | 0.38 |
| F2 | 1 | 0.63 | 0.86 | 0.96 | 0.38 |
| F2 | 2 | 0.63 | 0.87 | 1.07 | 0.40 |

| Condition | V-MAE | V-PCC | A-MAE | A-PCC |
|---|---:|---:|---:|---:|
| No L1 | 0.617±0.006 | 0.870±0.000 | 0.923±0.023 | 0.363±0.021 |
| L1 | 0.613±0.006 | 0.870±0.000 | 0.923±0.040 | 0.370±0.000 |
| E | 0.630±0.017 | 0.867±0.006 | 1.057±0.025 | 0.373±0.012 |
| F2 | 0.630±0.000 | 0.867±0.006 | 1.073±0.115 | 0.387±0.012 |

- L1 − No L1：V-MAE −0.003、V-PCC 0.000、A-MAE 0.000、A-PCC +0.007。
- E − L1：V-MAE +0.017、V-PCC −0.003、A-MAE +0.133、A-PCC +0.003。
- F2 − L1：V-MAE +0.017、V-PCC −0.003、A-MAE +0.150、A-PCC +0.017。

→ L1 對 test 的影響很小；E/F2 提高 arousal PCC，但同時提高 arousal MAE，F2 的
PCC–MAE trade-off 最明顯。

---

## 4. KG 生成增強消融（6 條件 × 3 seed，全提交 test）

所有條件：E4 架構 + `--extra_train data/train_aug_{X}.csv`（各 400 篇合成），差別只在種子策略與風格錨定。

| 條件 | seed_mode | 風格錨定 | 長度指令 | 資料檔 |
|---|---|---|---|---|
| N | 不給種子詞 | ✗ | fixed | `train_aug_N.csv` |
| A | 隨機抽詞 | ✗ | fixed | `train_aug_A.csv` |
| C | VA 查表（=實驗5） | ✗ | fixed | `train_aug_C.csv` |
| E | 圖擴散 G1+G3 | ✗ | fixed | `train_aug_E.csv` |
| F | 圖擴散 | ✓ Val 200 篇 | fixed | `train_aug_F.csv` |
| F2 | 圖擴散 | ✓ Val | match_real | `train_aug_F2.csv` |

### 官方 test A_PCC（mean±std over 3 seed）

| 條件 | test A_PCC | test A_MAE | 逐 seed A_PCC |
|---|---|---|---|
| N | 0.353±0.012 | 1.07 | 0.34 / 0.36 / 0.36 |
| **A** | **0.367±0.012** | 1.09 | 0.35 / 0.38 / 0.37 |
| C | 0.370±0.017 | 1.08 | 0.35 / 0.38 / 0.38 |
| E | 0.373±0.012 | 1.06 | 0.36 / 0.38 / 0.38 |
| F | 0.38（s42；s1/s2 待記） | 1.08 | 0.38 / — / — |
| F2 | 0.387±0.012 | 1.073±0.115 | 0.38 / 0.38 / 0.40 |
| **無增強 baseline（matched batch 32）** | 0.370±0.000 | **0.923±0.040** | 0.37 / 0.37 / 0.37 |

> **成分分解**：N 0.353 →(+任意種子詞 0.014)→ A 0.367
> →(+VA 過濾 **0.003，無效**)→ C 0.370 →(+圖結構 **0.003，無效**)→ E 0.373
> →(+錨定/長度 0.014)→ F2 0.387。增益主要來自「有 seed words」與風格/長度對齊，
> **不是 VA 過濾或圖結構**；F2 的 A-PCC +0.017 伴隨 A-MAE +0.150，呈現明確 trade-off。
> dev 分數六條件全在 0.598–0.603（失真，僅參考）。

跨 family 的無增強比較以 §3 的 matched batch-32 runs 為準；本節表格保留原六條件
component ladder，用來分析 N→A→C→E→F2 的內部變化。

---

## 5. Teacher pseudo-label：3-seed test

E13：以 3 個 teacher 重標 400 篇合成資料，移除離群值後保留 318 篇。

| seed | V-MAE | V-PCC | A-MAE | A-PCC |
|---|---|---|---|---|
| s1 | 0.62 | 0.87 | 0.94 | 0.37 |
| s2 | 0.62 | 0.87 | 0.92 | 0.37 |
| s42 | 0.62 | 0.87 | 0.92 | 0.37 |
| **mean±std** | **0.620±0.000** | **0.870±0.000** | **0.927±0.009** | **0.370±0.000** |

→ test 上與無增強 baseline（V-PCC 0.870、A-MAE 0.928、A-PCC 0.372）幾乎相同。
因此 val 上「teacher 偽標改善校準但犧牲排序」的故事沒有在 test 重現；
較安全的結論是：**teacher pseudo-label 對 test 沒有可測增益，也沒有明顯額外傷害。**

---

## 6. 需要提交的 CSV（`.csv.zip`，內部檔名必為 `submission.csv`）

**提交規則**：上傳 zip，內含 `submission.csv`（欄位 `ID,Valence,Arousal`，1,100 列）；每天 10 次額度。
本機路徑：`baseline/outputs/`。

### ✅ 已有官方分數（不用再交）
E4/macbert_s1–s4/s42、E18、E19、E20、roberta_s42、robertaL_s42、
nolex_s1/s2/s42、macbert_pseudo_s1/s2/s42、
aug_{N,A,C,E}_全3seed、aug_F_s42、aug_F2_全3seed、
controlled_{nolex,l1,aug_E,aug_F2}_b32_s{42,1,2}

### 🔲 已有 test ZIP，可直接上傳

| 優先 | 檔案 | 為什麼交 |
|---|---|---|
| ★★★ | `aug_F_s1_test.csv.zip`、`aug_F_s2_test.csv.zip` | 補齊 F 的 3 seed，拆開「style anchor」與「match_real 長度」的貢獻；目前唯一可直接提交且對論文仍有明顯價值的組別 |

### 🔲 模型/val CSV 已完成，但還沒有 test CSV

下列現有檔案都是 **200-row validation submission，不能直接上傳 test**；需先用 1,100-row
test predictions 重新產生 `{run}_test_submission.csv` 並打包。

| 優先 | 現有 CSV | 尚缺什麼 | 論文價值 |
|---|---|---|---|
| ★★ | `e21_dim_attention_rank_aug_submission.csv` | 從 `e21_dim_attention_rank_aug_best.pt` 推論 test | E21b 已有 val 0.394；補 test 可再增加一個 val→test reversal/transfer 檢驗點 |
| ★★ | `e10_seed_ens_submission.csv` | 將既有 5 顆 MacBERT test predictions 等權融合 | 直接檢驗 multi-seed ensemble 在 test 是否仍無效；5 顆 constituent test CSV 已齊 |
| ★ | `e12_enc_ens_submission.csv` | 依原 E12 權重重建 test ensemble | 檢驗跨 encoder ensemble；需先確認原 7 模型中每顆都有 test predictions |
| ★ | `e12_enc_mean_submission.csv` | 重建 test mean ensemble | 與 weighted E12 對照，但價值低於 E10/E21b |
| ☆ | `e21_dim_attention_submission.csv` | 從 `e21_dim_attention_best.pt` 推論 test | 純 dim-attention；目前缺可靠的 official val scalar，論文配對價值較低 |
| ☆ | `e13_pseudo_ens_submission.csv` | 融合 pseudo 三顆 test predictions | 單模型 pseudo 的 3-seed test 已顯示無增益，ensemble 只屬完整性補充 |

> **提交建議**：先交 `aug_F_s1/s2`；若願意產生新 test CSV，再依序做
> E21b、E10、E12。Pseudo ensemble 的邊際論文價值最低。

---

## 7. 復現指令對照

```bash
# 主線
python train.py --no_lexicon                               # 實驗3（無 L1）
python train_v2.py --run_name e4_repro                     # E4（預設=實驗4）
python train_v2.py --lex_mode l1_intensity --source_aware --run_name e19_source_aware  # E19（提交）
# 消融
python train_v2.py --lex_mode none --seed S --run_name nolex_sS                        # no-L1
python train_v2.py --extra_train data/train_aug_X.csv --seed S --run_name aug_X_sS     # KG 消融
# 推論任意 checkpoint 到 test
python predict.py --ckpt outputs/{run}_best.pt --lex_mode {l1|l1_intensity|none} \
    --input ../DSANIDF_TestSet.csv --run_name {run} --split test
# 打包提交
python fetch_results.py --pack {run}_test        # → outputs/{run}_test.csv.zip
```
生成合成資料：`augment_generate.py --seed_mode {none,random,lookup,graph} [--style_anchor ...] [--length match_real]`
