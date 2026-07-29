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
| 21 | E21：dim-attention | `--pooling mean_cls_dim_attention`（無 L1） | 0.638 | 0.865 | 0.923 | 0.394 | val |

### 同一批系統的官方 **test** 分數（n=1,100）

| 系統 | 方法 | V-MAE | V-PCC | A-MAE | A-PCC |
|---|---|---|---|---|---|
| **E4**（=macbert_s42, 10 維 L1） | 詞典融合 | 0.61 | 0.87 | 0.94 | **0.372** |
| E18 | 31 維 intensity | 0.61 | 0.87 | 0.93 | 0.370 |
| **E19**（正式提交） | + source-aware | 0.6200 | 0.8663 | 0.9259 | **0.3566** |
| E20 | ranking-only | 0.63 | 0.87 | 0.93 | 0.370 |
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

## 3. no-L1 消融（L1 詞典融合到底有沒有貢獻）

同 E4 但 `--lex_mode none`，3 seed 提交 test：

| seed | V-MAE | V-PCC | A-MAE | A-PCC |
|---|---|---|---|---|
| s1 | 0.62 | 0.87 | 0.91 | 0.34 |
| s2 | 0.62 | 0.87 | 0.91 | 0.37 |
| s42 | 0.61 | 0.87 | 0.95 | 0.38 |
| **no-L1 mean±std** | 0.617 | 0.870 | 0.923 | **0.363±0.021** |
| **有 L1 baseline** | 0.614 | 0.870 | 0.928 | **0.372±0.004** |

→ 差 0.009，**遠在雜訊內**。連 L1 對 valence 的 val 增益(0.870→0.880)在 test 也消失。
**結論：L1 在 test 上不構成可測貢獻。**

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
| C | 0.370±0.017 | 1.08 | 0.35 / 0.38 / 0.38 |
| E | 0.373±0.012 | 1.06 | 0.36 / 0.38 / 0.38 |
| F | 0.38（s42；s1/s2 待記） | 1.08 | 0.38 / — / — |
| F2 | 0.387±0.012 | 1.04 | 0.40 / 0.38 / 0.38 |
| **無增強 baseline** | 0.372±0.004 | **0.93** | — |

> **成分分解**：N 0.353 →(+種子詞 0.017)→ C 0.370 →(+圖結構 **0.003，無效**)→ E 0.373
> →(+錨定/長度 0.014)→ F2 0.387。**圖結構貢獻≈0；最強的 F2 仍打不贏不增強**（A_PCC 微增被 A_MAE +0.11 抵銷）。
> dev 分數六條件全在 0.598–0.603（失真，僅參考）。

---

## 5. 需要提交的 CSV（`.csv.zip`，內部檔名必為 `submission.csv`）

**提交規則**：上傳 zip，內含 `submission.csv`（欄位 `ID,Valence,Arousal`，1,100 列）；每天 10 次額度。
本機路徑：`baseline/outputs/`。

### ✅ 已有官方分數（不用再交）
E4/macbert_s1–s4/s42、E18、E19、E20、roberta_s42、robertaL_s42、
nolex_s1/s2/s42、aug_{N,C,E}_全3seed、aug_F_s42、aug_F2_全3seed

### 🔲 還沒交（本機已備 zip，直接上傳即可）

| 優先 | 檔案 | 為什麼交 |
|---|---|---|
| ★★ | `aug_A_s1_test.csv.zip`、`aug_A_s2_test.csv.zip`、`aug_A_s42_test.csv.zip` | 「隨機種子」對照組，補完消融表（確認 N→C 的 +0.017 是 VA 過濾還是隨便給詞） |
| ★ | `aug_F_s1_test.csv.zip`、`aug_F_s2_test.csv.zip` | 補齊 F 的 3 seed（目前只有 s42），讓 E→F→F2 遞進完整 |
| ☆ | `macbert_pseudo_s1_test.csv.zip`、`macbert_pseudo_s2_test.csv.zip`、`macbert_pseudo_s42_test.csv.zip` | E13 teacher 偽標的 test（驗證校準↔排序 trade-off 是否在 test 成立） |

### 🔲 想交但還沒有 test 預測（需先在 Colab 產生）
`e10_seed_ens`、`e12_enc_ens`、`e13_pseudo_ens`（ensemble 檔，本機無 test 預測）
→ 若要驗證「ensemble 在 test 也無效」，需在 Colab 用 `ensemble.py` 對 test 產生預測後再打包。
（優先度低：單模型的 macbert_pseudo 已能代表 teacher 路線。）

> **注**：論文主結論（§1–4）已由現有 test 分數完全支撐，上述待提交項屬**補完消融的完整性**，
> 非必要。若額度有限，優先交 `aug_A`（★★）即可。

---

## 6. 復現指令對照

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
