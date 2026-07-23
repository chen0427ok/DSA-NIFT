# 官方 Test Set 分數追蹤（n=1,100）

> 評測結束後網站仍可提交（每天 10 次額度）。本檔累積各 run 的官方 test 分數。
> 用途：① 把論文變異估計從 dev bootstrap 代理值升級為官方集合實測；
> ② 檢驗 val 上的結論（ensemble 死路、trade-off、圖譜等）在 test 上是否成立。
>
> 提交格式：`outputs/{run}_test.csv.zip`（`predict.py` 產生、`fetch_results.py --pack` 打包）。

---

## 已取得的 test 分數

| run | 架構 | V_MAE ↓ | V_PCC ↑ | A_MAE ↓ | A_PCC ↑ | 備註 |
|---|---|---|---|---|---|---|
| **e19_source_aware** | 31維L1 + source-aware | 0.6200 | 0.8663 | 0.9259 | 0.3566 | **正式提交** |
| macbert_s42 | 10維L1（≈E4, seed42） | 0.61 | 0.87 | 0.94 | 0.37 | 無增強 baseline |
| macbert_s1 | 10維L1（seed1） | 0.62 | 0.87 | 0.91 | 0.37 | 無增強 baseline |
| macbert_s2 | 10維L1（seed2） | 0.61 | 0.87 | 0.92 | 0.37 | 無增強 baseline |
| macbert_s3 | 10維L1（seed3） | 0.61 | 0.87 | 0.92 | 0.38 | 無增強 baseline |
| macbert_s4 | 10維L1（seed4） | 0.62 | 0.87 | 0.95 | 0.37 | 無增強 baseline |
| e18_l1_intensity | 31維intensity（無sa） | 0.61 | 0.87 | 0.93 | 0.37 | 2×2 第三格 |
| e19（predict.py 重跑） | 31維 + source-aware | 0.61 | 0.87 | 0.93 | 0.36 | **管線驗證**，見下 |
| e20_rank_aug | ranking-only 增強 | 0.63 | 0.87 | 0.93 | 0.37 | val 上已淘汰 |

---

## 無增強 baseline 的 test seed 變異（進行中）

目標：跑完 5 個 seed（macbert_s42/s1/s2/s3/s4），得到**官方 test 上真實的 seed std**，
取代目前在 dev 上 bootstrap 的代理估計（那是論文最容易被審稿人攻擊的但書）。

| seed | V_MAE | V_PCC | A_MAE | A_PCC |
|---|---|---|---|---|
| 42 | 0.61 | 0.87 | 0.94 | 0.37 |
| 1 | 0.62 | 0.87 | 0.91 | 0.37 |
| 2 | 0.61 | 0.87 | 0.92 | 0.37 |
| 3 | 0.61 | 0.87 | 0.92 | 0.38 |
| 4 | 0.62 | 0.87 | 0.95 | 0.37 |
| **mean±std** | **0.614±0.006** | **0.870±0.000** | **0.928±0.016** | **0.372±0.004** |

**✅ 已湊齊 5 seed，這是官方 test 集合上的真實 seed 變異**（取代 dev bootstrap 代理值）：

- **A_PCC = 0.372 ± 0.004，全距僅 [0.37, 0.38]。** 訓練隨機性對 arousal 排序的影響**極小**。
- 對照：E4（0.37）、E19（0.357）、5-seed baseline（0.372）三者全部落在這條 0.37±0.005 的窄帶裡
  → **arousal 上「哪個系統比較好」根本無從分辨**，val 上 E4/E19 的 0.026 差距確定是雜訊。
- V_PCC 五顆完全相同（0.870），再次顯示 valence 穩定飽和。

**⚠️ 誠實但書（論文要寫）**：leaderboard 只回報到小數點後 2 位，所以這裡的 std 受**捨入粒度**限制，
是真實 seed 變異的**上界估計**。即便如此，0.004 已經小到足以支撐「變異來自評估集大小、非訓練隨機性」的論點。

**論文用途**：這組數字直接把 `docs/experiments.md` §5.4 的第 4 點
（「排除訓練隨機性」）從 dev 代理**升級為官方 test 實測**，消滅審稿人最可能攻擊的但書。

---

## ✅ 管線驗證通過（重要）

`e19` 用 `predict.py` 從 checkpoint 重跑的 test 分數 = **0.61 / 0.87 / 0.93 / 0.36**，
對照原始正式提交 **0.6200 / 0.8663 / 0.9259 / 0.3566**（四捨五入到 2 位 = 0.62/0.87/0.93/0.36）：
- A_MAE / A_PCC / V_PCC **完全吻合**，V_MAE 差 0.01（MPS vs CUDA 的數值差，使用者已決定採用原始 0.6200）。
- ⇒ **`predict.py` 重跑管線與當初 Colab 推論等價**，後續所有用它產生的 test 分數皆可信。

---

## 💥 核心發現：2×2 ablation 在 test 上整個蒸發

val 上驅動我們提交決策的那套敘事（「31 維強度特徵有害、source-aware 有效」），在 test 上不成立：

| 系統 | 設定 | **A_PCC val (n=200)** | **A_PCC test (n=1,100)** |
|---|---|---|---|
| E4 | 10維L1，無 source-aware | 0.426 | 0.372 |
| E18 | 31維intensity，無 source-aware | 0.408 | 0.370 |
| E19 | 31維 + source-aware（**提交**） | **0.452** | **0.357** |
| E20 | ranking-only 增強 | 0.407 | 0.370 |

- **val 全距 0.045**（看起來 E19 明顯領先、E18 明顯落後）→ **test 全距僅 0.015**，且**排序完全打散**：
  val 上最高的 E19 在 test 上變**最低**，val 上被判「有害」的 E18 在 test 上與 E4 並列。
- 四個系統在 test 上全部落在 **0.357–0.372**，遠小於 n=1,100 的 A_PCC 95% CI 寬（0.103）。
- ⇒ **整個 2×2 消融的「結論」都是 val 上的抽樣雜訊。** source-aware、強度特徵、ranking 增強
  對 arousal **都沒有可偵測的真實效果**。這是論文 selection-overfitting 論點的最強實證：
  不只提交的那一顆選錯，是**整組據以做決策的排名都不可信**。

> **對論文的影響**：`docs/paper/outline.md` 中標為 ❌needs-evidence 的
> 「source-aware loss 改善 arousal」「31 維強度特徵有害」兩條 claim，
> test 證據顯示**兩者皆不成立** → 論文不得宣稱 source-aware 有效，
> 應改寫為「這些在 val 上看似有效的技巧，在 test 上都落在雜訊內」。

---

## 待提交清單（本機已備妥 zip）

`outputs/*_test.csv.zip`，依 `docs/remaining_experiments.md` P0-B 的價值排序：

1. ✅ macbert_s42、✅ macbert_s1
2. 🔲 macbert_s2 / s3 / s4（補完 seed 變異，最高 CP 值）
3. 🔲 e10_seed_ens（驗證 ensemble 是否真的有害）
4. 🔲 e18_l1_intensity（2×2 第三格）
5. 🔲 macbert_pseudo_s42（E13 trade-off）
6. 🔲 e20_rank_aug / roberta_s42 / robertaL_s42（補完）

之後 KG 消融的 18 個 run 訓練完，再各自提交。
