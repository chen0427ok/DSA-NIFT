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

---

## 無增強 baseline 的 test seed 變異（進行中）

目標：跑完 5 個 seed（macbert_s42/s1/s2/s3/s4），得到**官方 test 上真實的 seed std**，
取代目前在 dev 上 bootstrap 的代理估計（那是論文最容易被審稿人攻擊的但書）。

| seed | V_MAE | V_PCC | A_MAE | A_PCC |
|---|---|---|---|---|
| 42 | 0.61 | 0.87 | 0.94 | 0.37 |
| 1 | 0.62 | 0.87 | 0.91 | 0.37 |
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| 4 | — | — | — | — |
| **mean±std** | 待補 | 待補 | 待補 | 待補 |

**目前觀察（2 seed）**：A_PCC 兩顆都是 0.37，與 e19 的 0.357 幾乎無差 →
**再次印證 arousal 上這些系統彼此不可區分**（見 `docs/experiments.md` §0/§5.4）。
A_MAE 有 0.03 波動（0.94 vs 0.91），符合「arousal 量測本就雜訊大」的預期。

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
