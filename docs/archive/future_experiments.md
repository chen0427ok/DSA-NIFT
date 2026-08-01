# 未來實驗規劃 — DSA-NIDF ROCLING 2026

> 接續 `experiment.md`（實驗 1–9b）與 `handover.md` 的下一步清單。
> 現況：**實驗 4（L1 詞典融合）是提交基準**（V 0.600/0.880、A 0.882/0.426）；
> 唯一瓶頸是 **Arousal PCC**。實驗 9/9b 證明合成文本能提 A_PCC（→0.46）但 bin 弱標籤毀了 A_MAE。
>
> 本輪新增程式（`train.py` 凍結不動，實驗 1–9b 可復現）：
> `train_v2.py`、`ensemble.py`、`build_pseudo_labels.py`、`embed_regressor.py`、
> `lexicon_l2.py`、`calibrate.py`。統一預測格式輸出到 `outputs/preds/{run}_{dev,val}.csv`。

---

## 實驗總表（編號接續 9b，優先序 S > A > B）

| # | 實驗 | 假設 / 目的 | 程式 | GPU | 優先 |
|---|---|---|---|---|---|
| **E10** | Multi-seed ensemble（實驗 4 × 5 seeds 等權平均） | 兩篇得獎論文共識的最大槓桿；平掉 seed 方差、穩定 arousal 排序 | `train_v2.py` + `ensemble.py --mode mean` | Colab ×5 | **S** |
| **E11** | RoBERTa-wwm-ext(-large) + L1 | 換 encoder bias，large 對長反思文本可能更會排 arousal | `train_v2.py --model ...` | Colab | **S** |
| **E12** | 跨 encoder dimension-wise weighted ensemble | V/A 分開加權，A 維度只讓會排 arousal 的模型投票 | `ensemble.py --mode weighted` | 無（融合腳本） | **S** |
| **E13** | Teacher 偽標精修增強（多 teacher 逐篇重標 + 每 bin ±1.5SD 離群移除） | 實驗 9 的 A_PCC 增益來自文本、傷害來自標籤 → 換掉標籤即可能 4 指標全勝 | `build_pseudo_labels.py` + `train_v2.py --extra_train` | Colab | **A** |
| **E14** | Frozen embedding + SVR / Ridge（VA Regression Ensemble） | 補一條非 neural-head 分數分布，進 E12 當補位成員（TCU 路線） | `embed_regressor.py` | Colab（抽 emb） | **A** |
| **E15** | Arousal-weighted loss（1.2）＋ batch PCC loss（0.1） | 直接把訓練目標往 arousal 排序偏 | `train_v2.py --arousal_weight/--pcc_weight` | Colab | **B** |
| **E16** | L2 OOV 詞 VA 特徵接進 L1（10→16 維） | 詞典覆蓋不足的反思詞彙由 L2 補 VA 訊號 | `lexicon_l2.py` + `train_v2.py --lex_mode l1l2` | Colab | **B** |
| **E17** | Arousal 後校準（dev 網格搜尋 scale/shift） | 修 A_MAE 的整體偏移/壓縮；**只影響 MAE，PCC 不變** | `calibrate.py` | 無 | **B**（最後做） |

---

## 建議執行順序與指令

### 第一波（拿分主線，一個 Colab session 可跑完 E10+E11）

**E10 — multi-seed（先跑，之後所有實驗的對照都有 5 顆 teacher 可用）**
```bash
for s in 42 1 2 3 4; do
  python train_v2.py --seed $s --run_name macbert_s$s --epochs 4 --batch_size 32
done
python ensemble.py macbert_s42 macbert_s1 macbert_s2 macbert_s3 macbert_s4 \
    --mode mean --name e10_seed_ens
# → outputs/e10_seed_ens_submission.csv
```
判讀：dev 的 ensemble 指標應該全面 ≥ 單 seed；官方 A_PCC 目標 0.43+。

**E11 — RoBERTa-wwm-ext / large**
```bash
python train_v2.py --model hfl/chinese-roberta-wwm-ext       --run_name roberta_s42
python train_v2.py --model hfl/chinese-roberta-wwm-ext-large --batch_size 8 --lr 1e-5 \
    --run_name robertaL_s42   # T4 記憶體：large 一定要 batch 8 + lr 降到 1e-5
```
判讀：只看它 dev 的 **arousal** 指標是否比 macbert 好；就算整體輸，只要 A 維度有特色就值得進 E12。

**E12 — dimension-wise ensemble（工程量 ≈ 0，把手上所有 run 丟進去）**
```bash
python ensemble.py macbert_s42 macbert_s1 macbert_s2 macbert_s3 macbert_s4 \
    roberta_s42 robertaL_s42 --mode weighted --name e12_enc_ens
```
注意：dev 只有 253 筆，權重可能 overfit → 同時輸出 `--mode mean` 版比較，兩者 dev 差距大時選保守的 mean。

### 第二波（arousal 專攻）

**E13 — 偽標精修（實驗 9 的續章，最有機會 4 指標全勝的增強版）**
```bash
# teacher 用 E10/E11 訓好的、encoder 異質的 2–3 顆（錯誤不相關才有清理效果）
python build_pseudo_labels.py \
    --teacher hfl/chinese-macbert-base=outputs/macbert_s42_best.pt \
    --teacher hfl/chinese-macbert-base=outputs/macbert_s1_best.pt \
    --teacher hfl/chinese-roberta-wwm-ext=outputs/roberta_s42_best.pt \
    --out data/train_aug_pseudo.csv
python train_v2.py --extra_train data/train_aug_pseudo.csv --run_name macbert_pseudo_s42
```
變體（一次生成、多次實驗）：
- `--blend 0.3`：保留 30% bin 中心訊號（怕 teacher 把極端樣本全拉回中間、丟失 PCC 增益）。
- 減量：`--sd_k 1.0` 收緊離群移除 ≈ 只留 200–300 篇高信度樣本。
- ⚠️ **dev 對增強實驗不可信**（實驗 9 教訓：增強資料與 dev 同風格）。E13 的 dev 分數只看趨勢，
  結論必須靠官方提交；提交額度不夠就先跑 E10/E12。

**E14 — frozen embedding 回歸（跑一次，供 E12 補位）**
```bash
python embed_regressor.py --model hfl/chinese-macbert-base
# 然後把 svr_chinese-macbert-base 加進 ensemble：
python ensemble.py macbert_s42 ... svr_chinese-macbert-base --mode weighted --name e12b
```
判讀：SVR 單獨不用贏 BERT；只要它的 arousal 預測與 neural 模型**相關性低但 PCC 不太差**，加進 ensemble 就有貢獻。

### 第三波（小成本試探）

**E15 — loss 調整（各跑一顆，與 macbert_s42 對照）**
```bash
python train_v2.py --arousal_weight 1.2 --run_name macbert_aw12_s42
python train_v2.py --pcc_weight 0.1    --run_name macbert_pcc01_s42
```
規則：先 1.2 / 0.1，有效再加大（1.5 / 0.2）；valence 掉超過 0.01 PCC 就停。有效的顆也丟進 E12。

**E16 — L1+L2 特徵**
```bash
# 需 outputs/l2_word_va.pkl（814MB，Colab 由 notebook 的 L2 cell 重建；本機一定要用 .venv）
python train_v2.py --lex_mode l1l2 --run_name macbert_l1l2_s42
```

**E17 — 校準（最後、只對最終要提交的 ensemble 做）**
```bash
python calibrate.py --run e12_enc_ens --dim arousal --name e12_cal
```
提醒：線性校準 **PCC 不變**，純粹修 A_MAE；dev 上省不到 0.01 就別用（過擬合 253 筆的風險）。

---

## 提交策略（官方提交額度珍貴）

1. 保底：實驗 4 已提交的成績留著。
2. 第 2 份：**E10**（multi-seed mean）— 幾乎不可能比實驗 4 差。
3. 第 3 份：**E12**（加了 E11/E14/E15 成員的 weighted ensemble），與 E10 dev 比較後選一。
4. 第 4 份起：**E13**（偽標精修）— 高風險高報酬，dev 不可信，必須用真提交驗證；
   若 A_MAE 回到 0.9 以下且 A_PCC ≥ 0.45，把 `macbert_pseudo_*` 也加進最終 ensemble。

## 論文敘事對應

- 方法章主線：L1（詞典融合）→ L2（OOV VA 擴充，E16）→ L3（圖譜控制生成）→
  multi-teacher 偽標精修（E13）→ ensemble（E10/E12/E14）。
- E13 就算官方分數沒全勝，「合成文本提升 arousal 排序、偽標精修 vs 整批收縮的對照」
  （實驗 9 → 9b → E13）本身就是完整的分析章節。

## 統一預測格式（所有腳本共用）

```
outputs/preds/{run}_dev.csv : ID,valence_true,arousal_true,valence_pred,arousal_pred,model_name,split
outputs/preds/{run}_val.csv : ID,valence_pred,arousal_pred,model_name,split
outputs/{run}_submission.csv: ID,Valence,Arousal   （官方格式）
```
ensemble 的輸出也寫回 `outputs/preds/`，所以 ensemble 之上可以再疊 ensemble 或接 `calibrate.py`。
