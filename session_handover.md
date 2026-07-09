# Session Handover — E10–E17 實驗基礎設施 + Colab/VS Code 執行

> 本次 session 的交接。承接 `handover.md`（實驗 1–9b）與 `future_experiments.md`（E10–E17 規劃）。
> 現況：**實驗 4（L1 詞典融合）仍是提交基準**（官方 V 0.600/0.880、A 0.882/0.426）；瓶頸是 **Arousal PCC**。
> 本次沒有改動任何既有模型結論，只**新增了跑後續實驗需要的程式與執行流程**，並已在 A100 上啟動 E10。

---

## 1. 本次做了什麼

### 1.1 新增 6 支程式（`train.py` 凍結不動，實驗 1–9b 仍可復現）
| 檔案 | 用途 | 對應實驗 |
|---|---|---|
| `train_v2.py` | 擴充訓練器：統一預測輸出 + multi-seed / 多 encoder / arousal-weighted / PCC-loss / L1L2 旋鈕。**預設參數 = 完全復現實驗 4** | E10/E11/E15/E16 |
| `ensemble.py` | dimension-wise weighted / mean 融合（V、A 分開算權重） | E10/E12 |
| `build_pseudo_labels.py` | multi-teacher 逐篇偽標 + teacher 分歧過濾 + 每 bin ±1.5SD 離群移除 | E13 |
| `embed_regressor.py` | frozen embedding + SVR/Ridge 回歸（非 neural 補位） | E14 |
| `lexicon_l2.py` | L1+L2 OOV 詞 VA 特徵（10→16 維） | E16 |
| `calibrate.py` | arousal 後校準（網格搜尋 scale/shift；**只改 MAE，PCC 不變**） | E17 |

### 1.2 規劃與 notebook
- `future_experiments.md`：**E10–E17 完整規劃 + 每個實驗的 Colab 指令 + 提交策略**（權威，先看這份）。
- `Rocling2026_Colab_ensemble.ipynb`：E10/E11/E12 專用 notebook（見下方 §3）。

### 1.3 git 狀態
- 分支：**`feat/ensemble-teacher-student-experiments`**（remote: `https://github.com/chen0427ok/DSA-NIFT.git`，**private**）。
- 已 push commit `3f07af0`（6 支程式 + 規劃 + notebook + 交接文件 + experiment.md 的 9/9b 記錄）。
- **未 commit**：`Rocling2026_Colab_ensemble.ipynb` 的最新修改（git clone 版 cell 2、A100 batch 調整、GPU 檢查 cell）——這些是本 session 在 VS Code 裡邊跑邊改的，值得再 commit 一次同步。
- `.gitignore` 已加 `*.zip` 與 `data/train_aug_pseudo.csv`。

---

## 2. 統一預測格式（所有腳本共用，這是 ensemble/校準能串起來的關鍵）
```
outputs/preds/{run}_dev.csv : ID,valence_true,arousal_true,valence_pred,arousal_pred,model_name,split
outputs/preds/{run}_val.csv : ID,valence_pred,arousal_pred,model_name,split
outputs/{run}_submission.csv: ID,Valence,Arousal   （官方格式）
outputs/{run}_best.pt       : 權重（E13 的 teacher 會用到，務必保留）
```
ensemble 的輸出也寫回 `outputs/preds/`，所以可以「ensemble 之上再疊 ensemble」或接 `calibrate.py`。

---

## 3. Colab + VS Code 擴充套件執行流程（本次採用）

**環境**：Google 官方「Google Colab」VS Code 擴充套件 + Colab **Pro**，本次連到 **A100-SXM4-40GB**。

**關鍵觀念**：VS Code 只是介面，**程式跑在 Colab 遠端 VM 上**。因此：
- **資料**：不是讀本機 `data/`，是 cell 2 在 VM 上 **git clone** 下來的 repo 的 `data/`（內容相同，位置在雲端）。
- **產出**：submission / 權重都在 VM 的 `repo/outputs/`，**session 一斷就消失** → 必須用 cell 8 打包拉回本地或推回 GitHub。
- **private repo**：cell 2 用 `getpass` 執行時輸入 GitHub token（不寫進 notebook 檔案）。

**notebook cell 流程**：
1. cell 1：裝套件 + **印出連到哪顆 GPU**（A100/L4/V100/T4 判定 + nvidia-smi）。
2. cell 2：git clone branch → `cd repo` →（token 用 getpass 輸入）。
3. cell 3–4：**E10** MacBERT+L1 × 5 seeds（本次用 `--batch_size 64`，dev A_PCC≈0.61 與實驗 4 一致，OK）。
4. cell 5：E10 融合（`--mode mean`）。
5. cell 7–8：**E11** RoBERTa-wwm-ext（batch 64）+ large（batch 16, lr 1e-5）。
6. cell 10：**E12** 跨 encoder ensemble（weighted + mean 兩版對照）。
7. cell 12：打包 `outputs/` → 拉回本地（files.download / VS Code 檔案總管 / 掛 Drive）。

> **batch size 註**：實驗 4 官方是 batch 32；A100 上本次用 64，dev 分數與 32 幾乎相同，可接受。若要嚴格對齊實驗 4 復現，改回 32 即可（A100 一樣快）。

---

## 4. 目前進度（本 session 結束時）

- ✅ 資料已驗證：repo 的 `train=9435 / dev=253 / val=200`，本地與 remote blob hash 一致，無 train_aug 污染、無空文本、標籤範圍正常。
- 🔄 **E10 進行中**：A100 上跑 5 seeds，seed 42/1 已完成、seed 2 訓練中。
  - seed 42 best dev：V_MAE 0.472 / V_PCC 0.819 / A_MAE 0.824 / A_PCC 0.615
  - seed 1  best dev：V_MAE 0.491 / V_PCC 0.813 / A_MAE 0.839 / A_PCC 0.615
  - （dev = DSA-MST，分數天生高於官方；**別當官方分數**）
- ⏳ E11/E12 尚未跑。

---

## 5. 下一步（優先序）

### 5.1 先收尾 E10–E12
- 跑完 5 seeds → cell 5 融合 → 跑 E11 兩顆 → cell 10 做 E12。
- **cell 8 打包 `outputs/` 拉回本地**（含所有 `*_best.pt`，E13 要當 teacher）。
- 提交策略（見 `future_experiments.md` §提交策略）：實驗 4 保底 → E10(mean) → E12(weighted/mean 擇一)。
- 判讀 E12：dev 只有 253 筆，weighted 有 overfit 風險；weighted 與 mean 差很多時**選保守的 mean**。

### 5.2 主攻 arousal：E13 偽標精修（第二波，最有機會 4 指標全勝的增強）
```bash
python build_pseudo_labels.py \
    --teacher hfl/chinese-macbert-base=outputs/macbert_s42_best.pt \
    --teacher hfl/chinese-macbert-base=outputs/macbert_s1_best.pt \
    --teacher hfl/chinese-roberta-wwm-ext=outputs/roberta_s42_best.pt \
    --out data/train_aug_pseudo.csv
python train_v2.py --extra_train data/train_aug_pseudo.csv --run_name macbert_pseudo_s42
```
- 假設：實驗 9 的 A_PCC 增益來自**文本**、傷害來自**bin 弱標籤** → 用 teacher 逐篇重標可能保住增益、修回 MAE。
- 變體：`--blend 0.3`（保留部分 bin 訊號）、`--sd_k 1.0`（收緊留高信度）。
- ⚠️ **E13 的 dev 不可信**（增強與 dev 同風格），結論必看官方提交。
- 前置：需 `data/train_aug.csv`（400 篇，本機生成、**gitignore 不在 repo**）——Colab 上要另外上傳或改從本機帶入。

### 5.3 平行低成本：E14 frozen embedding + SVR
```bash
python embed_regressor.py --model hfl/chinese-macbert-base
# 再把 svr_chinese-macbert-base 加進 ensemble.py 的 runs（E12 補位）
```

### 5.4 第三波（小成本試探，有效再進 E12）
- E15：`train_v2.py --arousal_weight 1.2` / `--pcc_weight 0.1`
- E16：`train_v2.py --lex_mode l1l2`（需 `outputs/l2_word_va.pkl`，814MB，Colab 由 L2 cell 重建；本機務必用 `.venv`）
- E17：`calibrate.py --run <最終ensemble> --dim arousal`（最後才做，只修 A_MAE）

---

## 6. 雷點（延續 handover.md + 本次新增）
- **VS Code + Colab 是遠端運算**：本機檔案不在 VM 上，一律 git clone；產出要主動拉回，斷線即失。
- **private repo clone**：cell 2 用 getpass 輸入 token，**別把 token 寫進 notebook**（會隨 commit 外洩）。
- **`data/train_aug.csv` 不在 git**（gitignore）：E13 在 Colab 跑前要確認 VM 上有這個檔（上傳或重新生成，生成要 API key + 計費）。
- **`.venv` 是 uv、無 pip**：本機裝套件用 `uv pip install`；載 `l2_word_va.pkl` 必須用 `.venv`（anaconda base 會炸 scipy/gensim）。
- **改 `.ipynb`**：VS Code 開著且在跑時會自動存檔覆蓋外部修改 → 要改 cell 請在 VS Code 介面改，或先關閉/停止再改檔。
- **dev ≠ 官方**：dev 是 DSA-MST 反思，分數系統性高於官方 val；增強實驗的 dev 尤其失真。

---

## 7. 專案約束（務必遵守）
- **一律用繁體中文回覆。**
- **commit message 不得提及 Claude 共同作者。**
- 功能開在各自 branch；已跑過的實驗程式**不可覆蓋**、必須可復現。
- repo 為 **private**（`chen0427ok/DSA-NIFT`）。
