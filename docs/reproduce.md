# 復現與環境 — DSA-NIFT @ ROCLING 2026

## 1. 復現各實驗

```bash
# 資料重建
bash download_external.sh && python prepare_data.py

# 實驗 1（baseline）
python train.py --no_lexicon --epochs 4 --batch_size 32   # train set 需切回 orign_train_data.csv
# 實驗 2（DAPT）
python dapt.py && python train.py --no_lexicon
# 實驗 3（反思語料）
python train.py --no_lexicon
# 實驗 4（L1 詞典融合，valence 最佳）
python train.py --epochs 4 --batch_size 32 --model hfl/chinese-macbert-base
#   等價：python train_v2.py --run_name e4_repro   （train_v2 預設即實驗 4）

# 實驗 5 / 5b（L3 生成增強）
#   notebooks/Rocling2026_Colab_aug.ipynb，cell 3 設 SHRINK（1.0 = 實驗 5、0.6 = 實驗 5b）

# E10–E12（ensemble）
for s in 42 1 2 3 4; do python train_v2.py --seed $s --run_name macbert_s$s; done
python ensemble.py macbert_s42 macbert_s1 macbert_s2 macbert_s3 macbert_s4 --mode mean --name e10_seed_ens

# E13（teacher 偽標）
python build_pseudo_labels.py \
    --teacher hfl/chinese-macbert-base=outputs/macbert_s42_best.pt \
    --teacher hfl/chinese-macbert-base=outputs/macbert_s1_best.pt \
    --teacher hfl/chinese-roberta-wwm-ext=outputs/roberta_s42_best.pt \
    --out data/train_aug_pseudo.csv
python train_v2.py --extra_train data/train_aug_pseudo.csv --run_name macbert_pseudo_s42

# E18 / E19（最終提交）/ E20 / E21
python train_v2.py --lex_mode l1_intensity                  --run_name e18_l1_intensity
python train_v2.py --lex_mode l1_intensity --source_aware   --run_name e19_source_aware   # ⭐ 最終提交
python train_v2.py --rank_aug data/train_aug.csv            --run_name e20_rank_aug
python train_v2.py --pooling mean_cls_dim_attention         --run_name e21_dim_attention
```

**除實驗 1–5b 外，一律用 `train_v2.py`。`train.py` 已凍結不得覆蓋。**

---

## 2. Colab 執行流程（VS Code + 官方 Colab 擴充 + Colab Pro A100）

**關鍵觀念：VS Code 只是介面，程式跑在遠端 VM 上。**
- **資料**：不是本機 `data/`，是 notebook cell 2 在 VM 上 `git clone` 下來的 repo。
- **產出**：在 VM 的 `repo/outputs/`，**斷線即失** → 必須主動打包拉回或 push。
- **private repo**：cell 2 用 `getpass` 輸入 GitHub token，**絕不寫進 notebook**。

notebook 對照：

| notebook | 涵蓋 |
|---|---|
| `notebooks/Rocling2026_Colab.ipynb` | L2 → L3 → L1 全流程 |
| `notebooks/Rocling2026_Colab_aug.ipynb` | 實驗 5 / 5b 增強（cell 3 有 `SHRINK` 旋鈕） |
| `notebooks/Rocling2026_Colab_ensemble.ipynb` | E10–E12 |
| `notebooks/Rocling2026_Colab_e13.ipynb` | E13 |
| `notebooks/Rocling2026_Colab_e18_e21.ipynb` | E18–E21（cell 18 有 E22 指令） |

本機收尾：
```bash
python fetch_results.py                        # 解 ~/Downloads/*_results.zip 進 outputs/
python fetch_results.py --pack <run>           # 打包成 submission.csv.zip
```

---

## 3. 提交格式（CodaLab / Codabench）

上傳的是 **zip**，內部檔名必須是 **`submission.csv`**（欄位 `ID,Valence,Arousal`）。
裸 csv 或錯檔名 → 報「Could not find scores file」。
`fetch_results.py --pack` 會自動正名並清掉 macOS 雜項檔。

---

## 4. 環境雷點（都是踩過的坑）

### Python 環境
- **`.venv` 是 `uv venv`，沒有 pip**：裝套件用 `uv pip install ...`，
  別用 `.venv/bin/python -m pip`（報 No module named pip）。
- **系統 `python3` 沒有 pandas**：跑分析腳本一律用 `.venv/bin/python`。
- **anaconda base 載 `l2_word_va.pkl` 會炸**（`cannot import name 'triu' from 'scipy.linalg'`，
  gensim/scipy 版本衝突）→ **必須用 `.venv`**；Colab 端已 pin `gensim>=4.3.3`。

### 本機訓練（M2 MacBook Air）
- **⚠️ `mps` 在正式 batch size 下會 hang，且不易察覺**：
  `--batch_size 32 --max_len 256` 跑 `train_v2.py` 時 process 存活但卡在第一個 training step，
  log 永遠停在 HF `BertModel LOAD REPORT`，不會印出 `[epoch 1] step50/...`。
- **判斷方法**：`ps -o etime,time -p <pid>` 比對 wall time 與 CPU time。
  實測案例：跑了 **5 小時 19 分**只累積 **9 分 58 秒 CPU** → 卡住而非慢跑，直接 `kill`。
- **教訓**：本機只跑小 batch（8）+ 短 max_len（64）的 smoke test 驗證程式邏輯；
  **正式訓練一律上 Colab A100**。

### 背景任務
- **Claude Code session 中斷會殺掉背景訓練**：`run_in_background: true` 啟動的 process，
  session 重啟時會被一併終止且不留錯誤訊息。
  需要長跑就在**使用者自己的終端機**用 `nohup ... &`，與 Claude Code 脫鉤。

### git / notebook
- **`git checkout` 報 "Unable to read current working directory"**：shell 抓到被刪的 inode → 重新 `cd` 進 baseline。
- **git push 常被擋**：先 `git fetch` → rebase；遠端 commit 常加 `outputs/*.csv`（tracked），
  與本機同名未追蹤檔衝突時先 `rm` 那些 csv（保留 `.pt` / `.pkl`）再 rebase。
- **改 `.ipynb`**：用 `NotebookEdit` 或直接改 JSON（改完 `json.load` 驗證）。
  **VS Code 開著且在跑時會自動存檔覆蓋外部修改** → 要改正在跑的 notebook 請在 VS Code 介面改。

### API key
- `augment_generate.py` / `build_silver_pairs.py` 需 `ANTHROPIC_API_KEY`
  （或 `OPENAI_API_KEY` + `--provider openai`）。
  **Claude Code 的 Bash sandbox 沒有這些 key**，要在使用者自己的終端機或 Colab Secrets 跑。**會計費。**

### 安全
- `notebooks/Rocling2026_Colab_ensemble.ipynb` 曾在工作目錄版本內嵌一個 GitHub PAT
  （`ghp_qljV...`）。**已於整理時 redact，且經 `git log --all -S` 確認從未進入 git 歷史。**
  但該 token 曾以明文存在本機檔案，**仍建議去 GitHub 撤銷重發**。
  新版 notebook 已改用 `getpass`，無此問題。

---

## 5. 未追蹤檔案（gitignore）

- `data/train_aug.csv`（400 篇合成，**本機生成、不在 git**）、`data/train_base.csv`
- `outputs/`（`l2_word_va.pkl` 814MB、`l3_graph.pkl`、13 顆 `*_best.pt`、各 `*_submission.csv`）
- `*.pt`、`*.zip`、`.venv/`

本機 `outputs/` 已備妥 E10–E13 / E18–E21 的全部權重（含 **`e19_source_aware_best.pt`**，
即最終提交模型，可用來重跑 test set 推論）。**沒有 E22 的權重**（三次本機嘗試皆 hang）。
