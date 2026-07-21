# 資料集 — DSA-NIFT @ ROCLING 2026

## 0. 關鍵前提：官方沒給有標註的訓練資料

官方只釋出：

| 檔案 | 篇數 | 內容 | 角色 |
|---|---|---|---|
| `DSANIDF_ValidationSet.csv` | 200 | `ID, Text`（**無標籤**） | 評測期間的提交目標（leaderboard） |
| `DSANIDF_TestSet.csv` | 1,100 | `ID, Text`（**無標籤**） | 最終評測目標 |

**沒有任何目標域（新住民反思）的監督樣本。** 全部訓練標註都來自相鄰域的借用語料。
這是本任務 low-resource / zero in-domain label 的本質，也是 arousal 難的根源：
目標域的 arousal 分布完全沒有監督訊號。

> 兩個官方集合都**沒有釋出 gold 標籤**，我們手上只有 leaderboard 回傳的 4 個標量分數。
> 這一點直接限制了所有事後分析（無法算信賴區間、無法做 error analysis），
> 也是 `docs/experiments.md` §5.4 selection overfitting 分析的前提。

---

## 1. 訓練集 `data/train.csv`（9,435 筆）

由 `prepare_data.py` 合併四個來源：

| 來源 | 筆數 | `granularity` 欄位 | 語域 | 平均文長 | Arousal std |
|---|---|---|---|---|---|
| Chinese EmoBank — CVAS | 2,583 | `sentence` | 書評 / 新聞，句子級 | 57 | 1.05 |
| Chinese EmoBank — CVAT | 2,970 | `text` | 書評 / 新聞，篇章級 | 57 | 1.05 |
| DSA-MST（ROCLING-2025） | 2,282 | `reflection` | **醫療自我反思**（最貼近目標域） | 76 | 1.31 |
| ROCLING-2021 | 1,600 | `edu2021` | **教育反思**短文（arousal 訊號最強但偏短） | 19 | 1.32 |

`granularity` 欄位是 E19 `--source_aware` 加權的依據。

### dev `data/dev.csv`（253 筆）
從 DSA-MST 反思切出——最接近目標域，避免拿 EmoBank 當 dev 高估分數。

> ⚠️ **兩個已知失真**：
> 1. dev 分數系統性高於官方（dev A_PCC ~0.61 vs official ~0.42），**絕不可當官方分數看**。
> 2. dev 與合成資料同風格 → **所有增強類實驗的 dev 分數不可信**（實驗 5 / E13 的教訓）。

### 舊版 `data/orign_train_data.csv`（4,998 筆）
只用 EmoBank 的版本，實驗 1–2 用，保留供復現。

---

## 2. 合成資料（獨立檔，非預設併入 train.csv）

| 檔案 | 筆數 | 說明 | git |
|---|---|---|---|
| `data/train_aug.csv` | 400 | `claude-opus-4-8` 生成的新住民第一人稱反思，L3 圖譜引導，5 個 VA 象限各 80 | gitignore（**僅本機**） |
| `data/train_aug_pseudo.csv` | 318 | 上者經 3 顆 teacher 逐篇重標 + 每 bin ±1.5SD 離群移除（E13 用） | `git add -f` 追蹤 |

> `train_aug.csv` 是**唯一「目標域風格 × 目標 arousal 區間」的監督樣本**——
> 這正是實驗 5 能把 A_PCC 推到 0.461 的原因（它補了別處都沒有的洞）。
> 品質經人工抽檢：完全在域內、情緒象限正確，低喚醒兩端與中性高喚醒（原訓練集最缺）補得好。

生成細節與 5 個 bin 的設定見 `docs/experiments.md` §4「實驗 5 / 5b」。

---

## 3. 外部詞典（特徵資源，非訓練樣本）

`external/emobank/`：

| 子集 | 內容 | 數量 |
|---|---|---|
| CVAW | 單字 | 5,512 |
| CVAP | 片語 | 2,249 |
| **合計** | 餵 L1 / L2 / L3 | **7,761** |

CVAS（2,582 句）/ CVAT（2,969 篇）則作為訓練樣本併入 `train.csv`。

---

## 4. 官方推薦資源的採用評估

| 資源 | 內容 | 判斷 | 採用 |
|---|---|---|---|
| ROCLING-2025 DSA-MST | 醫師 ICU 反思，文檔級 VA 1–9，繁中，2,535 篇 | 同型任務、文長/語域/尺度全對齊 | ✅ 訓練 + dev |
| ROCLING-2021 教育 DSA | 學生課堂回饋短文，VA 1–9，繁中，1,600 篇 | 反思語域、arousal 變異最大，但文本偏短 | ✅ 訓練 |
| DimABSA2026 | 6 語言 aspect-based VA | aspect 級、評論域，需轉換、域不合 | ❌ |
| SIGHAN2024-dimABSA | 中文餐廳評論 aspect-based VA | aspect 級、評論域 | ❌ |

採用原則：**文檔級 + 反思語域 + VA 1–9 直接可用**者優先。

---

## 5. 重建指令

```bash
bash download_external.sh   # 下載 DSA-MST（第一次才要）
python prepare_data.py      # 產生 data/train.csv, dev.csv, val_unlabeled.csv
```
