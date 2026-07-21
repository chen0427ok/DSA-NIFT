# 論文骨架 — ROCLING-2026 Shared Task Paper

- **投稿**：EasyChair，track = *Special-Session Track – Shared Task on Chinese Dimension Sentiment Analysis*
- **截止**：2026-08-10（AoE）
- **語言**：英文 | **格式**：`rocling2026.sty`（範本在 `/Users/brian/Rocling2026/rocling2026-submission/`，工作副本在 `/Users/brian/Rocling2026/paper/`）
- **匿名**：**不需匿名**，要寫作者與單位
- **標題格式（強制）**：`TEAM_NAME at ROCLING-2026 Shared Task: <descriptive title>`
- **必引**：Lin, Liu, Tang, Lee. 2026. *ROCLING-2026 Shared Task: Chinese Dimensional Sentiment Analysis for New Immigrants' Feeling Texts.* （已放進 `paper/rocling2026.bib`，key = `lin-etal-2026-rocling`）

---

## 0. 敘事決策：走「系統描述 + 方法論發現」

兩種寫法：

| | A. 純系統描述 | **B. 系統描述 + selection overfitting 分析（建議）** |
|---|---|---|
| 主張 | 我們做了 L1/L3/source-aware，拿到 X 名 | 同左，**外加**：小型 validation 上的 arousal 模型選擇不可靠，我們有完整證據 |
| 風險 | 分數不突出時很單薄 | 需要 P0-3 的統計證據撐住 |
| 審稿人反應 | 「還行」 | shared task 場合**特別吃香**——負面結果 + 誠實分析是這類 workshop 的高價值內容 |

**採 B。** 理由：我們的 test 分數不會是前段班，但我們手上有一條非常完整的證據鏈
（13 次官方提交 + 三種增強策略的 trade-off + val→test 崩落），這是多數隊伍寫不出來的。
把「為什麼我們選錯模型」講清楚，比假裝分數很好有價值得多。

**候選標題**：
> `TEAM_NAME` at ROCLING-2026 Shared Task: Lexicon-Fused Regression and Why Arousal Gains on a 200-Sample Validation Set Do Not Transfer

較保守版：
> `TEAM_NAME` at ROCLING-2026 Shared Task: Affective-Lexicon Fusion and Source-Aware Loss for Chinese Dimensional Sentiment Analysis of New Immigrants' Texts

---

## 1. Abstract（~150 字）

段落角色：任務 → 方法 → 結果 → **發現**。

- **任務**：預測新住民反思文本的 VA（1–9），**官方未提供任何 in-domain 標註訓練資料**。
- **方法**：MacBERT 雙回歸頭 + CVAW/CVAP 情感詞典特徵融合（L1）；
  探索了情感知識圖譜引導的可控生成增強、multi-teacher 偽標、多模型融合、source-aware loss。
- **提交結果**：test 上 V-MAE 0.6200 / V-PCC 0.8663 / A-MAE 0.9259 / A-PCC 0.3566。
- **發現（賣點，一句話）**：valence 跨集合穩定，但 arousal PCC 從 validation 的 0.452 掉到 test 的 0.357；
  我們以 bootstrap 分析顯示 n=200 上的 arousal PCC 信賴區間寬到足以吞掉我們所有 ablation 的差距。

> ⚠️ 最後那句在 P0-3 跑完前**不可寫死**。沒有 CI 數字就只能寫「we observe a large drop」，不能宣稱原因。

---

## 2. Introduction（4 段）

1. **opening**：維度情緒分析（VA）與本任務；新住民文本的社會意義。
2. **challenge**：本任務的真正難點不是模型，是 **zero in-domain supervision**——
   官方只給無標籤的 200/1,100 篇，所有監督訊號都得從相鄰域借（EmoBank 書評/新聞、醫療反思、教育反思）。
   而 **arousal 對 domain shift 遠比 valence 敏感**（本文全篇的主軸對比）。
3. **approach**：我們的系統以 L1 詞典融合為核心，並系統性地測試了四類 arousal 專攻策略
   （生成增強 / 偽標精修 / 模型融合 / source-aware loss）。
4. **contributions**（條列 3 點，逐一對應到後面的證據）：
   - C1：顯性情感詞典特徵與 contextual embedding **互補**，同時改善 arousal 的校準與排序。
   - C2：三條常見的 arousal 強化路線在本任務上**各自失敗於不同原因**，我們給出診斷
     （ensemble 壓縮、偽標的自我參照陷阱、合成資料的校準↔排序 trade-off）。
   - C3：**小型 validation 上的 arousal 模型選擇不可靠**——我們用 13 次官方提交 + test 落差 + bootstrap 量化了它。

---

## 3. Task and Data（1 頁內）

- 任務定義、評分（4 指標 mean rank）。
- **強調資料前提**：官方無標註訓練資料 → 表 1 列出四個借用來源（9,435 筆）與其 arousal std。
- 內部 dev（DSA-MST 253）的角色，以及它的**兩個已知失真**（分數系統性偏高、與合成資料同風格）。
  這裡先埋伏筆，第 6 節才引爆。

---

## 4. System Description（Method，2 頁）

放 **pipeline 圖**（可直接由 `docs/method.md` 的 mermaid 轉出）。

- **4.1 Backbone**：MacBERT → mask 加權 mean pooling → 雙回歸頭；標籤 1–9 → [0,1] + sigmoid；SmoothL1。
- **4.2 L1 情感詞典融合**：CVAW+CVAP 共 7,761 詞 → 10 維聚合特徵 → concat 進 pooled embedding。
  **寫清楚 motivation**：arousal 缺乏 in-domain 監督，顯性詞級 VA 是唯一不依賴目標域標籤的訊號源。
- **4.3 L2/L3 情感知識圖譜與可控生成**：FastText+SVR 學詞→VA（補 OOV）→ 建 7,761 節點 kNN 圖 →
  `seeds_for_target()` 對 5 個代表性不足的 VA 區間撈種子詞 → LLM 生成 400 篇新住民反思。
- **4.4 Source-aware arousal loss**：依 `granularity` 對 arousal loss 加權
  （CVAS 0.25 / CVAT 0.5 / edu2021 0.75 / DSA-MST 1.0），valence 全 1.0。
  motivation = arousal 的 domain shift 比 valence 嚴重。
- **4.5 Submitted system**：E19 = 4.1 + 31 維 L1++ + 4.4。**誠實交代**這是依 validation 分數選出來的。

---

## 5. Experiments（2 頁）

- **表 2（主表）**：13 次官方 validation 提交的 4 指標全表（`docs/experiments.md` §3 直接搬）。
- **表 3（ablation）**：`lex_mode` × `source_aware` 的 2×2，**帶 seed mean ± std** ← 需 P1-1
- **表 4（最終 test）**：E19 在 test 的 4 指標 + 與 val 的對照。

三個 findings 各一小節，每節 = 現象 → 診斷 → 證據：

- **5.1 詞典融合有效**（實驗 3 → 4：A_PCC 0.388→0.426、A_MAE 0.944→0.882，4 指標全勝）
- **5.2 三條 arousal 路線的失敗診斷**
  - Ensemble：E10/E12 四指標輸單模型 → 多模型平均把**已壓縮**的 arousal 再壓一次
    （證據：拿掉 roberta 後 E10≈E12，傷害不在某顆爛模型而在「平均」本身）
  - 生成增強：實驗 5（k=1.0）A_PCC 0.461 但 A_MAE 1.100 → 5b（k=0.6）預測 std 收回、
    **PCC 守住而 MAE 沒回來** ⇒ 傷害在文本不在標籤
  - 偽標精修：E13 A_MAE 修回 0.898 但 A_PCC 掉回 0.423 ⇒ **自我參照陷阱**（teacher 自己就壓縮）
  - 綜合成一句：**arousal 增強存在校準 ↔ 排序的根本 trade-off**
- **5.3 Selection overfitting**（論文的高潮）
  - 現象：E19 A_PCC val 0.452 → test 0.357（−0.095），而 V_PCC 0.869 → 0.866（−0.003）
  - 診斷：我們對著一個 200 篇、只回傳 4 個標量的 leaderboard 做了 13 次選擇
  - 證據（**P0-3 已完成**）：bootstrap n=200 → A_PCC 95% CI 寬 **0.193** vs V_PCC **0.083**（2.31 倍）；
    選擇差距僅 0.026（小 7.4 倍）；seed std 僅 0.0055（排除訓練隨機性）；
    test 預測分布與 val 幾乎相同（arousal std 0.730 vs 0.729，排除模型漂移）
  - 附帶證據：silver ranking benchmark 也失敗（押 e20/e18 > e19，官方相反）——
    **兩個獨立的代理驗證訊號同時失效**，強化「這不是運氣不好，是任務本身的評估變異」

---

## 6. Limitations / Lessons（0.5 頁，別省）

- 兩個官方集合都無 gold 標籤 → 無法做 error analysis 或算官方分數的 CI（我們只能用 dev 代理）。
- 多數 ablation 是單 seed（除非 P1-1 補完）。
- 合成資料僅 400 篇、單一 LLM 生成、未做 L3 引導 vs 隨機種子的對照（除非 P2-1 補完）。
- **給後續參賽者的建議**：此類任務應在小 validation 上報告 CI，
  並優先鎖定跨集合穩定的維度（valence），對 arousal 的排名差異保持懷疑。

---

## 7. Conclusion（1 段）
重述 C1–C3，收在「valence 已近飽和、arousal 的瓶頸是評估與監督雙重稀缺」。

---

## 📋 Claim–Evidence Map（寫作時逐條核對）

| Claim | Evidence | Status |
|---|---|---|
| C1 詞典融合同時改善 arousal 校準與排序 | 實驗 3→4 官方分數，4 指標全勝 | ✅ supported |
| C2a Ensemble 對壓縮型 arousal 無效 | E10 / E12 官方分數，且 E10≈E12 | ✅ supported |
| C2b 合成資料的傷害在文本不在標籤 | 實驗 5b：pred std 收回但 A_MAE 未回 | ✅ supported |
| C2c 偽標的自我參照陷阱 | E13：A_MAE 修回、A_PCC 掉回 | ✅ supported（診斷屬推論，需寫成 "we hypothesize"） |
| C3 小 validation 的 arousal 選擇不可靠 | val→test 落差 0.452→0.357；bootstrap n=200 的 A_PCC CI 寬 0.193（V 只有 0.083）；選擇差距僅 0.026；seed std 0.0055；test 預測分布與 val 相同 | ✅ **supported**（but 但書：CI 是在 dev 上估的代理值） |
| source-aware loss 改善 arousal | 僅 E19 單 seed 的 val 分數，且 test 已崩 | ❌ **needs evidence — 必須 P1-1，否則降級為 "we adopted" 不宣稱有效** |
| L3 圖譜引導優於隨機種子 | **無任何對照組** | ❌ **needs evidence — 需 P2-1，否則方法節只描述、不宣稱優越** |
| 31 維強度特徵有害 | E18 < E4（單 seed，val） | ⚠️ partial，需 P1-1 |

> **紅線**：標 ❌ 的兩條，在對應實驗跑完前，Abstract 與 Introduction **一律不得宣稱有效**。
> 這正是 `docs/remaining_experiments.md` 的 P0-3 / P1-1 / P2-1 被排在最前面的原因。

---

## 🔍 五維自審清單（完稿前逐項回答）

1. **Contribution**：如果拿掉 selection-overfitting 分析，這篇還有什麼？（→ 有 C1/C2，但明顯變弱）
2. **Clarity**：valence/arousal 的對比是否貫穿全篇？術語（calibration = MAE、ranking = PCC）是否從頭定義並一致使用？
3. **Experimental strength**：ablation 有沒有誤差棒？審稿人一定會問「這 0.026 的差距顯著嗎」。
4. **Evaluation completeness**：有沒有誠實說明「我們無法取得 gold 標籤，所有分析都靠 dev 代理」？
5. **Method soundness**：source-aware 的權重（0.25/0.5/0.75/1.0）是怎麼決定的？
   —— **是憑直覺設的，沒有調參**。必須誠實寫出來，別假裝是搜出來的。
