# DSA-NIFT 下一階段實驗規劃

> 任務：ROCLING 2026 Shared Task — **Chinese Dimensional Sentiment Analysis for New Immigrants' Feeling Texts (DSA-NIFT)**  
> 目標：對新住民反思 / 感受文本預測 **Valence / Arousal**，範圍為 1–9 實數。  
> 目前核心瓶頸：**Arousal PCC 偏低**。  
> 本文件整理目前討論後的實驗優先順序、背後原因、預期效果、實作方向與參考論文。

---

## 0. 目前任務設定與已知結論

### 0.1 資料組成

目前 `data/train.csv` 共 **9,435 筆**，但不是單一同分布資料集，而是由多個相鄰 domain 組成：

| 來源 | 筆數 | 粒度 | 說明 |
|---|---:|---|---|
| Chinese EmoBank — CVAS | 2,583 | sentence | 中文情感庫，句子級 VA |
| Chinese EmoBank — CVAT | 2,970 | text | 中文情感庫，篇章級 VA |
| DSA-MST / ROCLING-2025 | 2,282 | reflection | 醫療自我反思文本，另 253 筆切為 dev |
| ROCLING-2021 | 1,600 | edu2021 | 教育反思短文 |
| **合計** | **9,435** | — | — |

另外有合成資料：

| 檔案 | 筆數 | 說明 |
|---|---:|---|
| `data/train_aug.csv` | 400 | Opus 4.8 生成的新住民第一人稱反思文本，由 L3 情緒圖譜引導，5 個 VA 象限各 80 篇 |
| `data/train_aug_pseudo.csv` | 318 | 對上述 400 篇以 3 顆 teacher 重新偽標並移除離群樣本後的資料 |

官方只提供：

| 檔案 | 筆數 | 說明 |
|---|---:|---|
| `data/val_unlabeled.csv` | 200 | 新住民文本，無標籤，為提交目標 |

### 0.2 關鍵任務特性

本任務不是一般 supervised VA regression，而是：

> **Zero in-domain labeled target adaptation**  
> 有標籤資料都來自通用情緒庫、醫療反思與教育反思；真正目標域「新住民反思文本」沒有任何標註資料。

因此目前 Arousal 困難的根本原因不是模型容量不足，而是：

1. **沒有目標域 Arousal supervision**
2. **Arousal 比 Valence 更受 domain shift 影響**
3. **目標域的新住民文本可能有不同的喚醒度表達方式**
4. **合成資料能補 target-style arousal pattern，但絕對標籤不可靠**

---

## 1. 目前最佳結果與診斷

### 1.1 目前最佳模型：實驗 4 — L1 詞典融合

目前最佳提交基準為：

```text
hfl/chinese-macbert-base
→ attention-mask mean pooling
→ concat L1 lexicon features
→ Linear regression head
→ Valence / Arousal
```

成績：

| 實驗 | Valence MAE ↓ | Valence PCC ↑ | Arousal MAE ↓ | Arousal PCC ↑ |
|---|---:|---:|---:|---:|
| Baseline | 0.654 | 0.867 | 0.985 | 0.412 |
| DAPT | 0.649 | 0.866 | 1.009 | 0.395 |
| 反思語料 | 0.627 | 0.870 | 0.944 | 0.388 |
| **L1 詞典融合** | **0.600** | **0.880** | **0.882** | **0.426** |
| L3 生成增強 raw label | 0.611 | 0.870 | 1.100 | **0.461** |
| L3 生成增強 shrink | 0.632 | 0.874 | 1.058 | **0.460** |
| multi-seed ensemble | 0.614 | 0.881 | 0.907 | 0.415 |
| multi-encoder ensemble | 0.613 | 0.882 | 0.906 | 0.418 |
| teacher pseudo-label | 0.617 | 0.878 | 0.898 | 0.423 |
| E18 L1++ intensity | 0.641 | 0.865 | 0.909 | 0.408 |
| **E19 source-aware loss** | 0.666 | 0.869 | **0.870** | **0.452** |
| E20 ranking-only 增強 | 0.631 | 0.877 | 0.923 | 0.407 |
| E21b dim-attention + rank | 0.638 | 0.865 | 0.923 | 0.394 |
| E11b RoBERTa-large 單顆 | 0.691 | 0.869 | 0.914 | 0.407 |

> **2026-07-11 官方結果更新**：E19 為 arousal 雙贏新突破（A_MAE/A_PCC 皆優於實驗 4，但 valence 退化，
> 兩者各贏 2 指標）；E18/E20/E21b/robertaL 官方確認淘汰（E21a 未提交）。
> **E18 消融發現：E19 增益全來自 source-aware，31 維強度特徵有害 → 下一發首選
> `--lex_mode l1 --source_aware`。** 詳見 `experiment.md` 實驗 18–21 章。

### 1.2 目前最重要的診斷

#### L1 詞典融合有效

L1 將 Chinese EmoBank 的 CVAW / CVAP 詞典轉為 10 維 VA 情緒特徵，補充 MacBERT embedding 沒有明確表示的情緒先驗。這對 zero in-domain labeled 的 DSA-NIFT 特別重要。

#### 一般 ensemble 不適合目前問題

E10 / E12 顯示 multi-seed / multi-encoder 平均融合沒有改善 Arousal PCC，反而可能進一步壓縮 arousal 預測分布。

#### 生成資料有排序價值，但絕對標籤不可靠

`train_aug.csv` 使用 raw bin center label 時：

```text
Arousal PCC: 0.426 → 0.461
Arousal MAE: 0.882 → 1.100
```

這代表合成資料確實讓模型更會分辨高低喚醒，但絕對分數校準變差。

#### teacher pseudo-label 修 MAE，但丟掉排序

E13 中 teacher 重新偽標後：

```text
Arousal MAE: 1.100 → 0.898
Arousal PCC: 0.461 → 0.423
```

原因是 teacher 本身 arousal 預測壓縮，會把低喚醒樣本也標到接近中間值，導致排序訊號消失。

---

## 2. 目前 loss 是什麼？

### 2.1 目前不是 Cross Entropy

目前模型做的是 **連續值回歸**，不是分類。

標籤：

```text
Valence, Arousal ∈ [1, 9]
```

訓練時會先正規化：

```text
y_norm = (y - 1) / 8
```

模型輸出經 sigmoid 限制到 `[0, 1]`：

```text
y_pred_norm = sigmoid(model_output)
```

目前主要 loss 是：

```text
SmoothL1(y_pred_norm, y_true_norm)
```

所以目前不是 cross entropy。

### 2.2 Cross Entropy 什麼時候會用？

只有當我們把 VA 離散成 bins，例如：

```text
Arousal low / mid / high
Valence low / mid / high
```

才會使用 cross entropy 作為 auxiliary loss。

建議未來若使用 ordinal/bin head，採用：

```text
total_loss =
  SmoothL1 regression loss
  + small_weight * CrossEntropy(bin prediction)
```

而不是完全改成 cross entropy。

---

## 3. 參考方法來源

### 3.1 ROCLING 2025 前三名方法

#### Paper R1 — CYUT-NLP at ROCLING-2025 Shared Task

方法重點：

- 使用 LLM / RAG 生成增強資料
- 多 teacher pseudo-label
- outlier removal
- 提升低資源 domain generalization

對本任務的啟發：

- 可控生成對低資源 VA 預測有幫助
- 但我們的 E13 顯示 teacher pseudo-label 可能壓縮 Arousal
- 因此不宜照抄，應改成 **ranking-only augmentation**

Reference:  
Jian et al. 2025. *CYUT-NLP at ROCLING-2025 Shared Task: Valence–Arousal Prediction for Chinese Medical Self-Reflection Texts Using Domain-Specific Augmentation and Multi-Teacher Pseudo-Labeling*.  
https://aclanthology.org/2025.rocling-main.42/

#### Paper R2 — NTULAW at ROCLING-2025 Shared Task

方法重點：

- 分析 domain-specific annotation 對 Arousal 的重要性
- 發現 Valence 可受益於外部情緒資源
- Arousal 更需要接近目標 domain 的資料

對本任務的啟發：

- Valence / Arousal 不應完全用同一套資料策略
- Arousal loss 應使用 source-aware weighting
- CVAS / CVAT 對 Valence 有幫助，但 Arousal 可能需要降權

Reference:  
Huang et al. 2025. *NTULAW at ROCLING-2025 Shared Task: Domain Adaptation for Valence-Arousal Prediction in Chinese Medical Self-Reflection Texts*.  
https://aclanthology.org/2025.rocling-main.43/

#### Paper R3 — TCU at ROCLING-2025 Shared Task

方法重點：

- 使用 LLM / pretrained model embedding
- frozen embedding + SVR / Ridge / regression models
- ensemble / regression-based prediction

對本任務的啟發：

- Frozen embedding + SVR / Ridge 可作為非 neural-head 分支
- 不建議直接做 prediction averaging
- 更適合放進 stacking / meta-regression

Reference:  
Li et al. 2025. *TCU at ROCLING-2025 Shared Task: Leveraging LLM Embeddings for Chinese Dimensional Sentiment Analysis*.  
https://aclanthology.org/2025.rocling-main.44/

---

### 3.2 現有 VA / dimensional emotion SOTA

#### Paper A — Transformer Fusion for Chinese VA

方法重點：

- 針對中文 Valence-Arousal intensity prediction
- 使用 transformer fusion 整合不同語言表示
- 不只依賴單一 pooled embedding

對本任務的啟發：

- 將目前單一 mean pooling 升級為 multi-granularity fusion
- 融合 token-level、sentence-level、document-level、lexicon features

Reference:  
Deng et al. 2023. *Toward Transformer Fusions for Chinese Sentiment Intensity Prediction in Valence-Arousal Dimensions*. IEEE Access.  
https://doi.org/10.1109/ACCESS.2023.3322436

#### Paper B — Adversarial Attention Network for Multi-dimensional Emotion Regression

方法重點：

- 多維情緒回歸
- 對不同 emotion dimension 學不同 attention
- 讓 Valence / Arousal 看不同 token

對本任務的啟發：

- 新增 Valence-specific attention pooling
- 新增 Arousal-specific attention pooling
- 避免 mean pooling 稀釋少數高喚醒 cue

Reference:  
Zhu et al. 2019. *Adversarial Attention Modeling for Multi-dimensional Emotion Regression*. ACL 2019.  
https://aclanthology.org/P19-1045/

#### Paper C — Combined Valence-Arousal Ordinal Classification

方法重點：

- 不把所有錯誤視為同等嚴重
- 將 Valence / Arousal 空間視為 ordinal structure
- 透過 ordinal / ranking 思路降低錯誤嚴重程度

對本任務的啟發：

- 在 SmoothL1 外加入 ranking / ordinal objective
- 對 synthetic data 不使用絕對標籤，而是使用相對排序約束
- 直接針對 Arousal PCC 訓練

Reference:  
Mitsios et al. 2024. *Improved Text Emotion Prediction Using Combined Valence and Arousal Ordinal Classification*. NAACL 2024.  
https://aclanthology.org/2024.naacl-short.72/

#### Paper D — LLM Text Enrichment

方法重點：

- 使用 LLM 對文本產生額外心理 / 情緒指標
- 再用這些 enriched text / features fine-tune model
- 可提升 Pearson correlation

對本任務的啟發：

- 不一定要用 LLM 生成更多文本
- 可用 LLM 產生 arousal-related features
- 例如焦慮、生理激動、睡眠困擾、疲憊、社會壓力、平靜接受

Reference:  
Furniturewala et al. 2024. *Empaths at WASSA 2024 Empathy and Personality Shared Task: LLM Text Enrichment and DeBERTa Fine-tuning*. WASSA 2024.  
https://aclanthology.org/2024.wassa-1.35/

---

## 4. 下一階段實驗總表

| 優先 | 實驗 ID | 實驗名稱 | 主要目標 | 參考方法 |
|---|---|---|---|---|
| S0 | E4-check | 固定目前最佳 L1 baseline | 確保 train_v2 可復現實驗 4 | 現有最佳 |
| S1 | E18 | L1++ intensity features | 補 Arousal 強度特徵，不動資料分布 | Paper D + L1 |
| S2 | E19 | Source-aware Arousal loss / sampling | 解決不同來源資料的 Arousal domain shift | Paper R2 |
| S3 | E20 | Synthetic ranking-only augmentation | 保留 train_aug 的排序價值，避免 MAE 爆掉 | Paper R1 + Paper C |
| S4 | E21 | Arousal-specific attention + multi-pooling fusion | 避免 mean pooling 稀釋高喚醒 token | Paper A + Paper B |
| A1 | E22 | 2×2 controlled VA generation | 生成同主題四象限 pair，提升可比較性 | Paper R1 + Paper C |
| A2 | E23 | Target-style RAG generation | 用 official 200 unlabeled 當風格範例 | Paper R1 |
| A3 | E24 | LLM arousal enrichment features | 讓 LLM 產生結構化 arousal indicators | Paper D |
| A4 | E25 | Frozen embedding + SVR/Ridge stacking | 補一條非 fine-tune regression 分支 | Paper R3 |
| B1 | E26 | Arousal calibration | 只修 MAE，不改排序 | Post-processing |
| B2 | E27 | Ablation experiments | 支撐論文貢獻 | 所有方法 |

---

## 5. 實驗細節

---

## S0 — E4-check：固定目前最佳 L1 baseline

### 目的

在改任何架構前，先確保 `train_v2.py` 預設設定可復現目前最佳實驗 4。

### 原因

目前實驗 4 是唯一四指標全勝的提交基準。後續所有實驗都應該與它比較，而不是與較弱 baseline 比較。

### 實作

```bash
python train_v2.py \
  --model hfl/chinese-macbert-base \
  --epochs 4 \
  --batch_size 32 \
  --lr 2e-5 \
  --run_name e4_l1_reproduce
```

### 評估

比較：

```text
Valence MAE / PCC
Arousal MAE / PCC
official submission score
```

### 預期

應接近：

```text
V_MAE ≈ 0.600
V_PCC ≈ 0.880
A_MAE ≈ 0.882
A_PCC ≈ 0.426
```

---

## S1 — E18：L1++ intensity features

### 目的

在目前 L1 詞典特徵基礎上，加入更直接的 Arousal intensity features。

### 背後原因

L1 已證明有效，但目前只使用 CVAW / CVAP 的 VA 聚合訊號。Arousal 不只來自情緒詞，也來自：

- 程度副詞
- 生理反應
- 睡眠與疲憊
- 句子節奏
- 重複與標點
- 社會壓力 / 語言適應 / 工作壓力

這些是新住民反思文本中可能非常重要的喚醒度 cue。

### 新增 features

建議將 L1 由 10 維擴充為約 25–35 維。

#### 詞典 VA features

沿用原本：

```text
lex_v_mean
lex_a_mean
lex_v_max
lex_a_max
lex_v_min
lex_a_min
lex_v_std
lex_a_std
lex_count
lex_coverage
```

#### Arousal intensity features

```text
exclamation_density
question_density
ellipsis_density
degree_adverb_count
body_reaction_count
sleep_disturbance_count
anxiety_fear_count
pressure_event_count
low_arousal_word_count
repeated_char_ratio
sentence_length_mean
sentence_length_std
short_sentence_ratio
negation_count
contrast_marker_count
```

#### 新住民 domain-specific cues

```text
language_barrier_count
document_identity_count
work_pressure_count
family_separation_count
culture_adaptation_count
financial_pressure_count
```

### 實作

新增：

```text
lexicon_intensity.py
```

或在 `lexicon.py` 中新增 mode：

```bash
--lex_mode l1_intensity
```

訓練：

```bash
python train_v2.py \
  --lex_mode l1_intensity \
  --run_name e18_l1_intensity \
  --epochs 4 \
  --batch_size 32 \
  --lr 2e-5
```

### 預期效果

希望：

```text
Arousal PCC ↑
Arousal MAE 不明顯變差
Valence 不明顯下降
```

### 風險

- feature 太多可能 overfit DSA-MST dev
- 需要檢查 official 200 unlabeled 上的 feature 分布

### 參考論文

- Paper D：LLM / feature enrichment 思路
- Paper A：fusion of representation and affective features
- 目前 L1 baseline

---

## S2 — E19：Source-aware Arousal loss / sampling

### 目的

不要把 CVAS、CVAT、DSA-MST、ROCLING-2021 等來源資料等權對待，尤其是 Arousal。

### 背後原因

目前 `train.csv` 不是同分布資料：

```text
CVAS / CVAT: 通用情緒庫
DSA-MST: 醫療反思
ROCLING-2021: 教育反思
Official target: 新住民反思
```

Valence 相對穩定，Arousal 更受 domain shift 影響。因此 Arousal 的 loss 應根據來源加權。

### 初始 source weights

建議第一版：

| 來源 | Valence loss weight | Arousal loss weight | 原因 |
|---|---:|---:|---|
| CVAS | 1.0 | 0.25 | 句子級通用情緒，對 arousal domain mismatch 較大 |
| CVAT | 1.0 | 0.5 | 篇章級比 CVAS 接近，但仍非反思目標域 |
| DSA-MST | 1.0 | 1.0 | 醫療反思，形式最接近 |
| ROCLING-2021 | 1.0 | 0.75 | 教育反思，形式接近但主題不同 |
| train_aug raw | 0.0–0.25 | 0.0 for SmoothL1 | 絕對標籤不可靠 |
| train_aug pseudo | 0.25 | 0.25 | 校準較好，但排序被壓縮 |

### Loss 設計

```text
loss_real =
  w_v(source) * SmoothL1(V_pred, V_true)
  + w_a(source) * SmoothL1(A_pred, A_true)
```

### 實作

需要在 `data/train.csv` 加上 `source` 欄位，或在 `Dataset` 載入時根據 ID / granularity 判斷來源。

訓練：

```bash
python train_v2.py \
  --lex_mode l1_intensity \
  --source_aware \
  --run_name e19_l1_intensity_source_weight
```

### 預期效果

希望改善：

```text
Arousal PCC
Arousal MAE
official target generalization
```

### 風險

- 權重需要調整
- dev 是 DSA-MST，可能偏向 DSA-MST 權重，不一定代表 official

### 參考論文

- Paper R2：NTULAW domain adaptation / Arousal domain-specific annotation 觀察

---

## S3 — E20：Synthetic ranking-only augmentation

### 目的

保留 `train_aug.csv` 提升 Arousal PCC 的效果，但避免 raw bin labels 讓 Arousal MAE 爆掉。

### 背後原因

既有實驗顯示：

```text
raw synthetic labels:
  A_PCC ↑
  A_MAE 爆掉

teacher pseudo labels:
  A_MAE 修回
  A_PCC 掉回
```

這代表 synthetic texts 對排序有用，但絕對標籤不可靠。

因此應將 synthetic data 從 supervised regression data 改為 ranking constraint data。

### Ranking loss

對於同一批合成資料，根據原始 bin 建立 pair：

```text
high arousal text > low arousal text
high valence text > low valence text
```

Arousal ranking loss：

```text
loss_rank_A = max(0, margin - (A_pred_high - A_pred_low))
```

Valence ranking loss：

```text
loss_rank_V = max(0, margin - (V_pred_high - V_pred_low))
```

建議 margin：

```text
margin = 0.125 或 0.25
```

因為 label 已正規化到 `[0, 1]`。

### 總 loss

```text
loss =
  real_supervised_loss
  + λ_A * arousal_ranking_loss
  + λ_V * valence_ranking_loss
```

第一版建議：

```text
λ_A = 0.1
λ_V = 0.05
```

因為主要瓶頸是 Arousal。

### 實作

先不重新生成資料，直接使用現有 `data/train_aug.csv`。

```bash
python train_v2.py \
  --lex_mode l1_intensity \
  --source_aware \
  --rank_aug data/train_aug.csv \
  --rank_lambda_a 0.1 \
  --rank_lambda_v 0.05 \
  --run_name e20_rank_aug
```

### 預期效果

希望達到：

```text
Arousal PCC 接近 raw augmentation 的 0.46
Arousal MAE 接近 L1 baseline 的 0.88–0.90
```

### 風險

- Pair 組合方式會影響結果
- ranking loss 權重太大可能破壞 calibration
- 需要監控 official target 上 Arousal prediction std 是否過度放大

### 參考論文

- Paper R1：ROCLING 2025 augmentation + pseudo-label
- Paper C：VA ordinal / ranking idea

---

## S4 — E21：Arousal-specific attention + multi-pooling fusion

### 目的

讓模型不只使用 mean pooling，而是用不同 pooling 抽取不同層次的情緒訊號。

### 背後原因

目前模型使用 mask mean pooling：

```text
encoder hidden states → mean pooling
```

但 Arousal cue 常常只出現在少數詞，例如：

```text
焦慮、心跳、睡不著、害怕、崩潰、喘不過氣、很緊張
```

mean pooling 可能會將這些 cue 稀釋。

### 架構

```text
MacBERT hidden states
├── mean pooling
├── CLS pooling
├── Valence-specific attention pooling
├── Arousal-specific attention pooling
└── sentence-level pooling
      ↓
concat L1++ features
      ↓
V head / A head
```

### Dimension-specific attention

```text
shared_pool = mean_pool(hidden_states)
v_pool = attention_pool_v(hidden_states)
a_pool = attention_pool_a(hidden_states)

V_pred = V_head([shared_pool, v_pool, L1++])
A_pred = A_head([shared_pool, a_pool, L1++])
```

### 實作

```bash
python train_v2.py \
  --pooling mean_cls_dim_attention \
  --lex_mode l1_intensity \
  --source_aware \
  --run_name e21_dim_attention
```

若 E20 有效，再跑：

```bash
python train_v2.py \
  --pooling mean_cls_dim_attention \
  --lex_mode l1_intensity \
  --source_aware \
  --rank_aug data/train_aug.csv \
  --run_name e21_dim_attention_rank_aug
```

### 預期效果

希望：

```text
Arousal PCC ↑
Valence PCC 不下降
模型可解釋性提升
```

### 風險

- 參數增加，可能 overfit DSA-MST dev
- 需要 dropout / weight decay 控制

### 參考論文

- Paper A：Transformer fusion
- Paper B：dimension-specific attention for multi-dimensional emotion regression

---

## A1 — E22：2×2 controlled VA generation

### 目的

重新生成更適合 ranking learning 的資料，不再只生成單點 VA 樣本，而是同主題四象限對照文本。

### 背後原因

目前 `train_aug.csv` 雖然包含 5 個 VA 象限，但每篇文本彼此主題不一定相同。若要讓模型學會 VA 軸的區分，最好讓同一主題生成四個版本：

| 版本 | Valence | Arousal |
|---|---|---|
| A | low | high |
| B | low | low |
| C | high | high |
| D | high | low |

這樣可以建立更乾淨的 ranking constraints：

```text
Arousal(A) > Arousal(B)
Arousal(C) > Arousal(D)

Valence(C) > Valence(A)
Valence(D) > Valence(B)
```

### Prompt 設計

對每個新住民主題，例如：

```text
語言不通
找工作壓力
家庭分離
文化適應
身份文件
孩子教育
經濟壓力
```

要求 LLM 生成四個版本：

```text
同主題、同長度、同第一人稱反思風格
但分別控制為：
lowV_highA, lowV_lowA, highV_highA, highV_lowA
```

### 實作

新增：

```text
augment_generate_pairs.py
```

輸出格式：

```csv
group_id,variant,topic,text,valence_bin,arousal_bin,seed_words
001,lowV_highA,language_barrier,...
001,lowV_lowA,language_barrier,...
001,highV_highA,language_barrier,...
001,highV_lowA,language_barrier,...
```

### 預期效果

比原始 `train_aug.csv` 更適合 ranking loss，降低 synthetic label calibration 問題。

### 參考論文

- Paper R1：controlled augmentation
- Paper C：ordinal / ranking VA space

---

## A2 — E23：Target-style RAG generation

### 目的

讓 LLM 生成的文本更像官方 200 篇新住民文本。

### 背後原因

目前 LLM 生成主要根據：

```text
L3 graph seed words
+ 手寫新住民反思 prompt
```

但還沒有充分利用 official `val_unlabeled.csv` 的 200 篇目標域無標籤文本。

建議改成：

```text
official 200 unlabeled texts
→ retrieve style examples
→ L3 graph affective seed words
→ LLM 生成 target-style VA controlled texts
```

### 實作

對每個生成主題：

1. 從 `val_unlabeled.csv` 檢索 2–3 篇風格相近的文本
2. 從 L3 graph 選 VA seed words
3. Prompt 中同時給 style examples 與 affective seeds
4. 要求生成同主題四象限文本

### 預期效果

降低 synthetic texts 與 official target domain 的風格偏移。

### 參考論文

- Paper R1：RAG + augmentation
- DSA-NIFT 任務本身的 zero in-domain labeled 設定

---

## A3 — E24：LLM arousal enrichment features

### 目的

不讓 LLM 直接產生標籤，而是產生可解釋的 arousal-related structured features。

### 背後原因

LLM 直接產生 VA label 風險高，可能造成 calibration 問題。但讓 LLM 產生心理 / 情緒中介指標較穩。

### 建議 LLM 輸出

```json
{
  "anxiety_fear": 0,
  "body_activation": 0,
  "sleep_disturbance": 0,
  "fatigue_numbness": 0,
  "calm_acceptance": 0,
  "social_pressure": 0,
  "language_barrier_pressure": 0,
  "family_separation": 0,
  "work_financial_pressure": 0,
  "urgency": 0
}
```

每個欄位取值：

```text
0 = 無
1 = 輕微
2 = 明顯
```

### 使用方式

```text
MacBERT embedding
+ L1++ features
+ LLM enrichment features
→ V/A regression head
```

### 訓練

```bash
python train_v2.py \
  --lex_mode l1_intensity \
  --llm_features data/llm_enrichment_features.csv \
  --run_name e24_llm_enrichment
```

### 預期效果

- 主要改善 Arousal PCC
- 增加模型可解釋性
- 不直接污染 VA label

### 參考論文

- Paper D：LLM text enrichment + downstream fine-tuning

---

## A4 — E25：Frozen embedding + SVR/Ridge stacking

### 目的

建立一條與 fine-tuned MacBERT 不同的 regression 分支，用於 stacking，而不是普通平均 ensemble。

### 背後原因

你目前 E10 / E12 的 prediction averaging 已確認對 Arousal 沒有幫助，可能因為平均會壓縮 arousal 分布。

但 frozen embedding + SVR / Ridge 可以提供不同錯誤型態。若用 stacking，而不是直接平均，可能能補足 MacBERT L1 的偏差。

### Frozen embedding

```text
encoder 不更新參數
只抽文本 embedding
```

例如：

```text
text → MacBERT / bge-m3 / e5 embedding → vector
```

### Ridge

線性回歸 + L2 正則化，穩定、不易 overfit。

### SVR

Support Vector Regression，適合小資料與高維 embedding。

### Stacking

不要直接平均：

```text
final = mean(model outputs)
```

而是：

```text
base model predictions + L1++ features
→ Ridge meta-regressor
→ final V/A
```

### 重要：必須用 OOF predictions

避免 leakage，需使用 K-fold out-of-fold predictions 訓練 meta-regressor。

### 實作

```bash
python embed_regressor.py \
  --model hfl/chinese-macbert-base \
  --mode ridge_svr \
  --oof 5 \
  --run_name e25_embed_reg

python stacking.py \
  --runs e4_l1_reproduce e18_l1_intensity e25_embed_reg \
  --features l1_intensity \
  --meta ridge \
  --run_name e25_stacking
```

### 參考論文

- Paper R3：LLM embeddings + regression methods

---

## B1 — E26：Arousal calibration

### 目的

在不重新訓練模型的情況下，對 Arousal 預測做線性後處理，主要修 MAE。

### 背後原因

若模型排序大致正確，但整體分數偏高 / 偏低 / 太分散 / 太集中，可用 calibration 改善 MAE。

### 公式

```text
A_calibrated = alpha * A_pred + beta
```

其中：

```text
alpha: 控制分布縮放
beta: 控制整體平移
```

若 `alpha > 0`，通常 PCC 幾乎不變，因為排序不改。

### 實作

```bash
python calibrate.py \
  --run e20_rank_aug \
  --dim arousal \
  --name e26_e20_calibrated
```

### 使用時機

只在以下情況使用：

```text
A_PCC 已經提高
但 A_MAE 偏差仍大
```

### 不適合情況

若模型排序本身錯，calibration 無法修 PCC。

---

## B2 — E27：Ablation experiments

### 目的

支撐論文方法章與分析章，證明每個模組有實際貢獻。

### 必做消融

| 消融 | 目的 |
|---|---|
| no lexicon vs L1 vs L1++ | 證明詞典與強度特徵有效 |
| source-aware off/on | 證明 source shift 處理有效 |
| raw augmentation vs pseudo-label vs ranking-only | 證明校準 / 排序 trade-off |
| L3 guided seeds vs random seeds | 證明情緒圖譜不是裝飾 |
| mean pooling vs dimension-specific attention | 證明 Arousal token attention 有效 |
| no target-style RAG vs target-style RAG | 證明 official unlabeled style examples 有用 |
| calibration before/after | 分析 MAE/PCC 解耦 |

---

## 6. 建議執行順序

### 第一波：最穩、最可能提升分數

```text
S0 E4-check
S1 E18 L1++ intensity
S2 E19 source-aware loss
S3 E20 ranking-only augmentation
```

建議順序：

```bash
# 1. 復現目前最佳
python train_v2.py --run_name e4_l1_reproduce

# 2. 加 L1++ intensity
python train_v2.py --lex_mode l1_intensity --run_name e18_l1_intensity

# 3. 加 source-aware arousal weighting
python train_v2.py --lex_mode l1_intensity --source_aware --run_name e19_source_aware

# 4. 加 train_aug ranking-only
python train_v2.py --lex_mode l1_intensity --source_aware \
  --rank_aug data/train_aug.csv \
  --rank_lambda_a 0.1 \
  --rank_lambda_v 0.05 \
  --run_name e20_rank_aug
```

### 第二波：架構升級

```text
S4 E21 dimension-specific attention
```

若 E18 / E19 / E20 有效，再加 attention。不要一開始就改太多，避免無法知道是哪個模組有效。

### 第三波：生成資料升級

```text
A1 E22 2×2 controlled generation
A2 E23 target-style RAG generation
```

先用現有 `train_aug.csv` 做 E20，若 ranking-only 有效，再花 API 成本重新生成四象限 pair。

### 第四波：輔助特徵與 stacking

```text
A3 E24 LLM enrichment features
A4 E25 frozen embedding stacking
```

這些作為補強，不應優先於 E18–E20。

### 第五波：後處理與論文消融

```text
B1 E26 calibration
B2 E27 ablations
```

---

## 7. 提交策略

官方提交額度有限時，建議：

| 提交順序 | 版本 |
|---|---|
| 1 | 實驗 4 / E4-check baseline |
| 2 | E18 L1++ intensity |
| 3 | E19 source-aware |
| 4 | E20 ranking-only augmentation |
| 5 | E21 attention 或 E20 + calibration |

不要優先提交：

```text
普通 mean ensemble
raw train_aug supervised append
teacher pseudo-label only augmentation
```

原因是這三者已有實驗顯示無法穩定超越實驗 4。

---

## 8. 論文故事線建議

可以將方法章寫成：

```text
1. Zero in-domain labeled DSA-NIFT setting
2. L1 lexicon fusion as affective prior
3. L1++ intensity features for immigrant reflection arousal cues
4. Source-aware Arousal training for cross-domain VA regression
5. Graph-guided synthetic texts as ranking constraints
6. Optional dimension-specific attention for V/A-specific evidence extraction
```

核心貢獻可以描述為：

### Contribution 1 — Source-aware affective regression

針對 DSA-NIFT 沒有新住民標註資料的設定，將不同來源資料以不同權重使用，特別降低通用情緒庫對 Arousal 的干擾。

### Contribution 2 — Enhanced lexicon-intensity fusion

在 CVAW / CVAP 詞典 VA features 外，加入新住民反思文本常見的 arousal intensity cues，例如焦慮、生理反應、睡眠困擾、語言壓力與家庭分離。

### Contribution 3 — Graph-guided ranking augmentation

使用 L3 affective knowledge graph 引導 LLM 生成 target-style synthetic texts，但不把 synthetic labels 視為 gold labels，而是將其轉為 VA ranking constraints，以緩解 calibration / ranking trade-off。

### Contribution 4 — Calibration vs ranking analysis

系統性分析 raw augmentation、label shrink、teacher pseudo-label 與 ranking-only 的差異，指出 Arousal regression 中 MAE 與 PCC 的張力。

---

## 9. 最終建議

下一步不要先做大型架構或 ensemble。最穩的實驗路線是：

```text
E4 baseline
→ E18 L1++ intensity
→ E19 source-aware Arousal loss
→ E20 synthetic ranking-only augmentation
→ E21 dimension-specific attention
```

其中最有機會超越目前實驗 4 的是：

```text
E20 = L1++ + source-aware + train_aug ranking-only
```

原因是它直接利用目前已觀察到的現象：

```text
synthetic texts 能提升 Arousal PCC
但 synthetic absolute labels 會傷 Arousal MAE
```

因此應保留 synthetic 的排序訊號，而不是相信其絕對分數。

---

## 10. 實作注意事項

- 不要覆蓋既有 `train.py`，新實驗放在 `train_v2.py` 或新 branch。
- `train_aug.csv` 是未追蹤檔，Colab 需要額外帶上去或從本機打包。
- dev set 是 DSA-MST，不等於 official target，增強實驗不能只看 dev。
- 監控 official 200 unlabeled 上的 Arousal prediction std、range，以及與 L1++ intensity features 的相關性。
- commit message 不要提及外部共同作者或不適當 attribution。
