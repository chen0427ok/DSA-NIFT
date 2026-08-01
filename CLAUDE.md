# DSA-NIFT

ROCLING 2026 中文情感維度（valence / arousal）迴歸任務。

## Agent skills

### Issue tracker

Issue 追蹤在 GitHub（`chen0427ok/DSA-NIFT`），透過 `gh` CLI 操作。詳見 `docs/agents/issue-tracker.md`。

### Triage labels

使用五個標準角色標籤：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。詳見 `docs/agents/triage-labels.md`。

### Domain docs

單一 context 佈局：根目錄 `CONTEXT.md` + `docs/adr/`。詳見 `docs/agents/domain.md`。

### Git commit 頻率（2026-07-21 確認）

不用等使用者明確要求，**每個有意義的變更完成後就自動 commit**（例如：新增/修改一個實驗腳本、
更新一份實驗記錄文件、跑完一輪結果並記錄下來）。commit message 照舊不得提及 Claude 共同作者。
**push 前仍要跟使用者確認一次**，不要自動 push。
