#!/usr/bin/env bash
# 下載官方推薦的外部 VA 情感資源到 external/。在本機或 Colab 都可跑。
set -e
cd "$(dirname "$0")"

echo "== DSA-MST (ROCLING-2025 醫療反思, 含 VA 標籤, 最貼近目標域) =="
mkdir -p external/DSA-MST
BASE="https://raw.githubusercontent.com/NYCU-NLP/ROCLING-2025-ST-DSA-MST/main/Dataset"
for f in DSAMST-ValidationSet_ans.csv DSAMST-TestSet_ans.csv; do
  curl -fsSL "$BASE/$f" -o "external/DSA-MST/$f" && echo "  ✓ $f"
done

echo "完成。接著跑: python prepare_data.py"
