"""貼進 Colab / Jupyter cell 就能跑的 test set 推論片段（不需要整本 notebook）。

用途：手上已有 checkpoint，要對官方 1,100 篇 test set 重跑推論並產生
`submission.csv.zip`（壓縮檔內部檔名固定 `submission.csv`，評分網站要的格式）。

- `lex_mode` 與 `pooling` 會從 checkpoint 的 state_dict 形狀自動推斷
  （head 輸入維度 - hidden = lex_dim；有 `pool_v.*` 就是 E21 的 dim-attention），
  所以不會發生「lex_mode 填錯 → 載入報錯或靜默錯誤」。
- 與 `predict.py` 走同一條管線（同樣的 VADataset / predict / clip），
  重跑結果可與論文既有的 test 提交直接比較。

也可以直接 `python notebooks/snippet_infer_test.py`（先改好下面的常數）。
"""
import csv, os, shutil, sys, zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO      = "/content/DSA-NIFT"                       # repo 根目錄
TEST_CSV  = "data/DSANIDF_TestSet.csv"                # 官方 test（ID,Text），相對 REPO
MODEL     = "hfl/chinese-macbert-base"                # 或本機 snapshot 資料夾
OUT_DIR   = "outputs"                                 # 相對 REPO
BATCH_SIZE, MAX_LEN, EXPECTED_ROWS = 32, 256, 1100
PRIMARY   = None          # 要另存成 submission.csv.zip 的 run 名；None = 用第一個
CKPTS = {                 # run_name -> checkpoint 路徑（相對 REPO 或絕對路徑皆可）
    "cell2x2_e22_l1_sa_s42": "outputs/cell2x2_e22_l1_sa_s42_best.pt",
}
# lex_mode / pooling 預設從 checkpoint 自動推斷；只有 l1l2 模型要手動指定：
OVERRIDE = {}             # 例：{"some_run": {"lex_mode": "l1l2", "pooling": "mean"}}

sys.path.insert(0, REPO)
os.chdir(REPO)
from transformers import AutoTokenizer
from train import VADataset, VARegressor, pick_device, LABEL_MIN, LABEL_MAX
from train_v2 import VARegressorDimAttn, predict


def read_rows(path):
    """官方檔是 ID,Text；內部檔是 id,text —— 一律正規化成 id/text。"""
    with open(path, encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        low = {(k or "").strip().lower(): v for k, v in r.items()}
        assert "text" in low, f"{path} 缺 text/Text 欄位：{list(r.keys())}"
        rows.append({"id": low.get("id", ""), "text": low["text"] or ""})
    return rows


def inspect_ckpt(state):
    """從 state_dict 形狀反推 pooling 與 lex_mode，避免填錯而載入失敗或靜默錯誤。"""
    hidden = state["encoder.embeddings.word_embeddings.weight"].shape[1]
    if any(k.startswith("pool_v.") for k in state):          # E21 dim-attention
        pooling, in_dim = "mean_cls_dim_attention", state["head_v.1.weight"].shape[1]
        lex_dim = in_dim - 3 * hidden
    else:                                                    # 一般 mean pooling
        pooling, in_dim = "mean", state["head.1.weight"].shape[1]
        lex_dim = in_dim - hidden
    lex_mode = {0: "none", 10: "l1", 31: "l1_intensity"}.get(lex_dim)
    assert lex_mode, f"未知 lex_dim={lex_dim}（l1l2 之類請用 OVERRIDE 指定）"
    return pooling, lex_mode, lex_dim


def make_featurizer(lex_mode):
    if lex_mode == "none":
        return None, 0
    if lex_mode == "l1":
        from lexicon import LexiconFeaturizer, FEATURE_DIM
        return LexiconFeaturizer(), FEATURE_DIM
    if lex_mode == "l1_intensity":
        from lexicon_intensity import L1IntensityFeaturizer, FEATURE_DIM_INTENSITY
        return L1IntensityFeaturizer(), FEATURE_DIM_INTENSITY
    from lexicon_l2 import L1L2Featurizer, FEATURE_DIM_L2
    return L1L2Featurizer(os.path.join(OUT_DIR, "l2_word_va.pkl")), FEATURE_DIM_L2


def check_submission(path, expected_rows):
    """上傳前把關：欄位、列數、ID 唯一、無缺值、預測落在 [1,9]。"""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == ["ID", "Valence", "Arousal"], list(rows[0].keys())
    assert len(rows) == expected_rows, f"{path}: {len(rows)} 列，應為 {expected_rows}"
    assert len({r["ID"] for r in rows}) == len(rows), f"{path}: ID 重複"
    for r in rows:
        for col in ("Valence", "Arousal"):
            assert r[col] not in (None, ""), f"{path}: {r['ID']} 缺 {col}"
            assert 1.0 <= float(r[col]) <= 9.0, f"{path}: {r['ID']} 的 {col} 超出 [1,9]"
    return rows


def zip_submission(csv_path, zip_path):
    """壓成評分網站要的 zip：內部檔名固定 submission.csv。"""
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["submission.csv"], zf.namelist()
    return zip_path


device = pick_device()
rows = read_rows(TEST_CSV)
print(f"device={device} | test rows={len(rows)}")
assert len(rows) == EXPECTED_ROWS, f"{TEST_CSV} 有 {len(rows)} 列，預期 {EXPECTED_ROWS}"
os.makedirs(os.path.join(OUT_DIR, "preds"), exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)

made = []
for run, ckpt in CKPTS.items():
    state = torch.load(ckpt, map_location="cpu")
    pooling, lex_mode, lex_dim = inspect_ckpt(state)
    cfg = OVERRIDE.get(run, {})
    pooling, lex_mode = cfg.get("pooling", pooling), cfg.get("lex_mode", lex_mode)
    featurizer, feat_dim = make_featurizer(lex_mode)
    assert feat_dim == lex_dim or cfg, f"{run}: featurizer {feat_dim} 維 != checkpoint {lex_dim} 維"
    print(f"\n[{run}] pooling={pooling} lex_mode={lex_mode} lex_dim={feat_dim}")

    model_cls = VARegressorDimAttn if pooling == "mean_cls_dim_attention" else VARegressor
    model = model_cls(MODEL, lex_dim=feat_dim).to(device)
    model.load_state_dict(state)      # 形狀不合會直接報錯，不會靜默載錯
    loader = DataLoader(VADataset(rows, tok, MAX_LEN, has_label=False, featurizer=featurizer),
                        batch_size=BATCH_SIZE)
    pred = np.clip(predict(model, loader, device), LABEL_MIN, LABEL_MAX)

    pred_path = os.path.join(OUT_DIR, "preds", f"{run}_test.csv")
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "valence_pred", "arousal_pred", "model_name", "split"])
        for r, p in zip(rows, pred):
            w.writerow([r["id"], f"{p[0]:.4f}", f"{p[1]:.4f}", run, "test"])

    sub_path = os.path.join(OUT_DIR, f"{run}_test_submission.csv")
    with open(sub_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Valence", "Arousal"])
        for r, (v, a) in zip(rows, pred):
            w.writerow([r["id"], f"{v:.4f}", f"{a:.4f}"])

    check_submission(sub_path, EXPECTED_ROWS)
    zip_path = zip_submission(sub_path, os.path.join(OUT_DIR, f"{run}_test_submission.csv.zip"))
    made.append((run, zip_path))
    print(f"  preds      -> {pred_path}")
    print(f"  submission -> {sub_path}")
    print(f"  zip        -> {zip_path}")
    print(f"  [dist] V mean={pred[:, 0].mean():.3f} std={pred[:, 0].std():.3f} | "
          f"A mean={pred[:, 1].mean():.3f} std={pred[:, 1].std():.3f} "
          f"range [{pred[:, 1].min():.2f}, {pred[:, 1].max():.2f}]")
    del model, state
    if device == "cuda":
        torch.cuda.empty_cache()

primary = PRIMARY or made[0][0]
assert primary in dict(made), f"PRIMARY={primary} 不在 CKPTS 裡"
shutil.copy(dict(made)[primary], os.path.join(OUT_DIR, "submission.csv.zip"))
print(f"\n主提交檔 {OUT_DIR}/submission.csv.zip <- {primary}")
