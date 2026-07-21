"""
fetch_results.py — 把 Colab 下載回來的結果 zip 解進 outputs/，並可打包 submission 準備上傳。

每輪 Colab 實驗跑完會下載一包 `*_results.zip` 到 ~/Downloads。這支腳本把它解到
`outputs/`（保留既有 l2/l3 pkl、不覆蓋），列出新增的 submission / 權重，
並可一步把指定 run 的 submission 打包成評分網站要的 `submission.csv.zip`
（內部檔名正名為 submission.csv、去除 macOS 雜項）。

用法：
    # 自動抓 ~/Downloads 最新的 *_results.zip，解到 outputs/
    python fetch_results.py
    # 指定 zip
    python fetch_results.py --zip ~/Downloads/e13_results.zip
    # 解完順便把某個 run 打包成 submission.csv.zip 準備上傳
    python fetch_results.py --pack e13_pseudo_ens
    # 只打包、不解壓（outputs 已有該檔時）
    python fetch_results.py --pack e13_pseudo_ens --no_extract
"""
import os
import glob
import zipfile
import argparse
import tempfile
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
DOWNLOADS = os.path.expanduser("~/Downloads")


def newest_zip():
    zips = glob.glob(os.path.join(DOWNLOADS, "*_results.zip"))
    if not zips:
        raise FileNotFoundError(f"{DOWNLOADS} 下找不到 *_results.zip")
    return max(zips, key=os.path.getmtime)


def extract(zip_path):
    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
    before = set(glob.glob(os.path.join(OUT, "*")))
    with zipfile.ZipFile(zip_path) as z:
        # 跳過 macOS 雜項；不動 outputs 既有的 *.pkl（zip 裡本來就沒有）
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        z.extractall(OUT, members=members)
    added = sorted(os.path.basename(p) for p in set(glob.glob(os.path.join(OUT, "*"))) - before)
    subs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "*_submission.csv")))
    pts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "*_best.pt")))
    print(f"✅ 解壓 {os.path.basename(zip_path)} → {OUT}")
    print(f"   本次新增 top-level：{added or '（皆為覆蓋更新）'}")
    print(f"   submission 共 {len(subs)} 個、權重 {len(pts)} 顆、pkl 保留：",
          [os.path.basename(p) for p in glob.glob(os.path.join(OUT, '*.pkl'))])


def pack(run):
    """outputs/{run}_submission.csv → submission.csv.zip（內部檔名 submission.csv，無 macOS 雜項）。"""
    src = os.path.join(OUT, f"{run}_submission.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"找不到 {src}；先確認 run 名稱（省略 _submission.csv）")
    dst_zip = os.path.join(os.path.dirname(HERE), "submission.csv.zip")  # 放在專案上層（提交暫存區）
    stage = tempfile.mkdtemp()
    try:
        staged = os.path.join(stage, "submission.csv")
        shutil.copy(src, staged)
        subprocess.run(["xattr", "-c", staged], check=False)
        if os.path.exists(dst_zip):
            os.remove(dst_zip)
        # -X 不塞額外屬性；從 stage 目錄打包，內部就是純 submission.csv
        subprocess.run(["zip", "-q", "-X", "-j", dst_zip, staged], check=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    with zipfile.ZipFile(dst_zip) as z:
        names = z.namelist()
    print(f"📦 已打包 {run} → {dst_zip}")
    print(f"   內部檔案：{names}（應只有 ['submission.csv']）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="結果 zip 路徑；預設抓 ~/Downloads 最新的 *_results.zip")
    ap.add_argument("--pack", default=None, help="解完把此 run 打包成 submission.csv.zip（run 名省略 _submission.csv）")
    ap.add_argument("--no_extract", action="store_true", help="只打包、不解壓")
    args = ap.parse_args()

    if not args.no_extract:
        extract(args.zip or newest_zip())
    if args.pack:
        pack(args.pack)


if __name__ == "__main__":
    main()
