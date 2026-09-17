"""
从 weibo/*.csv 的 image_id 列批量下载新浪图床图片。

保存目录（与 clip_data_pre / clip_dataloader 一致）：
  weibo/nonrumor_images/  label=0
  weibo/rumor_images/      label=1

文件名 = URL 最后一段，例如 62b31d36gw1expshp70k2j20hm0btmyk.jpg
预处理脚本用「去掉扩展名、转小写」作为字典键匹配。

用法（在项目根目录或 src 目录均可）：
  python download_weibo_images.py
  python download_weibo_images.py --workers 8 --delay 0.2
  python download_weibo_images.py --csv ../weibo/train_2_domain.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

# 默认路径：脚本在 src/ 下，weibo 在项目根目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_WEIBO_DIR = os.path.join(os.environ.get("DAMMFND_DATA", _PROJECT_ROOT), "weibo")
_DEFAULT_CSVS = [
    os.path.join(_DEFAULT_WEIBO_DIR, "train_2_domain.csv"),
    os.path.join(_DEFAULT_WEIBO_DIR, "val_2_domain.csv"),
    os.path.join(_DEFAULT_WEIBO_DIR, "test_2_domain.csv"),
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://weibo.com/",
}

_OUT_DIRS = {
    0: "nonrumor_images",
    1: "rumor_images",
}


def url_to_filename(url: str) -> str | None:
    url = url.strip()
    if not url or url.lower() == "null":
        return None
    name = url.split("/")[-1].split("?")[0]
    if not name or "." not in name:
        return None
    return name


def iter_image_tasks(csv_paths: Iterable[str]) -> list[tuple[str, int, str]]:
    """返回 (url, label, filename) 列表，按文件名去重。"""
    seen_url: set[str] = set()
    stem_to_label: dict[str, int] = {}
    tasks: list[tuple[str, int, str]] = []
    conflicts: list[tuple[str, int, int]] = []

    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"[warn] CSV 不存在，跳过: {csv_path}")
            continue
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "image_id" not in (reader.fieldnames or []):
                print(f"[warn] 缺少 image_id 列: {csv_path}")
                continue
            if "label" not in (reader.fieldnames or []):
                print(f"[warn] 缺少 label 列: {csv_path}")
                continue
            for row in reader:
                try:
                    label = int(row["label"])
                except (TypeError, ValueError):
                    continue
                if label not in _OUT_DIRS:
                    continue
                for part in str(row["image_id"]).split("|"):
                    fn = url_to_filename(part)
                    if not fn:
                        continue
                    stem = fn.rsplit(".", 1)[0].lower()
                    if stem in stem_to_label and stem_to_label[stem] != label:
                        conflicts.append((fn, stem_to_label[stem], label))
                        continue
                    url = part.strip()
                    if url.startswith("http://"):
                        url = "https://" + url[7:]
                    elif not url.startswith("https://"):
                        url = "https://" + url
                    if url in seen_url:
                        continue
                    seen_url.add(url)
                    stem_to_label[stem] = label
                    tasks.append((url, label, fn))

    if conflicts:
        print(f"[warn] {len(conflicts)} 个文件名同时出现在 0/1 两类，已跳过重复（保留先出现的）")
        for fn, a, b in conflicts[:5]:
            print(f"       {fn}: label {a} vs {b}")
        if len(conflicts) > 5:
            print("       ...")

    return tasks


def download_one(
    url: str,
    label: int,
    filename: str,
    out_root: str,
    timeout: float,
    retries: int,
) -> tuple[str, bool, str]:
    """下载单张图。返回 (filename, ok, message)。"""
    sub = _OUT_DIRS[label]
    out_dir = os.path.join(out_root, sub)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return filename, True, "skip_exists"

    last_err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < 100:
                last_err = f"too_small({len(data)}B)"
                continue
            with open(out_path, "wb") as f:
                f.write(data)
            return filename, True, "ok"
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = f"URL {e.reason}"
        except TimeoutError:
            last_err = "timeout"
        except OSError as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))

    return filename, False, last_err


def _download_worker(args: tuple) -> tuple[str, bool, str]:
    return download_one(*args)


def main() -> int:
    parser = argparse.ArgumentParser(description="从 weibo CSV 批量下载图片到 nonrumor_images / rumor_images")
    parser.add_argument(
        "--weibo-dir",
        default=_DEFAULT_WEIBO_DIR,
        help=f"weibo 数据目录（默认 {_DEFAULT_WEIBO_DIR}）",
    )
    parser.add_argument(
        "--csv",
        action="append",
        dest="csvs",
        help="CSV 路径，可多次指定；默认 train/val/test 三个 _2_domain.csv",
    )
    parser.add_argument("--workers", type=int, default=4, help="并发下载线程数")
    parser.add_argument("--timeout", type=float, default=30.0, help="单次请求超时（秒）")
    parser.add_argument("--retries", type=int, default=2, help="失败重试次数")
    parser.add_argument("--delay", type=float, default=0.0, help="每张图下载完成后的间隔（秒）")
    parser.add_argument("--limit", type=int, default=0, help="仅下载前 N 张（0=全部，便于试跑）")
    args = parser.parse_args()

    weibo_dir = os.path.normpath(args.weibo_dir)
    csv_paths = args.csvs or _DEFAULT_CSVS
    if not args.csvs:
        csv_paths = [
            os.path.join(weibo_dir, "train_2_domain.csv"),
            os.path.join(weibo_dir, "val_2_domain.csv"),
            os.path.join(weibo_dir, "test_2_domain.csv"),
        ]

    print("weibo 目录:", weibo_dir)
    print("CSV:", *csv_paths, sep="\n  ")
    tasks = iter_image_tasks(csv_paths)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"待处理唯一图片: {len(tasks)}（已跳过 CSV 内重复 URL）")

    if not tasks:
        print("没有可下载任务，退出。")
        return 1

    ok = skip = fail = 0
    fail_log_path = os.path.join(weibo_dir, "download_failed.txt")
    fail_lines: list[str] = []

    work_args = [
        (url, label, fn, weibo_dir, args.timeout, args.retries)
        for url, label, fn in tasks
    ]

    if args.workers <= 1:
        for i, wa in enumerate(work_args, 1):
            fn, success, msg = download_one(*wa)
            if msg == "skip_exists":
                skip += 1
            elif success:
                ok += 1
            else:
                fail += 1
                fail_lines.append(f"{fn}\t{msg}\t{wa[0]}")
            if i % 50 == 0 or i == len(work_args):
                print(f"进度 {i}/{len(work_args)}  ok={ok} skip={skip} fail={fail}")
            if args.delay > 0:
                time.sleep(args.delay)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_download_worker, wa): wa for wa in work_args}
            done = 0
            for fut in as_completed(futures):
                done += 1
                fn, success, msg = fut.result()
                if msg == "skip_exists":
                    skip += 1
                elif success:
                    ok += 1
                else:
                    fail += 1
                    wa = futures[fut]
                    fail_lines.append(f"{fn}\t{msg}\t{wa[0]}")
                if done % 100 == 0 or done == len(work_args):
                    print(f"进度 {done}/{len(work_args)}  ok={ok} skip={skip} fail={fail}")
                if args.delay > 0:
                    time.sleep(args.delay)

    if fail_lines:
        with open(fail_log_path, "w", encoding="utf-8") as f:
            f.write("filename\treason\turl\n")
            f.write("\n".join(fail_lines))
        print(f"失败列表已写入: {fail_log_path}")

    print("—— 完成 ——")
    print(f"  新下载成功: {ok}")
    print(f"  已存在跳过: {skip}")
    print(f"  失败:       {fail}")
    print(f"  非谣言目录: {os.path.join(weibo_dir, 'nonrumor_images')}")
    print(f"  谣言目录:   {os.path.join(weibo_dir, 'rumor_images')}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
