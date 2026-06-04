import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_REPO = "AND-Q/Facial-Expression-Recognition"
DEFAULT_TAG = "v1.0.0"

MODEL_ASSETS = {
    "yolov11n-face.pt": "yolov11n-face.pt",
    "runs/classify/datasets_plus_optimized/weights/best.pt": "datasets_plus_best.pt",
    "runs/classify/fer2013_plus_optimized/weights/best.pt": "fer2013_plus_best.pt",
    "runs/classify/affectnet_optimized/weights/best.pt": "affectnet_best.pt",
    "runs/classify/my_datasets_optimized/weights/best.pt": "my_datasets_best.pt",
}

DATASET_ASSETS = {
    "datasets/affectnet.zip": "affectnet.zip",
    "datasets/fer2013plus.zip": "fer2013plus.zip",
}

CHUNK_SIZE = 1024 * 1024
URLLIB_RETRIES = 3
CURL_MAX_TIME_SECONDS = 900
CURL_MIN_SPEED_BYTES = 1024
CURL_MIN_SPEED_SECONDS = 60


def release_url(repo, tag, asset_name):
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def format_not_found_message(url):
    return (
        f"Release 资产不存在: {url}\n"
        f"请先在 GitHub Releases 的 {DEFAULT_TAG} 中上传对应文件，或用 --tag 指定实际版本。"
    )


def download_with_urllib(url, tmp_path):
    with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as file:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                print(f"\r进度: {percent:5.1f}%", end="", flush=True)
    if total:
        print()


def download_with_curl(url, tmp_path):
    curl = shutil.which("curl")
    if not curl:
        return False

    print("Python 下载失败，改用 curl 重试。")
    result = subprocess.run(
        [
            curl,
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "30",
            "--max-time",
            str(CURL_MAX_TIME_SECONDS),
            "--speed-limit",
            str(CURL_MIN_SPEED_BYTES),
            "--speed-time",
            str(CURL_MIN_SPEED_SECONDS),
            "--output",
            str(tmp_path),
            url,
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl 下载失败，退出码: {result.returncode}")
    return True


def download_file(url, destination, overwrite=False):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        print(f"已存在，跳过: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    print(f"下载: {url}")
    print(f"保存: {destination}")

    last_error = None
    for attempt in range(1, URLLIB_RETRIES + 1):
        try:
            download_with_urllib(url, tmp_path)
            tmp_path.replace(destination)
            return
        except urllib.error.HTTPError as exc:
            tmp_path.unlink(missing_ok=True)
            if exc.code == 404:
                raise RuntimeError(format_not_found_message(url)) from exc
            last_error = exc
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            last_error = exc

        if attempt < URLLIB_RETRIES:
            print(f"下载中断，准备重试 ({attempt}/{URLLIB_RETRIES})...")
            time.sleep(2 * attempt)

    try:
        if download_with_curl(url, tmp_path):
            tmp_path.replace(destination)
            return
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"下载失败: {last_error}; curl 备用下载也失败: {exc}") from exc

    raise RuntimeError(f"下载失败: {last_error}")


def selected_assets(args):
    assets = {}
    if args.models or args.all:
        assets.update(MODEL_ASSETS)
    if args.datasets or args.all:
        assets.update(DATASET_ASSETS)
    if not assets:
        assets.update(MODEL_ASSETS)
    return assets


def main():
    parser = argparse.ArgumentParser(description="下载项目运行所需模型和可选数据集")
    parser.add_argument("--repo", default=os.getenv("FER_RELEASE_REPO", DEFAULT_REPO), help="GitHub 仓库，如 owner/repo")
    parser.add_argument("--tag", default=os.getenv("FER_RELEASE_TAG", DEFAULT_TAG), help="Release 标签")
    parser.add_argument("--models", action="store_true", help="下载运行所需模型权重")
    parser.add_argument("--datasets", action="store_true", help="下载可选数据集压缩包")
    parser.add_argument("--all", action="store_true", help="下载模型和数据集")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在文件")
    args = parser.parse_args()

    for destination, asset_name in selected_assets(args).items():
        url = release_url(args.repo, args.tag, asset_name)
        download_file(url, destination, overwrite=args.overwrite)

    print("资源准备完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"下载失败: {exc}", file=sys.stderr)
        sys.exit(1)
