import argparse
import os
import sys
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


def release_url(repo, tag, asset_name):
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def download_file(url, destination, overwrite=False):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        print(f"已存在，跳过: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    print(f"下载: {url}")
    print(f"保存: {destination}")

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as file:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(f"\r进度: {percent:5.1f}%", end="", flush=True)
        if total:
            print()
        tmp_path.replace(destination)
    except urllib.error.HTTPError as exc:
        tmp_path.unlink(missing_ok=True)
        if exc.code == 404:
            raise RuntimeError(
                f"Release 资产不存在: {url}\n"
                f"请先在 GitHub Releases 的 {DEFAULT_TAG} 中上传对应文件，或用 --tag 指定实际版本。"
            ) from exc
        raise
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


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
