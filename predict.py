import argparse
import os

from ultralytics import YOLO


DEFAULT_MODEL_PATH = "runs/classify/fer2013_plus_optimized/weights/best.pt"
DEFAULT_SOURCE = "datasets/封面.png"
DOWNLOAD_HINT = "请先运行: python scripts/download_assets.py --models"


def main():
    parser = argparse.ArgumentParser(description="使用表情识别模型进行预测")
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE, help="图片、视频或目录路径")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="模型权重路径")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}\n{DOWNLOAD_HINT}")
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"输入源不存在: {args.source}")

    model = YOLO(args.model)
    results = model(args.source)
    print(f"预测完成，共返回 {len(results)} 个结果")


if __name__ == "__main__":
    main()
