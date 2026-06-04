import argparse
import os

from ultralytics import YOLO


DEFAULT_MODEL_PATH = "runs/classify/fer2013_plus_optimized/weights/best.pt"
DOWNLOAD_HINT = "请先运行: python scripts/download_assets.py --models"


def main():
    parser = argparse.ArgumentParser(description="验证表情识别模型")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="模型权重路径")
    parser.add_argument("--data", default=None, help="数据集路径或名称；不填则使用模型保存的训练配置")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}\n{DOWNLOAD_HINT}")

    model = YOLO(args.model)
    metrics = model.val(data=args.data) if args.data else model.val()
    print(f"top1: {metrics.top1:.4f}")
    print(f"top5: {metrics.top5:.4f}")


if __name__ == "__main__":
    main()
