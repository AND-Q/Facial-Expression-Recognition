import argparse
import os

from ultralytics import YOLO


DEFAULT_MODEL_PATH = "runs/classify/fer2013_plus_optimized/weights/best.pt"
DOWNLOAD_HINT = "请先运行: python scripts/download_assets.py --models"


def main():
    parser = argparse.ArgumentParser(description="导出表情识别模型")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="模型权重路径")
    parser.add_argument("--format", default="onnx", help="导出格式，例如 onnx、torchscript、openvino")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}\n{DOWNLOAD_HINT}")

    model = YOLO(args.model)
    output_path = model.export(format=args.format)
    print(f"模型已导出: {output_path}")


if __name__ == "__main__":
    main()
