# Facial Expression Recognition

这是一个基于 YOLOv11 的人脸检测与表情识别项目，支持图片、视频和摄像头输入。图形界面入口是 `UI.py`，命令行入口是 `yolo_face_detection.py`。

## 环境配置

推荐使用 Conda 创建独立环境：

```bash
conda env create -f environment.yml
conda activate fer
```

如果已经有 Python 环境，也可以直接安装：

```bash
pip install -e ".[fer]"
```

## 运行

首次运行前下载模型权重：

```bash
python scripts/download_assets.py --models
```

如需训练数据集压缩包，可按需下载：

```bash
python scripts/download_assets.py --datasets
```

启动图形界面：

```bash
python UI.py
```

命令行处理图片：

```bash
python yolo_face_detection.py --image 图片路径
```

命令行处理视频：

```bash
python yolo_face_detection.py --video 视频路径
```

摄像头实时检测：

```bash
python yolo_face_detection.py --camera
```

## 模型文件

为保持 `git clone` 轻量，模型权重、数据集压缩包、训练过程产物不直接提交到 Git 仓库。项目默认从 GitHub Release 下载以下资产：

| Release 文件名 | 下载后路径 | 用途 |
| --- | --- | --- |
| `yolov11n-face.pt` | `yolov11n-face.pt` | 人脸检测 |
| `datasets_plus_best.pt` | `runs/classify/datasets_plus_optimized/weights/best.pt` | 综合数据集表情识别 |
| `fer2013_plus_best.pt` | `runs/classify/fer2013_plus_optimized/weights/best.pt` | FER2013 增强模型 |
| `affectnet_best.pt` | `runs/classify/affectnet_optimized/weights/best.pt` | AffectNet 模型 |
| `my_datasets_best.pt` | `runs/classify/my_datasets_optimized/weights/best.pt` | 自定义数据集模型 |

可选数据集资产：

| Release 文件名 | 下载后路径 |
| --- | --- |
| `affectnet.zip` | `datasets/affectnet.zip` |
| `fer2013plus.zip` | `datasets/fer2013plus.zip` |

如果使用不同 Release 标签，可以通过参数指定：

```bash
python scripts/download_assets.py --models --tag v1.0.1
```

也可以用环境变量覆盖默认仓库和版本：

```bash
FER_RELEASE_REPO=AND-Q/Facial-Expression-Recognition FER_RELEASE_TAG=v1.0.0 python scripts/download_assets.py
```
