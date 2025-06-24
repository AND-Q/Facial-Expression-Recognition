# 人脸检测与表情识别系统

本项目是一个基于深度学习的人脸检测与表情识别系统，使用YOLOv11进行人脸检测，并使用自定义训练的YOLO模型进行表情识别。系统支持图像、视频文件和实时摄像头输入，具有直观的图形用户界面，可以轻松进行人脸检测和表情分析。

## 目录结构

```
├── UI.py                       # 图形用户界面主程序
├── yolo_face_detection.py      # 人脸检测核心功能
├── image_emotion_recognition.py # 图像表情识别功能
├── train.py                    # 模型训练脚本
├── val.py                      # 模型验证脚本
├── predict.py                  # 模型预测脚本
├── export.py                   # 模型导出脚本
├── yolov11n-face.pt            # YOLOv11人脸检测预训练模型
├── yolo11n.pt                  # YOLO基础模型
├── yolo11n-cls.pt              # YOLO分类基础模型
├── 数据集/                      # 数据集相关文件
│   ├── yolo_emotion_dataset_plus/   # 增强的YOLO情绪数据集
│   ├── my_emotion_datasets/         # 自定义情绪数据集
│   ├── my_yolo_emotion_dataset/     # 处理后的YOLO格式情绪数据集
│   ├── split_dataset_for_yolo.py    # 数据集分割脚本
│   ├── shuffle_and_rename.py        # 数据集随机化和重命名脚本
│   ├── 提取每一帧.py                 # 从视频提取帧的脚本
│   └── 下载b站视频.py                # B站视频下载脚本
├── test_data/                  # 测试数据
├── runs/                       # 训练运行结果目录
│   └── classify/               # 分类模型训练结果
│       ├── datasets_plus_optimized/    # 综合数据集模型
│       ├── fer2013_plus_optimized/     # FER2013增强模型
│       ├── affectnet_optimized/        # AffectNet模型
│       └── my_datasets_optimized/      # 自定义数据集模型
└── ultralytics/                # YOLO核心库
```

## 系统特点

- **多模型支持**：集成了多个训练模型，包括综合数据集模型、FER2013增强模型、AffectNet模型和自定义数据集模型
- **多输入源**：支持图像文件、视频文件和实时摄像头输入
- **美观的图形界面**：使用PyQt5开发的现代化界面，支持暗色主题
- **实时处理**：高效的视频处理，支持实时表情识别
- **结果保存**：可将处理结果保存为图像或视频文件
- **灵活配置**：可调整置信度阈值，选择不同的表情识别模型

## 核心模块说明

### 1. UI.py

图形用户界面主程序，集成了所有功能，包括：

- 输入源选择（摄像头、图像文件、视频文件）
- 模型选择（多种预训练模型）
- 置信度阈值调整
- 结果显示和保存
- 多线程处理避免UI卡顿

主要类：
- `VideoThread`：视频处理线程，处理实时视频流
- `FaceDetectionApp`：主应用窗口，提供用户界面和控制功能

### 2. yolo_face_detection.py

人脸检测核心功能模块，提供以下功能：

- 人脸检测模型加载和管理
- 实时视频人脸检测
- 图像人脸检测
- 视频文件人脸检测
- 中文文本渲染支持

主要函数：
- `download_face_model()`：下载YOLOv11人脸检测模型
- `load_font()`：加载中文字体
- `cv2_add_chinese_text()`：在OpenCV图像上添加中文文本
- `detect_faces_video()`：视频人脸检测（摄像头）
- `detect_faces_image()`：图像人脸检测
- `detect_faces_video_file()`：视频文件人脸检测

### 3. image_emotion_recognition.py

图像表情识别功能模块，提供以下功能：

- 静态图像中的人脸检测
- 表情识别与分析
- 结果可视化与保存

主要函数：
- `recognize_emotion()`：识别图片中的人脸表情

### 4. train.py

模型训练脚本，使用ultralytics库训练YOLO分类模型，包含以下特性：

- 支持多种数据集（FER2013Plus、AffectNet、自定义数据集）
- 高级优化器设置（AdamW）
- 学习率调度（余弦退火）
- 正则化技术（权重衰减、Dropout）
- 数据增强（内置增强、Mixup）
- 训练管理（早停、定期保存）

### 5. 数据集处理

- `split_dataset_for_yolo.py`：将数据集分割为训练集、验证集和测试集
- `shuffle_and_rename.py`：随机化和重命名数据集文件
- `提取每一帧.py`：从视频中提取帧作为训练数据
- `下载b站视频.py`：从B站下载视频用于数据收集

## 预训练模型

项目包含多个预训练模型：

1. **综合数据集模型**：使用多个数据集联合训练的模型，路径：`runs/classify/datasets_plus_optimized/weights/best.pt`
2. **FER2013增强模型**：使用增强的FER2013数据集训练的模型，路径：`runs/classify/fer2013_plus_optimized/weights/best.pt`
3. **AffectNet模型**：使用AffectNet数据集训练的模型，路径：`runs/classify/affectnet_optimized/weights/best.pt`
4. **自定义数据集模型**：使用自定义收集和标注的数据集训练的模型，路径：`runs/classify/my_datasets_optimized/weights/best.pt`

## 使用方法

### 运行图形界面

```bash
python UI.py
```

### 命令行使用（单张图片表情识别）

```bash
python image_emotion_recognition.py 图片路径
```

### 命令行使用（人脸检测）

```bash
python yolo_face_detection.py --image 图片路径  # 图片模式
python yolo_face_detection.py --video 视频路径  # 视频模式
python yolo_face_detection.py --camera          # 摄像头模式
```

### 训练自己的模型

1. 准备数据集并按YOLO格式组织（可使用`split_dataset_for_yolo.py`辅助）
2. 修改`train.py`中的数据集路径和参数
3. 运行训练脚本
```bash
python train.py
```

## 表情类别

系统可以识别的表情类别包括：
- 愤怒（Angry）
- 厌恶（Disgust）
- 高兴（Happy）
- 中性（Neutral）
- 悲伤（Sad）
- 惊讶（Surprise）

## 环境要求

- Python 3.8+
- PyQt5
- OpenCV
- PyTorch
- Ultralytics
- Pillow
- NumPy

可以使用以下命令安装依赖：

```bash
pip install ultralytics opencv-python PyQt5 pillow numpy torch torchvision
```

## 性能优化

- 使用多线程处理视频流，避免UI卡顿
- 对人脸区域进行预处理，提高表情识别准确率
- 支持调整置信度阈值，平衡检测速度和准确率
- 优化边界框处理，避免越界错误

## 注意事项

1. 首次运行时，系统会自动下载所需的预训练模型
2. 表情识别在灯光良好的环境下效果更佳
3. 对于视频文件处理，建议使用较高配置的计算机以获得更流畅的体验
4. 系统会自动检测并使用系统中可用的中文字体，如果没有找到合适的字体，中文显示可能会出现乱码 