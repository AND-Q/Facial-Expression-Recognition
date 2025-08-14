
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n-cls.yaml")  # 从YAML构建新模型
# model = YOLO("yolo11l-cls.pt")

results = model.train(
    # fer2013plus
    # affectnet
    # my_yolo_emotion_dataset
    # yolo_emotion_dataset_plus
    data="fer2013plus",
    epochs=200,
    batch=256,
    imgsz=224,
    workers=6,
    # 优化器设置
    optimizer="AdamW",  # 使用具有自适应动量的现代优化器
    lr0=0.001,  # 初始学习率
    lrf=0.001,  # 最终学习率因子
    warmup_epochs=5,  # 逐渐预热以防止早期不稳定
    cos_lr=True,  # 余弦退火学习率调度
    # 正则化
    weight_decay=0.0005,  # L2正则化
    dropout=0.2,  # 添加dropout以提高泛化能力
    # 数据增强
    augment=True,  # 启用内置增强
    mixup=0.1,  # 应用mixup增强
    # 训练管理
    patience=20,  # 早停耐心值
    save_period=10,  # 每10个epoch保存一次检查点
    # 项目设置
    project="runs/classify",
    name="fer2013_plus_optimized",
    exist_ok=True,
)
