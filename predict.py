from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load an official model
model = YOLO("runs/classify/train2/weights/best.pt")  # load a custom model

# Predict with the model
results = model("test_data/images/8.png")  # predict on an image
