import kagglehub
import shutil
import os

# path = kagglehub.dataset_download("fatihkgg/affectnet-yolo-format")
path = kagglehub.dataset_download("mstjebashazida/affectnet")

target_dir = "./datasets/affectnet-yolo-format"

shutil.copytree(path, target_dir, dirs_exist_ok=True)

print("数据集已移动到：", target_dir)
