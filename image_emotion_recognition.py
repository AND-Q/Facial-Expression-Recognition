import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont


def recognize_emotion(image_path):
    """
    识别图片中的人脸表情

    参数:
        image_path: 图片路径
    """
    # 加载训练好的模型
    model_path = 'runs/classify/train/weights/best.pt'
    model = YOLO(model_path)

    # 表情标签（根据fer2013plus数据集）
    emotion_labels = ['生气', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '中性']

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 设置中文字体
    # 首先尝试使用系统字体
    font_path = None
    if os.path.exists("C:/Windows/Fonts/simhei.ttf"):
        font_path = "C:/Windows/Fonts/simhei.ttf"
    elif os.path.exists("C:/Windows/Fonts/simsun.ttc"):
        font_path = "C:/Windows/Fonts/simsun.ttc"
    elif os.path.exists("C:/Windows/Fonts/msyh.ttc"):
        font_path = "C:/Windows/Fonts/msyh.ttc"

    # 如果找不到系统字体，尝试使用fonts目录下的字体
    if font_path is None:
        os.makedirs("fonts", exist_ok=True)
        font_path = "fonts/font.ttf"
        if not os.path.exists(font_path):
            print(f"未找到字体文件: {font_path}")
            print("将使用默认字体，中文可能显示为乱码")

    # 创建字体对象
    try:
        font_size = 24
        font = ImageFont.truetype(font_path, font_size)
        print(f"已加载字体: {font_path}")
    except Exception as e:
        print(f"加载字体失败: {e}")
        font = ImageFont.load_default()

    def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), font=font):
        """
        在OpenCV图像上添加中文文本
        """
        # 判断是否需要使用PIL绘制
        if font is not None and font != ImageFont.load_default():
            # 转换图像从OpenCV格式(BGR)到PIL格式(RGB)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建绘图对象
            draw = ImageDraw.Draw(img_pil)
            # 绘制文本
            draw.text(position, text, font=font, fill=text_color[::-1])  # PIL颜色顺序是RGB
            # 转换回OpenCV格式
            img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_with_text
        else:
            # 如果没有合适的字体，使用OpenCV默认方法（可能显示乱码）
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
            return img

    # 转换为灰度图像（人脸检测用）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # 如果没有检测到人脸
    if len(faces) == 0:
        print("未检测到人脸")
        image = cv2_add_chinese_text(image, "未检测到人脸", (20, 60), (0, 0, 255))

    # 对每个检测到的人脸进行表情识别
    for i, (x, y, w, h) in enumerate(faces):
        # 绘制人脸框
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 提取人脸区域
        face_roi = image[y:y + h, x:x + w]

        try:
            # 将人脸区域转换为灰度图像，与训练数据保持一致
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # 将灰度图像转换为3通道，因为YOLO模型需要3通道输入
            face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

            # 使用YOLO模型进行表情识别
            results = model(face_roi_gray_3ch)

            # 获取预测结果
            probs = results[0].probs.data.tolist()
            class_id = probs.index(max(probs))
            confidence = max(probs)

            # 获取表情标签
            emotion = emotion_labels[class_id]

            # 在图像上显示预测结果
            text = f"人脸 {i + 1}: {emotion} ({confidence:.2f})"
            image = cv2_add_chinese_text(image, text, (x, y - 30), (36, 255, 12))

            print(f"人脸 {i + 1}: {emotion} (置信度: {confidence:.2f})")
        except Exception as e:
            print(f"处理人脸 {i + 1} 时出错: {e}")

    # 显示结果
    cv2.imshow("表情识别结果", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"result_{filename}")
    cv2.imwrite(output_path, image)
    print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片人脸表情识别")
    parser.add_argument("image_path", help="图片路径")
    args = parser.parse_args()

    recognize_emotion(args.image_path)