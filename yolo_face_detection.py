import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import torch


def download_face_model():
    """下载YOLOv11人脸检测模型"""
    model_path = "yolov11n-face.pt"

    return model_path


def load_font():
    """加载中文字体"""
    # 设置中文字体
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
        return font
    except Exception as e:
        print(f"加载字体失败: {e}")
        return ImageFont.load_default()


def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), font=None, adjust_position=True):
    """在OpenCV图像上添加中文文本，可自动调整位置避免边缘遮挡"""
    if font is None:
        font = load_font()

    # 获取图像尺寸
    img_height, img_width = img.shape[:2]
    x_pos, y_pos = position

    # 判断是否需要使用PIL绘制
    if font is not None and font != ImageFont.load_default():
        # 转换图像从OpenCV格式(BGR)到PIL格式(RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建绘图对象
        draw = ImageDraw.Draw(img_pil)

        # 估算文本尺寸
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]

        # 如果需要调整位置以避免边缘遮挡
        if adjust_position:
            # 检查是否会超出左边界
            if x_pos < 0:
                x_pos = 5

            # 检查是否会超出右边界
            if x_pos + text_width > img_width:
                x_pos = img_width - text_width - 5

            # 检查是否会超出上边界
            if y_pos < text_height:
                # 如果文本在人脸框上方且会超出上边界，将其移到人脸框下方
                y_pos = position[1] + 30  # 移到人脸框下方

        # 绘制文本
        draw.text((x_pos, y_pos), text, font=font, fill=text_color[::-1])  # PIL颜色顺序是RGB
        # 转换回OpenCV格式
        img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_with_text
    else:
        # 如果没有合适的字体，使用OpenCV默认方法（可能显示乱码）
        # 估算文本尺寸
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

        # 如果需要调整位置以避免边缘遮挡
        if adjust_position:
            # 检查是否会超出左边界
            if x_pos < 0:
                x_pos = 5

            # 检查是否会超出右边界
            if x_pos + text_width > img_width:
                x_pos = img_width - text_width - 5

            # 检查是否会超出上边界
            if y_pos < text_height:
                # 如果文本在人脸框上方且会超出上边界，将其移到人脸框下方
                y_pos = position[1] + 30  # 移到人脸框下方

        cv2.putText(img, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        return img

def detect_faces_video():
    """使用YOLOv11人脸检测模型进行视频人脸检测（摄像头）"""
    # 下载并加载YOLOv11人脸检测模型
    face_model_path = download_face_model()
    face_model = YOLO(face_model_path)  # 使用专门的人脸检测模型

    # 加载表情识别模型
    emotion_model_path = 'runs/classify/fer2013_plus_optimized/weights/best.pt'
    if not os.path.exists(emotion_model_path):
        print(f"警告：表情识别模型文件不存在: {emotion_model_path}")
        print("将仅进行人脸检测，不进行表情识别")
        emotion_model = None
    else:
        emotion_model = YOLO(emotion_model_path)

    # 表情标签
    emotion_labels = ['愤怒', '厌恶', '高兴', '中性', '悲伤', '惊讶']

    # 加载字体
    font = load_font()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    print("按 'q' 键退出程序")

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            print("无法获取视频帧")
            break

        # 使用YOLOv11检测人脸

        results = face_model(frame, conf=0.8)
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 扩大边界框（调整人脸框大小）
                frame_height, frame_width = frame.shape[:2]
                # 计算边界框的扩展量（框的20%）
                expand_x = int((x2 - x1) * 0.2)
                expand_y = int((y2 - y1) * 0.2)

                # 应用扩展，但确保不超出图像边界
                x1_expanded = max(0, x1 - expand_x)
                y1_expanded = max(0, y1 - expand_y)
                x2_expanded = min(frame_width, x2 + expand_x)
                y2_expanded = min(frame_height, y2 + expand_y)

                # 绘制扩大后的人脸框
                cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

                # 提取扩大后的人脸区域
                face_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                # 如果没有加载表情识别模型，则跳过表情识别
                if emotion_model is None or face_roi.size == 0:
                    continue

                try:
                    # 将人脸区域转换为灰度图像，与训练数据保持一致
                    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                    # 将灰度图像转换为3通道，因为YOLO模型需要3通道输入
                    face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                    # 使用YOLO模型进行表情识别
                    emotion_results = emotion_model(face_roi_gray_3ch)

                    # 获取预测结果
                    probs = emotion_results[0].probs.data.tolist()
                    class_id = probs.index(max(probs))
                    confidence = max(probs)

                    # 获取表情标签
                    emotion = emotion_labels[class_id]

                    # 在图像上显示预测结果
                    text = f"{emotion}: {confidence:.2f}"

                    # 确定文本位置，智能调整以避免边缘遮挡
                    text_position = (x1_expanded, y1_expanded - 30)
                    # 如果文本位置太靠近上边界，则将其移到人脸框下方
                    if text_position[1] < 30:
                        text_position = (x1_expanded, y2_expanded + 30)

                    # 使用自定义函数添加中文文本
                    frame = cv2_add_chinese_text(frame, text, text_position, (36, 255, 12), font, adjust_position=True)
                except Exception as e:
                    print(f"处理人脸时出错: {e}")

        # 显示结果
        cv2.imshow("YOLO人脸检测与表情识别", frame)

        # 检查是否按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


def detect_faces_image(image_path):
    """使用YOLOv11人脸检测模型进行图片人脸检测"""
    # 下载并加载YOLOv11人脸检测模型
    face_model_path = download_face_model()
    face_model = YOLO(face_model_path)  # 使用专门的人脸检测模型

    # 加载表情识别模型
    emotion_model_path = 'runs/classify/fer2013_plus_optimized/weights/best.pt'
    if not os.path.exists(emotion_model_path):
        print(f"警告：表情识别模型文件不存在: {emotion_model_path}")
        print("将仅进行人脸检测，不进行表情识别")
        emotion_model = None
    else:
        emotion_model = YOLO(emotion_model_path)

    # 表情标签
    emotion_labels = ['愤怒', '厌恶', '高兴', '中性', '悲伤', '惊讶']
    # {0: 'anger', 1: 'disgust', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}
    # 加载字体
    font = load_font()

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 使用YOLOv11检测人脸

    results = face_model(image, conf=0.8)
    # 检查是否检测到人脸
    if len(results[0].boxes) == 0:
        print("未检测到人脸")
        image = cv2_add_chinese_text(image, "未检测到人脸", (20, 60), (0, 0, 255), font)
    else:
        print(f"检测到 {len(results[0].boxes)} 个人脸")

    # 处理检测结果
    for i, result in enumerate(results):
        boxes = result.boxes
        for j, box in enumerate(boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 扩大边界框（调整人脸框大小）
            image_height, image_width = image.shape[:2]
            # 计算边界框的扩展量（框的20%）
            expand_x = int((x2 - x1) * 0.2)
            expand_y = int((y2 - y1) * 0.2)

            # 应用扩展，但确保不超出图像边界
            x1_expanded = max(0, x1 - expand_x)
            y1_expanded = max(0, y1 - expand_y)
            x2_expanded = min(image_width, x2 + expand_x)
            y2_expanded = min(image_height, y2 + expand_y)

            # 绘制扩大后的人脸框
            cv2.rectangle(image, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

            # 如果没有加载表情识别模型，则跳过表情识别
            if emotion_model is None:
                continue

            # 提取扩大后的人脸区域
            face_roi = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

            if face_roi.size == 0:
                continue

            try:
                # 将人脸区域转换为灰度图像，与训练数据保持一致
                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                # 将灰度图像转换为3通道，因为YOLO模型需要3通道输入
                face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                # 使用YOLO模型进行表情识别
                emotion_results = emotion_model(face_roi_gray_3ch)

                # 获取预测结果
                probs = emotion_results[0].probs.data.tolist()
                class_id = probs.index(max(probs))
                confidence = max(probs)

                # 获取表情标签
                emotion = emotion_labels[class_id]

                # 在图像上显示预测结果
                text = f"人脸 {j + 1}: {emotion} ({confidence:.2f})"

                # 确定文本位置，智能调整以避免边缘遮挡
                text_position = (x1_expanded, y1_expanded - 30)
                # 如果文本位置太靠近上边界，则将其移到人脸框下方
                if text_position[1] < 30:
                    text_position = (x1_expanded, y2_expanded + 30)

                image = cv2_add_chinese_text(image, text, text_position, (36, 255, 12), font, adjust_position=True)
                print(f"人脸 {j + 1}: {emotion} (置信度: {confidence:.2f})")
            except Exception as e:
                print(f"处理人脸 {j + 1} 时出错: {e}")

    # 显示结果
    cv2.imshow("YOLO人脸检测与表情识别结果", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"yolo_result_{filename}")
    cv2.imwrite(output_path, image)
    print(f"结果已保存至: {output_path}")


def detect_faces_video_file(video_path):
    """使用YOLOv11人脸检测模型进行视频文件人脸检测"""
    # 下载并加载YOLOv11人脸检测模型
    face_model_path = download_face_model()
    face_model = YOLO(face_model_path)  # 使用专门的人脸检测模型

    # 加载表情识别模型
    emotion_model_path = 'runs/classify/fer2013_plus_optimized/weights/best.pt'
    if not os.path.exists(emotion_model_path):
        print(f"警告：表情识别模型文件不存在: {emotion_model_path}")
        print("将仅进行人脸检测，不进行表情识别")
        emotion_model = None
    else:
        emotion_model = YOLO(emotion_model_path)

    # 表情标签
    emotion_labels = ['愤怒', '厌恶', '高兴', '中性', '悲伤', '惊讶']

    # 加载字体
    font = load_font()

    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出目录
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置输出视频
    video_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"yolo_result_{video_filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"正在处理视频: {video_path}")
    print("按 'q' 键退出程序")

    frame_count = 0

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            print("视频处理完成")
            break

        frame_count += 1

        # 使用YOLOv11检测人脸
        # results = face_model(frame)
        results = face_model(frame, conf=0.8)
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 扩大边界框（调整人脸框大小）
                # 计算边界框的扩展量（框的20%）
                expand_x = int((x2 - x1) * 0.2)
                expand_y = int((y2 - y1) * 0.2)

                # 应用扩展，但确保不超出图像边界
                x1_expanded = max(0, x1 - expand_x)
                y1_expanded = max(0, y1 - expand_y)
                x2_expanded = min(frame_width, x2 + expand_x)
                y2_expanded = min(frame_height, y2 + expand_y)

                # 绘制扩大后的人脸框
                cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

                # 如果没有加载表情识别模型，则跳过表情识别
                if emotion_model is None or (y2_expanded - y1_expanded) <= 0 or (x2_expanded - x1_expanded) <= 0:
                    continue

                try:
                    # 提取扩大后的人脸区域
                    face_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                    if face_roi.size == 0:
                        continue

                    # 将人脸区域转换为灰度图像，与训练数据保持一致
                    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                    # 将灰度图像转换为3通道，因为YOLO模型需要3通道输入
                    face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                    # 使用YOLO模型进行表情识别
                    emotion_results = emotion_model(face_roi_gray_3ch)

                    # 获取预测结果
                    probs = emotion_results[0].probs.data.tolist()
                    class_id = probs.index(max(probs))
                    confidence = max(probs)

                    # 获取表情标签
                    emotion = emotion_labels[class_id]

                    # 在图像上显示预测结果
                    text = f"{emotion}: {confidence:.2f}"

                    # 确定文本位置，智能调整以避免边缘遮挡
                    text_position = (x1_expanded, y1_expanded - 30)
                    # 如果文本位置太靠近上边界，则将其移到人脸框下方
                    if text_position[1] < 30:
                        text_position = (x1_expanded, y2_expanded + 30)

                    # 使用自定义函数添加中文文本
                    frame = cv2_add_chinese_text(frame, text, text_position, (36, 255, 12), font, adjust_position=True)
                except Exception as e:
                    print(f"处理人脸时出错: {e}")

        # 显示进度
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")

        # 写入输出视频
        out.write(frame)

        # 显示结果
        cv2.imshow("YOLO人脸检测与表情识别", frame)

        # 检查是否按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"处理完成，结果已保存至: {output_path}")


def main():
    """主函数，根据输入类型选择相应的处理方式"""
    # 获取用户输入
    input_path = input("请输入图片/视频文件路径（直接回车使用摄像头）：").strip()

    # 根据输入类型选择相应的处理方式
    if not input_path:
        # 没有提供输入路径，使用摄像头
        print("启动视频流人脸检测模式...")
        detect_faces_video()
    else:
        # 检查文件是否存在
        if not os.path.exists(input_path):
            print(f"错误：文件不存在: {input_path}")
            return

        # 获取文件扩展名
        _, ext = os.path.splitext(input_path.lower())

        if ext in ['.jpg', '.jpeg', '.png']:
            # 图片处理
            print(f"正在处理图片: {input_path}")
            detect_faces_image(input_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 视频处理
            print(f"正在处理视频: {input_path}")
            detect_faces_video_file(input_path)
        else:
            print(f"不支持的文件类型: {ext}")
            print("支持的文件类型: .jpg, .jpeg, .png, .mp4, .avi, .mov, .mkv")


if __name__ == "__main__":
    main()

