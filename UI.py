# 修改后的代码

import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

from ultralytics import YOLO

# 导入原有的人脸检测函数
from yolo_face_detection import cv2_add_chinese_text, download_face_model, load_font


class VideoThread(QThread):
    """视频处理线程，避免UI卡顿."""

    change_pixmap_signal = pyqtSignal(np.ndarray)
    progress_signal = pyqtSignal(int)

    def __init__(self, mode="camera", file_path=None):
        super().__init__()
        self.mode = mode
        self.file_path = file_path
        self.running = True
        self.face_model = None
        self.emotion_model = None
        self.conf_threshold = 0.5

    def set_models(self, face_model, emotion_model):
        self.face_model = face_model
        self.emotion_model = emotion_model

    def set_conf_threshold(self, value):
        self.conf_threshold = value / 100.0

    def run(self):
        # 加载字体
        font = load_font()

        # 表情标签
        emotion_labels = ["愤怒", "厌恶", "高兴", "中性", "悲伤", "惊讶"]

        # 初始化视频源
        if self.mode == "camera":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.file_path)

        if not cap.isOpened():
            return

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.mode == "video" else 0

        # 图片模式特殊处理
        if self.mode == "image":
            # 读取单张图片
            ret, frame = cap.read()
            if ret:
                # 人脸检测
                if self.face_model:
                    results = self.face_model(frame, conf=self.conf_threshold)

                    # 处理检测结果
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                            # 扩大边界框
                            frame_height, frame_width = frame.shape[:2]
                            expand_x = int((x2 - x1) * 0.2)
                            expand_y = int((y2 - y1) * 0.2)

                            x1_expanded = max(0, x1 - expand_x)
                            y1_expanded = max(0, y1 - expand_y)
                            x2_expanded = min(frame_width, x2 + expand_x)
                            y2_expanded = min(frame_height, y2 + expand_y)

                            # 绘制美观的人脸框
                            # 使用渐变色边框
                            cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

                            # 如果有表情识别模型，进行表情识别
                            if self.emotion_model:
                                try:
                                    # 提取人脸区域
                                    face_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                                    if face_roi.size == 0:
                                        continue

                                    # 转换为灰度图
                                    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                                    face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                                    # 表情识别
                                    emotion_results = self.emotion_model(face_roi_gray_3ch)

                                    # 获取预测结果
                                    probs = emotion_results[0].probs.data.tolist()
                                    class_id = probs.index(max(probs))
                                    confidence = max(probs)

                                    # 获取表情标签
                                    emotion = emotion_labels[class_id]

                                    # 在图像上显示预测结果
                                    text = f"{emotion}: {confidence:.2f}"

                                    # 确定文本位置 - 修改文本位置逻辑，避免与人脸框重叠
                                    # 优先显示在人脸框上方，如果空间不足则显示在下方，并且增加一定的距离
                                    if y1_expanded > 40:  # 上方有足够空间
                                        text_position = (x1_expanded, y1_expanded - 35)  # 上移更多像素
                                    else:  # 上方空间不足，显示在下方
                                        text_position = (x1_expanded, y2_expanded + 55)  # 下移更多像素

                                    # 使用自定义函数添加中文文本
                                    frame = cv2_add_chinese_text(frame, text, text_position, (36, 255, 12), font)
                                except Exception as e:
                                    print(f"处理人脸时出错: {e}")

                # 发送处理后的帧
                self.change_pixmap_signal.emit(frame)

            # 图片处理完成后释放资源
            cap.release()
            return

        # 视频和摄像头模式处理
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if self.mode == "video" and total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                self.progress_signal.emit(progress)

            # 人脸检测
            if self.face_model:
                results = self.face_model(frame, conf=self.conf_threshold)

                # 处理检测结果
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # 扩大边界框
                        frame_height, frame_width = frame.shape[:2]
                        expand_x = int((x2 - x1) * 0.2)
                        expand_y = int((y2 - y1) * 0.2)

                        x1_expanded = max(0, x1 - expand_x)
                        y1_expanded = max(0, y1 - expand_y)
                        x2_expanded = min(frame_width, x2 + expand_x)
                        y2_expanded = min(frame_height, y2 + expand_y)

                        # 绘制美观的人脸框
                        # 使用渐变色边框
                        cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

                        # 如果有表情识别模型，进行表情识别
                        if self.emotion_model:
                            try:
                                # 提取人脸区域
                                face_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                                if face_roi.size == 0:
                                    continue

                                # 转换为灰度图
                                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                                face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                                # 表情识别
                                emotion_results = self.emotion_model(face_roi_gray_3ch)

                                # 获取预测结果
                                probs = emotion_results[0].probs.data.tolist()
                                class_id = probs.index(max(probs))
                                confidence = max(probs)

                                # 获取表情标签
                                emotion = emotion_labels[class_id]

                                # 在图像上显示预测结果
                                text = f"{emotion}: {confidence:.2f}"

                                # 确定文本位置 - 修改文本位置逻辑，避免与人脸框重叠
                                # 优先显示在人脸框上方，如果空间不足则显示在下方，并且增加一定的距离
                                if y1_expanded > 40:  # 上方有足够空间
                                    text_position = (x1_expanded, y1_expanded - 35)  # 上移更多像素
                                else:  # 上方空间不足，显示在下方
                                    text_position = (x1_expanded, y2_expanded + 55)  # 下移更多像素

                                # 使用自定义函数添加中文文本
                                frame = cv2_add_chinese_text(frame, text, text_position, (36, 255, 12), font)
                            except Exception as e:
                                print(f"处理人脸时出错: {e}")

            # 发送处理后的帧
            self.change_pixmap_signal.emit(frame)

            # 控制帧率
            if self.mode == "camera":
                cv2.waitKey(1)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class FaceDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置应用标题和图标
        self.setWindowTitle("人脸检测与表情识别系统")

        # 设置窗口大小
        self.setMinimumSize(1700, 1100)

        # 定义可用的模型路径和名称
        self.model_paths = {
            "综合数据集模型": "runs/classify/datasets_plus_optimized/weights/best.pt",
            "FER2013增强模型": "runs/classify/fer2013_plus_optimized/weights/best.pt",
            "AffectNet模型": "runs/classify/affectnet_optimized/weights/best.pt",
            "我的自定义数据集模型": "runs/classify/my_datasets_optimized/weights/best.pt",
        }

        # 初始化UI
        self.init_ui()

        # 初始化模型
        self.init_models()

        # 初始化视频线程
        self.video_thread = None

    def init_ui(self):
        # 设置中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主垂直布局
        full_layout = QVBoxLayout()
        full_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(full_layout)

        # 创建内容水平布局
        content_layout = QHBoxLayout()

        # 创建左侧控制面板
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setMaximumWidth(600)  # 将控制面板宽度从650px增加到800px
        control_panel_layout = QVBoxLayout(control_panel)
        # 设置布局边距，特别是增加底部边距
        control_panel_layout.setContentsMargins(30, 30, 30, 40)  # 进一步增加内边距

        # 应用标题
        title_label = QLabel("人脸检测与表情识别")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(26)  # 进一步增加字体大小
        title_font.setBold(True)
        title_label.setFont(title_font)
        # 设置内边距，防止文字被边框遮挡，添加渐变背景
        title_label.setStyleSheet("""
            padding: 30px; 
            margin-top: 15px;
            margin-bottom: 15px;
            font-size: 20px;  
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3A1C71, stop:0.5 #D76D77, stop:1 #FFAF7B);
            border-radius: 12px;
            color: white;
            font-weight: bold;
            letter-spacing: 2px;
        """)
        control_panel_layout.addWidget(title_label)

        # 添加分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        control_panel_layout.addWidget(line)

        # 模式选择组
        mode_group = QGroupBox("检测模式")
        mode_layout = QVBoxLayout()
        mode_layout.setContentsMargins(25, 35, 25, 25)  # 增加内边距
        mode_layout.setSpacing(20)  # 增加控件之间的间距

        # 模式下拉框
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("摄像头实时检测")
        self.mode_combo.addItem("图片文件检测")
        self.mode_combo.addItem("视频文件检测")
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        mode_layout.addWidget(self.mode_combo)

        # 文件选择按钮
        self.file_button = QPushButton("选择文件")
        self.file_button.clicked.connect(self.select_file)
        self.file_button.setEnabled(False)
        self.file_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4A6572, stop:1 #547980);
                padding: 15px;  /* 增加内边距 */
                font-size: 16px;  /* 增加字体大小 */
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #547980, stop:1 #619199);
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        mode_layout.addWidget(self.file_button)

        # 文件路径显示
        self.file_label = QLabel("未选择文件")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("font-size: 14px; margin-top: 5px;")  # 增加字体大小
        mode_layout.addWidget(self.file_label)

        mode_group.setLayout(mode_layout)
        control_panel_layout.addWidget(mode_group)

        # 检测参数组
        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout()
        param_layout.setContentsMargins(25, 35, 25, 25)  # 增加内边距
        param_layout.setSpacing(20)  # 增加控件之间的间距

        # 添加模型选择下拉框
        model_layout = QVBoxLayout()
        model_label = QLabel("选择表情识别模型:")
        model_label.setStyleSheet("font-size: 15px;")
        self.model_combo = QComboBox()
        for model_name in self.model_paths.keys():
            self.model_combo.addItem(model_name)
        self.model_combo.currentIndexChanged.connect(self.change_emotion_model)
        self.model_combo.setStyleSheet("""
            padding: 10px;
            font-size: 15px;
        """)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        param_layout.addLayout(model_layout)

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        conf_label.setStyleSheet("font-size: 15px;")  # 增加字体大小
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        self.conf_value_label = QLabel("0.50")
        self.conf_value_label.setStyleSheet("font-size: 15px;")  # 增加字体大小

        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        param_layout.addLayout(conf_layout)

        param_group.setLayout(param_layout)
        control_panel_layout.addWidget(param_group)

        # 操作按钮组
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout()
        action_layout.setContentsMargins(25, 35, 25, 25)  # 增加内边距
        action_layout.setSpacing(25)  # 增加控件之间的间距

        # 开始/停止按钮
        self.start_button = QPushButton("开始检测")
        self.start_button.clicked.connect(self.toggle_detection)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2E8B57, stop:1 #3CB371);
                padding: 22px;  /* 进一步增加内边距 */
                font-size: 20px;  /* 进一步增加字体大小 */
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3CB371, stop:1 #4BC387);
            }
        """)
        action_layout.addWidget(self.start_button)

        # 保存结果按钮
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5F4B8B, stop:1 #7B68EE);
                padding: 18px;  /* 进一步增加内边距 */
                font-size: 18px;  /* 进一步增加字体大小 */
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7B68EE, stop:1 #9370DB);
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        action_layout.addWidget(self.save_button)

        action_group.setLayout(action_layout)
        control_panel_layout.addWidget(action_group)

        # 进度条
        self.progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(25, 35, 25, 25)  # 增加内边距
        progress_layout.setSpacing(20)  # 增加控件之间的间距
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.progress_group.setLayout(progress_layout)
        self.progress_group.setVisible(False)
        control_panel_layout.addWidget(self.progress_group)

        # 添加弹性空间，但不要占用太多空间
        spacer = QWidget()
        spacer.setMinimumHeight(5)
        spacer.setMaximumHeight(10)
        control_panel_layout.addWidget(spacer)

        # 将左侧控制面板添加到内容布局
        content_layout.addWidget(control_panel)

        # 创建右侧显示区域
        display_panel = QFrame()
        display_panel.setFrameShape(QFrame.StyledPanel)
        display_layout = QVBoxLayout(display_panel)

        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(900, 650)  # 增加图像显示区域尺寸
        self.image_label.setStyleSheet("""
            background-color: #1E1E1E; 
            border-radius: 15px;
            border: 2px solid #444444;
            padding: 8px;
        """)
        self.image_label.setText("等待开始检测...")
        display_layout.addWidget(self.image_label)

        # 添加显示区域到内容布局
        content_layout.addWidget(display_panel, 1)

        # 将内容布局添加到主布局
        full_layout.addLayout(content_layout, 1)

        # 设置样式
        self.apply_styles()

    def apply_styles(self):
        # 设置应用整体样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 14px;  /* 增加字体大小 */
            }
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078D7, stop:1 #00A2FF);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1C97EA, stop:1 #33B1FF);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00559B, stop:1 #0078D7);
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QComboBox {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: 1px solid #555555;
                padding: 10px;  /* 增加内边距 */
                border-radius: 5px;
                font-size: 15px;  /* 增加字体大小 */
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3C3C3C;
                color: #E0E0E0;
                selection-background-color: #0078D7;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 20px;
                font-weight: bold;
                font-size: 16px;  /* 增加字体大小 */
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                margin-top: 8px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 12px;  /* 增加高度 */
                background: #3C3C3C;
                margin: 2px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #0078D7;
                width: 22px;  /* 增加宽度 */
                height: 22px;  /* 增加高度 */
                margin: -6px 0;
                border-radius: 11px;
            }
            QSlider::handle:horizontal:hover {
                background: #1C97EA;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                color: white;
                font-size: 14px;
            }
            QProgressBar::chunk {
                background-color: #0078D7;
                border-radius: 5px;
            }
            QFrame {
                background-color: #252526;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)

    def init_models(self):
        try:
            # 下载并加载人脸检测模型
            face_model_path = download_face_model()
            self.face_model = YOLO(face_model_path)

            # 加载表情识别模型
            self.emotion_models = {}
            model_loaded = False

            # 尝试加载所有模型
            for model_name, model_path in self.model_paths.items():
                if os.path.exists(model_path):
                    try:
                        self.emotion_models[model_name] = YOLO(model_path)
                        model_loaded = True
                    except Exception as e:
                        print(f"加载模型 {model_name} 失败: {e}")

            # 设置当前使用的模型
            if model_loaded:
                current_model_name = self.model_combo.currentText()
                self.emotion_model = self.emotion_models.get(current_model_name)
                if not self.emotion_model and self.emotion_models:
                    # 如果当前选择的模型不可用，使用第一个可用的模型
                    first_available = list(self.emotion_models.keys())[0]
                    self.emotion_model = self.emotion_models[first_available]
                    # 更新下拉框选择
                    self.model_combo.setCurrentText(first_available)
            else:
                self.emotion_model = None
                QMessageBox.warning(self, "警告", "未找到任何表情识别模型，将只进行人脸检测")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.face_model = None
            self.emotion_model = None
            self.emotion_models = {}

    def change_emotion_model(self):
        """切换表情识别模型."""
        model_name = self.model_combo.currentText()

        # 如果模型已加载，直接使用
        if model_name in self.emotion_models:
            self.emotion_model = self.emotion_models[model_name]
            # 如果视频线程正在运行，更新其模型
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.set_models(self.face_model, self.emotion_model)
            return

        # 如果模型未加载，尝试加载
        model_path = self.model_paths.get(model_name)
        if model_path and os.path.exists(model_path):
            try:
                self.emotion_models[model_name] = YOLO(model_path)
                self.emotion_model = self.emotion_models[model_name]
                # 如果视频线程正在运行，更新其模型
                if self.video_thread and self.video_thread.isRunning():
                    self.video_thread.set_models(self.face_model, self.emotion_model)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型 {model_name} 失败: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", f"模型文件 {model_path} 不存在")

    def mode_changed(self):
        """处理模式变更."""
        mode = self.mode_combo.currentIndex()

        # 启用/禁用文件选择按钮
        self.file_button.setEnabled(mode > 0)

        # 重置文件选择
        if mode == 0:  # 摄像头模式
            self.file_label.setText("未选择文件")
            self.selected_file = None

    def select_file(self):
        """选择文件对话框."""
        mode = self.mode_combo.currentIndex()

        if mode == 1:  # 图片模式
            file_filter = "图片文件 (*.jpg *.jpeg *.png)"
        else:  # 视频模式
            file_filter = "视频文件 (*.mp4 *.avi *.mov *.mkv)"

        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)

        if file_path:
            self.selected_file = file_path
            self.file_label.setText(os.path.basename(file_path))

    def update_conf_threshold(self):
        """更新置信度阈值."""
        value = self.conf_slider.value() / 100.0
        self.conf_value_label.setText(f"{value:.2f}")

        # 如果视频线程正在运行，更新其阈值
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_conf_threshold(self.conf_slider.value())

    def toggle_detection(self):
        """开始/停止检测."""
        if self.video_thread and self.video_thread.isRunning():
            # 停止检测
            self.video_thread.stop()
            self.start_button.setText("开始检测")
            self.save_button.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.file_button.setEnabled(self.mode_combo.currentIndex() > 0)
            self.model_combo.setEnabled(True)  # 启用模型选择
            self.progress_group.setVisible(False)
        elif self.video_thread and not self.video_thread.isRunning() and self.mode_combo.currentIndex() == 1:
            # 图片模式特殊处理：线程可能已经结束但UI状态未更新
            self.start_button.setText("开始检测")
            self.save_button.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.file_button.setEnabled(True)
            self.model_combo.setEnabled(True)  # 启用模型选择
            self.progress_group.setVisible(False)

            # 清除视频线程引用，允许再次检测同一图片
            self.video_thread = None
        else:
            # 开始检测
            mode = self.mode_combo.currentIndex()

            if mode > 0 and not self.selected_file:
                QMessageBox.warning(self, "警告", "请先选择文件")
                return

            if not self.face_model:
                QMessageBox.critical(self, "错误", "人脸检测模型未加载")
                return

            if not self.emotion_model:
                QMessageBox.warning(self, "警告", "表情识别模型未加载，将只进行人脸检测")

            # 设置UI状态
            self.start_button.setText("停止检测")
            self.save_button.setEnabled(False)
            self.mode_combo.setEnabled(False)
            self.file_button.setEnabled(False)
            self.model_combo.setEnabled(False)  # 禁用模型选择

            # 显示进度条（仅视频模式）
            self.progress_group.setVisible(mode == 2)
            self.progress_bar.setValue(0)

            # 创建并启动视频线程
            if mode == 0:  # 摄像头模式
                self.video_thread = VideoThread(mode="camera")
            elif mode == 1:  # 图片模式
                self.video_thread = VideoThread(mode="image", file_path=self.selected_file)
            else:  # 视频模式
                self.video_thread = VideoThread(mode="video", file_path=self.selected_file)

            # 设置模型
            self.video_thread.set_models(self.face_model, self.emotion_model)
            self.video_thread.set_conf_threshold(self.conf_slider.value())

            # 连接信号
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.progress_signal.connect(self.update_progress)

            # 启动线程
            self.video_thread.start()

            # 对于图片模式，需要在线程结束后自动更新UI状态
            if mode == 1:  # 图片模式
                self.video_thread.finished.connect(lambda: self.image_processed_callback())

    def update_image(self, cv_img):
        """更新图像显示."""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        self.current_image = cv_img.copy()

    def update_progress(self, value):
        """更新进度条."""
        self.progress_bar.setValue(value)

    def convert_cv_qt(self, cv_img):
        """将OpenCV图像转换为QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def save_result(self):
        """保存结果."""
        if not hasattr(self, "current_image"):
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "图片文件 (*.jpg *.png)")

        if file_path:
            try:
                # 创建目录
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # 保存图像
                cv2.imwrite(file_path, self.current_image)
                QMessageBox.information(self, "成功", f"结果已保存至: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭事件."""
        # 停止视频线程
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        event.accept()

    def image_processed_callback(self):
        """图片处理完成后的回调函数."""
        # 更新UI状态
        self.start_button.setText("开始检测")
        self.save_button.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.file_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.progress_group.setVisible(False)

        # 清除视频线程引用，允许再次检测同一图片
        self.video_thread = None


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle(QStyleFactory.create("Fusion"))

    # 创建并显示主窗口
    window = FaceDetectionApp()
    window.show()

    sys.exit(app.exec_())
