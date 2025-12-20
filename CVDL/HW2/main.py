import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchsummary import summary

# --- 匯入自定義模組 ---
from utils import set_seed
from model import get_modified_resnet18, get_faster_rcnn
from inference import preprocess_image, predict_resnet, show_probability_histogram, run_rcnn_inference
# 設定 Random Seed
set_seed(43)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVDL HW2 - ResNet & Faster R-CNN")
        self.setGeometry(100, 100, 900, 600)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image_path = None
        self.loaded_resnet_model = None 
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout() # 左右佈局

        # --- 左側：按鈕區 ---
        btn_layout = QVBoxLayout()
        
        # Group 1: Faster R-CNN
        self.label_q1 = QLabel("== Q1. Faster R-CNN ==")
        btn_layout.addWidget(self.label_q1)
        
        self.btn_1_1 = QPushButton("1.1 Show Architecture")
        self.btn_1_1.clicked.connect(self.q1_1_show_structure)
        btn_layout.addWidget(self.btn_1_1)
        
        self.btn_1_2 = QPushButton("1.2 Show Training Loss")
        self.btn_1_2.clicked.connect(self.q1_2_show_loss)
        btn_layout.addWidget(self.btn_1_2)
        
        self.btn_1_3 = QPushButton("1.3 Inference")
        self.btn_1_3.clicked.connect(self.q1_3_inference)
        btn_layout.addWidget(self.btn_1_3)
        
        btn_layout.addSpacing(20)

        # Group 2: ResNet18
        self.label_q2 = QLabel("== Q2. ResNet18 (CIFAR-10) ==")
        btn_layout.addWidget(self.label_q2)

        self.btn_2_1 = QPushButton("2.1 Load and Show Image")
        self.btn_2_1.clicked.connect(self.q2_1_load_image)
        btn_layout.addWidget(self.btn_2_1)

        self.btn_2_2 = QPushButton("2.2 Show Model Structure")
        self.btn_2_2.clicked.connect(self.q2_2_show_structure)
        btn_layout.addWidget(self.btn_2_2)

        self.btn_2_3 = QPushButton("2.3 Show Acc and Loss")
        self.btn_2_3.clicked.connect(self.q2_3_show_acc_loss)
        btn_layout.addWidget(self.btn_2_3)

        self.btn_2_4 = QPushButton("2.4 Inference")
        self.btn_2_4.clicked.connect(self.q2_4_inference)
        btn_layout.addWidget(self.btn_2_4)
        
        btn_layout.addStretch() # 推擠按鈕到上方
        main_layout.addLayout(btn_layout, 1)

        # --- 右側：圖片顯示區 ---
        self.image_label = QLabel("Image Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black; background-color: white;")
        main_layout.addWidget(self.image_label, 3)

        central_widget.setLayout(main_layout)

    # ==========================
    #      Q1 Functions
    # ==========================
    def q1_1_show_structure(self):
        """1.1 Load Model And Show Model Structure [cite: 72, 73]"""
        model = get_faster_rcnn()
        print(model) # 輸出到 Terminal
        # 你也可以選擇顯示一個簡單的訊息框告訴使用者「已輸出到 Terminal」

    def q1_2_show_loss(self):
        """1.2 Show Training Loss [cite: 82]"""
        # 讀取你訓練好存下來的圖檔
        filename = "training_loss_q1.png" 
        self.show_image_in_gui(filename)

    def q1_3_inference(self):
        # 1. 選擇圖片
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "./", "Images (*.png *.jpg *.jpeg)")
        if not filename: return
        
        # 2. 載入模型 (如果還沒載入)
        if self.loaded_rcnn_model is None:
            self.image_label.setText("Loading R-CNN model...")
            QApplication.processEvents() # 刷新 UI 讓字顯示出來
            
            try:
                self.loaded_rcnn_model = get_faster_rcnn()
                # 載入權重：請確認檔名與你訓練存檔的一致
                pth_path = "fasterrcnn_voc.pth"
                if os.path.exists(pth_path):
                    self.loaded_rcnn_model.load_state_dict(torch.load(pth_path, map_location=self.device))
                    self.loaded_rcnn_model.to(self.device)
                    print(f"Loaded weights from {pth_path}")
                else:
                    self.image_label.setText(f"Error: {pth_path} not found!")
                    return
            except Exception as e:
                print(f"Error loading model: {e}")
                self.image_label.setText("Error loading model.")
                return

        # 3. 執行推論 (直接呼叫完整邏輯)
        try:
            # 呼叫 inference.py 中的函式
            # threshold 可依需求調整，作業範例為 0.99 or 1.00，這裡設 0.8
            result_img_bgr = run_rcnn_inference(self.loaded_rcnn_model, filename, self.device, threshold=0.8)
            
            # 4. 顯示結果 (OpenCV BGR -> PyQt QImage)
            self.show_image_from_array(result_img_bgr)
            print(f"Inference done for {filename}")
            
        except Exception as e:
            print(f"Inference failed: {e}")
            self.image_label.setText(f"Inference failed:\n{e}")

    # ==========================
    #      Q2 Functions
    # ==========================
    def q2_1_load_image(self):
        """2.1 Load and Show the Image"""
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "./", "Images (*.png *.jpg *.jpeg)")
        if filename:
            self.current_image_path = filename
            # Q2 需要顯示 Resize 後的 32x32，或原圖
            # 這裡顯示原圖，推論時再 Resize
            self.show_image_in_gui(filename)
            print(f"Image loaded: {filename}")

    def q2_2_show_structure(self):
        """2.2 Show Architecture of ResNet18"""
        model = get_modified_resnet18()
        print(model) # Print to terminal
        try:
            # 使用 torchsummary 輸出詳細結構 (Input size: 3x32x32)
            summary(model.cuda(), (3, 32, 32)) 
        except Exception as e:
            print("torchsummary failed or cuda not available, skipping summary.")

    def q2_3_show_acc_loss(self):
        """2.3 Show Training/Validating Loss and Accuracy"""
        # 讀取你訓練好存下來的 Accuracy/Loss 圖表
        filename = "resnet_acc_loss.png" 
        self.show_image_in_gui(filename)

    def q2_4_inference(self):
        """2.4 Inference with Probability Distribution"""
        if not self.current_image_path:
            print("Please load an image first (Button 2.1).")
            return

        # 1. 初始化模型 (如果尚未載入)
        if self.loaded_resnet_model is None:
            print("Loading ResNet18 model...")
            self.loaded_resnet_model = get_modified_resnet18().to(self.device)
            
            # [重要] 這裡要載入你訓練好的權重 .pth
            model_path = "resnet18_best.pth" 
            if os.path.exists(model_path):
                self.loaded_resnet_model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Model weights loaded.")
            else:
                print(f"Warning: {model_path} not found. Using random weights.")

        # 2. 預處理圖片
        img_tensor, _ = preprocess_image(self.current_image_path)
        
        # 3. 執行推論
        label, max_prob, probs_np = predict_resnet(self.loaded_resnet_model, img_tensor, self.device, threshold=0.5)
        
        # 4. 顯示結果 (Terminal & GUI Label)
        result_text = f"Predicted: {label} (Conf: {max_prob:.4f})"
        print(result_text)
        self.image_label.setText(f"Result: {label}\nConfidence: {max_prob:.2f}")
        
        # 5. 彈出直方圖
        show_probability_histogram(probs_np, label)

    # --- Helper: 顯示圖片到 QLabel ---
    def show_image_in_gui(self, file_path):
        if os.path.exists(file_path):
            pixmap = QPixmap(file_path)
            # 縮放以適應視窗
            scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText(f"File not found:\n{file_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())