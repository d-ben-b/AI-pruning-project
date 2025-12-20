import torch.nn as nn
import torchvision.models as models

def get_modified_resnet18():
    """
    Q2.2: 修改 ResNet18 架構以適應 CIFAR-10 (32x32)
    1. 將第一層 7x7 conv (stride 2) 改為 3x3 conv (stride 1)
    2. 移除 MaxPool 層
    3. 修改 FC 層為 10 類別
    """
    # 建立模型 (不載入 ImageNet 權重，因為我們要自己訓練或是載入自己的pth)
    model = models.resnet18(weights=None)
    
    # 修改 Conv1: 輸入通道3, 輸出64, Kernel 3x3, Stride 1, Padding 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 移除 MaxPool: 使用 Identity 讓資料直接通過不做處理
    model.maxpool = nn.Identity()
    
    # 修改 Fully Connected Layer: ResNet18 原始為 512 -> 1000，改為 512 -> 10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    return model

def get_faster_rcnn():
    """
    Q1.1: 載入 Faster R-CNN 模型
    設定 num_classes = 21 (20 類別 + 1 背景)
    """
    model = models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=21)
    return model