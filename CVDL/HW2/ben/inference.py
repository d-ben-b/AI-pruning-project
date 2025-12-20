import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image

# CIFAR-10 類別定義 (給 Q2 用)
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                 'dog', 'frog', 'horse', 'ship', 'truck']

# Pascal VOC 類別定義 (給 Q1 用)
VOC_CLASSES = [
    'background', # class 0
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def preprocess_image(image_path):
    """Q2 用：讀取圖片並預處理 (Resize to 32x32)"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img

def predict_resnet(model, img_tensor, device, threshold=0.5):
    """Q2 用：ResNet 推論"""
    model.eval()
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    probs_np = probabilities.cpu().numpy()
    max_prob = np.max(probs_np)
    predicted_idx = np.argmax(probs_np)
    
    if max_prob < threshold:
        label = "Others"
    else:
        label = CIFAR_CLASSES[predicted_idx]
        
    return label, max_prob, probs_np

def show_probability_histogram(probs_np, predicted_label):
    """Q2 用：畫直方圖"""
    plt.figure(figsize=(10, 5))
    bars = plt.bar(CIFAR_CLASSES, probs_np, color='blue')
    
    plt.title(f"Inference Result - Predicted: {predicted_label}")
    plt.ylabel("Probability")
    plt.xlabel("Class")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    if predicted_label != "Others":
        try:
            idx = CIFAR_CLASSES.index(predicted_label)
            bars[idx].set_color('green')
        except ValueError:
            pass
            
    plt.tight_layout()
    plt.show()

def run_rcnn_inference(model, image_path, device, threshold=0.5):
    """
    Q1.3: 執行 Faster R-CNN 推論並在圖片上畫框
    """
    model.eval()
    
    # 1. 讀取圖片並轉 Tensor
    img_pil = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_pil).to(device)
    
    # 2. 推論
    with torch.no_grad():
        prediction = model([img_tensor])
        
    # 3. 解析結果
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    # 4. 準備用 OpenCV 畫圖 (轉為 BGR 格式)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 5. 過濾並畫框
    for i in range(len(scores)):
        if scores[i] >= threshold:
            x1, y1, x2, y2 = boxes[i].astype(int)
            score = scores[i]
            label_id = labels[i]
            
            if label_id < len(VOC_CLASSES):
                label_name = VOC_CLASSES[label_id]
                text = f"{label_name}: {score:.2f}"
                
                # 畫矩形 (紅色)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 畫文字
                cv2.putText(img_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # 回傳 BGR 圖片
    return img_cv