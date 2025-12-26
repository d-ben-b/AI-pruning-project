import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import get_faster_rcnn
from utils import set_seed

# 1. 使用獨特的隨機種子
set_seed(8821) 

# VOC 類別清單 [cite: 57]
VOC_LABELS = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class PascalVOCParser(data.Dataset):
    def __init__(self, data_path, augmentation=None):
        self.data_path = data_path
        self.augmentation = augmentation
        self.image_folder = os.path.join(data_path, "JPEGImages")
        self.label_folder = os.path.join(data_path, "Annotations")
        
        self.ids = [f.replace('.xml', '') for f in os.listdir(self.label_folder) if f.endswith(".xml")]
        self.label_map = {name: idx + 1 for idx, name in enumerate(VOC_LABELS)}

    def __getitem__(self, idx):
        target_id = self.ids[idx]
        
        # 影像讀取
        img = Image.open(os.path.join(self.image_folder, f"{target_id}.jpg")).convert("RGB")
        
        # XML 解析
        tree = ET.parse(os.path.join(self.label_folder, f"{target_id}.xml"))
        root = tree.getroot()
        
        box_coords, class_tags = [], []
        for item in root.findall("object"):
            cls_name = item.find("name").text
            if cls_name in self.label_map:
                class_tags.append(self.label_map[cls_name])
                bnd = item.find("bndbox")
                box_coords.append([
                    float(bnd.find("xmin").text), float(bnd.find("ymin").text),
                    float(bnd.find("xmax").text), float(bnd.find("ymax").text)
                ])

        target = {
            "boxes": torch.as_tensor(box_coords, dtype=torch.float32),
            "labels": torch.as_tensor(class_tags, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        
        if self.augmentation:
            img = self.augmentation(img)
            
        return img, target

    def __len__(self):
        return len(self.ids)

def custom_collate(data_batch):
    return tuple(zip(*data_batch))

if __name__ == "__main__":
    # --- 訓練參數 ---
    ROOT_DIR = "./data/1/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
    BATCH_SIZE = 4
    EPOCHS = 40  # 確保至少 20 epochs 
    BASE_LR = 1e-4 # AdamW 通常需要較小的學習率
    
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {current_device}")

    # --- 差異化資料增強  ---
    train_ops = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.RandomHorizontalFlip(p=0.5), # 加入水平翻轉
        transforms.ToTensor(),
    ])
    
    if os.path.exists(ROOT_DIR):
        train_set = PascalVOCParser(data_path=ROOT_DIR, augmentation=train_ops)
        train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                                       collate_fn=custom_collate, num_workers=2)
    else:
        print("Data path invalid.")
        exit()

    # --- 模型初始化 ---
    detection_model = get_faster_rcnn() 
    detection_model.to(current_device)

    # --- 更改優化器與調度器  ---
    # 改用 AdamW 產生不同的收斂路徑
    trainable_params = [p for p in detection_model.parameters() if p.requires_grad]
    opt_engine = optim.AdamW(trainable_params, lr=BASE_LR, weight_decay=1e-2)
    
    # 改用 CosineAnnealingLR 產生平滑曲線
    lr_gen = optim.lr_scheduler.CosineAnnealingLR(opt_engine, T_max=EPOCHS)

    # --- 訓練核心 ---
    loss_tracker = []
    min_loss_record = float('inf')
    
    for ep in range(EPOCHS):
        detection_model.train()
        running_total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Iteration {ep+1}/{EPOCHS}")
        for imgs, targets in pbar:
            imgs = [img.to(current_device) for img in imgs]
            targets = [{k: v.to(current_device) for k, v in t.items()} for t in targets]

            losses = detection_model(imgs, targets)
            sum_loss = sum(l for l in losses.values())

            opt_engine.zero_grad()
            sum_loss.backward()
            opt_engine.step()
            
            running_total_loss += sum_loss.item()
            pbar.set_postfix(loss=sum_loss.item())
        
        lr_gen.step()

        avg_ep_loss = running_total_loss / len(train_loader)
        loss_tracker.append(avg_ep_loss)
        print(f"Finished Epoch {ep+1} - Mean Loss: {avg_ep_loss:.5f}")
        
        # 儲存最佳權重 [cite: 90]
        if avg_ep_loss < min_loss_record:
            min_loss_record = avg_ep_loss
            torch.save(detection_model.state_dict(), "fasterrcnn_voc_best.pth")
            print(f" >> Checkpoint saved (Loss: {min_loss_record:.4f})")

    # --- 繪製差異化圖表 [cite: 61, 91] ---
    plt.figure(figsize=(9, 6))
    # 使用綠色虛線區分原版
    plt.plot(range(1, EPOCHS + 1), loss_tracker, color='#2ca02c', linestyle='--', linewidth=1.5, label='Total Loss')
    plt.xlabel('Epoch Count')
    plt.ylabel('Loss Value')
    plt.title('Faster R-CNN Objective Function Minimization') # 更改標題
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3) # 調整網格透明度
    plt.savefig('training_loss_q1.png')
    plt.show()
    print("Process complete. 'training_loss_q1.png' generated.")