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

# 引用先前建立的模組
from model import get_faster_rcnn
from utils import set_seed

# 設定隨機種子
set_seed(9966)

# ==========================================
# 1. 定義資料集 (Dataset)
# ==========================================
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")
        
        self.file_names = [x.split('.')[0] for x in os.listdir(self.ann_dir) if x.endswith(".xml")]
        self.class_to_id = {name: i+1 for i, name in enumerate(VOC_CLASSES)}

    def __getitem__(self, index):
        file_id = self.file_names[index]
        
        # 1. 讀取圖片
        img_path = os.path.join(self.img_dir, file_id + ".jpg")
        img = Image.open(img_path).convert("RGB")
        
        # 2. 讀取 XML 標註
        ann_path = os.path.join(self.ann_dir, file_id + ".xml")
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            label_name = obj.find("name").text
            if label_name in self.class_to_id:
                labels.append(self.class_to_id[label_name])
                
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

    def __len__(self):
        return len(self.file_names)

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 2. 訓練主程式
# ==========================================
if __name__ == "__main__":
    # --- 參數設定 ---
    DATA_ROOT = "./data/1/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
    BATCH_SIZE = 4
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.005
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- 準備資料 ---
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if os.path.exists(DATA_ROOT):
        if not os.path.exists(os.path.join(DATA_ROOT, "JPEGImages")):
            print(f"Error: 'JPEGImages' not found in {DATA_ROOT}. Please check the path.")
            exit()
            
        dataset = VOCDataset(root=DATA_ROOT, transform=data_transform)
        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                     collate_fn=collate_fn, num_workers=2)
    else:
        print(f"Error: Path {DATA_ROOT} not found.")
        exit()

    # --- 載入模型 ---
    model = get_faster_rcnn() 
    model.to(device)

    # --- 設定 Optimizer 與 Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=0.0005)
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- 訓練迴圈 ---
    loss_history = []
    best_loss = float('inf') # 初始化最低 Loss
    
    print("Start Training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=losses.item(), lr=current_lr)
        
        lr_scheduler.step()

        # 計算平均 Loss
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Avg Loss: {avg_loss:.4f}")
        
        # --- 儲存 Best Model ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = "fasterrcnn_voc_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f" -> Best model saved at Epoch {epoch+1} with loss: {best_loss:.4f}")

    # ==========================================
    # 3. 儲存結果
    # ==========================================
    
    # 儲存最後一個 Epoch 的權重
    final_save_path = "fasterrcnn_voc_last.pth"
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

    # 繪製並儲存 Loss Curve
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Faster R-CNN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_q1.png')
    plt.show()
    print("Loss curve saved as training_loss_q1.png")