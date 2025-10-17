# HW2: UAV Segmentation – Baseline Experiment Report

## 第一次實驗：DeepLabV3 基線模型 (Baseline)

### 🧩 實驗目的
本次實驗目標為建立 UAV segmentation 任務的第一個可行 **baseline 模型**，  
以 DeepLabV3 架構為主，驗證資料前處理、模型設定與評分流程是否正確，  
並作為後續改進 (U-Net / DeepLabV3+) 的比較基準。

---

### ⚙️ 實驗設定

| 項目 | 設定 |
|------|------|
| **Dataset** | UAV Segmentation Dataset (train: 4000 imgs + masks, test: 1000 imgs) |
| **輸入影像尺寸** | Resize 至 `512×512`（訓練）<br>推論後再依原圖動態放大至 `(600×800)` |
| **資料分割** | Train/Validation = 80% / 20% |
| **模型架構** | `DeepLabV3 (ResNet-50 backbone)` |
| **Loss 函數** | CrossEntropyLoss |
| **Optimizer** | Adam (`lr=1e-4`) |
| **Scheduler** | CosineAnnealingLR (`T_max=50`, `eta_min=1e-6`) |
| **Batch Size / Epochs** | 8 / 50 |
| **Normalization** | 無（baseline 不使用 ImageNet mean/std） |
| **資料增強 (Augmentation)** | RandomHorizontalFlip, ColorJitter, RandomRotation |
| **Evaluation Metric** | mean Intersection over Union (mIoU) |

---

### 🧠 訓練過程

模型以 ResNet-50 為 encoder backbone，進行 50 epochs 的訓練，  
整體 loss 呈現穩定下降趨勢，mIoU 在 validation set 上逐步提升至 0.64 左右。  

下圖為訓練曲線記錄：

![training_curves](./ML_HW2/figure/loss_curve.png)

- **橘色線**：Training Loss 逐漸收斂  
- **綠色線**：Validation mIoU 持續上升並趨於穩定  
- **藍色線**：Learning Rate 隨 cosine 曲線週期性下降  

---

### 🖼️ 結果可視化

以下為訓練集中的預測結果示例：
![img1](./ML_HW2/figure/model_pred.png)

模型能成功區分道路、天空、樹木與建築物等主要結構，  
但在邊界細節（如電線桿與陰影區）仍有明顯誤差。

---

### 💾 Kaggle 提交流程

訓練完成後，將模型最佳權重 (`best_deeplabv3.pth`)  
對測試集進行推論，輸出 segmentation mask，  
並透過 **RLE 編碼 (Run-Length Encoding)** 轉換為 submission.csv 上傳至 Kaggle 平台：

```bash
!kaggle competitions submit \
  -c 2025-ncku-ee-ml-16-classes-segmentation \
  -f submission.csv \
  -m "DeepLabV3+ 512x512 | cosine LR | multi-loss tuned"
```
### ⚠️ 遇到的困難與解決方式

在建立 baseline 模型的過程中，遇到了多項實作與資料處理上的問題，  
以下為主要困難與對應的解決策略：

| 困難項目 | 問題說明 | 解決方式 |
|-----------|------------|-----------|
| **1️⃣ Kaggle 無法提交 (`Solution and submission values for img do not match`)** | 初期輸出的 `submission.csv` 影像名稱為 `mask_0000.png` 格式，與官方 `sample_submission.csv` 的 `0000.png` 不一致，導致平台無法比對。 | 檢查官方 CSV，將檔名前綴移除並依照 `sample_submission.csv` 的順序重新生成 submission。 |
| **2️⃣ 模型輸出尺寸錯誤 (mIoU ≈ 0.26)** | 預測結果固定 resize 為 `1024×1024`，而實際 test 影像為 `(600×800)`，導致 segmentation mask 與原圖錯位。 | 修改推論階段程式，使輸出依據原始影像的實際尺寸動態 resize (`cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)`)。 |
| **3️⃣ Validation 成效不穩** | 初期未統一訓練與推論的 transform，test 阶段仍包含 Normalize，導致顏色分佈不一致。 | 移除 Normalize，保持 train/test 前處理一致，模型收斂速度與結果穩定度明顯改善。 |
| **4️⃣ mIoU 分數偏低 (≈0.27)** | 原因為上述兩項錯位與 Normalize 問題並存，使評分下降超過 20%。 | 修正尺寸與前處理後重新上傳，Public Score 由 0.27 → 0.51 (+0.24)。 |
| **5️⃣ 訓練過程中顯示 GPU 記憶體不足** | 嘗試同時載入大量 batch 時出現 CUDA OOM。 | 將 batch size 由 16 降為 8，並適度減少 augmentation 強度，確保穩定訓練。 |

---

經過多次修正與驗證後，整個 pipeline（資料載入、模型訓練、推論、RLE 編碼、Kaggle 提交）已可穩定運作。  
這次實驗最大的學習重點在於：  
> segmentation 任務中「影像尺寸與 mask 的正確對齊」比模型架構本身更關鍵。  
只要 pipeline 正確，後續的模型改進（如 U-Net 或 DeepLabV3+）就能在此基礎上穩定提升表現。
