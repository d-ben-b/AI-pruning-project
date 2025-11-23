# **Lab 1 Task 1 Report**

## **1. 實驗目的**

本實驗目標是以 **Numpy 自行實作神經網路 (MLP)**，透過逐步調整模型架構與訓練參數，觀察不同設定對 MNIST 分類任務的影響。

---

## **2. 實驗設定**

* **Dataset**: MNIST (csv 格式)，28×28 灰階影像，共 10 類別
* **Preprocessing**:
  * 實驗 1–4 使用 `transform(x) = np.asarray(x) * 0.99 + 0.01`
  * 實驗 5 使用 `transform(x) = np.asarray(x) / 255.0`
* **Optimizer**: SGD
* **Loss function**: CrossEntropyLoss
* **Scheduler**: Step Decay (`init_lr=0.1, step_size=10, gamma=0.5`)

---

## **3. 實驗結果**

### (a) 實驗比較表

| 實驗編號  | 模型設定                               | Batch Size | Epochs | LR    | 其他設定                              | 最佳 Train Acc | 最佳 Val Acc | 最佳 Test Acc | 效果評估                 |
| ----- | ---------------------------------- | ---------- | ------ | ----- | --------------------------------- | ------------ | ---------- | ----------- | -------------------- |
| **1** | Baseline MLP (兩層 linear + softmax) | 1          | 1000   | 0.001 | 無                                 | ~10%         | ~10%       | ~10%        | 幾乎無法學習，等同隨機猜測        |
| **2** | Baseline MLP                       | 64         | 1000   | 0.1   | 無                                 | ~10%         | ~10%       | ~10%        | loss 有下降趨勢，但 acc 無提升 |
| **3** | MLP + `ReLU + Linear(10,10)`       | 32         | 100    | 0.05  | 無                                 | ~80%         | ~80%       | 提早終止實驗           | 成效大幅提升，但仍未達 90%      |
| **4** | MLP + `ReLU + Linear(10,10)`       | 16         | 100    | 0.05  | **LR scheduler**                  | ~100%        | ~97.5%     | **97.55%**  | 成效顯著，達到要求            |
| **5** | **MLP + `ReLU + Linear(10,10)` + Dropout(p=0.2)**           | 16         | 100    | 0.05  | LR scheduler + transform `/255.0` | ~99.3%       | ~98.3%     | **98.28%**  | 最佳結果，泛化能力提升          |

---

### (b) 詳細實驗紀錄

#### **第四次實驗紀錄**

使用預設的參數，但在模型最後加上 `ReLU + Linear(10,10)`，並搭配 **learning rate scheduler**。

**訓練參數**

* Batch size: 16
* Epochs: 100
* Learning rate: 0.05
* Optimizer: SGD
* Learning rate scheduler: `adjust_lr(optimizer, epoch, init_lr=0.1, step_size=10, gamma=0.5)`

**模型結構**

```
MLP(
  (fc1): Linear(in_features=784, out_features=128)
  (relu1): ReLU()
  (fc2): Linear(in_features=128, out_features=10)
  (relu2): ReLU()
  (fc3): Linear(in_features=10, out_features=10)
  (softmax): Softmax()
)
```

**訓練 Log（部分截取）**

```
[Epoch 0] LR = 0.05000
epoch 0: train_loss = 0.3333, train_acc = 0.9047
epoch 0: valid_loss = 0.2625, valid_acc = 0.9323

[Epoch 1] LR = 0.05000
epoch 1: train_loss = 0.1503, train_acc = 0.9586
epoch 1: valid_loss = 0.1937, valid_acc = 0.9498

...

[Epoch 98] LR = 0.00010
epoch 98: train_loss = 0.000755, train_acc = 1.0
epoch 98: valid_loss = 0.1839, valid_acc = 0.9745

[Epoch 99] LR = 0.00010
epoch 99: train_loss = 0.000755, train_acc = 1.0
epoch 99: valid_loss = 0.1839, valid_acc = 0.9745

Test: test_loss = 0.1718, test_acc = 0.9755
```

**結果**

* Validation Accuracy: 97.5%
* Test Accuracy: 97.55%
* 模型收斂穩定，達到實驗要求。
* ![第四次實驗曲線](image.png)

---

#### **第五次實驗紀錄**

在模型中加入 Dropout (p=0.2)，進行測試。

**訓練參數**

* 與第四次實驗相同
* 額外：Dropout(p=0.2)
* Transform 改為 `/255.0`

**模型結構**

```
MLP_WithDropOut(
  (fc1): Linear(in_features=784, out_features=128)
  (relu1): ReLU()
  (drop1): Dropout(p=0.2)
  (fc2): Linear(in_features=128, out_features=64)
  (relu2): ReLU()
  (drop2): Dropout(p=0.2)
  (fc3): Linear(in_features=64, out_features=10)
  (softmax): Softmax()
)
```

**訓練 Log（部分截取）**

```
[Epoch 0] LR = 0.05000
epoch 0: train_loss = 0.9296, train_acc = 0.6805
epoch 0: valid_loss = 0.2464, valid_acc = 0.9263

[Epoch 1] LR = 0.05000
epoch 1: train_loss = 0.2441, train_acc = 0.9290
epoch 1: valid_loss = 0.1471, valid_acc = 0.9553

...

[Epoch 98] LR = 0.00010
epoch 98: train_loss = 0.0203, train_acc = 0.9937
epoch 98: valid_loss = 0.0696, valid_acc = 0.9837

[Epoch 99] LR = 0.00010
epoch 99: train_loss = 0.0210, train_acc = 0.9934
epoch 99: valid_loss = 0.0696, valid_acc = 0.9833

Test: test_loss = 0.0681, test_acc = 0.9828
```

**結果**

* Validation Accuracy: 98.3%
* Test Accuracy: 98.28%
* Dropout + Normalize 有效改善 overfitting
* ![第五次實驗曲線](image-1.png)

---

## **4. 問題與解決方法**

* **問題 1**: Batch size = 1、lr = 0.001 → 模型無法收斂，acc 長期停在 10%。
  **解決方法**: 將 batch size 調至 32、lr=0.05，並新增 `ReLU+Linear`，準確率提升至 ~80%。

* **問題 2**: 模型 overfitting，train acc 接近 100%，但 valid acc 停滯。
  **解決方法**: 引入 **Dropout (p=0.2)** 並改用 `x/255.0` 正規化，valid/test acc 提升至 **98%**。

* **問題 3**: 收斂速度不穩定。
  **解決方法**: 使用 **Learning Rate Scheduler**（Step Decay），使訓練更平滑，測試集表現達 97.55%。

---

## **5. 結論**

* **Baseline MLP** 幾乎無法學習，準確率僅約 10%。
* **增加一層 ReLU+Linear** 後，模型學習效果明顯改善 (80%)。
* **加入 Learning Rate Scheduler**，模型收斂穩定，測試集達 **97.55%**。
* **最佳實驗 (Dropout + Normalize)**，最終 **Test Accuracy = 98.28%**，證明正則化與數據正規化對提升泛化能力極為重要。

---

