# report
## 第一次實驗紀錄
使用預設的參數，進行訓練，並記錄結果。
### 訓練參數
- Batch size: 1
- Epochs: 1000
- Learning rate: 0.001
- Optimizer: SGD
實驗效果不佳
loss下不去，準確率也無法提升停留在10%左右等同亂猜

##第二次實驗紀錄
使用預設的參數，進行訓練，並記錄結果。
### 訓練參數
- Batch size: 64
- Epochs: 1000
- Learning rate: 0.1
- Optimizer: SGD
實驗效果不佳
loss同樣下不去，準確率也無法提升停留在10%左右等同亂猜，不過改大了batch size後，loss下降的趨勢同時降第training時間，但準確率依然無法提升
## 第三次實驗紀錄
使用預設的參數，但在模型最後加上reLU + linear(10,10)，進行訓練，並記錄結果。
### 訓練參數
- Batch size: 32
- Epochs: 100
- Learning rate: 0.05
- Optimizer: SGD
實驗效果顯著提升
loss下降的趨勢明顯，準確率也提升到80%左右，不過仍無法達到90%要求
## 第四次實驗紀錄
使用預設的參數，但在模型最後加上reLU + linear(10,10)，進行訓練並加上learning rate scheduler，並記錄結果。
### 訓練參數
- Batch size: 16
- Epochs: 100
- Learning rate: 0.05
- Optimizer: SGD
- Learning rate scheduler: adjust_lr(optimizer, epoch, init_lr=0.1, step_size=10, gamma=0.5):
實驗效果顯著提升
loss下降的趨勢明顯，準確率也提升到90%左右，達到要求
train set: 3375 images
validation set: 375 images
test set: 625 images
x shape: (16, 784)
y shape: (16, 10)
MLP(
  (fc1): Linear(in_features=784, out_features=128)
  (relu1): ReLU()
  (fc2): Linear(in_features=128, out_features=10)
...
  (fc3): Linear(in_features=10, out_features=10)
  (softmax): Softmax()
)
[Epoch 0] LR = 0.05000
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
                                                              1.52it/s]
epoch 0: train_loss = 0.3332593670639739, train_acc = 0.9047407407407407
100%|██████████| 375/375 [00:00<00:00, 1035.06it/s]
epoch 0: valid_loss = 0.2624764573401304, valid_acc = 0.9323333333333333

[Epoch 1] LR = 0.05000
                                                              7.26it/s]
epoch 1: train_loss = 0.15032843449332559, train_acc = 0.9586296296296296
100%|██████████| 375/375 [00:00<00:00, 933.49it/s]
epoch 1: valid_loss = 0.19367940799960748, valid_acc = 0.9498333333333333

[Epoch 2] LR = 0.05000
                                                              5.00it/s]
epoch 2: train_loss = 0.10827465360962817, train_acc = 0.9698703703703704
100%|██████████| 375/375 [00:00<00:00, 962.95it/s]
epoch 2: valid_loss = 0.16895705752678253, valid_acc = 0.958

[Epoch 3] LR = 0.05000
                                                              0.50it/s]
epoch 3: train_loss = 0.08666862253616356, train_acc = 0.9756111111111111
100%|██████████| 375/375 [00:00<00:00, 1027.78it/s]
epoch 3: valid_loss = 0.16781154678591642, valid_acc = 0.9605

[Epoch 97] LR = 0.00010
                                                              8.29it/s]
epoch 97: train_loss = 0.0007550379245473957, train_acc = 1.0
100%|██████████| 375/375 [00:00<00:00, 1000.28it/s]
epoch 97: valid_loss = 0.1839228366028364, valid_acc = 0.9745

[Epoch 98] LR = 0.00010
                                                              9.22it/s]
epoch 98: train_loss = 0.0007549311816707423, train_acc = 1.0
100%|██████████| 375/375 [00:00<00:00, 1013.44it/s]
epoch 98: valid_loss = 0.1839235318000411, valid_acc = 0.9745

[Epoch 99] LR = 0.00010
                                                              8.14it/s]
epoch 99: train_loss = 0.0007547895325522016, train_acc = 1.0
100%|██████████| 375/375 [00:00<00:00, 1007.61it/s]
epoch 99: valid_loss = 0.1839273866260106, valid_acc = 0.9745
100%|██████████| 625/625 [00:00<00:00, 927.48it/s]
test_loss = 0.17179670928819213, test_acc = 0.9755

![alt text](image.png)

## 第五次實驗紀錄
### 實驗參數
加上 Dropout (p=0.2)做測試
訓練參數同第四次實驗
但是訓練效果不一樣明顯
模型便會亂猜
不確定為神麼將transform改成以下會有比較好的效果
def transform(x):
    return np.asarray(x) / 255.0
能有達到原有標準
train set: 3375 images
validation set: 375 images
test set: 625 images
x shape: (16, 784)
y shape: (16, 10)
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
[Epoch 0] LR = 0.05000
100%|██████████| 375/375 [00:00<00:00, 1080.48it/s]           2.44it/s]
epoch 0: train_loss = 0.9295655523813666, train_acc = 0.6804814814814815
epoch 0: valid_loss = 0.24635886953723, valid_acc = 0.9263333333333333

[Epoch 1] LR = 0.05000
100%|██████████| 375/375 [00:00<00:00, 1020.02it/s]           6.44it/s]
epoch 1: train_loss = 0.24407915359635465, train_acc = 0.929037037037037
epoch 1: valid_loss = 0.14708027534496543, valid_acc = 0.9553333333333334

[Epoch 2] LR = 0.05000
100%|██████████| 375/375 [00:00<00:00, 1089.71it/s]           1.50it/s]
epoch 2: train_loss = 0.17334394483489068, train_acc = 0.9493333333333334
epoch 2: valid_loss = 0.12944224778027213, valid_acc = 0.9603333333333334

[Epoch 3] LR = 0.05000
100%|██████████| 375/375 [00:00<00:00, 1040.62it/s]           2.77it/s]
epoch 3: train_loss = 0.13868462913605212, train_acc = 0.9582592592592593
epoch 3: valid_loss = 0.1052475268465002, valid_acc = 0.9676666666666667

[Epoch 97] LR = 0.00010
100%|██████████| 375/375 [00:00<00:00, 1065.63it/s]           7.93it/s]
epoch 97: train_loss = 0.02033351284101533, train_acc = 0.9936666666666667
epoch 97: valid_loss = 0.0695626054718611, valid_acc = 0.9836666666666667

[Epoch 98] LR = 0.00010
100%|██████████| 375/375 [00:00<00:00, 1062.96it/s]           7.31it/s]
epoch 98: train_loss = 0.020512425662186078, train_acc = 0.9934074074074074
epoch 98: valid_loss = 0.06956552409666791, valid_acc = 0.9835

[Epoch 99] LR = 0.00010
100%|██████████| 375/375 [00:00<00:00, 1089.67it/s]           1.68it/s]epoch 99: train_loss = 0.02101053723296293, train_acc = 0.9933518518518518
epoch 99: valid_loss = 0.0695904163546706, valid_acc = 0.9833333333333333

100%|██████████| 625/625 [00:00<00:00, 971.08it/s]test_loss = 0.06814059690622187, test_acc = 0.9828
![alt text](image-1.png)