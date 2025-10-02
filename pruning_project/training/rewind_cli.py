import torch
import torch.nn as nn
import torch.optim as optim
from rewind import save_rewind_point, load_rewind_point


def main():
    # 建立一個簡單模型
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 做一次 forward/backward 改變權重
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    print("Before saving, weight[0,0] =", model.weight[0, 0].item())

    # 存檔
    save_rewind_point(model, optimizer, "checkpoints/rewind_test.pth")

    # 修改權重（模擬 finetune）
    with torch.no_grad():
        model.weight.add_(10.0)
    print("After modification, weight[0,0] =", model.weight[0, 0].item())

    # 載入 rewind point
    load_rewind_point(model, optimizer, "checkpoints/rewind_test.pth")
    print("After rewind, weight[0,0] =", model.weight[0, 0].item())


if __name__ == "__main__":
    main()
