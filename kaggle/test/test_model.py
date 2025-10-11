# kaggle/test/test_model.py
import torch
from kaggle.code.model import build_model


def test_model_forward():
    device = "cpu"  # 單元測試固定 CPU，避免佔 GPU
    model = build_model(
        model_name="DeepLabV3Plus",
        encoder="resnet18",  # 小一點的 backbone
        num_classes=16,
        pretrained=False,  # 減少下載及記憶體
    ).to(device)

    x = torch.randn(1, 3, 256, 256, device=device)  # 小尺寸輸入
    model.eval()
    with torch.no_grad():
        y = model(x)

    print("✅ 模型 forward 測試成功")
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    test_model_forward()
