import torch

def main():
    print("Hello from pruning!")
    print(torch.__version__, torch.cuda.is_available())
    x = torch.randn(10000, 10000, device="cuda")
    y = torch.matmul(x, x)
    print(y.shape, y.device)


if __name__ == "__main__":
    main()
