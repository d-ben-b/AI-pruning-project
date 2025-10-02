import torch
from pathlib import Path


def save_rewind_point(model, optimizer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"[Rewind] Saved checkpoint to {path}")


def load_rewind_point(model, optimizer, path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"[Rewind] Loaded checkpoint from {path}")
    return model, optimizer
