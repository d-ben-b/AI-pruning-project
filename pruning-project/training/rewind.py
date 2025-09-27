import torch
from pathlib import Path


def save_rewind_point(model, optimizer, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # TODO: 存 model.state_dict() & optimizer.state_dict()
    pass


def load_rewind_point(model, optimizer, path, map_location=None):
    # TODO: 載入 state dicts
    pass
