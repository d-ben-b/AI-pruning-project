import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    設定隨機種子以確保結果可重現 (Reproducibility)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Info] Random seed set to: {seed}")