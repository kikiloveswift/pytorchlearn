import numpy as np
import torch
import torchvision

def current_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def train_pytorch():
    print(f"current_device = {current_device()}")
    data = torch.ones((3, 3))
    data = data.to(current_device())
    print(data.device)

if __name__ == '__main__':
    train_pytorch()
