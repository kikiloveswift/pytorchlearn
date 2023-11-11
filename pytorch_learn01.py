import numpy as np
import torch
import torchvision

def train_pytorch():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data = torch.ones((3, 3))
    print(data.device)

if __name__ == '__main__':
    train_pytorch()
