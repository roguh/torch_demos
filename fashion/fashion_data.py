import os.path

import torch
import torchvision
import torchvision.transforms

from visualize import *

# let me introduce you to my friend
# her name is fashion
# she's always my plus one
fashion_train = torchvision.datasets.FashionMNIST(
    download=True,
    train=True,
    root=os.path.expanduser("~/datasets"),
    transform=torchvision.transforms.ToTensor(),
)

fashion = torchvision.datasets.FashionMNIST(
    download=True,
    train=False,
    root=os.path.expanduser("~/datasets"),
    transform=torchvision.transforms.ToTensor(),
)

batch_size = 6
trainloader = torch.utils.data.DataLoader(
    fashion_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
testloader = torch.utils.data.DataLoader(
    fashion,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)
