import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS     = 40
BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data/',
                          train=True,
                          )
)