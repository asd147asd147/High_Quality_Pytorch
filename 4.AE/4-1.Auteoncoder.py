from pickle import decode_long
import torch 
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

EPOCH = 10
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

trainset = datasets.FashionMNIST(
    root = './data/',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2
)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

if __name__ == "__main__":
    print("Using Device:", DEVICE)
    autoencoder = Autoencoder().to(DEVICE)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    view_data = trainset.data[:5].view(-1, 28*28)
    view_data = view_data.type(torch.FloatTensor)/255.

    def train(autoencoder, train_loader):
        autoencoder.train()
        for step, (x, label) in enumerate(train_loader):
            x = x.view(-1, 28*28).to(DEVICE)
            y = x.view(-1, 28*28).to(DEVICE)
            label = label.to(DEVICE)

            encoded, decoded = autoencoder(x)

            loss = criterion(decoded, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for epoch in range(1,EPOCH+1):
        train(autoencoder, train_loader)

    test_x = view_data.to(DEVICE)
    _, decoded_data = autoencoder(test_x)

    f, a = plt.subplots(2, 5, figsize=(5,2))
    print("[Epoch {}]".format(epoch))

    for i in range(5):
        img = np.reshape(view_data.data.numpy()[i],(28,28))
        a[0][i].imshow(img, cmap='gray')
    for i in range(5):
        img = np.reshape(decoded_data.to("cpu").data.numpy()[i], (28,28))
        a[1][i].imshow(img, cmap='gray')
    plt.show()