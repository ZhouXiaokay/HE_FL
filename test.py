from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # (16, 28, 28)                           #  output: (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (16, 14, 14)
        )
        self.Sq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.Sq1(x)
        x = self.Sq2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def train():
    epoches = 1
    mnist_net = CNN().cuda()
    mnist_net.train()
    loss_fn = nn.CrossEntropyLoss()
    opitimizer = optim.SGD(mnist_net.parameters(), lr=0.002)
    mnist_train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=5, shuffle=True)

    loss = 0
    for epoch in range(epoches):
        for batch_X, batch_Y in train_loader:
            batch_X= batch_X.cuda()
            batch_Y = batch_Y.cuda()
            opitimizer.zero_grad()
            outputs = mnist_net(batch_X)
            loss = loss_fn(outputs, batch_Y)
            loss.backward()
            opitimizer.step()
            loss += loss.item()
            print(loss)


if __name__ == '__main__':
    train()

