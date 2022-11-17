import torchvision
import torch.utils.data as Data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

train_set = torchvision.datasets.MNIST(root='data', train=True, transform=torchvision.transforms.ToTensor(),
                                       download=False)
test_set = torchvision.datasets.MNIST(root='data', train=False, transform=torchvision.transforms.ToTensor(),
                                      download=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
train_loader = Data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=16)
test_loader = Data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=16)
print(train_set)
input_size = 28 * 28
num_classes = 10
lr = 0.01


class LogisticModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes).to(device)

    def forward(self, x):
        y_hat = self.linear(x.to(device))
        return y_hat


model = LogisticModel(input_size, num_classes).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_set = []

for epoch in range(100):
    for step, (x, y) in enumerate(train_loader):
        x = torch.autograd.Variable(x.view(-1, input_size)).cuda()
        y = torch.autograd.Variable(y).cuda()
        y_pred = model(x).cuda()
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            loss_set.append(loss.item())
            print(f"epoch = {epoch} current loss = {loss.data}")

correct = 0
total = 0

for x, y in test_loader:
    x = torch.autograd.Variable(x.view(-1, 28 * 28)).to(device)
    y_pred = model(x)
    _, pred = torch.max(y_pred.data, 1)
    total += y.size(0)
    correct += (pred == y.to(device)).sum()

print(100 * correct / total)
plt.plot(loss_set)
plt.show()
