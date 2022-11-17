from model import LogisticModel
from trainer.base_trainer import BaseTrainer
import torch.optim as optim
from torch.utils.data import DataLoader
import torch


class LogisticTrainer(BaseTrainer):
    def __init__(self, args, train_set, test_set):
        super().__init__(args, train_set, test_set)
        self.model = LogisticModel(self.input_size, self.num_class).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_train(self):
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle_dataset,
                                  num_workers=4)

        for epoch in range(self.epoch):
            for step, (x, y) in enumerate(train_loader):
                x = torch.autograd.Variable(x.view(-1, self.input_size)).cuda()
                y = torch.autograd.Variable(y).cuda()
                y_pred = self.model(x).cuda()
                loss = self.criterion(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print("before ", self.model.linear.bias)
        self.average_params()
        print("after ", self.model.linear.bias)

    def test(self):
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=self.shuffle_dataset,
                                  num_workers=4)
        total = 0
        correct = 0
        for x, y in test_loader:
            x = torch.autograd.Variable(x.view(-1, 28 * 28)).cuda()
            y_pred = self.model(x)
            _, pred = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        print(100 * correct / total)
