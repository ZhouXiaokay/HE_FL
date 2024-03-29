from model import LogisticModel
from trainer.tenseal.base_trainer import BaseTrainer
import torch.optim as optim
import torch


class LogisticTrainer(BaseTrainer):
    def __init__(self, args, train_set, test_set):
        super().__init__(args, train_set, test_set)
        self.model = LogisticModel(self.input_size, self.num_classes).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_train(self):

        for epoch in range(self.epoch):
            for step, (x, y) in enumerate(self.train_loader):
                x = x.view(-1, self.input_size).cuda()
                y = y.cuda()
                y_pred = self.model(x).cuda()
                loss = self.criterion(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print("before ", self.model.linear.bias)
        self.average_params()
        print("after ", self.model.linear.bias)

    def test(self, test_loader):
        self.test_loader = test_loader
        total = 0
        correct = 0
        for x, y in self.test_loader:
            x = torch.autograd.Variable(x.view(-1, 28 * 28)).cuda()
            y = y.cuda()
            y_pred = self.model(x)
            _, pred = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        print(100 * correct / total)
