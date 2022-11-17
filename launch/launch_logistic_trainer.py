import torchvision
from conf.args import args_parser
from trainer.logistic_trainer import LogisticTrainer
import torch


def run(arg):
    train_set = torchvision.datasets.MNIST(root='../data', train=True, transform=torchvision.transforms.ToTensor(),
                                           download=False)
    test_set = torchvision.datasets.MNIST(root='../data', train=False, transform=torchvision.transforms.ToTensor(),
                                          download=False)
    lr_trainer = LogisticTrainer(arg, train_set, test_set)

    for rnd in range(args.rounds):
        print("round: ", rnd)

        lr_trainer.local_train()



if __name__ == '__main__':
    args = args_parser()

    run(args)
