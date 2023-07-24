import torchvision
from conf.args import args_parser
from trainer.tenseal.cnn_trainer import CNNTrainer
import pickle
from torch.utils.data import DataLoader


def run(arg):
    with open("../data/sampling.pkl", "rb") as tf:
        dict_users = pickle.load(tf)
    trans_mnist = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(root='../data', train=True, transform=trans_mnist,
                                           download=True)
    test_set = torchvision.datasets.MNIST(root='../data', train=False, transform=trans_mnist,
                                          download=True)
    test_loader = DataLoader(test_set, batch_size=arg.batch_size,
                             shuffle=True,
                             num_workers=0)
    cnn_trainer = CNNTrainer(arg, train_set, dict_users[arg.id])

    for rnd in range(args.rounds):
        print("round: ", rnd)

        cnn_trainer.local_train()
        cnn_trainer.test(test_loader)

if __name__ == '__main__':
    args = args_parser()
    args.id =1
    run(args)
