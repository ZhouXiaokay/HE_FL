from comm.utils import flatten_tensors, unflatten_tensors
import torch.nn
import sys

sys.path.append("../../../")
from comm.client import Client
from conf.args import args_parser
from torch.utils.data import DataLoader
from dataset import DatasetSplit


class BaseTrainer(object):

    def __init__(self, args, data_set, idxs):
        self.args = args

        # initialize settings
        self.sample_num = args.sample_num
        self.input_size = self.args.input_size
        self.num_class = self.args.num_classes
        self.epoch = self.args.epoch
        self.batch_size = self.args.batch_size
        self.shuffle_dataset = True
        self.train_loader = DataLoader(DatasetSplit(data_set, idxs), batch_size=self.batch_size,
                                       shuffle=self.shuffle_dataset,
                                       num_workers=0)
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.criterion = None

        # initialize the communication params with server
        self.max_msg_size = 900000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]

        self.server_address = args.server_address
        self.client = Client(self.server_address, args.id, self.sample_num, args.ctx_file)

    # send model params to server, and get the sum params
    def transmit(self, params_list):
        flat_tensor = flatten_tensors(params_list).detach()
        # get the average params from server
        received_list = self.client.transmit(flat_tensor)
        received_tensors = torch.tensor(received_list, dtype=flat_tensor.dtype, device=flat_tensor.device)

        return received_tensors

    # from optimizer get the model params,return a list
    def get_params_list(self):
        param_list = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    p.mul_(self.sample_num)
                    param_list.append(p)

        return param_list

    # update the model params with average params
    def average_params(self):
        params_list = self.get_params_list()
        average_params = self.transmit(params_list)
        # set average params as the new params
        for f, t in zip(unflatten_tensors(average_params, params_list), params_list):
            with torch.no_grad():
                t.set_(f)

    # one communication round
    def local_train(self):
        raise NotImplementedError

    def test(self,test_loader):
        raise NotImplementedError

    def launch(self):
        for rnd in range(self.args.rounds):
            print("round: ", rnd)
            self.local_train()


if __name__ == '__main__':
    arg = args_parser()
    # trainer = BaseTrainer(arg)
    # logistic_trainer.one_round()
