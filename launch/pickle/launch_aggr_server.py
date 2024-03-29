import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torchvision
from torch.utils.data import DataLoader
from comm.aggregation_server import AggregationServer
import comm.aggregation_server_pb2_grpc as aggregation_server_pb2_grpc
import grpc
from concurrent import futures
from model import LogisticModel, MLPModel, CNNMnistModel
from conf.args import args_parser


def launch_aggregate_server(host, port):
    args = args_parser()
    aggr_server_address = host + ":" + str(port)
    max_msg_size = 1000000000
    trans_mnist = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    test_set = torchvision.datasets.MNIST(root='../data', train=False, transform=trans_mnist,
                                          download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=0)

    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    # model = LogisticModel(input_size=args.input_size,num_classes=args.num_class)
    model = MLPModel(args.input_size, 200, args.num_classes).cuda()
    # model = CNNMnistModel(args)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    aggregation_server_pb2_grpc.add_AggregationServerServiceServicer_to_server(
        AggregationServer(2, model, test_loader),
        server)
    server.add_insecure_port(aggr_server_address)
    server.start()
    print("Aggregate Server start")
    server.wait_for_termination()


def main():
    host = '127.0.0.1'
    port = 50000
    launch_aggregate_server(host, port)


if __name__ == '__main__':
    main()
