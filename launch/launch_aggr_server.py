import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from comm.aggregation_server import AggregationServer
import comm.aggregation_server_pb2_grpc as aggregation_server_pb2_grpc
import grpc
from concurrent import futures
from model import LogisticModel,MLPModel
from conf.args import args_parser


def launch_aggregate_server(host, port):
    args = args_parser()
    aggr_server_address = host + ":" + str(port)
    max_msg_size = 1000000000
    pk_ctx_file = "../h_e/ts_ckks_pk.config"
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    # model = LogisticModel(input_size=args.input_size,num_classes=args.num_class)
    model = MLPModel(args.input_size, 200, args.num_classes).cuda()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    aggregation_server_pb2_grpc.add_AggregationServerServiceServicer_to_server(
        AggregationServer(2, pk_ctx_file, model),
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
