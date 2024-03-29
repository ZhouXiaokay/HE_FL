import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grpc
import pickle

from comm import aggregation_server_pb2
from comm import aggregation_server_pb2_grpc


class Client:

    def __init__(self, server_address, client_rank, sample_num):
        self.server_address = server_address
        self.client_rank = client_rank
        self.sample_num = sample_num

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = aggregation_server_pb2_grpc.AggregationServerServiceStub(channel)

    def __sum(self, plain_vector):
        # params_dict = params_dict.detach().cpu()
        vector_msg = pickle.dumps(plain_vector)

        request = aggregation_server_pb2.local_params(
            client_rank=self.client_rank,
            sample_num=self.sample_num,
            params_msg=vector_msg
        )
        # comm with server
        response = self.stub.fed_avg(request)

        # deserialize summed vector from response
        assert self.client_rank == response.client_rank
        summed_vector = pickle.loads(response.params_msg)
        # summed_plain_vector = summed_vector

        return summed_vector

    def reweight(self, params_dict):
        vector_msg = pickle.dumps(params_dict)

        request = aggregation_server_pb2.local_params(
            client_rank=self.client_rank,
            sample_num=self.sample_num,
            params_msg=vector_msg
        )
        # comm with server
        response = self.stub.fed_shapley(request)

        # deserialize summed vector from response
        assert self.client_rank == response.client_rank
        reweight_vector = pickle.loads(response.params_msg)


        return reweight_vector

    def transmit(self, params_list):
        received = self.__sum(params_list)

        return received


if __name__ == '__main__':
    serv_address = "127.0.0.1:59000"
    # ctx_file = "../../transmission/ts_ckks.config"
    # client_rank = 0
    #
    # client = TensealClient(serv_address, client_rank, ctx_file)
