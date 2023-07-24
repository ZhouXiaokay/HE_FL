import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grpc
import tenseal as ts

from comm import aggregation_server_pb2
from comm import aggregation_server_pb2_grpc


class TensealClient:

    def __init__(self, server_address, client_rank, sample_num, ctx_file):
        self.server_address = server_address
        self.client_rank = client_rank
        self.sample_num = sample_num
        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = aggregation_server_pb2_grpc.AggregationServerServiceStub(channel)

    def __sum_encrypted(self, plain_vector):
        plain_vector = plain_vector.detach().cpu()
        enc_vector = ts.ckks_vector(self.ctx, plain_vector)

        request = aggregation_server_pb2.local_params(
            client_rank=self.client_rank,
            sample_num=self.sample_num,
            params_msg=enc_vector.serialize()
        )
        # comm with server
        response = self.stub.fed_avg(request)

        # deserialize summed vector from response
        assert self.client_rank == response.client_rank
        summed_encrypted_vector = ts.ckks_vector_from(self.ctx, response.params_msg)

        # decrypt vector
        summed_plain_vector = summed_encrypted_vector.decrypt()

        return summed_plain_vector

    def transmit(self, params_list):
        received = self.__sum_encrypted(params_list)

        return received


if __name__ == '__main__':
    serv_address = "127.0.0.1:59000"
    # ctx_file = "../../transmission/ts_ckks.config"
    # client_rank = 0
    #
    # client = TensealClient(serv_address, client_rank, ctx_file)
