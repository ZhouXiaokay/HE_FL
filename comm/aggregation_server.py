import comm.aggregation_server_pb2 as aggregate_server_pb2
import comm.aggregation_server_pb2_grpc as aggregate_server_pb2_grpc
import torch.optim as optim
import tenseal as ts
from comm.utils import *
import time


class AggregationServer(aggregate_server_pb2_grpc.AggregationServerServiceServicer):
    def __init__(self, num_clients, pk_ctx_file, model):
        # initial params
        self.num_clients = num_clients
        self.num_clients = 2
        pk_ctx_bytes = open(pk_ctx_file, "rb").read()
        self.pk_ctx = ts.context_from(pk_ctx_bytes)
        self.reline_keys = self.pk_ctx.relin_keys()
        self.sleep_time = 0.1
        ###########
        self.global_model = model
        self.optimizer = optim.Adam(self.global_model.parameters())
        self.latest_model_params = self.__get_init_params()

        # for sum_encrypted
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.n_sum_round = 0
        self.params_list = []
        self.avg_enc_params = []
        self.avg_completed = False

        self.count_dict = {}
        self.sum_count = 0


    def __get_init_params(self):
        param_list = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    param_list.append(p)
        param_flat_tensor = flatten_tensors(param_list).detach()
        plain_params_tensor = ts.plain_tensor(param_flat_tensor)
        enc_param_vector = ts.ckks_vector(self.pk_ctx, plain_params_tensor)
        return enc_param_vector

    def __reset_sum(self):
        self.avg_completed = False
        self.avg_enc_params.clear()
        self.params_list.clear()
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_count = 0

    def __avg_params(self):
        self.sum_count = sum(self.count_dict.values())
        sum_enc_params = sum(self.params_list)
        latest_enc_params = 1 / self.sum_count * sum_enc_params

        self.avg_enc_params.append(latest_enc_params)

    def fed_avg(self, request, context):
        client_rank = request.client_rank
        sample_num = request.sample_num
        enc_params_msg = request.params_msg
        enc_params_vector = ts.ckks_vector_from(self.pk_ctx, enc_params_msg)
        self.params_list.append(enc_params_vector)
        self.n_sum_request += 1
        self.count_dict[client_rank] = sample_num
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)

        # if client_rank == self.update_size - 1:
        if client_rank == self.num_clients - 1:
            self.__avg_params()
            self.avg_completed = True
        while not self.avg_completed:
            time.sleep(self.sleep_time)
        # create response
        enc_sum_params_msg = self.avg_enc_params[0].serialize()
        response = aggregate_server_pb2.avg_params(client_rank=client_rank,
                                                   params_msg=enc_sum_params_msg)
        # wait until all response created
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        # clear cache
        if client_rank == self.num_clients - 1:
            self.__reset_sum()

        # wait until cache for sum reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)

        return response
