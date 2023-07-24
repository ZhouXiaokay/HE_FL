import comm.aggregation_server_pb2 as aggregate_server_pb2
import comm.aggregation_server_pb2_grpc as aggregate_server_pb2_grpc
import torch.optim as optim
import pickle
from comm.utils import *
import time


class AggregationServer(aggregate_server_pb2_grpc.AggregationServerServiceServicer):
    def __init__(self, num_clients, model, test_loader):
        # initial params
        self.num_clients = num_clients
        self.num_clients = 2
        self.sleep_time = 0.1
        self.global_model = model
        self.optimizer = optim.Adam(self.global_model.parameters())
        self.latest_model_params = self.__get_init_params()
        self.test_loader = test_loader

        # for sum_encrypted
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.n_sum_round = 0
        self.params_list = []
        self.avg_params = []
        self.reweight_params = []
        self.avg_completed = False
        self.reweight_completed = False

        self.count_dict = {}
        self.sum_count = 0

    def __get_init_params(self):
        param_list = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    param_list.append(p)
        param_flat_tensor = flatten_tensors(param_list).cpu().detach()

        return param_flat_tensor

    def __reset_sum(self):
        self.avg_completed = False
        self.avg_params.clear()
        self.reweight_completed = False
        self.reweight_params.clear()
        self.params_list.clear()
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_count = 0

    def __avg_params(self):
        self.sum_count = sum(self.count_dict.values())
        sum_params = sum(self.params_list)
        latest_params = 1 / self.sum_count * sum_params

        self.avg_params.append(latest_params)

    def __reweight_params(self):
        shapley_value = cal_shapley(world_size=self.num_clients,
                                    params_list=self.params_list,
                                    model=self.global_model,
                                    test_loader=self.test_loader)
        weight_list = []
        sum_shapley_value = sum(shapley_value)
        for rank in range(self.num_clients):
            w = shapley_value[rank] / sum_shapley_value
            weight_list.append(w)

        # latest_params = sum(weight_params_list)
        latest_params = reweight_state_dict(weight_list, self.params_list)
        self.global_model.load_state_dict(latest_params)
        re_acc = test_model(self.global_model, self.test_loader)
        print("reweight accuracy:", re_acc)
        self.reweight_params.append(latest_params)

    def fed_avg(self, request, context):
        client_rank = request.client_rank
        sample_num = request.sample_num
        params_msg = request.params_msg
        params_vector = pickle.loads(params_msg)
        self.params_list.append(params_vector)
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
        sum_params_msg = pickle.dumps(self.avg_params[0])
        response = aggregate_server_pb2.avg_params(client_rank=client_rank,
                                                   params_msg=sum_params_msg)
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

    def fed_shapley(self, request, context):
        client_rank = request.client_rank
        sample_num = request.sample_num
        params_msg = request.params_msg
        params_vector = pickle.loads(params_msg)
        self.params_list.append(params_vector)
        self.n_sum_request += 1
        self.count_dict[client_rank] = sample_num
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)

        # if client_rank == self.update_size - 1:
        if client_rank == self.num_clients - 1:
            self.__reweight_params()
            self.reweight_completed = True
        while not self.reweight_completed:
            time.sleep(self.sleep_time)
        # create response
        sum_params_msg = pickle.dumps(self.reweight_params[0])
        response = aggregate_server_pb2.avg_params(client_rank=client_rank,
                                                   params_msg=sum_params_msg)
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
