import torch
import math
import collections


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def communicate(tensors, communication_op):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    flat_tensor = flatten_tensors(tensors)
    # print("before",flat_tensor)
    communication_op(tensor=flat_tensor)
    # print("after",flat_tensor)
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        with torch.no_grad():
            t.set_(f)


def get_utility_key(client_attendance):
    key = 0
    for i in reversed(client_attendance):
        key = 2 * key + i
    return key


def utility_key_to_groups(key, world_size):
    client_attendance = [0] * world_size
    for i in range(world_size):
        flag = key % 2
        client_attendance[i] = flag
        key = key // 2
    return client_attendance


def cal_shapley(world_size, params_list, model, test_loader):
    utility_value = dict()
    start_key = 1
    end_key = int(math.pow(2, world_size)) - 1
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        group_acc = shapley_model(group_flags, params_list, model, test_loader)
        utility_value[group_key] = group_acc

    group_acc_sum = [0 for _ in range(world_size)]
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        n_participant = sum(group_flags)
        group_acc_sum[n_participant - 1] += utility_value[group_key]
        print("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))
    print("accuracy sum of different size: {}".format(group_acc_sum))

    # cal factorial
    factor = [1] * world_size
    for epoch_idx in range(1, world_size):
        factor[epoch_idx] = factor[epoch_idx - 1] * epoch_idx

    # shapley value of all clients
    shapley_value = [0.0] * world_size
    n_shapley_round = 0

    for epoch_idx in range(world_size):
        score = 0.0
        # loop all possible groups including the current client
        start_key = 1
        end_key = int(math.pow(2, world_size)) - 1
        for group_key in range(start_key, end_key + 1):
            group_flags = utility_key_to_groups(group_key, world_size)
            group_size = sum(group_flags)
            # the current client is in the group
            if group_flags[epoch_idx] == 1 and group_size > 1:
                u_with = utility_value[group_key]
                group_flags[epoch_idx] = 0
                group_key = get_utility_key(group_flags)
                u_without = utility_value[group_key]
                score += factor[group_size - 1] / float(factor[world_size - 1]) * (u_with - u_without)
        score /= float(math.pow(2, world_size - 1))
        shapley_value[epoch_idx] = score
        n_shapley_round += 1
    print("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

    # shapley_ind = np.argsort(np.array(shapley_value))
    return shapley_value


def shapley_model(group, params_list, model, test_loader):
    attend_list = []

    for rank in range(len(group)):
        if group[rank]:
            attend_list.append(params_list[rank])
    group_params = sum_state_dict(attend_list)
    model.load_state_dict(group_params)
    acc = test_model(model, test_loader)
    return acc


def sum_state_dict(dict_list):
    key_list = list(dict_list[0].keys())
    sum_params_dict = collections.OrderedDict()
    # count = 0
    for key in key_list:
        value_list = []
        for d in dict_list:
            value_list.append(d[key])

        sum_params_dict[key] = sum(value_list) / len(value_list)
        # if count == 0 and len(value_list)==2:
        #     print(value_list[0])
        #     print(value_list[1])
        #     print(sum_params_dict[key])
        #     count += 1

    return sum_params_dict


def test_model(model, test_loader):
    total = 0
    correct = 0
    for x, y in test_loader:
        x = torch.autograd.Variable(x.view(-1, 28 * 28)).cuda()
        y = y.cuda()
        y_pred = model(x)
        _, pred = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    acc = correct / total
    # print("accuracy", correct / total)

    return acc


def reweight_state_dict(weight_list, dict_list):
    key_list = list(dict_list[0].keys())
    weight_params_dict = collections.OrderedDict()
    for key in key_list:
        value_list = []
        for rank in range(len(weight_list)):
            value_list.append(dict_list[rank][key]*weight_list[rank])

        weight_params_dict[key] = sum(value_list)


    return weight_params_dict
