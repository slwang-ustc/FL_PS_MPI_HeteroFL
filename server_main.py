import asyncio
from typing import List

from comm_utils import send_data, get_data
from config import cfg
import copy
import os
import time
from collections import OrderedDict
import random
from random import sample

import numpy as np
import torch
from client import *
import datasets
from models import utils
from training_utils import test

from mpi4py import MPI

import logging

random.seed(cfg['client_selection_seed'])

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['server_cuda']
device = torch.device("cuda" if cfg['server_use_cuda'] and torch.cuda.is_available() else "cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

RESULT_PATH = os.getcwd() + '/server_log/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

comm_tags = np.ones(cfg['client_num'] + 1)


def main():
    # 所有客户端的数量。后面每一轮会在所有客户端中选择一定数量的客户端进行本地训练
    client_num = cfg['client_num']
    logger.info("Total number of clients: {}".format(client_num))
    logger.info("Model type: {}".format(cfg["model_type"]))
    logger.info("Dataset: {}".format(cfg["dataset_type"]))

    # 初始化全局模型
    global_model = utils.create_model_instance(cfg['model_type'], model_ratio=1.0)
    global_model.to(device)
    global_params_dict = global_model.state_dict()
    # global_params_dict = global_model.state_dict()
    global_params = torch.nn.utils.parameters_to_vector(global_model.parameters())
    para_nums = global_params.nelement()
    model_size = global_params.nelement() * 4 / 1024 / 1024
    logger.info("Global params num: {}".format(para_nums))
    logger.info("Global model size: {} MB".format(model_size))

    # 划分数据集，客户端有多少个，就把训练集分成多少份
    train_data_partition, partition_sizes = partition_data(
        dataset_type=cfg['dataset_type'], 
        partition_pattern=cfg['data_partition_pattern'], 
        non_iid_ratio=cfg['non_iid_ratio'], 
        client_num=client_num
    )

    logger.info('\nData partition: ')
    for i in range(len(partition_sizes)):
        s = ""
        for j in range(len(partition_sizes[i])):
            s += "{:.3f}".format(partition_sizes[i][j]) + " "
        logger.info(s)

    # label_split = {}
    # for m in range(len(partition_sizes[0])):
    #     label_split[m] = partition_sizes[:, m]

    # create workers
    all_clients: List[ClientConfig] = list()
    for client_idx in range(client_num):
        client = ClientConfig(client_idx)
        client.lr = cfg['lr']
        client.model_ratio = cfg['model_ratio'][client_idx]
        client.train_data_idxes = train_data_partition.use(client_idx)
        all_clients.append(client)

    # 加载测试集
    _, test_dataset = datasets.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)

    best_epoch = 1
    best_acc = 0
    # 开始每一轮的训练
    for epoch_idx in range(1, 1+cfg['epoch_num']):
        logger.info("_____****_____\nEpoch: {:04d}".format(epoch_idx))
        print("_____****_____\nEpoch: {:04d}".format(epoch_idx))

        # 在这里可以实现客户端选择算法，本处实现随机选择客户端。选择客户端的id，而不是实际生成客户端。
        selected_num = cfg['active_client_num']
        model_ratios = np.array(cfg['model_ratio'])
        while True:
            selected_client_idxes = random.sample(range(client_num), selected_num)
            selected_model_ratios = model_ratios[selected_client_idxes]
            flag = True
            for model_ratio in np.unique(model_ratios):
                if len(np.where(selected_model_ratios == model_ratio)[0]) < int(
                        selected_num / len(np.unique(model_ratios))):
                    flag = False
            if flag:
                break

        logger.info("Selected clients' idxes: {}".format(selected_client_idxes))
        logger.info("Selected clients' model ratio: {}".format(np.array(cfg['model_ratio'])[selected_client_idxes]))
        print("Selected clients' idxes: {}".format(selected_client_idxes))
        print("Selected clients' model ratio: {}".format(np.array(cfg['model_ratio'])[selected_client_idxes]))

        # 对全局模型进行分解 （参照论文 HeteroFL 作者的源代码）
        local_params_dicts, client_params_idxes = split_model(
            client_idxes=selected_client_idxes,
            global_params_dict=global_params_dict,
            model_ratio=cfg['model_ratio']
        )
        if epoch_idx == 1:
            temp = []
            for m, params_dict in enumerate(local_params_dicts):
                if cfg['model_ratio'][selected_client_idxes[m]] in temp:
                    continue
                temp.append(cfg['model_ratio'][selected_client_idxes[m]])
                para_nums = 0
                for k, v in params_dict.items():
                    para_nums += v.nelement()
                model_size = para_nums * 4 / 1024 / 1024
                logger.info("\nModel ratio: {}".format(cfg['model_ratio'][selected_client_idxes[m]]))
                logger.info("Model params num: {}".format(para_nums))
                logger.info("Model size: {} MB".format(model_size))


        # 将选中的客户端创建实例，初始化这些客户端的配置，然后将这些客户端实例放入列表中
        selected_clients = []
        for m, client_idx in enumerate(selected_client_idxes):
            all_clients[client_idx].epoch_idx = epoch_idx
            all_clients[client_idx].params_dict = local_params_dicts[m]
            selected_clients.append(all_clients[client_idx])

        # 每一轮都需要将选中的客户端的配置（client.config）发送给相应的客户端
        communication_parallel(selected_clients, action="send_config")

        # 从选中的客户端那里接收配置，此时选中的客户端均已完成本地训练。配置包括训练好的本地模型，学习率等
        communication_parallel(selected_clients, action="get_config")

        # 聚合客户端的本地模型
        global_params_dict = aggregate_model(
            selected_clients,
            global_params_dict,
            client_params_idxes,
            # label_split
        )

        # 加载全局模型
        global_model.load_state_dict(global_params_dict)

        # 对全局模型进行测试
        test_loss, test_acc = test(global_model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch_idx
            model_save_path = cfg['model_save_path'] + now + '/'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path, exist_ok=True)
            torch.save(global_model.state_dict(), model_save_path + now + '.pth')
        logger.info(
            "Test_Loss: {:.4f}\n".format(test_loss) +
            "Test_ACC: {:.4f}\n".format(test_acc) +
            "Best_ACC: {:.4f}\n".format(best_acc) +
            "Best_Epoch: {:04d}\n".format(best_epoch)
        )

        for m in range(len(selected_clients)):
            comm_tags[m + 1] += 1


# 论文 HeteroFL 中 3.1 Heterogeneous Models 对全局模型进行分解 （参照论文作者源代码，略微修改）
# https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients
def split_model(client_idxes, global_params_dict, model_ratio):
    if cfg['model_type'] == 'cnn_hetero':
        idx_i = [None for _ in range(len(client_idxes))]
        local_params_idxes = [OrderedDict() for _ in range(len(client_idxes))]

        output_weight_name = [k for k in global_params_dict.keys() if 'weight' in k][-1]
        output_bias_name = [k for k in global_params_dict.keys() if 'bias' in k][-1]

        for k, v in global_params_dict.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(client_idxes)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if idx_i[m] is None:
                                idx_i[m] = torch.arange(input_size, device=device)
                            input_idx_i_m = idx_i[m]
                            if k == output_weight_name:
                                output_idx_i_m = torch.arange(output_size, device=device)
                            else:
                                scaler_ratio = model_ratio[client_idxes[m]]
                                local_output_size = int(np.ceil(output_size * scaler_ratio))
                                output_idx_i_m = torch.arange(output_size, device=device)[:local_output_size]
                            local_params_idxes[m][k] = output_idx_i_m, input_idx_i_m
                            idx_i[m] = output_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            local_params_idxes[m][k] = input_idx_i_m
                    else:
                        if k == output_bias_name:
                            input_idx_i_m = idx_i[m]
                            local_params_idxes[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            local_params_idxes[m][k] = input_idx_i_m
                else:
                    pass
    # TODO
    # elif 'resnet' in cfg['model_name']:
    #     idx_i = [None for _ in range(len(user_idx))]
    #     idx = [OrderedDict() for _ in range(len(user_idx))]
    #     for k, v in self.global_parameters.items():
    #         parameter_type = k.split('.')[-1]
    #         for m in range(len(user_idx)):
    #             if 'weight' in parameter_type or 'bias' in parameter_type:
    #                 if parameter_type == 'weight':
    #                     if v.dim() > 1:
    #                         input_size = v.size(1)
    #                         output_size = v.size(0)
    #                         if 'conv1' in k or 'conv2' in k:
    #                             if idx_i[m] is None:
    #                                 idx_i[m] = torch.arange(input_size, device=v.device)
    #                             input_idx_i_m = idx_i[m]
    #                             scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
    #                             local_output_size = int(np.ceil(output_size * scaler_rate))
    #                             output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
    #                             idx_i[m] = output_idx_i_m
    #                         elif 'shortcut' in k:
    #                             input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
    #                             output_idx_i_m = idx_i[m]
    #                         elif 'linear' in k:
    #                             input_idx_i_m = idx_i[m]
    #                             output_idx_i_m = torch.arange(output_size, device=v.device)
    #                         else:
    #                             raise ValueError('Not valid k')
    #                         idx[m][k] = (output_idx_i_m, input_idx_i_m)
    #                     else:
    #                         input_idx_i_m = idx_i[m]
    #                         idx[m][k] = input_idx_i_m
    #                 else:
    #                     input_size = v.size(0)
    #                     if 'linear' in k:
    #                         input_idx_i_m = torch.arange(input_size, device=v.device)
    #                         idx[m][k] = input_idx_i_m
    #                     else:
    #                         input_idx_i_m = idx_i[m]
    #                         idx[m][k] = input_idx_i_m
    #             else:
    #                 pass
    # elif cfg['model_name'] == 'transformer':
    #     idx_i = [None for _ in range(len(user_idx))]
    #     idx = [OrderedDict() for _ in range(len(user_idx))]
    #     for k, v in self.global_parameters.items():
    #         parameter_type = k.split('.')[-1]
    #         for m in range(len(user_idx)):
    #             if 'weight' in parameter_type or 'bias' in parameter_type:
    #                 if 'weight' in parameter_type:
    #                     if v.dim() > 1:
    #                         input_size = v.size(1)
    #                         output_size = v.size(0)
    #                         if 'embedding' in k.split('.')[-2]:
    #                             output_idx_i_m = torch.arange(output_size, device=v.device)
    #                             scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
    #                             local_input_size = int(np.ceil(input_size * scaler_rate))
    #                             input_idx_i_m = torch.arange(input_size, device=v.device)[:local_input_size]
    #                             idx_i[m] = input_idx_i_m
    #                         elif 'decoder' in k and 'linear2' in k:
    #                             input_idx_i_m = idx_i[m]
    #                             output_idx_i_m = torch.arange(output_size, device=v.device)
    #                         elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
    #                             input_idx_i_m = idx_i[m]
    #                             scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
    #                             local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
    #                                                             * scaler_rate))
    #                             output_idx_i_m = (torch.arange(output_size, device=v.device).reshape(
    #                                 cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
    #                             idx_i[m] = output_idx_i_m
    #                         else:
    #                             input_idx_i_m = idx_i[m]
    #                             scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
    #                             local_output_size = int(np.ceil(output_size * scaler_rate))
    #                             output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
    #                             idx_i[m] = output_idx_i_m
    #                         idx[m][k] = (output_idx_i_m, input_idx_i_m)
    #                     else:
    #                         input_idx_i_m = idx_i[m]
    #                         idx[m][k] = input_idx_i_m
    #                 else:
    #                     input_size = v.size(0)
    #                     if 'decoder' in k and 'linear2' in k:
    #                         input_idx_i_m = torch.arange(input_size, device=v.device)
    #                         idx[m][k] = input_idx_i_m
    #                     elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
    #                         input_idx_i_m = idx_i[m]
    #                         idx[m][k] = input_idx_i_m
    #                         if 'linear_v' not in k:
    #                             idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
    #                     else:
    #                         input_idx_i_m = idx_i[m]
    #                         idx[m][k] = input_idx_i_m
    #             else:
    #                 pass
    # else:
    #     raise ValueError('Not valid model name')

    else:
        raise ValueError('Not valid model')
    local_params_dicts = [OrderedDict() for _ in range(len(client_idxes))]
    for k, v in global_params_dict.items():
        parameter_type = k.split('.')[-1]
        for m in range(len(client_idxes)):
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if 'weight' in parameter_type:
                    if v.dim() > 1:
                        local_params_dicts[m][k] = copy.deepcopy(v[torch.meshgrid(local_params_idxes[m][k])].detach())
                    else:
                        local_params_dicts[m][k] = copy.deepcopy(v[local_params_idxes[m][k]].detach())
                else:
                    local_params_dicts[m][k] = copy.deepcopy(v[local_params_idxes[m][k]].detach())
            else:
                local_params_dicts[m][k] = copy.deepcopy(v.detach())
    return local_params_dicts, local_params_idxes


# # 论文 HeteroFL 中 3.1 Heterogeneous Models 对本地模型进行聚合 （参照论文作者源代码，略微修改）
# # https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients
# def aggregate_model(selected_clients, global_params_dict, params_idxes, label_split):
#     count = OrderedDict()
#     if cfg['model_type'] == 'CNN':
#         output_weight_name = [k for k in global_params_dict.keys() if 'weight' in k][-1]
#         output_bias_name = [k for k in global_params_dict.keys() if 'bias' in k][-1]
#         for k, v in global_params_dict.items():
#             parameter_type = k.split('.')[-1]
#             count[k] = v.new_zeros(v.size(), dtype=torch.float32, device=device)
#             tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device=device)
#             for m in range(len(selected_clients)):
#                 if 'weight' in parameter_type or 'bias' in parameter_type:
#                     if parameter_type == 'weight':
#                         if v.dim() > 1:
#                             if k == output_weight_name:
#                                 label = label_split[selected_clients[m].idx]
#                                 params_idxes[m][k] = list(params_idxes[m][k])
#                                 params_idxes[m][k][0] = params_idxes[m][k][0][label]
#                                 tmp_v[torch.meshgrid(params_idxes[m][k])] += selected_clients[m].params_dict[k][label]
#                                 count[k][torch.meshgrid(params_idxes[m][k])] += 1
#                             else:
#                                 tmp_v[torch.meshgrid(params_idxes[m][k])] += selected_clients[m].params_dict[k]
#                                 count[k][torch.meshgrid(params_idxes[m][k])] += 1
#                         else:
#                             tmp_v[params_idxes[m][k]] += selected_clients[m].params_dict[k]
#                             count[k][params_idxes[m][k]] += 1
#                     else:
#                         if k == output_bias_name:
#                             label = label_split[selected_clients[m].idx]
#                             params_idxes[m][k] = params_idxes[m][k][label]
#                             tmp_v[params_idxes[m][k]] += selected_clients[m].params_dict[k][label]
#                             count[k][params_idxes[m][k]] += 1
#                         else:
#                             tmp_v[params_idxes[m][k]] += selected_clients[m].params_dict[k]
#                             count[k][params_idxes[m][k]] += 1
#                 else:
#                     tmp_v += selected_clients[m].params_dict[k]
#                     count[k] += 1
#             tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
#             v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
#     # elif 'resnet' in cfg['model_name']:
#     #     for k, v in self.global_parameters.items():
#     #         parameter_type = k.split('.')[-1]
#     #         count[k] = v.new_zeros(v.size(), dtype=torch.float32)
#     #         tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
#     #         for m in range(len(local_parameters)):
#     #             if 'weight' in parameter_type or 'bias' in parameter_type:
#     #                 if parameter_type == 'weight':
#     #                     if v.dim() > 1:
#     #                         if 'linear' in k:
#     #                             label_split = self.label_split[user_idx[m]]
#     #                             param_idx[m][k] = list(param_idx[m][k])
#     #                             param_idx[m][k][0] = param_idx[m][k][0][label_split]
#     #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
#     #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
#     #                         else:
#     #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
#     #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
#     #                     else:
#     #                         tmp_v[param_idx[m][k]] += local_parameters[m][k]
#     #                         count[k][param_idx[m][k]] += 1
#     #                 else:
#     #                     if 'linear' in k:
#     #                         label_split = self.label_split[user_idx[m]]
#     #                         param_idx[m][k] = param_idx[m][k][label_split]
#     #                         tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
#     #                         count[k][param_idx[m][k]] += 1
#     #                     else:
#     #                         tmp_v[param_idx[m][k]] += local_parameters[m][k]
#     #                         count[k][param_idx[m][k]] += 1
#     #             else:
#     #                 tmp_v += local_parameters[m][k]
#     #                 count[k] += 1
#     #         tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
#     #         v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
#     # elif cfg['model_name'] == 'transformer':
#     #     for k, v in self.global_parameters.items():
#     #         parameter_type = k.split('.')[-1]
#     #         count[k] = v.new_zeros(v.size(), dtype=torch.float32)
#     #         tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
#     #         for m in range(len(local_parameters)):
#     #             if 'weight' in parameter_type or 'bias' in parameter_type:
#     #                 if 'weight' in parameter_type:
#     #                     if v.dim() > 1:
#     #                         if k.split('.')[-2] == 'embedding':
#     #                             label_split = self.label_split[user_idx[m]]
#     #                             param_idx[m][k] = list(param_idx[m][k])
#     #                             param_idx[m][k][0] = param_idx[m][k][0][label_split]
#     #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
#     #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
#     #                         elif 'decoder' in k and 'linear2' in k:
#     #                             label_split = self.label_split[user_idx[m]]
#     #                             param_idx[m][k] = list(param_idx[m][k])
#     #                             param_idx[m][k][0] = param_idx[m][k][0][label_split]
#     #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
#     #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
#     #                         else:
#     #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
#     #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
#     #                     else:
#     #                         tmp_v[param_idx[m][k]] += local_parameters[m][k]
#     #                         count[k][param_idx[m][k]] += 1
#     #                 else:
#     #                     if 'decoder' in k and 'linear2' in k:
#     #                         label_split = self.label_split[user_idx[m]]
#     #                         param_idx[m][k] = param_idx[m][k][label_split]
#     #                         tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
#     #                         count[k][param_idx[m][k]] += 1
#     #                     else:
#     #                         tmp_v[param_idx[m][k]] += local_parameters[m][k]
#     #                         count[k][param_idx[m][k]] += 1
#     #             else:
#     #                 tmp_v += local_parameters[m][k]
#     #                 count[k] += 1
#     #         tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
#     #         v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
#     else:
#         raise ValueError('Not valid model name')
#
#     return global_params_dict


# 论文 HeteroFL 中 3.1 Heterogeneous Models 对本地模型进行聚合 （参照论文作者源代码，略微修改）
# https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients
def aggregate_model(selected_clients, global_params_dict, params_idxes):
    count = OrderedDict()
    if cfg['model_type'] == 'cnn_hetero':
        for k, v in global_params_dict.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32, device=device)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device=device)
            for m in range(len(selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            tmp_v[torch.meshgrid(params_idxes[m][k])] += copy.deepcopy(selected_clients[m].params_dict[k].detach())
                            count[k][torch.meshgrid(params_idxes[m][k])] += 1
                        else:
                            tmp_v[params_idxes[m][k]] += copy.deepcopy(selected_clients[m].params_dict[k].detach())
                            count[k][params_idxes[m][k]] += 1
                    else:
                        tmp_v[params_idxes[m][k]] += copy.deepcopy(selected_clients[m].params_dict[k].detach())
                        count[k][params_idxes[m][k]] += 1
                else:
                    tmp_v += copy.deepcopy(selected_clients[m].params_dict[k].detach())
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = copy.deepcopy(tmp_v[count[k] > 0].to(v.dtype).detach())
    # elif 'resnet' in cfg['model_name']:
    #     for k, v in self.global_parameters.items():
    #         parameter_type = k.split('.')[-1]
    #         count[k] = v.new_zeros(v.size(), dtype=torch.float32)
    #         tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
    #         for m in range(len(local_parameters)):
    #             if 'weight' in parameter_type or 'bias' in parameter_type:
    #                 if parameter_type == 'weight':
    #                     if v.dim() > 1:
    #                         if 'linear' in k:
    #                             label_split = self.label_split[user_idx[m]]
    #                             param_idx[m][k] = list(param_idx[m][k])
    #                             param_idx[m][k][0] = param_idx[m][k][0][label_split]
    #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
    #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
    #                         else:
    #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
    #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
    #                     else:
    #                         tmp_v[param_idx[m][k]] += local_parameters[m][k]
    #                         count[k][param_idx[m][k]] += 1
    #                 else:
    #                     if 'linear' in k:
    #                         label_split = self.label_split[user_idx[m]]
    #                         param_idx[m][k] = param_idx[m][k][label_split]
    #                         tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
    #                         count[k][param_idx[m][k]] += 1
    #                     else:
    #                         tmp_v[param_idx[m][k]] += local_parameters[m][k]
    #                         count[k][param_idx[m][k]] += 1
    #             else:
    #                 tmp_v += local_parameters[m][k]
    #                 count[k] += 1
    #         tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
    #         v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
    # elif cfg['model_name'] == 'transformer':
    #     for k, v in self.global_parameters.items():
    #         parameter_type = k.split('.')[-1]
    #         count[k] = v.new_zeros(v.size(), dtype=torch.float32)
    #         tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
    #         for m in range(len(local_parameters)):
    #             if 'weight' in parameter_type or 'bias' in parameter_type:
    #                 if 'weight' in parameter_type:
    #                     if v.dim() > 1:
    #                         if k.split('.')[-2] == 'embedding':
    #                             label_split = self.label_split[user_idx[m]]
    #                             param_idx[m][k] = list(param_idx[m][k])
    #                             param_idx[m][k][0] = param_idx[m][k][0][label_split]
    #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
    #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
    #                         elif 'decoder' in k and 'linear2' in k:
    #                             label_split = self.label_split[user_idx[m]]
    #                             param_idx[m][k] = list(param_idx[m][k])
    #                             param_idx[m][k][0] = param_idx[m][k][0][label_split]
    #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
    #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
    #                         else:
    #                             tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
    #                             count[k][torch.meshgrid(param_idx[m][k])] += 1
    #                     else:
    #                         tmp_v[param_idx[m][k]] += local_parameters[m][k]
    #                         count[k][param_idx[m][k]] += 1
    #                 else:
    #                     if 'decoder' in k and 'linear2' in k:
    #                         label_split = self.label_split[user_idx[m]]
    #                         param_idx[m][k] = param_idx[m][k][label_split]
    #                         tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
    #                         count[k][param_idx[m][k]] += 1
    #                     else:
    #                         tmp_v[param_idx[m][k]] += local_parameters[m][k]
    #                         count[k][param_idx[m][k]] += 1
    #             else:
    #                 tmp_v += local_parameters[m][k]
    #                 count[k] += 1
    #         tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
    #         v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
    else:
        raise ValueError('Not valid model name')

    return global_params_dict


async def send_config(client, client_rank, comm_tag):
    await send_data(comm, client, client_rank, comm_tag)


async def get_config(client, client_rank, comm_tag):
    config_received = await get_data(comm, client_rank, comm_tag)
    for k, v in config_received.__dict__.items():
        setattr(client, k, v)


def communication_parallel(client_list, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for m, client in enumerate(client_list):
        if action == "send_config":
            task = asyncio.ensure_future(send_config(client, m + 1, comm_tags[m+1]))
        elif action == "get_config":
            task = asyncio.ensure_future(get_config(client, m+1, comm_tags[m+1]))
        else:
            raise ValueError('Not valid action')
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


def partition_data(dataset_type, partition_pattern, non_iid_ratio, client_num=10):
    train_dataset, _ = datasets.load_datasets(dataset_type=dataset_type, data_path=cfg['dataset_path'])
    partition_sizes = np.ones((cfg['classes_size'], client_num))
    # iid
    if partition_pattern == 0:
        partition_sizes *= (1.0 / client_num)
    # non-iid
    # 对于每个客户端：包含所有种类的数据，但是某些种类的数据比例非常大
    #
    elif partition_pattern == 1:
        if 0 < non_iid_ratio < 10:
            partition_sizes *= ((1 - non_iid_ratio * 0.1) / (client_num - 1))
            for i in range(cfg['classes_size']):
                partition_sizes[i][i % client_num] = non_iid_ratio * 0.1
        else:
            raise ValueError('Non-IID ratio is too large')
    # non-iid
    # 对于每个客户端：缺少某一部分种类的数据，其余种类的数据按照总体分布分配
    elif partition_pattern == 2:
        if 0 < non_iid_ratio < 10:
            # 计算出每个 worker 缺少多少类数据
            missing_class_num = int(round(cfg['classes_size'] * (non_iid_ratio * 0.1)))

            # 初始化分配矩阵
            partition_sizes = np.ones((cfg['classes_size'], client_num))

            begin_idx = 0
            for worker_idx in range(client_num):
                for i in range(missing_class_num):
                    partition_sizes[(begin_idx + i) % cfg['classes_size']][worker_idx] = 0.
                begin_idx = (begin_idx + missing_class_num) % cfg['classes_size']

            for i in range(cfg['classes_size']):
                count = np.count_nonzero(partition_sizes[i])
                for j in range(client_num):
                    if partition_sizes[i][j] == 1.:
                        partition_sizes[i][j] = 1. / count
        else:
            raise ValueError('Non-IID ratio is too large')
    elif partition_pattern == 3:
        if 0 < non_iid_ratio < 10:
            most_data_proportion = cfg['classes_size'] / client_num * non_iid_ratio * 0.1
            minor_data_proportion = cfg['classes_size'] / client_num * (1 - non_iid_ratio * 0.1) / (cfg['classes_size'] - 1)
            partition_sizes *= minor_data_proportion
            for i in range(client_num):
                partition_sizes[i % cfg['classes_size']][i] = most_data_proportion
        else:
            raise ValueError('Non-IID ratio is too large')
    else:
        raise ValueError('Not valid partition pattern')

    train_data_partition = datasets.LabelwisePartitioner(
        train_dataset, partition_sizes=partition_sizes, seed=cfg['data_partition_seed']
    )

    return train_data_partition, partition_sizes


if __name__ == "__main__":
    main()
