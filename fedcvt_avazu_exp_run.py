import os

import numpy as np
import torch

from dataset.ctr_dataset import Avazu2party
from fedcvt_core.fedcvt_parties import VFTLGuest, VFLHost
from fedcvt_core.param import PartyModelParam, FederatedTrainingParam
from models.ctr_models import DNNFM
from run_experiment import batch_run_experiments
from utils import set_seed


class VFTLGuestConstructor(object):

    def __init__(self, party_param: PartyModelParam, fed_model_param: FederatedTrainingParam):
        self.party_param = party_param
        self.fed_model_param = fed_model_param

    def build(self, args, debug=False):
        print("[INFO] ===> Build Guest Local Model.")

        seed = args["seed"]
        device = args["device"]

        l2_reg_linear = 1e-4
        l2_reg_embedding = 1e-4
        l2_reg_dnn = 1e-4
        init_std = 0.00001
        col_names = args["col_names"][0]
        nn_prime = DNNFM(0, col_names, col_names, dnn_activation="leakyrelu", init_std=init_std,
                         l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
                         seed=seed).to(device)
        nn = DNNFM(1, col_names, col_names, dnn_activation="leakyrelu", init_std=init_std,
                   l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
                   seed=seed).to(device)
        print("[INFO] Guest NN_prime:")
        print(nn_prime)
        print("[INFO] Guest NN:")
        print(nn)

        guest = VFTLGuest(party_model_param=self.party_param, debug=debug)
        guest.set_model(nn, nn_prime)

        return guest


class VFTLHostConstructor(object):

    def __init__(self, party_param: PartyModelParam, fed_model_param: FederatedTrainingParam):
        self.party_param = party_param
        self.fed_model_param = fed_model_param

    def build(self, args, debug=False):
        print("[INFO] ===> Build Host Local Model.")

        device = args["device"]
        seed = args["seed"]

        l2_reg_linear = 1e-4
        l2_reg_embedding = 1e-4
        l2_reg_dnn = 1e-4
        init_std = 0.00001
        col_names = args["col_names"][1]
        nn_prime = DNNFM(2, col_names, col_names, dnn_activation="leakyrelu", init_std=init_std,
                         l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
                         seed=seed).to(device)
        nn = DNNFM(3, col_names, col_names, dnn_activation="leakyrelu", init_std=init_std,
                   l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
                   seed=seed).to(device)
        print("[INFO] Host NN_prime:")
        print(nn_prime)
        print("[INFO] Host NN:")
        print(nn)

        host = VFLHost(party_model_param=self.party_param, debug=debug)
        host.set_model(nn, nn_prime)

        return host


def get_valid_sample_indices(data):
    idx_valid_sample_list = []
    idx_invalid_sample_list = []
    for idx, data in enumerate(data):
        if np.all(data == 0):
            idx_invalid_sample_list.append(idx)
            # print("X_text_all_i", idx, np.sum(X_text_all_i), len(X_text_all_i))
        else:
            idx_valid_sample_list.append(idx)

    print("number of all-zero text sample:", len(idx_invalid_sample_list))
    print("number of not all-zero text sample:", len(idx_valid_sample_list))
    print("total number of samples: ", len(idx_invalid_sample_list) + len(idx_valid_sample_list))
    return idx_valid_sample_list


if __name__ == "__main__":
    data_dir = "../../dataset/"
    input_dims = [11, 10]
    num_classes = 2
    # sub_data_dir = "criteo"
    sub_data_dir = "avazu"
    data_dir = os.path.join(data_dir, sub_data_dir)
    # data_train = Criteo2party(data_dir=data_dir, data_type='Train', k=2, input_size=32)
    # data_test = Criteo2party(data_dir=data_dir, data_type='Test', k=2, input_size=32)
    train_dataset = Avazu2party(data_dir, 'Train', 2, 32)
    test_dataset = Avazu2party(data_dir, 'Test', 2, 32)
    # x_train, y_train = data_train.get_data()
    # x_test, y_test = data_test.get_data()
    col_names = train_dataset.feature_list

    # X_guest_train = np.array(x_train[0])
    # X_host_train = np.array(x_train[1])
    # Y_train = np.array(y_train)
    # Y_train = np.eye(2)[Y_train]  # convert to one hot vectors
    #
    # X_guest_test = np.array(x_test[0])
    # X_host_test = np.array(x_test[1])
    # Y_test = np.array(y_test)
    # Y_test = np.eye(2)[Y_test]  # convert to one hot vectors

    # print("### original data shape")
    # print("X_guest_train shape", X_guest_train.shape)
    # print("X_host_train shape", X_host_train.shape)
    # print("Y_train shape", Y_train.shape)
    # print("X_guest_test shape", X_guest_test.shape)
    # print("X_host_test shape", X_host_test.shape)
    # print("Y_test shape", Y_test.shape)

    # =====================================
    # Prepare optimizer hyper-parameters
    # =====================================

    learning_rate_list = [0.0002]
    # learning_rate_list = [0.001]
    # learning_rate_list = [0.01]

    optim_args = dict()
    # optim_args["weight_decay"] = 1e-5
    optim_args["weight_decay"] = 1e-7
    optim_args["learning_rate_list"] = learning_rate_list

    # =====================================
    # Prepare training hyper-parameters
    # =====================================

    epoch = 10
    estimation_block_size = 5000
    ll_overlap_sample_batch_size = 512
    ul_overlap_sample_batch_size = 512
    non_overlap_sample_batch_size = 512
    # ul_overlap_sample_batch_size = 1024
    # ll_overlap_sample_batch_size = 1024
    # non_overlap_sample_batch_size = 1024
    sharpen_temperature = 0.5
    is_hetero_reprs = False

    label_prob_sharpen_temperature = 0.1
    fed_label_prob_threshold = 0.9
    guest_label_prob_threshold = 0.6
    host_label_prob_threshold = 0.5

    # label_prob_sharpen_temperature = 0.1
    # fed_label_prob_threshold = 0.8
    # guest_label_prob_threshold = 0.8
    # host_label_prob_threshold = 0.8

    # num_overlap_list = [20000]
    num_overlap_list = [40000]
    num_labeled_overlap_list = [200, 400, 600, 800, 1000]
    # num_overlap_list = [600]
    # num_overlap_list = [500]
    training_args = dict()
    training_args["epoch"] = epoch
    training_args["num_overlap_list"] = num_overlap_list
    training_args["num_labeled_overlap_list"] = num_labeled_overlap_list
    training_args["ul_overlap_sample_batch_size"] = ul_overlap_sample_batch_size
    training_args["ll_overlap_sample_batch_size"] = ll_overlap_sample_batch_size
    training_args["non_overlap_sample_batch_size"] = non_overlap_sample_batch_size
    training_args["estimation_block_size"] = estimation_block_size
    training_args["sharpen_temperature"] = sharpen_temperature
    training_args["is_hetero_reprs"] = is_hetero_reprs
    training_args["label_prob_sharpen_temperature"] = label_prob_sharpen_temperature
    training_args["fed_label_prob_threshold"] = fed_label_prob_threshold
    training_args["guest_label_prob_threshold"] = guest_label_prob_threshold
    training_args["host_label_prob_threshold"] = host_label_prob_threshold

    training_args["data_type"] = "tab"
    training_args["normalize_repr"] = False
    training_args["epoch"] = epoch

    training_args["hidden_dim"] = 64
    training_args["num_class"] = num_classes

    training_args["vfl_guest_constructor"] = VFTLGuestConstructor
    training_args["vfl_host_constructor"] = VFTLHostConstructor

    # =====================================
    # Prepare loss hyper-parameters
    # =====================================

    # lambda_dis_shared_reprs = [0.1]
    # lambda_sim_shared_reprs_vs_uniq_reprs = [0.1]
    # lambda_host_dis_ested_lbls_vs_true_lbls = [100]
    # lambda_dis_ested_reprs_vs_true_reprs = [0.1]
    # lambda_host_dist_two_ested_lbls = [0.01]
    # learning_rate = [0.01]
    lambda_dis_shared_reprs = [0.1]
    lambda_sim_shared_reprs_vs_uniq_reprs = [0.01]
    lambda_host_dist_ested_lbls_vs_true_lbls = [100]
    # lambda_host_dist_ested_lbls_vs_true_lbls = [1]
    lambda_dist_ested_reprs_vs_true_reprs = [1.0]
    lambda_host_dist_two_ested_lbls = [0.01]

    loss_weight_args = dict()
    loss_weight_args["lambda_dist_shared_reprs"] = lambda_dis_shared_reprs
    loss_weight_args["lambda_sim_shared_reprs_vs_uniq_reprs"] = lambda_sim_shared_reprs_vs_uniq_reprs
    loss_weight_args["lambda_host_dist_ested_lbls_vs_true_lbls"] = lambda_host_dist_ested_lbls_vs_true_lbls
    loss_weight_args["lambda_dist_ested_reprs_vs_true_reprs"] = lambda_dist_ested_reprs_vs_true_reprs
    loss_weight_args["lambda_host_dist_two_ested_lbls"] = lambda_host_dist_two_ested_lbls

    # batch_run_experiments(X_guest_train, X_host_train, Y_train,
    #                       X_guest_test, X_host_test, Y_test,
    #                       optim_args, loss_weight_args, training_args, other_args)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set gpu
    gpu_device_id = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_device_id)
        print('gpu device = %d' % gpu_device_id)

    seed_list = [0, 1, 2]
    for seed in seed_list:
        set_seed(seed)

        other_args = {"col_names": col_names, "seed": seed, "name": "avazu", "aggregation_mode": "cat",
                      "monitor_metric": "auc", "device": device, "valid_iteration_interval": 3}
        batch_run_experiments(train_dataset, test_dataset, optim_args, loss_weight_args, training_args, other_args)
