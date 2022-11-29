import os

import numpy as np

from dataset.ctr_dataset import Criteo2party, Avazu2party
from fedcvt_core.fedcvt_parties import VFTLGuest, VFLHost, PartyDataLoader
from fedcvt_core.param import PartyModelParam, FederatedModelParam
from models.ctr_models import DNNFM
from run_experiment import batch_run_experiments


class VFTLGuestConstructor(object):

    def __init__(self, party_param: PartyModelParam, fed_model_param: FederatedModelParam):
        self.party_param = party_param
        self.fed_model_param = fed_model_param

    def build(self, X_train, Y_train, X_test, Y_test, args, debug=False):
        print("[INFO] # ===> Build Guest Local Model.")

        hidden_dim_list = self.party_param.hidden_dim_list
        data_folder = self.party_param.data_folder

        overlap_indices = self.fed_model_param.overlap_indices
        non_overlap_indices = self.fed_model_param.non_overlap_indices

        guest_non_overlap_indices = non_overlap_indices
        guest_all_indices = np.concatenate([overlap_indices, guest_non_overlap_indices])

        col_names = args["col_names"][0]
        nn_prime = DNNFM(0, col_names, col_names, dnn_activation="leakyrelu", init_std=0.0001)
        nn = DNNFM(1, col_names, col_names, dnn_activation="leakyrelu", init_std=0.0001)
        # nn_prime.build(input_dim=X_train.shape[1], hidden_dim=None, output_dim=hidden_dim_list[-1])
        # nn.build(input_dim=X_train.shape[1], hidden_dim=None, output_dim=hidden_dim_list[-1])
        print("[INFO] Guest NN_prime:")
        print(nn_prime)
        print("[INFO] Guest NN:")
        print(nn)

        guest_data_loader = PartyDataLoader(data_folder_path=data_folder,
                                            is_guest=True,
                                            X_ll_train=X_train[overlap_indices],
                                            Y_ll_train=Y_train[overlap_indices],
                                            X_nol_train=X_train[guest_non_overlap_indices],
                                            Y_nol_train=Y_train[guest_non_overlap_indices],
                                            X_ested_train=X_train[guest_all_indices],
                                            Y_ested_train=Y_train[guest_all_indices],
                                            X_test=X_test,
                                            Y_test=Y_test)
        guest = VFTLGuest(party_model_param=self.party_param, data_loader=guest_data_loader, debug=debug)
        guest.set_model(nn, nn_prime)

        return guest


class VFTLHostConstructor(object):

    def __init__(self, party_param: PartyModelParam, fed_model_param: FederatedModelParam):
        self.party_param = party_param
        self.fed_model_param = fed_model_param

    def build(self, X_train, X_test, args, debug=False):
        print("[INFO] # ===> Build Host Local Model.")

        data_folder = self.party_param.data_folder

        overlap_indices = self.fed_model_param.overlap_indices
        non_overlap_indices = self.fed_model_param.non_overlap_indices

        host_non_overlap_indices = non_overlap_indices
        host_all_indices = np.concatenate([overlap_indices, host_non_overlap_indices])

        col_names = args["col_names"][1]
        nn_prime = DNNFM(2, col_names, col_names, dnn_activation="leakyrelu", init_std=0.0001)
        nn = DNNFM(3, col_names, col_names, dnn_activation="leakyrelu", init_std=0.0001)
        # nn_prime.build(input_dim=X_train.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim_list[-1])
        # nn.build(input_dim=X_train.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim_list[-1])
        print("[INFO] Host NN_prime:")
        print(nn_prime)
        print("[INFO] Host NN:")
        print(nn)

        host_data_loader = PartyDataLoader(data_folder_path=data_folder,
                                           is_guest=False,
                                           X_ll_train=X_train[overlap_indices],
                                           X_nol_train=X_train[host_non_overlap_indices],
                                           X_ested_train=X_train[host_all_indices],
                                           X_test=X_test)
        host = VFLHost(party_model_param=self.party_param, data_loader=host_data_loader, debug=debug)
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
    data_train = Avazu2party(data_dir, 'Train', 2, 32)
    data_test = Avazu2party(data_dir, 'Test', 2, 32)
    x_train, y_train = data_train.get_data()
    x_test, y_test = data_test.get_data()
    col_names = data_train.feature_list

    X_guest_train = np.array(x_train[0])
    X_host_train = np.array(x_train[1])
    Y_train = np.array(y_train)
    Y_train = np.eye(2)[Y_train]  # convert to one hot vectors

    X_guest_test = np.array(x_test[0])
    X_host_test = np.array(x_test[1])
    Y_test = np.array(y_test)
    Y_test = np.eye(2)[Y_test]  # convert to one hot vectors

    print("### original data shape")
    print("X_guest_train shape", X_guest_train.shape)
    print("X_host_train shape", X_host_train.shape)
    print("Y_train shape", Y_train.shape)
    print("X_guest_test shape", X_guest_test.shape)
    print("X_host_test shape", X_host_test.shape)
    print("Y_test shape", Y_test.shape)

    # =====================================
    # Prepare optimizer hyper-parameters
    # =====================================

    learning_rate_list = [0.001]
    # learning_rate_list = [0.01]

    optim_args = dict()
    optim_args["weight_decay"] = 0
    # optim_args["weight_decay"] = 1e-6
    optim_args["learning_rate_list"] = learning_rate_list

    # =====================================
    # Prepare training hyper-parameters
    # =====================================

    epoch = 20
    estimation_block_size = 5000
    # overlap_sample_batch_size = 128
    # non_overlap_sample_batch_size = 128
    # overlap_sample_batch_size = 256
    # non_overlap_sample_batch_size = 256
    overlap_sample_batch_size = 512
    non_overlap_sample_batch_size = 512
    sharpen_temperature = 0.5
    is_hetero_reprs = False

    # label_prob_sharpen_temperature = 1.0
    # fed_label_prob_threshold = 0.5
    # host_label_prob_threshold = 0.5
    label_prob_sharpen_temperature = 0.1
    fed_label_prob_threshold = 0.9
    host_label_prob_threshold = 0.9

    num_overlap_list = [600]
    # num_overlap_list = [500]
    training_args = dict()
    training_args["epoch"] = epoch
    training_args["num_overlap_list"] = num_overlap_list
    training_args["overlap_sample_batch_size"] = overlap_sample_batch_size
    training_args["non_overlap_sample_batch_size"] = non_overlap_sample_batch_size
    training_args["estimation_block_size"] = estimation_block_size
    training_args["sharpen_temperature"] = sharpen_temperature
    training_args["is_hetero_reprs"] = is_hetero_reprs
    training_args["label_prob_sharpen_temperature"] = label_prob_sharpen_temperature
    training_args["fed_label_prob_threshold"] = fed_label_prob_threshold
    training_args["host_label_prob_threshold"] = host_label_prob_threshold

    training_args["normalize_repr"] = False
    training_args["epoch"] = 10

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
    lambda_sim_shared_reprs_vs_uniq_reprs = [0.1]
    lambda_host_dist_ested_lbls_vs_true_lbls = [100]
    # lambda_host_dist_ested_lbls_vs_true_lbls = [1]
    lambda_dist_ested_reprs_vs_true_reprs = [0.1]
    lambda_host_dist_two_ested_lbls = [0.01]

    loss_weight_args = dict()
    loss_weight_args["lambda_dist_shared_reprs"] = lambda_dis_shared_reprs
    loss_weight_args["lambda_sim_shared_reprs_vs_uniq_reprs"] = lambda_sim_shared_reprs_vs_uniq_reprs
    loss_weight_args["lambda_host_dist_ested_lbls_vs_true_lbls"] = lambda_host_dist_ested_lbls_vs_true_lbls
    loss_weight_args["lambda_dist_ested_reprs_vs_true_reprs"] = lambda_dist_ested_reprs_vs_true_reprs
    loss_weight_args["lambda_host_dist_two_ested_lbls"] = lambda_host_dist_two_ested_lbls

    other_args = {"col_names": col_names}

    batch_run_experiments(X_guest_train, X_host_train, Y_train, X_guest_test, X_host_test, Y_test,
                          optim_args, loss_weight_args, training_args, other_args)
