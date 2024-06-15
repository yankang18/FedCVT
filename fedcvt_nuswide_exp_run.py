import numpy as np
import torch

from dataset.nuswide_dataset import NUSWIDEDataset2Party
from fedcvt_core.fedcvt_parties import VFTLGuest, VFLHost
from fedcvt_core.param import PartyModelParam, FederatedTrainingParam
from models.mlp_models import SoftmaxRegression
from run_experiment import batch_run_experiments
from utils import set_seed


def convert_dict_to_str_row(input_dict):
    keys = input_dict.keys()
    values = input_dict.values()

    str_keys = ' '.join(keys)
    str_values = ' '.join(str(e) for e in values)
    return str_keys, str_values


class VFTLGuestConstructor(object):

    def __init__(self, party_param: PartyModelParam, fed_model_param: FederatedTrainingParam):
        self.party_param = party_param
        self.fed_model_param = fed_model_param

    def build(self, args=None, debug=False):
        print("[INFO] # ===> Build Guest Local Model.")

        device = args["device"]
        hidden_dim_list = self.party_param.hidden_dim_list

        input_dim = args["guest_input_dim"]
        nn_prime = SoftmaxRegression(0).to(device)
        nn = SoftmaxRegression(1).to(device)
        nn_prime.build(input_dim=input_dim, hidden_dim=None, output_dim=hidden_dim_list[-1])
        nn.build(input_dim=input_dim, hidden_dim=None, output_dim=hidden_dim_list[-1])

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

    def build(self, args=None, debug=False):
        print("[INFO] # ===> Build Host Local Model.")

        device = args["device"]
        hidden_dim_list = self.party_param.hidden_dim_list

        input_dim = args["host_input_dim"]
        hidden_dim = None
        nn_prime = SoftmaxRegression(2).to(device)
        nn = SoftmaxRegression(3).to(device)
        nn_prime.build(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim_list[-1])
        nn.build(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim_list[-1])
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
        else:
            idx_valid_sample_list.append(idx)

    print("number of all-zero text sample:", len(idx_invalid_sample_list))
    print("number of not all-zero text sample:", len(idx_valid_sample_list))
    print("total number of samples: ", len(idx_invalid_sample_list) + len(idx_valid_sample_list))
    return idx_valid_sample_list


if __name__ == "__main__":
    DATA_DIR = "../../dataset/"
    sel_lbls = ['sky', 'clouds', 'person', 'water', 'animal',
                'grass', 'buildings', 'window', 'plants', 'lake']

    train_dataset = NUSWIDEDataset2Party(DATA_DIR, sel_lbls, 'Train', 2)
    test_dataset = NUSWIDEDataset2Party(DATA_DIR, sel_lbls, 'Test', 2)

    # =====================================
    # Prepare optimizer hyper-parameters
    # =====================================

    # learning_rate_list = [0.0006]
    learning_rate_list = [0.005]
    # learning_rate_list = [0.005]

    optim_args = dict()
    # optim_args["weight_decay"] = 0
    optim_args["weight_decay"] = 1e-5
    optim_args["learning_rate_list"] = learning_rate_list

    # =====================================
    # Prepare training hyper-parameters
    # =====================================

    epoch = 10
    estimation_block_size = 5000

    ll_overlap_sample_batch_size = 256
    ul_overlap_sample_batch_size = 512
    non_overlap_sample_batch_size = 512

    sharpen_temperature = 0.7
    is_hetero_reprs = False

    # label_prob_sharpen_temperature = 0.1
    # fed_label_prob_threshold = 0.8
    # guest_label_prob_threshold = 0.8
    # host_label_prob_threshold = 0.8

    label_prob_sharpen_temperature = 0.1
    fed_label_prob_threshold = 0.8
    guest_label_prob_threshold = 0.7
    host_label_prob_threshold = 0.7

    num_overlap_list = [250]
    # num_overlap_list = [250, 500, 1000, 2000, 4000]
    num_labeled_overlap_list = num_overlap_list
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
    training_args["epoch"] = 20

    training_args["hidden_dim"] = 96
    training_args["num_class"] = 10

    training_args["vfl_guest_constructor"] = VFTLGuestConstructor
    training_args["vfl_host_constructor"] = VFTLHostConstructor

    # =====================================
    # Prepare loss hyper-parameters
    # =====================================

    lambda_dis_shared_reprs = [0.1]
    lambda_sim_shared_reprs_vs_uniq_reprs = [0.01]
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

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set gpu
    gpu_device_id = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_device_id)
        print('gpu device = %d' % gpu_device_id)

    seed_list = [0, 1, 2, 3, 4]
    for seed in seed_list:
        set_seed(seed)
        training_args["seed"] = seed

        other_args = {"guest_input_dim": 634, "host_input_dim": 1000, "aggregation_mode": "cat",
                      "seed": seed, "name": "nuswide", "monitor_metric": "acc", "device": device,
                      "valid_iteration_interval": 3}
        batch_run_experiments(train_dataset, test_dataset, optim_args, loss_weight_args, training_args, other_args)
