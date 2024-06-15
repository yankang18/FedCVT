import numpy as np
import torch

from dataset.bhi_dataset import BHIDataset2Party
from fedcvt_core.fedcvt_parties import VFTLGuest, VFLHost
from fedcvt_core.param import PartyModelParam, FederatedTrainingParam
from models.cnn_models import MyResnet18
from run_experiment import batch_run_experiments
from utils import set_seed


class VFTLGuestConstructor(object):

    def __init__(self, party_param: PartyModelParam, fed_model_param: FederatedTrainingParam):
        self.party_param = party_param
        self.fed_model_param = fed_model_param

    def build(self, args=None, debug=False):
        print("[INFO] # ===> Build Guest Local Model.")

        device = args["device"]
        hidden_dim_list = self.party_param.hidden_dim_list
        n_classes = self.party_param.n_classes

        print("hidden_dim_list[-1]:", hidden_dim_list[-1])
        nn_prime = MyResnet18(an_id=0, class_num=n_classes, output_dim=hidden_dim_list[-1]).to(device)
        nn = MyResnet18(an_id=1, class_num=n_classes, output_dim=hidden_dim_list[-1]).to(device)
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
        n_classes = self.party_param.n_classes

        print("hidden_dim_list[-1]:", hidden_dim_list[-1])
        nn_prime = MyResnet18(an_id=2, class_num=n_classes, output_dim=hidden_dim_list[-1]).to(device)
        nn = MyResnet18(an_id=3, class_num=n_classes, output_dim=hidden_dim_list[-1]).to(device)
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
    file_dir = "../../dataset/bhi/"

    seed_list = [0]
    for seed in seed_list:

        train_dataset = BHIDataset2Party(file_dir, 'train', 32, 32, 2, seed)
        test_dataset = BHIDataset2Party(file_dir, 'test', 32, 32, 2, seed)

        # =====================================
        # Prepare optimizer hyper-parameters
        # =====================================

        learning_rate_list = [0.001]
        # learning_rate_list = [0.05]
        # learning_rate_list = [0.01]

        optim_args = dict()
        # optim_args["weight_decay"] = 0
        optim_args["weight_decay"] = 0.0
        optim_args["learning_rate_list"] = learning_rate_list

        # =====================================
        # Prepare training hyper-parameters
        # =====================================

        epoch = 80
        estimation_block_size = 4000

        ll_overlap_sample_batch_size = 256
        ul_overlap_sample_batch_size = 512
        non_overlap_sample_batch_size = 512

        sharpen_temperature = 0.5
        is_hetero_reprs = False
        aggregation_mode = "cat"

        label_prob_sharpen_temperature = 0.1
        fed_label_prob_threshold = 0.9
        guest_label_prob_threshold = 0.9
        host_label_prob_threshold = 0.9

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

        training_args["data_type"] = "img"
        training_args["normalize_repr"] = True
        training_args["only_use_ll"] = False
        training_args["epoch"] = epoch

        training_args["hidden_dim"] = 128
        training_args["num_class"] = 2

        training_args["vfl_guest_constructor"] = VFTLGuestConstructor
        training_args["vfl_host_constructor"] = VFTLHostConstructor

        # =====================================
        # Prepare loss hyper-parameters
        # =====================================

        lambda_dis_shared_reprs = [0.1]
        lambda_sim_shared_reprs_vs_uniq_reprs = [0.1]
        lambda_host_dist_ested_lbls_vs_true_lbls = [1]
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
            print('[INFO] using gpu device = %d' % gpu_device_id)
        else:
            print('[INFO] using cpu.')

        # seed_list = [0, 1]
        # seed = 0
        set_seed(seed)
        other_args = {"seed": seed, "name": "bhi", "aggregation_mode": aggregation_mode,
                      "monitor_metric": "fscore", "device": device, "valid_iteration_interval": 1}
        batch_run_experiments(train_dataset, test_dataset, optim_args, loss_weight_args, training_args, other_args)
