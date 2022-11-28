import json

import torch
import torch.backends.cudnn as cudnn

from fedcvt_core.param import PartyModelParam, FederatedModelParam
from fedcvt_core.fedcvt_parties import VFTLGuest, VFLHost, PartyDataLoader
from fedcvt_core.fedcvt_repr_estimator import AttentionBasedRepresentationEstimator
from fedcvt_core.fedcvt_train import VerticalFederatedTransferLearning
from models.cnn_models import ClientVGG8
from utils import get_timestamp


class ExpandingVFTLGuestConstructor(object):

    def __init__(self, party_param: PartyModelParam):
        self.guest = None
        self.party_param = party_param
        self.device = party_param.device

    def build(self, data_folder, input_shape):
        print("Guest Setup")

        nn_prime = ClientVGG8("cnn_0").to(self.device)
        nn = ClientVGG8("cnn_1").to(self.device)

        guest_data_loader = PartyDataLoader(data_folder_path=data_folder,
                                            is_guest=True)

        self.guest = VFTLGuest(party_model_param=self.party_param,
                               data_loader=guest_data_loader)
        self.guest.set_model(nn, nn_prime)

        with open(data_folder + "meta_data.json", "r") as read_file:
            meta_data = json.load(read_file)

        val_block_num = meta_data["guest_val_block_num"]
        overlap_block_num = meta_data["guest_overlap_block_num"]
        guest_nonoverlap_block_num = meta_data["guest_non-overlap_block_num"]
        guest_ested_block_num = meta_data["guest_estimation_block_num"]

        print("val_block_num {0}".format(val_block_num))
        print("overlap_block_num {0}".format(overlap_block_num))
        print("guest_nonoverlap_block_num {0}".format(guest_nonoverlap_block_num))
        print("guest_ested_block_num {0}".format(guest_ested_block_num))

        self.guest.set_val_block_number(val_block_num)
        self.guest.set_ol_block_number(overlap_block_num)
        self.guest.set_nol_block_number(guest_nonoverlap_block_num)
        self.guest.set_ested_block_number(guest_ested_block_num)

        return self.guest


class ExpandingVFTLHostConstructor(object):

    def __init__(self, party_param: PartyModelParam):
        self.host = None
        self.party_param = party_param
        self.device = party_param.device

    def build(self, data_folder):
        print("Host Setup")

        nn_prime = ClientVGG8("cnn_2").to(self.device)
        nn = ClientVGG8("cnn_3").to(self.device)

        host_data_loader = PartyDataLoader(data_folder_path=data_folder,
                                           is_guest=False)

        self.host = VFLHost(party_model_param=self.party_param,
                            data_loader=host_data_loader)
        self.host.set_model(nn, nn_prime)

        with open(data_folder + "meta_data.json", "r") as read_file:
            meta_data = json.load(read_file)

        val_block_num = meta_data["host_val_block_num"]
        overlap_block_num = meta_data["host_overlap_block_num"]
        host_nonoverlap_block_num = meta_data["host_non-overlap_block_num"]
        host_ested_block_num = meta_data["host_estimation_block_num"]

        print("val_block_num {0}".format(val_block_num))
        print("overlap_block_num {0}".format(overlap_block_num))
        print("host_nonoverlap_block_num {0}".format(host_nonoverlap_block_num))
        print("host_ested_block_num {0}".format(host_ested_block_num))

        self.host.set_val_block_number(val_block_num)
        self.host.set_ol_block_number(overlap_block_num)
        self.host.set_nol_block_number(host_nonoverlap_block_num)
        self.host.set_ested_block_number(host_ested_block_num)

        return self.host


tag_PATH = "[INFO]"
if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = 7

    print(f"[INFO] device : {device}; GPU:{gpu}")
    if torch.cuda.is_available():
        print("[INFO] cuda is available")
        torch.cuda.set_device(gpu)
        cudnn.benchmark = True
        cudnn.enabled = True

    # dataset_folder_path = "../../data/cifar-10-batches-py_500/"
    dataset_folder_path = '/Users/yankang/Documents/Data/cifar-10-batches-py_500/'
    print("{0} dataset_folder_path: {1}".format(tag_PATH, dataset_folder_path))

    file_folder = "training_log_info/"
    timestamp = get_timestamp()
    file_name = file_folder + "test_csv_read_" + timestamp + ".csv"

    # configuration
    combine_axis = 1

    guest_model_param = PartyModelParam(n_class=10, keep_probability=0.75, apply_dropout=True,
                                        data_type="img", device=device)
    host_model_param = PartyModelParam(n_class=10, keep_probability=0.75, apply_dropout=True,
                                       data_type="img", device=device)

    # print("combine_axis:", combine_axis)
    input_dim = 48 * 2 * 2
    guest_input_dim = int(input_dim / 2)
    hidden_dim = None
    guest_hidden_dim = None
    parallel_iterations = 100
    epoch = 20

    overlap_sample_batch_size = 128
    non_overlap_sample_batch_size = 128
    learning_rate = 0.001
    # loss_weight_list = [1.0, 0.01, 0.01, 500, 0.1, 0.1, 0.1]
    # loss_weight_list = [100, 0.1, 0.1, 1000, 0.1, 0.1, 0.1]
    # loss_weight_list = [0.01, 0.001, 0.001, 100, 0.1, 0.1, 0.1]
    # loss_weight_list = [1.0, 0.1, 0.1, 1000, 0.1, 0.1, 0.1]
    # loss_weight_list = [0.1, 0.01, 0.01, 1000, 1000, 0.1, 0.1, 0.1]
    loss_weight_dict = {"lambda_dist_shared_reprs": 0.1,
                        "lambda_guest_sim_shared_reprs_vs_unique_repr": 0.01,
                        "lambda_host_sim_shared_reprs_vs_unique_repr": 0.01,
                        "lambda_host_dist_ested_uniq_lbl_vs_true_lbl": 1000,
                        "lambda_host_dist_ested_comm_lbl_vs_true_lbl": 1000,
                        "lambda_guest_dist_ested_repr_vs_true_repr": 0.1,
                        "lambda_host_dist_ested_repr_vs_true_repr": 0.1,
                        "lambda_host_dist_two_ested_lbl": 0.1}

    fed_model_param = FederatedModelParam(fed_input_dim=input_dim,
                                          guest_input_dim=guest_input_dim,
                                          host_input_dim=guest_input_dim,
                                          fed_hidden_dim=hidden_dim,
                                          guest_hidden_dim=None,
                                          using_block_idx=True,
                                          learning_rate=learning_rate,
                                          fed_reg_lambda=0.001,
                                          guest_reg_lambda=0.0,
                                          loss_weight_dict=loss_weight_dict,
                                          # overlap_indices=overlap_indices,
                                          epoch=epoch,
                                          top_k=1,
                                          combine_axis=combine_axis,
                                          parallel_iterations=parallel_iterations,
                                          overlap_sample_batch_size=overlap_sample_batch_size,
                                          non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                          all_sample_block_size=5000,
                                          is_hetero_repr=False,
                                          sharpen_temperature=0.1,
                                          fed_label_prob_threshold=0.5,
                                          host_label_prob_threshold=0.3,
                                          training_info_file_name=file_name,
                                          device=device)

    # set up and train model
    guest_constructor = ExpandingVFTLGuestConstructor(guest_model_param)
    host_constructor = ExpandingVFTLHostConstructor(host_model_param)

    input_shape = (32, 16, 3)
    guest = guest_constructor.build(data_folder=dataset_folder_path,
                                    input_shape=input_shape)
    host = host_constructor.build(data_folder=dataset_folder_path,
                                  input_shape=input_shape)

    VFTL = VerticalFederatedTransferLearning(guest, host, fed_model_param, debug=False)
    VFTL.set_representation_estimator(AttentionBasedRepresentationEstimator())
    VFTL.build()
    VFTL.train()
