import json
import time

import tensorflow as tf

from cnn_models import ClientVGG8
from expanding_vertical_transfer_learning_param import PartyModelParam, FederatedModelParam
from vertical_semi_supervised_transfer_learning_v4 import VerticalFederatedTransferLearning
from vertical_sstl_parties import ExpandingVFTLGuest, ExpandingVFTLHost, ExpandingVFTLDataLoader
from vertical_sstl_representation_learner import AttentionBasedRepresentationEstimator


def get_timestamp():
    local_time = time.localtime(time.time())
    timestamp = time.strftime("%Y%m%d%H%M%S", local_time)
    return timestamp


class ExpandingVFTLGuestConstructor(object):

    def __init__(self, party_param: PartyModelParam):
        self.guest = None
        self.party_param = party_param

    def build(self, data_folder, input_shape):
        print("Guest Setup")

        nn_prime = ClientVGG8("cnn_0")
        nn_prime.build(input_shape=input_shape)

        nn = ClientVGG8("cnn_1")
        nn.build(input_shape=input_shape)

        guest_data_loader = ExpandingVFTLDataLoader(data_folder_path=data_folder,
                                                    is_guest=True)

        self.guest = ExpandingVFTLGuest(party_model_param=self.party_param,
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

    def build(self, data_folder, input_shape):
        print("Host Setup")

        nn_prime = ClientVGG8("cnn_2")
        nn_prime.build(input_shape=input_shape)

        nn = ClientVGG8("cnn_3")
        nn.build(input_shape=input_shape)

        host_data_loader = ExpandingVFTLDataLoader(data_folder_path=data_folder,
                                                   is_guest=False)

        self.host = ExpandingVFTLHost(party_model_param=self.party_param,
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
    dataset_folder_path = "../../data/cifar-10-batches-py_500/"
    print("{0} dataset_folder_path: {1}".format(tag_PATH, dataset_folder_path))

    file_folder = "training_log_info/"
    timestamp = get_timestamp()
    file_name = file_folder + "test_csv_read_" + timestamp + ".csv"

    # configuration
    combine_axis = 1

    # num_non_overlap = num_train - num_overlap

    # guest_model_param = PartyModelParam(n_class=10)
    # host_model_param = PartyModelParam(n_class=10)
    guest_model_param = PartyModelParam(n_class=10, keep_probability=0.75, apply_dropout=True)
    host_model_param = PartyModelParam(n_class=10, keep_probability=0.75, apply_dropout=True)

    # print("combine_axis:", combine_axis)
    input_dim = 48 * 2 * 2
    guest_input_dim = int(input_dim / 2)
    hidden_dim = None
    guest_hidden_dim = None
    parallel_iterations = 100
    epoch = 20

    overlap_sample_batch_size = 128
    non_overlap_sample_batch_size = 128

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
                                          fed_hidden_dim=hidden_dim,
                                          guest_hidden_dim=None,
                                          using_block_idx=True,
                                          learning_rate=0.001,
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
                                          fed_label_prob_threshold=0.7,
                                          host_label_prob_threshold=0.7,
                                          training_info_file_name=file_name)

    # set up and train model
    guest_constructor = ExpandingVFTLGuestConstructor(guest_model_param)
    host_constructor = ExpandingVFTLHostConstructor(host_model_param)

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    # input_shape = (28, 14, 1)
    input_shape = (32, 16, 3)
    guest = guest_constructor.build(data_folder=dataset_folder_path,
                                    input_shape=input_shape)
    host = host_constructor.build(data_folder=dataset_folder_path,
                                  input_shape=input_shape)

    VFTL = VerticalFederatedTransferLearning(guest, host, fed_model_param)
    VFTL.set_representation_estimator(AttentionBasedRepresentationEstimator())
    VFTL.build()
    VFTL.train(debug=False)
