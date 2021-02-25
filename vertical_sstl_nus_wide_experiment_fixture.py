import csv
import time

import numpy as np
import tensorflow as tf

from autoencoder import Autoencoder
from data_util.nus_wide_processed_data_util import TwoPartyNusWideDataLoader
from expanding_vertical_transfer_learning_param import PartyModelParam, FederatedModelParam
from vertical_semi_supervised_transfer_learning_v4 import VerticalFederatedTransferLearning
from vertical_sstl_parties import ExpandingVFTLGuest, ExpandingVFTLHost, ExpandingVFTLDataLoader
from vertical_sstl_representation_learner import AttentionBasedRepresentationEstimator


def get_timestamp():
    local_time = time.localtime(time.time())
    timestamp = time.strftime("%Y%m%d%H%M%S", local_time)
    return timestamp


def convert_dict_to_str_row(input_dict):
    keys = input_dict.keys()
    values = input_dict.values()

    str_keys = ' '.join(keys)
    str_values = ' '.join(str(e) for e in values)
    return str_keys, str_values


class ExpandingVFTLGuestConstructor(object):

    def __init__(self, party_param: PartyModelParam, fed_model_param: FederatedModelParam):
        self.guest = None
        self.party_param = party_param
        self.fed_model_param = fed_model_param

    def build(self, X_train, Y_train, X_test, Y_test):
        print("[INFO] => Build Guest.")

        hidden_dim_list = self.party_param.hidden_dim_list
        learning_rate = self.party_param.nn_learning_rate
        data_folder = self.party_param.data_folder

        num_guest_nonoverlap_samples = self.fed_model_param.num_guest_nonoverlap_samples
        overlap_indices = self.fed_model_param.overlap_indices
        non_overlap_indices = self.fed_model_param.non_overlap_indices
        guest_non_overlap_indices = non_overlap_indices[:num_guest_nonoverlap_samples]

        nn_prime = Autoencoder(0)
        nn = Autoencoder(1)
        nn_prime.build(input_dim=X_train.shape[1],
                       hidden_dim_list=hidden_dim_list,
                       learning_rate=learning_rate)
        nn.build(input_dim=X_train.shape[1],
                 hidden_dim_list=hidden_dim_list,
                 learning_rate=learning_rate)

        guest_data_loader = ExpandingVFTLDataLoader(data_folder_path=data_folder,
                                                    is_guest=True,
                                                    X_ol_train=X_train[overlap_indices],
                                                    Y_ol_train=Y_train[overlap_indices],
                                                    X_nol_train=X_train[guest_non_overlap_indices],
                                                    Y_nol_train=Y_train[guest_non_overlap_indices],
                                                    X_ested_train=X_train,
                                                    Y_ested_train=Y_train,
                                                    X_test=X_test,
                                                    Y_test=Y_test)
        self.guest = ExpandingVFTLGuest(party_model_param=self.party_param,
                                        data_loader=guest_data_loader)
        self.guest.set_model(nn, nn_prime)

        return self.guest


class ExpandingVFTLHostConstructor(object):

    def __init__(self, party_param: PartyModelParam, fed_model_param: FederatedModelParam):
        self.host = None
        self.party_param = party_param
        self.fed_model_param = fed_model_param

    def build(self, X_train, X_test):
        print("[INFO] => Build Host.")

        hidden_dim_list = self.party_param.hidden_dim_list
        learning_rate = self.party_param.nn_learning_rate
        data_folder = self.party_param.data_folder

        num_guest_nonoverlap_samples = self.fed_model_param.num_guest_nonoverlap_samples
        num_host_nonoverlap_samples = self.fed_model_param.num_host_nonoverlap_samples
        overlap_indices = self.fed_model_param.overlap_indices
        non_overlap_indices = self.fed_model_param.non_overlap_indices
        host_non_overlap_indices = non_overlap_indices[num_guest_nonoverlap_samples:
                                                       num_guest_nonoverlap_samples + num_host_nonoverlap_samples]
        nn_prime = Autoencoder(2)
        nn = Autoencoder(3)
        nn_prime.build(input_dim=X_train.shape[1], hidden_dim_list=hidden_dim_list, learning_rate=learning_rate)
        nn.build(input_dim=X_train.shape[1], hidden_dim_list=hidden_dim_list, learning_rate=learning_rate)
        host_data_loader = ExpandingVFTLDataLoader(data_folder_path=data_folder,
                                                   is_guest=False,
                                                   X_ol_train=X_train[overlap_indices],
                                                   X_nol_train=X_train[host_non_overlap_indices],
                                                   X_ested_train=X_train,
                                                   X_test=X_test)
        self.host = ExpandingVFTLHost(party_model_param=self.party_param, data_loader=host_data_loader)
        self.host.set_model(nn, nn_prime)

        return self.host


def run_experiment(X_guest_all, X_host_all, Y_guest_all, num_overlap,
                   hyperparameter_dict, num_train, test_start_index,
                   epoch, non_overlap_sample_batch_num,
                   overlap_sample_batch_size,
                   non_overlap_sample_batch_size,
                   estimation_block_size,
                   training_info_file_name):
    X_guest_train, y_train = X_guest_all[:num_train], Y_guest_all[:num_train]
    X_guest_test, y_test = X_guest_all[test_start_index:], Y_guest_all[test_start_index:]

    X_host_train = X_host_all[:num_train]
    X_host_test = X_host_all[test_start_index:]

    print("############# Data Info ##############")
    print("X_guest_train shape", X_guest_train.shape)
    print("X_host_train shape", X_host_train.shape)
    print("y_train shape", y_train.shape)

    print("X_guest_test shape", X_guest_test.shape)
    print("X_host_test shape", X_host_test.shape)
    print("y_test shape", y_test.shape)

    sample_num = num_overlap / 10
    lbl_sample_idx_dict = {}
    for index, one_hot_lbl in enumerate(Y_all):
        lbl_index = np.argmax(one_hot_lbl)
        sample_idx_list = lbl_sample_idx_dict.get(lbl_index)
        if sample_idx_list is None:
            lbl_sample_idx_dict[lbl_index] = [index]
        elif len(sample_idx_list) < sample_num:
            lbl_sample_idx_dict[lbl_index].append(index)
    print("lbl_sample_idx_dict:\n", lbl_sample_idx_dict)

    overlap_indices = list()
    for k, v in lbl_sample_idx_dict.items():
        overlap_indices += lbl_sample_idx_dict[k]
    print("overlap_indices:\n", overlap_indices, len(overlap_indices))

    # configuration
    # overlap_indices = [i for i in range(num_overlap)]
    non_overlap_indices = np.setdiff1d(range(num_train), overlap_indices)
    combine_axis = 1

    num_non_overlap = num_train - num_overlap

    guest_hidden_dim = 32
    host_hidden_dim = 32

    num_class = len(target_label_list)
    guest_model_param = PartyModelParam(data_folder=None,
                                        apply_dropout=False,
                                        hidden_dim_list=[guest_hidden_dim],
                                        n_class=num_class)
    host_model_param = PartyModelParam(data_folder=None,
                                       apply_dropout=False,
                                       hidden_dim_list=[host_hidden_dim],
                                       n_class=num_class)

    print("combine_axis:", combine_axis)
    if combine_axis == 0:
        input_dim = host_hidden_dim + guest_hidden_dim
    else:
        input_dim = 2 * (host_hidden_dim + guest_hidden_dim)

    parallel_iterations = 100

    learning_rate = hyperparameter_dict["learning_rate"]
    lambda_dist_shared_reprs = hyperparameter_dict["lambda_dist_shared_reprs"]
    lambda_sim_shared_reprs_vs_unique_repr = hyperparameter_dict["lambda_sim_shared_reprs_vs_unique_repr"]
    lambda_host_dist_ested_lbl_vs_true_lbl = hyperparameter_dict["lambda_host_dist_ested_lbl_vs_true_lbl"]
    lambda_dist_ested_repr_vs_true_repr = hyperparameter_dict["lambda_dist_ested_repr_vs_true_repr"]
    lambda_host_dist_two_ested_lbl = hyperparameter_dict["lambda_host_dist_two_ested_lbl"]

    # lambda for auxiliary losses, which include:
    # (1) loss for minimizing distance between shared representations between host and guest
    # (2) loss for minimizing similarity between shared representation and unique representation for guest
    # (3) loss for minimizing similarity between shared representation and unique representation for host
    # (4) loss for minimizing distance between estimated host unique overlap labels and true overlap labels
    # (5) loss for minimizing distance between estimated host common overlap labels and true overlap labels
    # (6) loss for minimizing distance between estimated guest overlap representation and true guest representation
    # (7) loss for minimizing distance between estimated host overlap representation and true host representation
    # (8) loss for minimizing distance between shared-repr-estimated host label and uniq-repr-estimated host label
    loss_weight_dict = {"lambda_dist_shared_reprs": lambda_dist_shared_reprs,
                        "lambda_guest_sim_shared_reprs_vs_unique_repr": lambda_sim_shared_reprs_vs_unique_repr,
                        "lambda_host_sim_shared_reprs_vs_unique_repr": lambda_sim_shared_reprs_vs_unique_repr,
                        "lambda_host_dist_ested_uniq_lbl_vs_true_lbl": lambda_host_dist_ested_lbl_vs_true_lbl,
                        "lambda_host_dist_ested_comm_lbl_vs_true_lbl": lambda_host_dist_ested_lbl_vs_true_lbl,
                        "lambda_guest_dist_ested_repr_vs_true_repr": lambda_dist_ested_repr_vs_true_repr,
                        "lambda_host_dist_ested_repr_vs_true_repr": lambda_dist_ested_repr_vs_true_repr,
                        "lambda_host_dist_two_ested_lbl": lambda_host_dist_two_ested_lbl}

    print("* hyper-parameter_dict :{0}".format(hyperparameter_dict))
    print("* loss_weight_dict: {0}".format(loss_weight_dict))
    fed_model_param = FederatedModelParam(fed_input_dim=input_dim,
                                          guest_input_dim=int(input_dim / 2),
                                          using_block_idx=False,
                                          learning_rate=learning_rate,
                                          fed_reg_lambda=0.0001,
                                          guest_reg_lambda=0.0001,
                                          loss_weight_dict=loss_weight_dict,
                                          overlap_indices=overlap_indices,
                                          non_overlap_indices=non_overlap_indices,
                                          epoch=epoch,
                                          top_k=1,
                                          combine_axis=combine_axis,
                                          parallel_iterations=parallel_iterations,
                                          num_guest_nonoverlap_samples=int(num_non_overlap / 2),
                                          num_host_nonoverlap_samples=int(num_non_overlap / 2),
                                          non_overlap_sample_batch_num=non_overlap_sample_batch_num,
                                          overlap_sample_batch_size=overlap_sample_batch_size,
                                          non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                          overlap_sample_batch_num=num_overlap,
                                          all_sample_block_size=estimation_block_size,
                                          sharpen_temperature=0.1,
                                          fed_label_prob_threshold=0.6,
                                          host_label_prob_threshold=0.6,
                                          training_info_file_name=training_info_file_name)

    # set up and train model
    guest_constructor = ExpandingVFTLGuestConstructor(guest_model_param,
                                                      fed_model_param)
    host_constructor = ExpandingVFTLHostConstructor(host_model_param,
                                                    fed_model_param)

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    guest = guest_constructor.build(X_train=X_guest_train,
                                    Y_train=y_train,
                                    X_test=X_guest_test,
                                    Y_test=y_test)
    host = host_constructor.build(X_train=X_host_train,
                                  X_test=X_host_test)

    VFTL = VerticalFederatedTransferLearning(vftl_guest=guest,
                                             vftl_host=host,
                                             model_param=fed_model_param)
    VFTL.set_representation_estimator(AttentionBasedRepresentationEstimator())
    VFTL.build()
    VFTL.train(debug=False)


if __name__ == "__main__":

    file_dir = "../../data/"
    target_label_list = ['sky', 'clouds', 'person', 'water', 'animal',
                         'grass', 'buildings', 'window', 'plants', 'lake']
    # target_label_list = ['sky', 'clouds', 'water', 'flowers', 'ocean',
    #                      'grass', 'buildings', 'window', 'plants', 'lake']
    data_loader = TwoPartyNusWideDataLoader(file_dir)
    X_image_all, X_text_all, Y_all = data_loader.get_train_data(target_labels=target_label_list,
                                                                binary_classification=False)
    # Y_guest_all = np.expand_dims(Y_guest_all, axis=1)

    print("### original data shape")
    print("X_image_all shape", X_image_all.shape)
    print("X_text_all shape", X_text_all.shape)
    print("Y_all shape", Y_all.shape)

    print("### Remove all-zero samples")
    print("X_text_all: ", X_text_all.shape)
    idx_valid_sample_list = []
    idx_invalid_sample_list = []
    for idx, X_text_all_i in enumerate(X_text_all):
        if np.all(X_text_all_i == 0):
            idx_invalid_sample_list.append(idx)
            # print("X_text_all_i", idx, np.sum(X_text_all_i), len(X_text_all_i))
        else:
            idx_valid_sample_list.append(idx)

    print("number of all-zero text sample:", len(idx_invalid_sample_list))
    print("number of not all-zero text sample:", len(idx_valid_sample_list))
    print("total number of samples: ", len(idx_invalid_sample_list) + len(idx_valid_sample_list))

    X_image = X_image_all[idx_valid_sample_list]
    X_text = X_text_all[idx_valid_sample_list]
    Y = Y_all[idx_valid_sample_list]

    # print("X_image: ", X_image.shape)
    # has_invalid_image_sample = False
    # for idx, X_image_i in enumerate(X_image):
    #     if np.all(X_image_i == 0):
    #         has_invalid_image_sample = True
    #         print("X_image_i", idx, np.sum(X_image_i), len(X_image_i))
    #
    # has_invalid_text_sample = False
    # print("X_text: ", X_text.shape)
    # for idx, X_text_id in enumerate(X_text):
    #     if np.all(X_text_id == 0):
    #         has_invalid_text_sample = True
    #         print("X_text_id", idx, np.sum(X_text_id), len(X_text_id))
    #
    # print("has_invalid_image_sample: {0}".format(has_invalid_image_sample))
    # print("has_invalid_text_sample: {0}".format(has_invalid_text_sample))

    X_guest_all = X_image
    X_host_all = X_text
    Y_guest_all = Y

    print("### All-zero removed data shape")
    print("X_guest_all shape", X_guest_all.shape)
    print("X_host_all shape", X_host_all.shape)
    print("Y_guest_all shape", Y_guest_all.shape)

    #
    # Start training
    #

    # num_train = int(0.86 * X_guest_all.shape[0])
    # test_start_index = 500
    epoch = 25
    estimation_block_size = 5000
    non_overlap_sample_batch_num = 40
    overlap_sample_batch_size = 256
    non_overlap_sample_batch_size = 256
    num_overlap = 500
    num_train = 40000
    test_start_index = 40000
    # num_train = 25000
    # test_start_index = 25000
    print("num_train", num_train)
    print("test_start_index", test_start_index)

    # lambda_dis_shared_reprs = [1.0]
    # lambda_sim_shared_reprs_vs_distinct_repr = [0.01, 0.1]
    # lambda_host_dis_ested_lbl_vs_true_lbl = [1500, 1000, 500, 100]
    # lambda_dis_ested_repr_vs_true_repr = [0.1, 1.0, 10.0]
    # lambda_host_dis_two_ested_repr = [0.1, 1.0, 10.0]
    # learning_rate = [0.01, 0.05, 0.1]

    lambda_dis_shared_reprs = [1.0]
    # lambda_sim_shared_reprs_vs_distinct_repr = [0.1]
    # lambda_host_dis_ested_lbl_vs_true_lbl = [1]
    # lambda_dis_ested_repr_vs_true_repr = [1]
    lambda_sim_shared_reprs_vs_uniq_reprs = [0.1]
    lambda_host_dis_ested_lbls_vs_true_lbls = [1000]
    lambda_dis_ested_reprs_vs_true_reprs = [0.1]
    lambda_host_dist_two_ested_lbls = [0.1]
    # learning_rate = [0.01, 0.005]
    learning_rate = [0.01]

    file_folder = "training_log_info/"
    timestamp = get_timestamp()
    file_name = file_folder + "test_csv_read_" + timestamp + ".csv"

    # hyperparam_list = list()
    # for lr in learning_rate:
    #     for lbd_dis_shared_reprs in lambda_dis_shared_reprs:
    #         for lbd_sim_shared_vs_dist_reprs in lambda_sim_shared_reprs_vs_distinct_repr:
    #             for lbd_host_dis_ested_lbl_vs_true_lbl in lambda_host_dis_ested_lbl_vs_true_lbl:
    #                 for lbda_4 in lambda_dis_ested_repr_vs_true_repr:
    #                     for lbda_5 in lambda_host_dis_two_ested_repr:
    #                         hyperparam_list.append()

    # lambda for auxiliary losses, which include:
    # (1) loss for minimizing distance between shared representations between host and guest
    # (2) loss for minimizing similarity between shared representation and unique representation for guest
    # (3) loss for minimizing similarity between shared representation and unique representation for host
    # (4) loss for minimizing distance between estimated host unique overlap labels and true overlap labels
    # (5) loss for minimizing distance between estimated host common overlap labels and true overlap labels
    # (6) loss for minimizing distance between estimated guest overlap representation and true guest representation
    # (7) loss for minimizing distance between estimated host overlap representation and true host representation
    # (8) loss for minimizing distance between shared-repr-estimated host label and uniq-repr-estimated host label
    hyperparameter_dict = dict()
    for lbda_0 in learning_rate:
        for lbda_1 in lambda_dis_shared_reprs:
            for lbda_2 in lambda_sim_shared_reprs_vs_uniq_reprs:
                for lbda_3 in lambda_host_dis_ested_lbls_vs_true_lbls:
                    for lbda_4 in lambda_dis_ested_reprs_vs_true_reprs:
                        for lbda_5 in lambda_host_dist_two_ested_lbls:
                            hyperparameter_dict["learning_rate"] = lbda_0
                            hyperparameter_dict["lambda_dist_shared_reprs"] = lbda_1
                            hyperparameter_dict["lambda_sim_shared_reprs_vs_unique_repr"] = lbda_2
                            hyperparameter_dict["lambda_host_dist_ested_lbl_vs_true_lbl"] = lbda_3
                            hyperparameter_dict["lambda_dist_ested_repr_vs_true_repr"] = lbda_4
                            hyperparameter_dict["lambda_host_dist_two_ested_lbl"] = lbda_5
                            run_experiment(X_guest_all=X_guest_all, X_host_all=X_host_all, Y_guest_all=Y_guest_all,
                                           num_overlap=num_overlap, hyperparameter_dict=hyperparameter_dict,
                                           num_train=num_train, test_start_index=test_start_index,
                                           epoch=epoch, non_overlap_sample_batch_num=non_overlap_sample_batch_num,
                                           overlap_sample_batch_size=overlap_sample_batch_size,
                                           non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                           estimation_block_size=estimation_block_size,
                                           training_info_file_name=file_name)
