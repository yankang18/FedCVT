import time

import numpy as np
import tensorflow as tf

from models.autoencoder import Autoencoder
from data_util.nus_wide_data_util import TwoPartyNusWideDataLoader
from param import PartyModelParam, FederatedModelParam
from vertical_semi_supervised_transfer_learning import VerticalFederatedTransferLearning
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
        # guest_non_overlap_indices = non_overlap_indices[:num_guest_nonoverlap_samples]
        guest_non_overlap_indices = non_overlap_indices
        guest_all_indices = np.concatenate([overlap_indices, guest_non_overlap_indices])

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
                                                    X_ested_train=X_train[guest_all_indices],
                                                    Y_ested_train=Y_train[guest_all_indices],
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
        # host_non_overlap_indices = non_overlap_indices[num_guest_nonoverlap_samples:
        #                                                num_guest_nonoverlap_samples + num_host_nonoverlap_samples]
        host_non_overlap_indices = non_overlap_indices
        print("overlap_indices:", overlap_indices.shape, host_non_overlap_indices.shape)
        host_all_indices = np.concatenate([overlap_indices, host_non_overlap_indices])

        nn_prime = Autoencoder(2)
        nn = Autoencoder(3)
        nn_prime.build(input_dim=X_train.shape[1], hidden_dim_list=hidden_dim_list, learning_rate=learning_rate)
        nn.build(input_dim=X_train.shape[1], hidden_dim_list=hidden_dim_list, learning_rate=learning_rate)
        host_data_loader = ExpandingVFTLDataLoader(data_folder_path=data_folder,
                                                   is_guest=False,
                                                   X_ol_train=X_train[overlap_indices],
                                                   X_nol_train=X_train[host_non_overlap_indices],
                                                   X_ested_train=X_train[host_all_indices],
                                                   X_test=X_test)
        self.host = ExpandingVFTLHost(party_model_param=self.party_param, data_loader=host_data_loader)
        self.host.set_model(nn, nn_prime)

        return self.host


def run_experiment(X_guest_train, X_host_train, y_train,
                   X_guest_test, X_host_test, y_test,
                   num_overlap, hyperparameter_dict, epoch,
                   overlap_sample_batch_size,
                   non_overlap_sample_batch_size,
                   estimation_block_size,
                   training_info_file_name,
                   sharpen_temperature):

    print("############# Data Info ##############")
    print("X_guest_train shape", X_guest_train.shape)
    print("X_host_train shape", X_host_train.shape)
    print("y_train shape", y_train.shape)

    print("X_guest_test shape", X_guest_test.shape)
    print("X_host_test shape", X_host_test.shape)
    print("y_test shape", y_test.shape)
    print("######################################")

    # sample_num = num_overlap / 10
    # lbl_sample_idx_dict = {}
    # for index, one_hot_lbl in enumerate(y_train):
    #     lbl_index = np.argmax(one_hot_lbl)
    #     sample_idx_list = lbl_sample_idx_dict.get(lbl_index)
    #     if sample_idx_list is None:
    #         lbl_sample_idx_dict[lbl_index] = [index]
    #     elif len(sample_idx_list) < sample_num:
    #         lbl_sample_idx_dict[lbl_index].append(index)
    # print("lbl_sample_idx_dict:\n", lbl_sample_idx_dict)
    # # compute overlap and non-overlap indices
    # overlap_indices = list()
    # for k, v in lbl_sample_idx_dict.items():
    #     overlap_indices += lbl_sample_idx_dict[k]

    overlap_indices = [i for i in range(num_overlap)]
    overlap_indices = np.array(overlap_indices)
    num_train = X_guest_train.shape[0]
    non_overlap_indices = np.setdiff1d(range(num_train), overlap_indices)
    num_non_overlap = num_train - num_overlap
    print("overlap_indices:\n", overlap_indices, len(set(overlap_indices)))
    print("non_overlap_indices:\n", non_overlap_indices, len(set(non_overlap_indices)))

    combine_axis = 1
    guest_hidden_dim = 32
    host_hidden_dim = 32
    num_class = 10

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
                                          fed_reg_lambda=0.001,
                                          guest_reg_lambda=0.0,
                                          loss_weight_dict=loss_weight_dict,
                                          overlap_indices=overlap_indices,
                                          non_overlap_indices=non_overlap_indices,
                                          epoch=epoch,
                                          top_k=1,
                                          combine_axis=combine_axis,
                                          parallel_iterations=parallel_iterations,
                                          num_guest_nonoverlap_samples=int(num_non_overlap / 2),
                                          num_host_nonoverlap_samples=int(num_non_overlap / 2),
                                          overlap_sample_batch_size=overlap_sample_batch_size,
                                          non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                          overlap_sample_batch_num=num_overlap,
                                          all_sample_block_size=estimation_block_size,
                                          label_prob_sharpen_temperature=0.5,
                                          sharpen_temperature=sharpen_temperature,
                                          fed_label_prob_threshold=0.6,
                                          host_label_prob_threshold=0.4,
                                          training_info_file_name=training_info_file_name,
                                          valid_iteration_interval=5)

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

    file_dir = "../../data/"
    target_label_list = ['sky', 'clouds', 'person', 'water', 'animal',
                         'grass', 'buildings', 'window', 'plants', 'lake']
    data_loader = TwoPartyNusWideDataLoader(file_dir)
    X_image_train, X_text_train, Y_train = data_loader.get_train_data(target_labels=target_label_list,
                                                                      binary_classification=False)
    X_image_test, X_text_test, Y_test = data_loader.get_test_data(target_labels=target_label_list,
                                                                  binary_classification=False)

    print("### original data shape")
    print("X_image_all shape", X_image_train.shape)
    print("X_text_all shape", X_text_train.shape)
    print("Y_train shape", Y_train.shape)
    print("X_image_test shape", X_image_test.shape)
    print("X_text_test shape", X_text_test.shape)
    print("Y_test shape", Y_test.shape)

    print("### Remove all-zero samples")
    print("X_text_all: ", X_text_train.shape)
    idx_valid_sample_list = get_valid_sample_indices(X_text_train)
    X_guest_train = X_image_train[idx_valid_sample_list]
    X_host_train = X_text_train[idx_valid_sample_list]
    Y_train = Y_train[idx_valid_sample_list]

    idx_valid_sample_list = get_valid_sample_indices(X_text_test)
    X_guest_test = X_image_test[idx_valid_sample_list]
    X_host_test = X_text_test[idx_valid_sample_list]
    Y_test = Y_test[idx_valid_sample_list]

    test_indices = np.random.permutation(int(len(idx_valid_sample_list)/2))
    X_guest_test = X_guest_test[test_indices]
    X_host_test = X_host_test[test_indices]
    Y_test = Y_test[test_indices]

    print("### All-zero removed data shape")
    print("X_guest_test shape", X_guest_test.shape)
    print("X_host_test shape", X_host_test.shape)
    print("Y_test shape", Y_test.shape)

    #
    # Start training
    #

    epoch = 30
    estimation_block_size = 4000
    overlap_sample_batch_size = 128
    non_overlap_sample_batch_size = 256
    sharpen_temperature = 0.1

    # num_overlap = 500
    num_overlap_list = [1000]
    lambda_dis_shared_reprs = [0.01]
    lambda_sim_shared_reprs_vs_uniq_reprs = [0.01]
    lambda_host_dis_ested_lbls_vs_true_lbls = [100]
    lambda_dis_ested_reprs_vs_true_reprs = [0.01]
    lambda_host_dist_two_ested_lbls = [0.01]
    learning_rate = [0.005]

    file_folder = "training_log_info/"
    timestamp = get_timestamp()

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
    for n_ol in num_overlap_list:
        for lbda_0 in learning_rate:
            for lbda_1 in lambda_dis_shared_reprs:
                for lbda_2 in lambda_sim_shared_reprs_vs_uniq_reprs:
                    for lbda_3 in lambda_host_dis_ested_lbls_vs_true_lbls:
                        for lbda_4 in lambda_dis_ested_reprs_vs_true_reprs:
                            for lbda_5 in lambda_host_dist_two_ested_lbls:

                                file_name = file_folder + "nuswide_" + str(n_ol) + "_" + timestamp

                                hyperparameter_dict["learning_rate"] = lbda_0
                                hyperparameter_dict["lambda_dist_shared_reprs"] = lbda_1
                                hyperparameter_dict["lambda_sim_shared_reprs_vs_unique_repr"] = lbda_2
                                hyperparameter_dict["lambda_host_dist_ested_lbl_vs_true_lbl"] = lbda_3
                                hyperparameter_dict["lambda_dist_ested_repr_vs_true_repr"] = lbda_4
                                hyperparameter_dict["lambda_host_dist_two_ested_lbl"] = lbda_5
                                run_experiment(X_guest_train=X_guest_train, X_host_train=X_host_train, y_train=Y_train,
                                               X_guest_test=X_guest_test, X_host_test=X_host_test, y_test=Y_test,
                                               num_overlap=n_ol, hyperparameter_dict=hyperparameter_dict, epoch=epoch,
                                               overlap_sample_batch_size=overlap_sample_batch_size,
                                               non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                               estimation_block_size=estimation_block_size,
                                               training_info_file_name=file_name,
                                               sharpen_temperature=sharpen_temperature)
