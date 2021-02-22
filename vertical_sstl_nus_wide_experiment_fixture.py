import csv
import time

import numpy as np
import tensorflow as tf

from autoencoder import Autoencoder
from data_util.nus_wide_processed_data_util import TwoPartyNusWideDataLoader
from expanding_vertical_transfer_learning_param import PartyModelParam, FederatedModelParam
# from vertical_semi_supervised_transfer_learning_v3 import VerticalFederatedTransferLearning
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
                   training_info_file_name,
                   training_info_field_names):
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

    print("check data")
    print("X_guest_train: ", X_guest_train.shape)
    for idx, X_guest_train_i in enumerate(X_guest_train):
        if np.all(X_guest_train_i == 0):
            print("X_guest_train_i", idx, X_guest_train_i, np.sum(X_guest_train_i), len(X_guest_train_i))

    print("X_guest_test: ", X_guest_test.shape)
    for idx, X_guest_test_i in enumerate(X_guest_test):
        if np.all(X_guest_test_i == 0):
            print("X_guest_test_i", idx, X_guest_test_i, np.sum(X_guest_test_i), len(X_guest_test_i))

    print("X_host_train: ", X_host_train.shape)
    for idx, X_host_train_i in enumerate(X_host_train):
        if np.all(X_host_train_i == 0):
            print("X_host_train_i", idx, X_host_train_i, np.sum(X_host_train_i), len(X_host_train_i))

    print("X_host_test: ", X_host_test.shape)
    for idx, X_host_test_i in enumerate(X_host_test):
        if np.all(X_host_test_i == 0):
            print("X_host_test_i", idx, X_host_test_i, np.sum(X_host_test_i), len(X_host_test_i))

    # configuration
    overlap_indices = [i for i in range(num_overlap)]
    non_overlap_indices = np.setdiff1d(range(num_train), overlap_indices)
    combine_axis = 1

    num_non_overlap = num_train - num_overlap

    # guest_second_to_last_dim = 10
    # guest_hidden_dim = 8
    # host_second_to_last_dim = 10
    # host_hidden_dim = 8

    # guest_second_to_last_dim = 200
    guest_hidden_dim = 32
    # host_second_to_last_dim = 200
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
    # guest_model_param = PartyModelParam(hidden_dim_list=[guest_second_to_last_dim, guest_hidden_dim])
    # host_model_param = PartyModelParam(hidden_dim_list=[host_second_to_last_dim, host_hidden_dim])

    print("combine_axis:", combine_axis)
    if combine_axis == 0:
        input_dim = host_hidden_dim + guest_hidden_dim
    else:
        input_dim = 2 * (host_hidden_dim + guest_hidden_dim)

    parallel_iterations = 100

    # weights for auxiliary losses, which include:
    # (1) loss for minimizing distance between shared representations between host and guest
    # (2) (3) loss for minimizing similarity between shared representation and distinct representation
    # for host and guest respectively
    # (4) loss for minimizing distance between estimated host overlap labels and true overlap labels
    # (5) loss for minimizing distance between estimated guest overlap representation and true guest representation
    # (6) loss for minimizing distance between estimated host overlap representation and true host representation
    # (7) loss for minimizing distance between shared-repr-estimated host label and uniq-repr-estimated host label
    learnring_rate = hyperparameter_dict["learning_rate"]
    lambda_dis_shared_reprs = hyperparameter_dict["lambda_dis_shared_reprs"]
    lambda_sim_shared_reprs_vs_distinct_repr = hyperparameter_dict["lambda_sim_shared_reprs_vs_distinct_repr"]
    lambda_host_dis_ested_lbl_vs_true_lbl = hyperparameter_dict["lambda_host_dis_ested_lbl_vs_true_lbl"]
    lambda_dis_ested_repr_vs_true_repr = hyperparameter_dict["lambda_dis_ested_repr_vs_true_repr"]
    lambda_host_dis_two_ested_repr = hyperparameter_dict["lambda_host_dis_two_ested_repr"]

    loss_weight_list = [lambda_dis_shared_reprs,
                        lambda_sim_shared_reprs_vs_distinct_repr,
                        lambda_sim_shared_reprs_vs_distinct_repr,
                        lambda_host_dis_ested_lbl_vs_true_lbl,
                        lambda_dis_ested_repr_vs_true_repr,
                        lambda_dis_ested_repr_vs_true_repr,
                        lambda_host_dis_two_ested_repr]

    print("* hyperparameter_dict :{0}".format(hyperparameter_dict))
    print("* loss_weight_list: {0}".format(loss_weight_list))
    fed_model_param = FederatedModelParam(fed_input_dim=input_dim,
                                          guest_input_dim=int(input_dim / 2),
                                          using_block_idx=False,
                                          learning_rate=learnring_rate,
                                          fed_reg_lambda=0.001,
                                          guest_reg_lambda=0.0,
                                          loss_weight_list=loss_weight_list,
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
                                          is_hetero_repr=False,
                                          sharpen_temperature=0.1,
                                          fed_label_prob_threshold=0.6,
                                          host_label_prob_threshold=0.6)

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

    training_result_log_dict, loss_list = VFTL.train(debug=False)
    record_dict = dict()
    record_dict.update(hyperparameter_dict)
    record_dict.update(training_result_log_dict)

    print("record_dict", record_dict)
    print("training_info_field_names", training_info_field_names)
    print("diff:", record_dict.keys() - training_info_field_names)
    with open(training_info_file_name, "a") as logfile:
        logger = csv.DictWriter(logfile, fieldnames=training_info_field_names)
        logger.writerow(record_dict)


if __name__ == "__main__":

    # prepare datasets

    # infile = "./datasets/"
    # data_loader = TwoPartyBreastDataLoader(infile)
    # X_guest_all, X_host_all, Y_guest_all = data_loader.get_data()
    # Y_guest_all = np.expand_dims(Y_guest_all, axis=1)

    # infile = "./datasets/UCI_Credit_Card/UCI_Credit_Card.csv"
    # # split_index = 5, 11
    # data_loader = TwoPartyUCICreditCardDataLoader(infile, split_index=5, balanced=False)
    # X_host_all, X_guest_all, Y_guest_all = data_loader.get_data()
    # Y_guest_all = np.expand_dims(Y_guest_all, axis=1)

    # file_dir = "/datasets/app/fate/yankang/"
    file_dir = "../../data/"
    # file_dir = "../"
    # target_label_list = ["person", "animal", "sky"]
    target_label_list = ['sky', 'clouds', 'person', 'water', 'animal',
                         'grass', 'buildings', 'window', 'plants', 'lake']
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

    print("X_image: ", X_image.shape)
    has_invalid_image_sample = False
    for idx, X_image_i in enumerate(X_image):
        if np.all(X_image_i == 0):
            has_invalid_image_sample = True
            print("X_image_i", idx, np.sum(X_image_i), len(X_image_i))

    has_invalid_text_sample = False
    print("X_text: ", X_text.shape)
    for idx, X_text_id in enumerate(X_text):
        if np.all(X_text_id == 0):
            has_invalid_text_sample = True
            print("X_text_id", idx, np.sum(X_text_id), len(X_text_id))

    print("has_invalid_image_sample: {0}".format(has_invalid_image_sample))
    print("has_invalid_text_sample: {0}".format(has_invalid_text_sample))

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
    overlap_sample_batch_size = 128
    non_overlap_sample_batch_size = 128
    num_overlap = 500
    num_train = 40000
    test_start_index = 40000
    print("num_train", num_train)
    print("test_start_index", test_start_index)

    # lambda_dis_shared_reprs = [1.0]
    # lambda_sim_shared_reprs_vs_distinct_repr = [0.01, 0.1]
    # lambda_host_dis_ested_lbl_vs_true_lbl = [1500, 1000, 500, 100]
    # lambda_dis_ested_repr_vs_true_repr = [0.1, 1.0, 10.0]
    # lambda_host_dis_two_ested_repr = [0.1, 1.0, 10.0]
    # learning_rate = [0.01, 0.05, 0.1]

    lambda_dis_shared_reprs = [1.0]
    lambda_sim_shared_reprs_vs_distinct_repr = [0.01]
    lambda_host_dis_ested_lbl_vs_true_lbl = [1]
    lambda_dis_ested_repr_vs_true_repr = [0.1]
    lambda_host_dis_two_ested_repr = [0.1]
    learning_rate = [0.01]

    log_field_names = ["fscore",
                       "all_fscore", "g_fscore", "h_fscore",
                       "all_acc", "g_acc", "h_acc",
                       "all_auc", "g_auc", "h_auc",
                       "epoch", "batch"]

    hyperparam_field_names = ["lambda_dis_shared_reprs",
                              "lambda_sim_shared_reprs_vs_distinct_repr",
                              "lambda_host_dis_ested_lbl_vs_true_lbl",
                              "lambda_dis_ested_repr_vs_true_repr",
                              "lambda_host_dis_two_ested_repr",
                              "learning_rate"]

    all_field_names = hyperparam_field_names + log_field_names
    print("all fields of log: {0}".format(all_field_names))
    file_folder = "training_log_info/"
    timestamp = get_timestamp()
    file_name = file_folder + "test_csv_read_" + timestamp + ".csv"
    with open(file_name, "a", newline='') as logfile:
        logger = csv.DictWriter(logfile, fieldnames=all_field_names)
        logger.writeheader()

    # hyperparam_list = list()
    # for lr in learning_rate:
    #     for lbd_dist_shared_reprs in lambda_dis_shared_reprs:
    #         for lbd_sim_shared_reprs in lambda_sim_shared_reprs_vs_distinct_repr:
    #             for lbda_3 in lambda_host_dis_ested_lbl_vs_true_lbl:
    #                 for lbda_4 in lambda_dis_ested_repr_vs_true_repr:
    #                     for lbda_5 in lambda_host_dis_two_ested_repr:
    #                         hyperparam_list.append()

    # weights for auxiliary losses, which include:
    # (1) loss for minimizing distance between shared representations between host and guest
    # (2) (3) loss for minimizing similarity between shared representation and distinct representation
    # for host and guest respectively
    # (4) loss for minimizing distance between estimated host overlap labels and true overlap labels
    # (5) loss for minimizing distance between estimated guest overlap representation and true guest representation
    # (6) loss for minimizing distance between estimated host overlap representation and true host representation
    # (7) loss for minimizing distance between shared-repr-estimated host label and uniq-repr-estimated host label
    # loss_weight_list = [1.0, 0.01, 0.01, 1500, 1.0, 1.0, 1.0]
    hyperparameter_dict = dict()
    for lbda_7 in learning_rate:
        for lbda_1 in lambda_dis_shared_reprs:
            for lbda_2 in lambda_sim_shared_reprs_vs_distinct_repr:
                for lbda_3 in lambda_host_dis_ested_lbl_vs_true_lbl:
                    for lbda_4 in lambda_dis_ested_repr_vs_true_repr:
                        for lbda_5 in lambda_host_dis_two_ested_repr:
                            hyperparameter_dict["learning_rate"] = lbda_7
                            hyperparameter_dict["lambda_dis_shared_reprs"] = lbda_1
                            hyperparameter_dict["lambda_sim_shared_reprs_vs_distinct_repr"] = lbda_2
                            hyperparameter_dict["lambda_host_dis_ested_lbl_vs_true_lbl"] = lbda_3
                            hyperparameter_dict["lambda_dis_ested_repr_vs_true_repr"] = lbda_4
                            hyperparameter_dict["lambda_host_dis_two_ested_repr"] = lbda_5
                            run_experiment(X_guest_all=X_guest_all, X_host_all=X_host_all, Y_guest_all=Y_guest_all,
                                           num_overlap=num_overlap, hyperparameter_dict=hyperparameter_dict,
                                           num_train=num_train, test_start_index=test_start_index,
                                           epoch=epoch, non_overlap_sample_batch_num=non_overlap_sample_batch_num,
                                           overlap_sample_batch_size=overlap_sample_batch_size,
                                           non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                           estimation_block_size=estimation_block_size,
                                           training_info_file_name=file_name, training_info_field_names=all_field_names)
