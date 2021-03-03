import time

import numpy as np
import tensorflow as tf

from cnn_models import ClientVGG8
from data_util.modelnet_data_loader import get_two_party_data
from expanding_vertical_transfer_learning_param import PartyModelParam, FederatedModelParam
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

        input_shape = self.party_param.input_shape
        data_folder = self.party_param.data_folder
        dense_units = self.party_param.dense_units

        num_guest_nonoverlap_samples = self.fed_model_param.num_guest_nonoverlap_samples
        overlap_indices = self.fed_model_param.overlap_indices
        non_overlap_indices = self.fed_model_param.non_overlap_indices
        # guest_non_overlap_indices = non_overlap_indices[:num_guest_nonoverlap_samples]
        guest_non_overlap_indices = non_overlap_indices
        guest_all_indices = np.concatenate([overlap_indices, guest_non_overlap_indices])

        nn_prime = ClientVGG8("cnn_0", dense_units=dense_units)
        nn_prime.build(input_shape=input_shape)
        nn = ClientVGG8("cnn_1", dense_units=dense_units)
        nn.build(input_shape=input_shape)

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

        input_shape = self.party_param.input_shape
        data_folder = self.party_param.data_folder
        dense_units = self.party_param.dense_units

        num_guest_nonoverlap_samples = self.fed_model_param.num_guest_nonoverlap_samples
        num_host_nonoverlap_samples = self.fed_model_param.num_host_nonoverlap_samples
        overlap_indices = self.fed_model_param.overlap_indices
        non_overlap_indices = self.fed_model_param.non_overlap_indices
        # host_non_overlap_indices = non_overlap_indices[num_guest_nonoverlap_samples:
        #                                                num_guest_nonoverlap_samples + num_host_nonoverlap_samples]
        host_non_overlap_indices = non_overlap_indices
        print("overlap_indices:", overlap_indices.shape, host_non_overlap_indices.shape)
        host_all_indices = np.concatenate([overlap_indices, host_non_overlap_indices])

        nn_prime = ClientVGG8("cnn_2", dense_units=dense_units)
        nn_prime.build(input_shape=input_shape)
        nn = ClientVGG8("cnn_3", dense_units=dense_units)
        nn.build(input_shape=input_shape)
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
                   training_info_file_name):

    print("############# Data Info ##############")
    print("X_guest_train shape", X_guest_train.shape)
    print("X_host_train shape", X_host_train.shape)
    print("y_train shape", y_train.shape)

    print("X_guest_test shape", X_guest_test.shape)
    print("X_host_test shape", X_host_test.shape)
    print("y_test shape", y_test.shape)

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
    #
    # overlap_indices = list()
    # for k, v in lbl_sample_idx_dict.items():
    #     overlap_indices += lbl_sample_idx_dict[k]
    # print("overlap_indices:\n", overlap_indices, len(overlap_indices))

    # configuration
    overlap_indices = [i for i in range(num_overlap)]
    overlap_indices = np.array(overlap_indices)
    num_train = X_guest_train.shape[0]
    non_overlap_indices = np.setdiff1d(range(num_train), overlap_indices)
    num_non_overlap = num_train - num_overlap
    print("overlap_indices:\n", overlap_indices, len(set(overlap_indices)))
    print("non_overlap_indices:\n", non_overlap_indices, len(set(non_overlap_indices)))

    combine_axis = 1
    dense_units = 64
    input_shape = (32, 32, 3)
    input_dim = dense_units * 2 * 2
    guest_input_dim = int(input_dim / 2)

    num_class = 10
    guest_model_param = PartyModelParam(data_folder=None,
                                        apply_dropout=True,
                                        keep_probability=0.75,
                                        input_shape=input_shape,
                                        dense_units=dense_units,
                                        n_class=num_class)
    host_model_param = PartyModelParam(data_folder=None,
                                       apply_dropout=True,
                                       keep_probability=0.75,
                                       input_shape=input_shape,
                                       dense_units=dense_units,
                                       n_class=num_class)

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
                                          guest_input_dim=guest_input_dim,
                                          using_block_idx=False,
                                          learning_rate=learning_rate,
                                          fed_reg_lambda=0.0001,
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
                                          label_prob_sharpen_temperature=0.3,
                                          sharpen_temperature=0.1,
                                          fed_label_prob_threshold=0.6,
                                          host_label_prob_threshold=0.3,
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


if __name__ == "__main__":

    data_dir = "../../Data/modelnet40v1png/"

    num_classes = 10
    X_guest_train, X_host_train, Y_train = get_two_party_data(data_dir=data_dir, data_type="train", k=2, c=num_classes)
    X_guest_test, X_host_test, Y_test = get_two_party_data(data_dir=data_dir, data_type="test", k=2, c=num_classes)

    print(X_guest_train[0].shape)
    print(X_host_train[0].shape)
    print(Y_train[0].shape)
    #
    # Start training
    #
    epoch = 20
    estimation_block_size = 2000
    overlap_sample_batch_size = 128
    non_overlap_sample_batch_size = 128
    num_train = 3500
    test_start_index = 3500
    print("num_train", num_train)
    print("test_start_index", test_start_index)

    num_overlap = 500
    lambda_dis_shared_reprs = [0.1]
    lambda_sim_shared_reprs_vs_uniq_reprs = [0.01]
    lambda_host_dis_ested_lbls_vs_true_lbls = [100]
    lambda_dis_ested_reprs_vs_true_reprs = [0.01]
    lambda_host_dist_two_ested_lbls = [0.01]
    learning_rate = [0.001]

    file_folder = "training_log_info/"
    timestamp = get_timestamp()
    file_name = file_folder + "test_csv_read_" + timestamp + ".csv"

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
                            run_experiment(X_guest_train=X_guest_train, X_host_train=X_host_train, y_train=Y_train,
                                           X_guest_test=X_guest_test, X_host_test=X_host_test, y_test=Y_test,
                                           num_overlap=num_overlap, hyperparameter_dict=hyperparameter_dict, epoch=epoch,
                                           overlap_sample_batch_size=overlap_sample_batch_size,
                                           non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                           estimation_block_size=estimation_block_size,
                                           training_info_file_name=file_name)
