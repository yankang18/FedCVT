import tensorflow as tf
import json
# from cnn_models_bk import CNNFeatureExtractor
from cnn_models import ClientDeeperCNNFeatureExtractor, ClientCNNFeatureExtractor, ClientMiniVGG, ClientVGG8
from data_util.cifar_data_util import TwoPartyCifar10DataLoader
from expanding_vertical_transfer_learning_param import PartyModelParam, FederatedModelParam
from bm_vertical_semi_supervised_transfer_learning import VerticalFederatedTransferLearning
from bm_vertical_sstl_parties import ExpandingVFTLGuest, ExpandingVFTLHost, ExpandingVFTLDataLoader
from vertical_sstl_representation_learner import AttentionBasedRepresentationEstimator
import numpy as np

class ExpandingVFTLGuestConstructor(object):

    def __init__(self, party_param: PartyModelParam):
        self.guest = None
        self.party_param = party_param

    def build(self, data_folder, input_shape):
        print("Guest Setup")

        nn = ClientVGG8("cnn_1")
        nn.build(input_shape=input_shape)

        guest_data_loader = ExpandingVFTLDataLoader(data_folder_path=data_folder,
                                                    is_guest=True)

        self.guest = ExpandingVFTLGuest(party_model_param=self.party_param,
                                        data_loader=guest_data_loader)
        self.guest.set_model(nn)

        with open(data_folder + "meta_data.json", "r") as read_file:
            meta_data = json.load(read_file)

        val_block_num = meta_data["guest_val_block_num"]
        overlap_block_num = meta_data["guest_overlap_block_num"]

        print("val_block_num {0}".format(val_block_num))
        print("overlap_block_num {0}".format(overlap_block_num))

        self.guest.set_val_block_number(val_block_num)
        self.guest.set_ol_block_number(overlap_block_num)

        return self.guest


class ExpandingVFTLHostConstructor(object):

    def __init__(self, party_param: PartyModelParam):
        self.host = None
        self.party_param = party_param

    def build(self, data_folder, input_shape):
        print("Host Setup")

        nn = ClientVGG8("cnn_3")
        nn.build(input_shape=input_shape)

        host_data_loader = ExpandingVFTLDataLoader(data_folder_path=data_folder,
                                                   is_guest=False)

        self.host = ExpandingVFTLHost(party_model_param=self.party_param,
                                      data_loader=host_data_loader)
        self.host.set_model(nn)

        with open(data_folder + "meta_data.json", "r") as read_file:
            meta_data = json.load(read_file)

        val_block_num = meta_data["host_val_block_num"]
        overlap_block_num = meta_data["host_overlap_block_num"]

        print("val_block_num {0}".format(val_block_num))
        print("overlap_block_num {0}".format(overlap_block_num))

        self.host.set_val_block_number(val_block_num)
        self.host.set_ol_block_number(overlap_block_num)

        return self.host


tag_PATH = "[INFO]"
if __name__ == "__main__":
    # prepare datasets

    overlap_sample_number = 250
    # dataset_folder_path = "../data/cifar-10-batches-py/"
    # dataset_folder_path = "../data/fashionmnist_2000/"
    dataset_folder_path = "../data/cifar-10-batches-py_" + str(overlap_sample_number) + "/"

    print("{0} dataset_folder_path: {1}".format(tag_PATH, dataset_folder_path))

    # configuration
    combine_axis = 1

    # num_non_overlap = num_train - num_overlap

    # guest_model_param = PartyModelParam(n_class=10)
    # host_model_param = PartyModelParam(n_class=10)
    guest_model_param = PartyModelParam(n_class=10, keep_probability=0.75, apply_dropout=True)
    host_model_param = PartyModelParam(n_class=10, keep_probability=0.75, apply_dropout=True)

    # print("combine_axis:", combine_axis)
    input_dim = 48 * 2
    guest_input_dim = int(input_dim / 2)
    hidden_dim = None
    guest_hidden_dim = None
    parallel_iterations = 100
    epoch = 100

    overlap_sample_batch_size = 128
    non_overlap_sample_batch_size = 128
    fed_model_param = FederatedModelParam(fed_input_dim=input_dim,
                                          fed_hidden_dim=hidden_dim,
                                          learning_rate=0.001,
                                          fed_reg_lambda=0.0001,
                                          epoch=epoch,
                                          parallel_iterations=parallel_iterations,
                                          overlap_sample_batch_size=overlap_sample_batch_size)

    # set up and train model
    guest_constructor = ExpandingVFTLGuestConstructor(guest_model_param)
    host_constructor = ExpandingVFTLHostConstructor(host_model_param)

    tf.compat.v1.reset_default_graph()

    # input_shape = (28, 14, 1)
    input_shape = (32, 16, 3)
    guest = guest_constructor.build(data_folder=dataset_folder_path,
                                    input_shape=input_shape)
    host = host_constructor.build(data_folder=dataset_folder_path,
                                  input_shape=input_shape)

    VFTL = VerticalFederatedTransferLearning(guest, host, fed_model_param)
    VFTL.set_representation_estimator(AttentionBasedRepresentationEstimator())
    VFTL.build()

    num_try = 5
    all_best_acc_list = []
    for try_idx in range(num_try):
        print("[INFO] {0} try for testing with overlapping samples: {1}".format(try_idx, overlap_sample_number))
        result_log, loss_list = VFTL.fit(debug=False)
        best_acc = result_log["all_acc"]
        all_best_acc_list.append(best_acc)

    print("{0} overlap_samples test has acc with mean {1} and stddev:{2}".format(overlap_sample_number,
                                                                                 np.mean(all_best_acc_list),
                                                                                 np.std(all_best_acc_list)))
    # series_plot(losses=loss_list, fscores=all_fscore_list, aucs=all_acc_list)
    # print("fed_model_param: \n", fed_model_param.__dict__)
