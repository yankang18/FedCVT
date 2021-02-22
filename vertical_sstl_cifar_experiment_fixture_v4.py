import tensorflow as tf
import json
# from cnn_models_bk import CNNFeatureExtractor
from cnn_models import ClientDeeperCNNFeatureExtractor, ClientCNNFeatureExtractor, ClientMiniVGG, ClientVGG8
from data_util.cifar_data_util import TwoPartyCifar10DataLoader
from expanding_vertical_transfer_learning_param import PartyModelParam, FederatedModelParam
from vertical_semi_supervised_transfer_learning_v4 import VerticalFederatedTransferLearning
from vertical_sstl_parties import ExpandingVFTLGuest, ExpandingVFTLHost, ExpandingVFTLDataLoader
from vertical_sstl_representation_learner import AttentionBasedRepresentationEstimator


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
    # prepare datasets

    # dataset_folder_path = "../data/cifar-10-batches-py/"
    # dataset_folder_path = "../data/fashionmnist_2000/"
    dataset_folder_path = "../data/cifar-10-batches-py_500/"

    print("{0} dataset_folder_path: {1}".format(tag_PATH, dataset_folder_path))

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

    # weights for auxiliary losses, which include:
    # (1) loss for shared representations between host and guest
    # (2) (3) loss for orthogonal representation for host and guest respectively
    # (4) loss for distance between estimated host overlap labels and true overlap labels

    # (5) loss for distance between estimated guest overlap representation and true guest representation
    # (6) loss for distance between estimated host overlap representation and true host representation
    # (7) loss for distance between shared-repr-estimated host label and uniq-repr-estimated host label
    # loss_weight_list = [1.0, 0.01, 0.01, 500, 0.1, 0.1, 0.1]
    # loss_weight_list = [100, 0.1, 0.1, 1000, 0.1, 0.1, 0.1]
    # loss_weight_list = [0.01, 0.001, 0.001, 100, 0.1, 0.1, 0.1]
    # loss_weight_list = [1.0, 0.1, 0.1, 1000, 0.1, 0.1, 0.1]
    loss_weight_list = [0.1, 0.01, 0.01, 1000, 1000, 0.1, 0.1, 0.1]
    fed_model_param = FederatedModelParam(fed_input_dim=input_dim,
                                          guest_input_dim=guest_input_dim,
                                          fed_hidden_dim=hidden_dim,
                                          guest_hidden_dim=None,
                                          using_block_idx=True,
                                          learning_rate=0.001,
                                          fed_reg_lambda=0.001,
                                          guest_reg_lambda=0.0,
                                          loss_weight_list=loss_weight_list,
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
                                          host_label_prob_threshold=0.7)

    # set up and train model
    guest_constructor = ExpandingVFTLGuestConstructor(guest_model_param)
    host_constructor = ExpandingVFTLHostConstructor(host_model_param)

    tf.compat.v1.reset_default_graph()
    # tf.compat.v1.reset_default_graph()

    # input_shape = (28, 14, 1)
    input_shape = (32, 16, 3)
    guest = guest_constructor.build(data_folder=dataset_folder_path,
                                    input_shape=input_shape)
    host = host_constructor.build(data_folder=dataset_folder_path,
                                  input_shape=input_shape)

    VFTL = VerticalFederatedTransferLearning(guest, host, fed_model_param)
    VFTL.set_representation_estimator(AttentionBasedRepresentationEstimator())
    VFTL.build()
    result_log, loss_list = VFTL.train(debug=False)

    print("result_log:", result_log)
    # print("fed_model_param: \n", fed_model_param.__dict__)
