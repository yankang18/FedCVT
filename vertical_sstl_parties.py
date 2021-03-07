import pickle

import numpy as np
import tensorflow as tf

from models.autoencoder import FeatureExtractor
from param import PartyModelParam


def load_data_block(block_file_full_path, block_id):
    filename = block_file_full_path + str(block_id) + '.p'
    print("load data block: {0}".format(filename))
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


class ExpandingVFTLParty(object):

    def __init__(self):
        self.local_model = None
        self.local_model_prime = None

    def set_model(self, feature_extractor: FeatureExtractor, feature_extractor_prime: FeatureExtractor):
        self.local_model = feature_extractor
        self.local_model_prime = feature_extractor_prime

    def set_session(self, sess):
        self.local_model.set_session(sess)
        self.local_model_prime.set_session(sess)

    def get_model_parameters(self):
        return self.local_model.get_model_parameters(), self.local_model_prime.get_model_parameters()

    def fetch_feat_reprs(self):
        U_all_uniq = self.local_model.get_all_hidden_reprs()
        U_all_comm = self.local_model_prime.get_all_hidden_reprs()
        U_overlap_uniq = self.local_model.get_overlap_hidden_reprs()
        U_overlap_comm = self.local_model_prime.get_overlap_hidden_reprs()
        U_non_overlap_uniq = self.local_model.get_non_overlap_hidden_reprs()
        U_non_overlap_comm = self.local_model_prime.get_non_overlap_hidden_reprs()

        U_all_uniq = tf.math.l2_normalize(U_all_uniq, axis=1)
        U_all_comm = tf.math.l2_normalize(U_all_comm, axis=1)
        U_overlap_uniq = tf.math.l2_normalize(U_overlap_uniq, axis=1)
        U_overlap_comm = tf.math.l2_normalize(U_overlap_comm, axis=1)
        U_non_overlap_uniq = tf.math.l2_normalize(U_non_overlap_uniq, axis=1)
        U_non_overlap_comm = tf.math.l2_normalize(U_non_overlap_comm, axis=1)

        return (U_all_uniq, U_all_comm), (U_non_overlap_uniq, U_non_overlap_comm), (U_overlap_uniq, U_overlap_comm)


class ExpandingVFTLDataLoader(object):
    def __init__(self, data_folder_path=None, is_guest=True,
                 X_ol_train=None, X_nol_train=None, X_ested_train=None, X_test=None,
                 Y_ol_train=None, Y_nol_train=None, Y_ested_train=None, Y_test=None):
        self.data_folder_path = data_folder_path
        self.X_ol_train = X_ol_train
        self.Y_ol_train = Y_ol_train
        self.X_nol_train = X_nol_train
        self.Y_nol_train = Y_nol_train
        self.X_ested_train = X_ested_train
        self.Y_ested_train = Y_ested_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.is_pre_load_data = True if self.data_folder_path is None else False

        if not self.is_pre_load_data:
            print("# data will be loaded online")
            if is_guest:
                self.overlap_block_file_full_path = self.data_folder_path + 'guest_overlap_block_'
                self.nonoverlap_block_file_full_path = self.data_folder_path + 'guest_nonoverlap_block_'
                self.ested_block_file_full_path = self.data_folder_path + 'guest_ested_block_'
                self.val_block_file_full_path = self.data_folder_path + 'guest_val_block_'
            else:
                self.overlap_block_file_full_path = self.data_folder_path + 'host_overlap_block_'
                self.nonoverlap_block_file_full_path = self.data_folder_path + 'host_nonoverlap_block_'
                self.ested_block_file_full_path = self.data_folder_path + 'host_ested_block_'
                self.val_block_file_full_path = self.data_folder_path + 'host_val_block_'
        else:
            print("# data has been preloaded")
            if is_guest:
                print("self.X_ol_train: ", len(self.X_ol_train))
                print("self.Y_ol_train: ", len(self.Y_ol_train))
                print("self.X_nol_train: ", len(self.X_nol_train))
                print("self.Y_nol_train: ", len(self.Y_nol_train))
                print("self.X_ested_train: ", len(self.X_ested_train))
                print("self.Y_ested_train: ", len(self.Y_ested_train))
                print("self.X_test: ", len(self.X_test))
                print("self.Y_test: ", len(self.Y_test))
            else:
                print("self.X_ol_train: ", len(self.X_ol_train))
                print("self.X_nol_train: ", len(self.X_nol_train))
                print("self.X_ested_train: ", len(self.X_ested_train))
                print("self.X_test: ", len(self.X_test))

    def get_overlap_block_size(self):
        return len(self.X_ol_train)

    def get_nonoverlap_block_size(self):
        return len(self.X_nol_train)

    def get_ested_block_size(self):
        return len(self.X_ested_train)

    def get_val_block_size(self):
        return len(self.X_test)

    def load_overlap_block(self, block_id):
        if not self.is_pre_load_data:
            self.X_ol_train, self.Y_ol_train = load_data_block(self.overlap_block_file_full_path, block_id)
        return len(self.X_ol_train)

    def load_nonoverlap_block(self, block_id):
        if not self.is_pre_load_data:
            self.X_nol_train, self.Y_nol_train = load_data_block(self.nonoverlap_block_file_full_path, block_id)
        return len(self.X_nol_train)

    def load_ested_block(self, block_id):
        if not self.is_pre_load_data:
            self.X_ested_train, self.Y_ested_train = load_data_block(self.ested_block_file_full_path, block_id)
        return len(self.X_ested_train)

    def load_val_block(self, block_id):
        if not self.is_pre_load_data:
            self.X_test, self.Y_test = load_data_block(self.val_block_file_full_path, block_id)
        return len(self.X_test)

    @staticmethod
    def retrieve_X_y(X, Y, batch_indices=None, batch_range=None):
        if batch_range is None and batch_indices is None:
            raise Exception("Both batch_range and batch_indices are None")
        if Y is None:
            if batch_indices is None:
                return X[batch_range[0]: batch_range[1]], None
            else:
                return X[batch_indices], None
        else:
            if batch_indices is None:
                return X[batch_range[0]: batch_range[1]], Y[batch_range[0]: batch_range[1]]
            else:
                return X[batch_indices], Y[batch_indices]

    def retrieve_ol_train_X_y(self, batch_indices=None, batch_range=None):
        return self.retrieve_X_y(self.X_ol_train, self.Y_ol_train, batch_indices, batch_range)

    def retrieve_all_ol_train_X_y(self):
        return self.X_ol_train, self.Y_ol_train

    def retrieve_nol_train_X_y(self, batch_indices=None, batch_range=None):
        return self.retrieve_X_y(self.X_nol_train, self.Y_nol_train, batch_indices, batch_range)

    def retrieve_ested_train_X_y(self, batch_indices=None, batch_range=None):
        return self.retrieve_X_y(self.X_ested_train, self.Y_ested_train, batch_indices, batch_range)

    def retrieve_ested_block(self, block_idx):
        return load_data_block(self.ested_block_file_full_path, block_idx)

    def retrieve_test_X_y(self):
        return self.X_test, self.Y_test

    def get_all_training_sample_size(self):
        return len(self.X_ested_train)

    def get_y(self):
        return self.Y_test


class ExpandingVFTLGuest(ExpandingVFTLParty):

    def __init__(self, party_model_param: PartyModelParam, data_loader: ExpandingVFTLDataLoader):
        super(ExpandingVFTLGuest, self).__init__()
        self.party_model_param = party_model_param
        self.keep_probability = party_model_param.keep_probability
        self.data_loader = data_loader
        self.n_class = party_model_param.n_class
        self.apply_dropout = party_model_param.apply_dropout

        self.Y_all_in_for_est = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_class),
                                                         name="labels_input_all_for_est")
        self.Y_overlap_in_for_est = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_class),
                                                             name="labels_input_overlap_for_est")

        self.Y_all_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_class), name="labels_input_all")
        self.Y_overlap_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_class),
                                                     name="labels_input_overlap")
        self.Y_non_overlap_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_class),
                                                         name="labels_input_non_overlap")
        self.guest_train_labels = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_class),
                                                           name="guest_train_labels_input")
        self.current_ol_block_idx = None
        self.current_nol_block_idx = None
        self.current_ested_block_idx = None
        self.current_val_block_idx = None

        self.ol_block_num = 1
        self.nol_block_num = 1
        self.ested_block_num = 1
        self.val_block_num = 1

    def set_ol_block_number(self, block_number):
        self.ol_block_num = block_number

    def set_nol_block_number(self, block_number):
        self.nol_block_num = block_number

    def set_ested_block_number(self, block_number):
        self.ested_block_num = block_number

    def set_val_block_number(self, block_number):
        self.val_block_num = block_number

    def get_ol_block_number(self):
        return self.ol_block_num

    def get_nol_block_number(self):
        return self.nol_block_num

    def get_ested_block_number(self):
        return self.ested_block_num

    def get_val_block_number(self):
        return self.val_block_num

    def load_ol_block(self, block_idx):
        if self.current_ol_block_idx is not None and self.current_ol_block_idx == block_idx:
            return self.data_loader.get_overlap_block_size()
        self.current_ol_block_idx = block_idx
        return self.data_loader.load_overlap_block(block_idx)

    def load_nol_block(self, block_idx):
        if self.current_nol_block_idx is not None and self.current_nol_block_idx == block_idx:
            print("here guest self.current_nol_block_idx, block_idx", self.current_nol_block_idx, block_idx)
            return self.data_loader.get_nonoverlap_block_size()
        self.current_nol_block_idx = block_idx
        return self.data_loader.load_nonoverlap_block(block_idx)

    def load_ested_block(self, block_idx):
        if self.current_ested_block_idx is not None and self.current_ested_block_idx == block_idx:
            return self.data_loader.get_ested_block_size()
        self.current_ested_block_idx = block_idx
        return self.data_loader.load_ested_block(block_idx)

    def load_val_block(self, block_idx):
        if self.current_val_block_idx is not None and self.current_val_block_idx == block_idx:
            return self.data_loader.get_val_block_size()
        self.current_val_block_idx = block_idx
        return self.data_loader.load_val_block(block_idx)

    def get_all_training_sample_size(self):
        return self.data_loader.get_all_training_sample_size()

    def get_Y_test(self):
        return self.data_loader.get_y()

    def get_Y_all_for_est(self):
        return self.Y_all_in_for_est

    def get_Y_overlap_for_est(self):
        return self.Y_overlap_in_for_est

    def get_Y_all(self):
        return self.Y_all_in

    def get_Y_overlap(self):
        return self.Y_overlap_in

    def get_Y_non_overlap(self):
        return self.Y_non_overlap_in

    def get_number_of_class(self):
        return self.n_class

    def get_train_feed_dict(self, overlap_batch_range, non_overlap_batch_range, block_indices=None, block_idx=None):
        if block_indices is None and block_idx is None:
            raise Exception("Both block_indices and block_idx are None")

        # overlap_batch_indices = self.overlap_indices[overlap_batch_range[0]: overlap_batch_range[1]]
        # non_overlap_batch_indices = self.non_overlap_indices[non_overlap_batch_range[0]: non_overlap_batch_range[1]]
        # X_overlap, Y_overlap = self.data_loader.retrieve_ol_train_X_y(overlap_batch_indices)
        # X_non_overlap, Y_non_overlap = self.data_loader.retrieve_nol_train_X_y(non_overlap_batch_indices)
        # X_all_ref_block, Y_all_ref_block = self.data_loader.retrieve_ested_block(block_idx)
        print("[DEBUG] guest overlap_batch_range", overlap_batch_range)
        print("[DEBUG] guest non_overlap_batch_range", non_overlap_batch_range)

        if block_indices is None:
            print("block_idx", block_idx)

            X_overlap, Y_overlap = self.data_loader.retrieve_ol_train_X_y(batch_range=overlap_batch_range)
            X_non_overlap, Y_non_overlap = self.data_loader.retrieve_nol_train_X_y(batch_range=non_overlap_batch_range)
            X_all_ref_block, Y_all_ref_block = self.data_loader.retrieve_ested_block(block_idx=block_idx)
        else:
            # overlap_batch_indices = self.overlap_indices[overlap_batch_range[0]: overlap_batch_range[1]]
            # non_overlap_batch_indices = self.non_overlap_indices[non_overlap_batch_range[0]: non_overlap_batch_range[1]]
            # print("num overlap_batch_indices", len(overlap_batch_indices))
            # print("num non_overlap_batch_indices", len(non_overlap_batch_indices))
            print("[DEBUG] num block_indices", len(block_indices))

            # X_overlap, Y_overlap = self.data_loader.retrieve_ol_train_X_y(overlap_batch_indices)
            # X_non_overlap, Y_non_overlap = self.data_loader.retrieve_nol_train_X_y(non_overlap_batch_indices)
            # X_all_ref_block, Y_all_ref_block = self.data_loader.retrieve_ested_train_X_y(block_indices)
            X_overlap, Y_overlap = self.data_loader.retrieve_ol_train_X_y(batch_range=overlap_batch_range)
            X_non_overlap, Y_non_overlap = self.data_loader.retrieve_nol_train_X_y(batch_range=non_overlap_batch_range)
            X_all_ref_block, Y_all_ref_block = self.data_loader.retrieve_ested_train_X_y(batch_indices=block_indices)

        # X_overlap = self.X_train[overlap_batch_indices]
        # X_non_overlap = self.X_train[non_overlap_batch_indices, :]
        # X_all_ref_block = self.X_train[block_indices]
        #
        # # Y_overlap = self.Y_train[self.overlap_indices]
        # Y_overlap = self.Y_train[overlap_batch_indices]
        # Y_non_overlap = self.Y_train[non_overlap_batch_indices, :]
        # Y_all_ref_block = self.Y_train[block_indices]

        print("[DEBUG] Guest X_overlap: {0}".format(len(X_overlap)))
        print("[DEBUG] Guest X_non_overlap: {0}".format(len(X_non_overlap)))
        print("[DEBUG] Guest X_all_ref_block: {0}".format(len(X_all_ref_block)))
        print("[DEBUG] Guest Y_overlap: {0}".format(len(Y_overlap)))
        print("[DEBUG] Guest Y_all_ref_block: {0}".format(len(Y_all_ref_block)))
        print("[DEBUG] Guest Y_non_overlap: {0}".format(len(Y_non_overlap)))

        # Y_overlap_for_est = Y_overlap - 1e-10
        # Y_all_ref_block_for_est = Y_all_ref_block - 1e-10
        Y_overlap_for_est = Y_overlap - 0.0
        Y_all_ref_block_for_est = Y_all_ref_block - 0.0

        feed_dict = {self.Y_all_in: Y_all_ref_block,
                     self.Y_overlap_in: Y_overlap,
                     self.Y_non_overlap_in: Y_non_overlap,
                     self.Y_all_in_for_est: Y_all_ref_block_for_est,
                     self.Y_overlap_in_for_est: Y_overlap_for_est,
                     self.local_model.get_all_samples(): X_all_ref_block,
                     self.local_model.get_overlap_samples(): X_overlap,
                     self.local_model.get_non_overlap_samples(): X_non_overlap,
                     self.local_model.get_is_train(): True,
                     self.local_model_prime.get_all_samples(): X_all_ref_block,
                     self.local_model_prime.get_overlap_samples(): X_overlap,
                     self.local_model_prime.get_non_overlap_samples(): X_non_overlap,
                     self.local_model_prime.get_is_train(): True}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): True,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): True,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_two_sides_predict_feed_dict(self):
        X_test, _ = self.data_loader.retrieve_test_X_y()
        feed_dict = {
            self.local_model.get_overlap_samples(): X_test,
            self.local_model.get_is_train(): False,
            self.local_model_prime.get_overlap_samples(): X_test,
            self.local_model_prime.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): False,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_one_side_predict_feed_dict(self):
        # X_overlap, _ = self.data_loader.retrieve_ol_train_X_y(self.overlap_indices)
        X_overlap, _ = self.data_loader.retrieve_all_ol_train_X_y()
        X_test, _ = self.data_loader.retrieve_test_X_y()

        # print("### get_one_side_predict_feed_dict")
        # print("X_overlap shape: {0}".format(X_overlap.shape))
        # print("X_test shape: {0}".format(X_test.shape))

        feed_dict = {
            self.local_model.get_overlap_samples(): X_overlap,
            self.local_model.get_non_overlap_samples(): X_test,
            self.local_model.get_is_train(): False,
            self.local_model_prime.get_non_overlap_samples(): X_test,
            self.local_model_prime.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): False,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_assist_host_side_predict_feed_dict(self, block_indices=None, block_idx=None):
        # print("get_assist_host_side_predict_feed_dict")
        if block_indices is None and block_idx is None:
            raise Exception("Both block_indices and block_idx are None")

        X_overlap, Y_overlap = self.data_loader.retrieve_all_ol_train_X_y()
        # X_overlap, Y_overlap = self.data_loader.retrieve_ol_train_X_y(self.overlap_indices)

        if block_indices is None:
            print("get_assist_host_side_predict_feed_dict using block_idx {0}".format(block_idx))
            X_all_ref_block, Y_all_ref_block = self.data_loader.retrieve_ested_block(block_idx)
        else:
            X_all_ref_block, Y_all_ref_block = self.data_loader.retrieve_ested_train_X_y(block_indices)

        # print("X_all_ref_block: ")
        # for idx, X_all_ref_block_i in enumerate(X_all_ref_block):
        #     if np.isnan(X_all_ref_block_i).any():
        #         print("X_all_ref_block_i", idx, X_all_ref_block_i)
        #     if np.all(X_all_ref_block_i == 0):
        #         print("X_all_ref_block_i", idx, X_all_ref_block_i)
        #
        # print("Y_all_ref_block: ")
        # for idx, Y_all_ref_block_i in enumerate(Y_all_ref_block):
        #     if np.isnan(Y_all_ref_block_i).any():
        #         print("Y_all_ref_block_i", idx, Y_all_ref_block_i)
        #     if np.all(Y_all_ref_block_i == 0):
        #         print("Y_all_ref_block_i", idx, Y_all_ref_block_i)
        #
        # print("X_overlap: ")
        # for idx, X_overlap_i in enumerate(X_overlap):
        #     if np.isnan(X_overlap_i).any():
        #         print("X_overlap_i", idx, X_overlap_i)
        #     if np.all(X_overlap_i == 0):
        #         print("X_overlap_i", idx, X_overlap_i)
        #
        # print("Y_overlap: ")
        # for idx, Y_overlap_i in enumerate(Y_overlap):
        #     if np.isnan(Y_overlap_i).any():
        #         print("Y_overlap_i", idx, Y_overlap_i)
        #     if np.all(Y_overlap_i == 0):
        #         print("Y_overlap_i", idx, Y_overlap_i)

        Y_overlap_for_est = Y_overlap - 0.0
        Y_all_ref_block_for_est = Y_all_ref_block - 0.0

        # print("### get_assist_host_side_predict_feed_dict")
        # print("X_overlap shape: {0}".format(X_overlap.shape))
        # print("Y_overlap shape: {0}".format(Y_overlap.shape))
        # print("X_all_ref_block shape: {0}".format(X_all_ref_block.shape))
        # print("Y_all_ref_block shape: {0}".format(Y_all_ref_block.shape))
        # print("Y_overlap_for_est shape: {0}".format(Y_overlap_for_est.shape))
        # print("Y_all_ref_block_for_est shape: {0}".format(Y_all_ref_block_for_est.shape))

        feed_dict = {self.Y_all_in: Y_all_ref_block,
                     self.Y_overlap_in: Y_overlap,
                     self.Y_all_in_for_est: Y_all_ref_block_for_est,
                     self.Y_overlap_in_for_est: Y_overlap_for_est,
                     self.local_model.get_overlap_samples(): X_overlap,
                     self.local_model.get_is_train(): False,
                     self.local_model_prime.get_all_samples(): X_all_ref_block,
                     self.local_model_prime.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): False,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_assist_host_distance_based_predict_feed_dict(self, block_indices=None, block_idx=None):
        if block_indices is None and block_idx is None:
            raise Exception("Both block_indices and block_idx are None")

        X_overlap, Y_overlap = self.data_loader.retrieve_all_ol_train_X_y()
        # X_overlap, Y_overlap = self.data_loader.retrieve_ol_train_X_y(self.overlap_indices)

        if block_indices is None:
            X_all_ref_block, Y_all_ref_block = self.data_loader.retrieve_ested_block(block_idx)
        else:
            X_all_ref_block, Y_all_ref_block = self.data_loader.retrieve_ested_train_X_y(block_indices)

        Y_overlap_for_est = Y_overlap - 0.0
        Y_all_ref_block_for_est = Y_all_ref_block - 0.0

        # print("### get_assist_host_side_predict_feed_dict")
        # print("X_overlap shape: {0}".format(X_overlap.shape))
        # print("Y_overlap shape: {0}".format(Y_overlap.shape))
        # print("X_all_ref_block shape: {0}".format(X_all_ref_block.shape))
        # print("Y_all_ref_block shape: {0}".format(Y_all_ref_block.shape))
        # print("Y_overlap_for_est shape: {0}".format(Y_overlap_for_est.shape))
        # print("Y_all_ref_block_for_est shape: {0}".format(Y_all_ref_block_for_est.shape))

        feed_dict = {
            self.Y_all_in_for_est: Y_all_ref_block_for_est,
            self.Y_overlap_in_for_est: Y_overlap_for_est,
            self.local_model.get_overlap_samples(): X_overlap,
            self.local_model.get_is_train(): False,
            self.local_model_prime.get_all_samples(): X_all_ref_block,
            self.local_model_prime.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): False,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict


class ExpandingVFTLHost(ExpandingVFTLParty):

    def __init__(self, party_model_param: PartyModelParam, data_loader: ExpandingVFTLDataLoader):
        super(ExpandingVFTLHost, self).__init__()
        self.party_model_param = party_model_param
        self.data_loader = data_loader
        self.keep_probability = party_model_param.keep_probability
        self.apply_dropout = party_model_param.apply_dropout

        self.current_ol_block_idx = None
        self.current_nol_block_idx = None
        self.current_ested_block_idx = None
        self.current_val_block_idx = None

        self.ol_block_num = 1
        self.nol_block_num = 1
        self.ested_block_num = 1
        self.val_block_num = 1

    def set_ol_block_number(self, block_number):
        self.ol_block_num = block_number

    def set_nol_block_number(self, block_number):
        self.nol_block_num = block_number

    def set_ested_block_number(self, block_number):
        self.ested_block_num = block_number

    def set_val_block_number(self, block_number):
        self.val_block_num = block_number

    def get_ol_block_number(self):
        return self.ol_block_num

    def get_nol_block_number(self):
        return self.nol_block_num

    def get_ested_block_number(self):
        return self.ested_block_num

    def get_val_block_number(self):
        return self.val_block_num

    def load_ol_block(self, block_idx):
        if self.current_ol_block_idx is not None and self.current_ol_block_idx == block_idx:
            return self.data_loader.get_overlap_block_size()
        self.current_ol_block_idx = block_idx
        return self.data_loader.load_overlap_block(block_idx)

    def load_nol_block(self, block_idx):
        if self.current_nol_block_idx is not None and self.current_nol_block_idx == block_idx:
            print("host here self.current_nol_block_idx, block_idx", self.current_nol_block_idx, block_idx)
            return self.data_loader.get_nonoverlap_block_size()
        self.current_nol_block_idx = block_idx
        return self.data_loader.load_nonoverlap_block(block_idx)

    def load_ested_block(self, block_idx):
        if self.current_ested_block_idx is not None and self.current_ested_block_idx == block_idx:
            return self.data_loader.get_ested_block_size()
        self.current_ested_block_idx = block_idx
        return self.data_loader.load_ested_block(block_idx)

    def load_val_block(self, block_idx):
        if self.current_val_block_idx is not None and self.current_val_block_idx == block_idx:
            return self.data_loader.get_val_block_size()
        self.current_val_block_idx = block_idx
        return self.data_loader.load_val_block(block_idx)

    def get_all_training_sample_size(self):
        return self.data_loader.get_all_training_sample_size()

    def get_train_feed_dict(self, overlap_batch_range, non_overlap_batch_range, block_indices=None, block_idx=None):
        if block_indices is None and block_idx is None:
            raise Exception("Both block_indices and block_idx are None")

        print("[DEBUG] host overlap_batch_range", overlap_batch_range)
        print("[DEBUG] host non_overlap_batch_range", non_overlap_batch_range)

        if block_indices is None:
            X_overlap, _ = self.data_loader.retrieve_ol_train_X_y(batch_range=overlap_batch_range)
            X_non_overlap, _ = self.data_loader.retrieve_nol_train_X_y(batch_range=non_overlap_batch_range)
            X_all_ref_block, _ = self.data_loader.retrieve_ested_block(block_idx=block_idx)
        else:
            X_overlap, _ = self.data_loader.retrieve_ol_train_X_y(batch_range=overlap_batch_range)
            X_non_overlap, _ = self.data_loader.retrieve_nol_train_X_y(batch_range=non_overlap_batch_range)
            X_all_ref_block, _ = self.data_loader.retrieve_ested_train_X_y(batch_indices=block_indices)

        print("Host X_non_overlap {0}".format(len(X_non_overlap)))
        # for idx, X_non_overlap_i in enumerate(X_non_overlap):
        #     print("host X_non_overlap {0}:{1} with shape {2}".format(idx, X_non_overlap_i.flatten(),
        #                                                              X_non_overlap_i.shape))
        #     if idx >= 2:
        #         break

        print("Host X_overlap: {0}".format(len(X_overlap)))
        # for idx, X_overlap_i in enumerate(X_overlap):
        #     print("host X_overlap {0}:{1} with shape {2}".format(idx, X_overlap_i.flatten(), X_overlap_i.shape))
        #     if idx >= 2:
        #         break

        print("Host X_all_ref_block: {0}".format(len(X_all_ref_block)))

        feed_dict = {
            self.local_model.get_all_samples(): X_all_ref_block,
            self.local_model.get_overlap_samples(): X_overlap,
            self.local_model.get_non_overlap_samples(): X_non_overlap,
            self.local_model.get_is_train(): True,
            self.local_model_prime.get_all_samples(): X_all_ref_block,
            self.local_model_prime.get_overlap_samples(): X_overlap,
            self.local_model_prime.get_non_overlap_samples(): X_non_overlap,
            self.local_model_prime.get_is_train(): True}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): True,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): True,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_two_sides_predict_feed_dict(self):
        X_test, _ = self.data_loader.retrieve_test_X_y()
        feed_dict = {
            self.local_model.get_overlap_samples(): X_test,
            self.local_model.get_is_train(): False,
            self.local_model_prime.get_overlap_samples(): X_test,
            self.local_model_prime.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): False,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_one_side_predict_feed_dict(self):
        # print("host get_one_side_predict_feed_dict")
        # X_overlap, _ = self.data_loader.retrieve_ol_train_X_y(self.overlap_indices)
        X_overlap, _ = self.data_loader.retrieve_all_ol_train_X_y()
        X_test, _ = self.data_loader.retrieve_test_X_y()

        print("X_overlap: ")
        for idx, X_overlap_i in enumerate(X_overlap):
            if np.isnan(X_overlap_i).any():
                print("X_overlap_i", idx, X_overlap_i)
            if np.all(X_overlap_i == 0):
                print("X_overlap_i", idx, X_overlap_i)

        print("original X_test: ", X_test.shape)
        # X_test = X_test[:10]
        # X_test_show = X_test[:5]
        # print("X_test_show: ", X_test_show.shape)
        # for idx, X_test_i in enumerate(X_test_show):
        #     print("X_test_i", idx, X_test_i, np.sum(X_test_i))
        #     if np.isnan(X_test_i).any():
        #         print("X_test_i", idx, X_test_i, np.sum(X_test_i))
        #     if np.all(X_test_i == 0):
        #         print("X_test_i", idx, X_test_i, np.sum(X_test_i))
        # if idx > 50:
        #     break

        feed_dict = {
            self.local_model.get_overlap_samples(): X_overlap,
            self.local_model.get_non_overlap_samples(): X_test,
            self.local_model.get_is_train(): False,
            self.local_model_prime.get_non_overlap_samples(): X_test,
            self.local_model_prime.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): False,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_one_side_distance_based_predict_feed_dict(self):
        # X_overlap, _ = self.data_loader.retrieve_ol_train_X_y(self.overlap_indices)
        X_overlap, _ = self.data_loader.retrieve_all_ol_train_X_y()
        X_test, _ = self.data_loader.retrieve_test_X_y()

        # X_test = X_test[:10]
        feed_dict = {
            self.local_model.get_overlap_samples(): X_overlap,
            self.local_model.get_non_overlap_samples(): X_test,
            self.local_model.get_is_train(): False,
            self.local_model_prime.get_non_overlap_samples(): X_test,
            self.local_model_prime.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): False,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_assist_guest_predict_feed_dict(self, block_indices=None, block_idx=None):
        if block_indices is None and block_idx is None:
            raise Exception("Both block_indices and block_idx are None")

        # X_overlap, _ = self.data_loader.retrieve_ol_train_X_y(self.overlap_indices)
        X_overlap, _ = self.data_loader.retrieve_all_ol_train_X_y()

        if block_indices is None:
            print("get_assist_guest_predict_feed_dict using block_idx {0}".format(block_idx))
            X_all_ref_block, _ = self.data_loader.retrieve_ested_block(block_idx)
        else:
            X_all_ref_block, _ = self.data_loader.retrieve_ested_train_X_y(block_indices)

        # print("### get_assist_guest_predict_feed_dict")
        # print("X_overlap shape: {0}".format(X_overlap.shape))
        # print("X_all_ref_block shape: {0}".format(X_all_ref_block.shape))

        feed_dict = {
            self.local_model.get_overlap_samples(): X_overlap,
            self.local_model.get_is_train(): False,
            self.local_model_prime.get_all_samples(): X_all_ref_block,
            self.local_model_prime.get_is_train(): False
        }

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability,
                                self.local_model_prime.get_is_train(): False,
                                self.local_model_prime.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict
