import pickle

import tensorflow as tf
import numpy as np
from autoencoder import FeatureExtractor
from expanding_vertical_transfer_learning_param import PartyModelParam


class ExpandingVFTLParty(object):

    def __init__(self):
        self.local_model: FeatureExtractor = None

    def set_model(self, feature_extractor: FeatureExtractor):
        self.local_model = feature_extractor

    def set_session(self, sess):
        self.local_model.set_session(sess)

    def get_model_parameters(self):
        return self.local_model.get_model_parameters()

    def fetch_feat_reprs(self):
        return self.local_model.get_overlap_hidden_reprs()


def load_preprocess_training_minibatch(block_file_full_path, block_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """

    features, labels = load_training_batch(block_file_full_path, block_id)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def load_training_batch(block_file_full_path, block_id):
    filename = block_file_full_path + str(block_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


def load_data_block(block_file_full_path, block_id):
    filename = block_file_full_path + str(block_id) + '.p'
    print("load data block: {0}".format(filename))
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


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

        self.Y_overlap_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_class), name="labels_input_overlap")

        self.current_ol_block_idx = None
        self.current_val_block_idx = None

        self.ol_block_num = 1
        self.val_block_num = 1

    def set_ol_block_number(self, block_number):
        self.ol_block_num = block_number

    def set_val_block_number(self, block_number):
        self.val_block_num = block_number

    def get_ol_block_number(self):
        return self.ol_block_num

    def get_val_block_number(self):
        return self.val_block_num

    def load_ol_block(self, block_idx):
        if self.current_ol_block_idx is not None and self.current_ol_block_idx == block_idx:
            return self.data_loader.get_overlap_block_size()
        self.current_ol_block_idx = block_idx
        return self.data_loader.load_overlap_block(block_idx)

    def load_val_block(self, block_idx):
        if self.current_val_block_idx is not None and self.current_val_block_idx == block_idx:
            return self.data_loader.get_val_block_size()
        self.current_val_block_idx = block_idx
        return self.data_loader.load_val_block(block_idx)

    def get_all_training_sample_size(self):
        return self.data_loader.get_all_training_sample_size()

    def get_Y_test(self):
        return self.data_loader.get_y()

    def get_Y_overlap(self):
        return self.Y_overlap_in

    def get_number_of_class(self):
        return self.n_class

    def get_train_feed_dict(self, overlap_batch_range):

        print("overlap_batch_range", overlap_batch_range)
        X_overlap, Y_overlap = self.data_loader.retrieve_ol_train_X_y(batch_range=overlap_batch_range)

        print("Guest X_overlap: {0}".format(len(X_overlap)))
        print("Guest Y_overlap: {0}".format(len(Y_overlap)))

        feed_dict = {
            self.Y_overlap_in: Y_overlap,
            self.local_model.get_overlap_samples(): X_overlap,
            self.local_model.get_is_train(): True,
        }

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): True,
                                self.local_model.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_two_sides_predict_feed_dict(self):
        X_test, _ = self.data_loader.retrieve_test_X_y()
        feed_dict = {
            self.local_model.get_overlap_samples(): X_test,
            self.local_model.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability}
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
        self.current_val_block_idx = None

        self.ol_block_num = 1
        self.val_block_num = 1

    def set_ol_block_number(self, block_number):
        self.ol_block_num = block_number

    def set_val_block_number(self, block_number):
        self.val_block_num = block_number

    def get_ol_block_number(self):
        return self.ol_block_num

    def get_val_block_number(self):
        return self.val_block_num

    def load_ol_block(self, block_idx):
        if self.current_ol_block_idx is not None and self.current_ol_block_idx == block_idx:
            return self.data_loader.get_overlap_block_size()
        self.current_ol_block_idx = block_idx
        return self.data_loader.load_overlap_block(block_idx)

    def load_val_block(self, block_idx):
        if self.current_val_block_idx is not None and self.current_val_block_idx == block_idx:
            return self.data_loader.get_val_block_size()
        self.current_val_block_idx = block_idx
        return self.data_loader.load_val_block(block_idx)

    def get_all_training_sample_size(self):
        return self.data_loader.get_all_training_sample_size()

    def get_train_feed_dict(self, overlap_batch_range):

        X_overlap, _ = self.data_loader.retrieve_ol_train_X_y(batch_range=overlap_batch_range)

        feed_dict = {
            self.local_model.get_overlap_samples(): X_overlap,
            self.local_model.get_is_train(): True}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): True,
                                self.local_model.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict

    def get_two_sides_predict_feed_dict(self):
        X_test, _ = self.data_loader.retrieve_test_X_y()
        feed_dict = {
            self.local_model.get_overlap_samples(): X_test,
            self.local_model.get_is_train(): False}

        if self.apply_dropout:
            feed_dict_others = {self.local_model.get_is_train(): False,
                                self.local_model.get_keep_probability(): self.keep_probability}
            feed_dict.update(feed_dict_others)

        return feed_dict
