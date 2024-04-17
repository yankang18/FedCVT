import pickle

import torch
import torch.nn.functional as F

from fedcvt_core.param import PartyModelParam


def load_data_block(block_file_full_path, block_id):
    filename = block_file_full_path + str(block_id) + '.p'
    print("load data block: {0}".format(filename))
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


def get_len(X):
    return 0 if X is None else len(X)


def get_shape(X):
    return 0 if X is None else X.shape


class PartyDataLoader(object):
    def __init__(self, data_folder_path=None, is_guest=True, X_ul_train=None,
                 X_ll_train=None, X_nol_train=None, X_ested_train=None, X_test=None,
                 Y_ll_train=None, Y_nol_train=None, Y_ested_train=None, Y_test=None):
        self.data_folder_path = data_folder_path
        self.X_ul_train = X_ul_train
        self.X_ll_train = X_ll_train
        self.Y_ll_train = Y_ll_train
        self.X_nol_train = X_nol_train
        self.Y_nol_train = Y_nol_train
        self.X_ested_train = X_ested_train
        self.Y_ested_train = Y_ested_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.is_pre_load_data = True if self.data_folder_path is None else False

        party_type = "Guest" if is_guest else "Host"
        if not self.is_pre_load_data:
            print("[INFO] # ===> {}'s data will be loaded online.".format(party_type))
            if is_guest:
                self.ul_overlap_block_file_full_path = self.data_folder_path + 'guest_ul_overlap_block_'
                self.ll_overlap_block_file_full_path = self.data_folder_path + 'guest_ll_overlap_block_'
                self.nonoverlap_block_file_full_path = self.data_folder_path + 'guest_nonoverlap_block_'
                self.ested_block_file_full_path = self.data_folder_path + 'guest_ested_block_'
                self.val_block_file_full_path = self.data_folder_path + 'guest_val_block_'
            else:
                self.ul_overlap_block_file_full_path = self.data_folder_path + 'host_ul_overlap_block_'
                self.ll_overlap_block_file_full_path = self.data_folder_path + 'host_ll_overlap_block_'
                self.nonoverlap_block_file_full_path = self.data_folder_path + 'host_nonoverlap_block_'
                self.ested_block_file_full_path = self.data_folder_path + 'host_ested_block_'
                self.val_block_file_full_path = self.data_folder_path + 'host_val_block_'
        else:
            print("[INFO] #===> {}'s data has been preloaded.".format(party_type))
            if is_guest:
                print("[INFO] X_ul_train: ", get_len(self.X_ul_train))
                print("[INFO] X_ll_train: ", len(self.X_ll_train))
                print("[INFO] Y_ll_train: ", len(self.Y_ll_train))
                print("[INFO] X_nol_train: ", len(self.X_nol_train))
                print("[INFO] Y_nol_train: ", len(self.Y_nol_train))
                print("[INFO] X_ested_train: ", len(self.X_ested_train))
                print("[INFO] Y_ested_train: ", len(self.Y_ested_train))
                print("[INFO] X_test: ", len(self.X_test))
                print("[INFO] Y_test: ", len(self.Y_test))
            else:
                print("[INFO] X_ul_train: ", get_len(self.X_ul_train))
                print("[INFO] X_ll_train: ", len(self.X_ll_train))
                print("[INFO] X_nol_train: ", len(self.X_nol_train))
                print("[INFO] X_ested_train: ", len(self.X_ested_train))
                print("[INFO] X_test: ", len(self.X_test))

    def get_labeled_overlap_block_size(self):
        return len(self.X_ll_train)

    def get_unlabeled_overlap_block_size(self):
        return get_len(self.X_ul_train)

    def get_nonoverlap_block_size(self):
        return len(self.X_nol_train)

    def get_ested_block_size(self):
        return len(self.X_ested_train)

    def get_val_block_size(self):
        return len(self.X_test)

    def load_labeled_overlap_block(self, block_id):
        if not self.is_pre_load_data:
            self.X_ll_train, self.Y_ll_train = load_data_block(self.ll_overlap_block_file_full_path, block_id)
        return len(self.X_ll_train)

    def load_unlabeled_overlap_block(self, block_id):
        if not self.is_pre_load_data:
            self.X_ul_train = load_data_block(self.ul_overlap_block_file_full_path, block_id)
        return get_len(self.X_ul_train)

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

        if X is None and Y is None:
            return X, Y

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

    def retrieve_batch_ul_train_X(self, batch_indices=None, batch_range=None):
        X, _ = self.retrieve_X_y(self.X_ul_train, None, batch_indices, batch_range)
        return X

    def retrieve_all_ul_train_X(self):
        return self.X_ul_train

    def retrieve_batch_ll_train_X_y(self, batch_indices=None, batch_range=None):
        return self.retrieve_X_y(self.X_ll_train, self.Y_ll_train, batch_indices, batch_range)

    def retrieve_all_ll_train_X_y(self):
        return self.X_ll_train, self.Y_ll_train

    def retrieve_batch_nol_train_X_y(self, batch_indices=None, batch_range=None):
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


class BaseVFLParty(object):

    def __init__(self, party_model_param: PartyModelParam):
        self.party_model_param = party_model_param
        self.local_model = None
        self.local_model_prime = None

        self.X_all_in = None
        self.X_ll_overlap_in = None
        self.X_ul_overlap_in = None
        self.X_non_overlap_in = None

    def set_model(self, feature_extractor, feature_extractor_prime):
        self.local_model = feature_extractor
        self.local_model_prime = feature_extractor_prime

    def get_model_parameters(self):
        return list(self.local_model.parameters()) + list(self.local_model_prime.parameters())

    def to_train_mode(self):
        self.local_model.train()
        self.local_model_prime.train()

    def to_eval_mode(self):
        self.local_model.eval()
        self.local_model_prime.eval()

    def fetch_feat_reprs(self):
        # print("X_all_in:", self.X_all_in.shape)
        # print("local_model:", self.local_model)

        U_all_uniq = self.local_model(self.X_all_in)
        # print("X_all_in:", self.X_all_in.shape)
        # print("U_all_uniq:", U_all_uniq.shape)
        U_all_comm = self.local_model_prime(self.X_all_in)

        U_ll_overlap_uniq = self.local_model(self.X_ll_overlap_in)
        U_ll_overlap_comm = self.local_model_prime(self.X_ll_overlap_in)
        U_ul_overlap_uniq = self.X_ul_overlap_in if self.X_ul_overlap_in is None else self.local_model(
            self.X_ul_overlap_in)
        U_ul_overlap_comm = self.X_ul_overlap_in if self.X_ul_overlap_in is None else self.local_model_prime(
            self.X_ul_overlap_in)

        # print("X_ll_overlap_in:", self.X_ll_overlap_in.shape)
        # print("X_ul_overlap_in:", self.X_ul_overlap_in.shape)
        # print("X_non_overlap_in:", self.X_non_overlap_in.shape)

        U_non_overlap_uniq = self.local_model(self.X_non_overlap_in)
        U_non_overlap_comm = self.local_model_prime(self.X_non_overlap_in)

        if self.party_model_param.normalize_repr:
            print("[DEBUG] normalize features.")
            U_all_uniq = F.normalize(U_all_uniq, p=2.0, dim=1)
            U_all_comm = F.normalize(U_all_comm, p=2.0, dim=1)
            U_ll_overlap_uniq = F.normalize(U_ll_overlap_uniq, p=2.0, dim=1)
            U_ll_overlap_comm = F.normalize(U_ll_overlap_comm, p=2.0, dim=1)
            U_ul_overlap_uniq = U_ul_overlap_uniq if U_ul_overlap_uniq is None else F.normalize(U_ul_overlap_uniq,
                                                                                                p=2.0, dim=1)
            U_ul_overlap_comm = U_ul_overlap_comm if U_ul_overlap_comm is None else F.normalize(U_ul_overlap_comm,
                                                                                                p=2.0, dim=1)
            U_non_overlap_uniq = F.normalize(U_non_overlap_uniq, p=2.0, dim=1)
            U_non_overlap_comm = F.normalize(U_non_overlap_comm, p=2.0, dim=1)

        U_ul_overlap_tuple = (U_ul_overlap_uniq, U_ul_overlap_comm) \
            if U_ul_overlap_uniq is not None and U_ul_overlap_comm is not None else None
        return (U_all_uniq, U_all_comm), \
               (U_non_overlap_uniq, U_non_overlap_comm), \
               (U_ll_overlap_uniq, U_ll_overlap_comm), \
               U_ul_overlap_tuple

    def local_predict(self, x):
        U_uniq = self.local_model(x)
        U_comm = self.local_model_prime(x)

        if self.party_model_param.normalize_repr:
            print("[DEBUG] normalize features.")
            U_uniq = F.normalize(U_uniq, p=2.0, dim=1)
            U_comm = F.normalize(U_comm, p=2.0, dim=1)

        return U_uniq, U_comm


class VFTLGuest(BaseVFLParty):

    def __init__(self, party_model_param: PartyModelParam, debug=False):
        super(VFTLGuest, self).__init__(party_model_param)
        self.party_model_param = party_model_param
        self.keep_probability = party_model_param.keep_probability
        self.n_classes = party_model_param.n_classes
        self.apply_dropout = party_model_param.apply_dropout
        self.device = party_model_param.device
        self.debug = debug

        self.Y_all_in_for_est = None
        self.Y_ll_overlap_in_for_est = None

        self.Y_all_in = None
        self.Y_ll_overlap_in = None
        self.Y_non_overlap_in = None

    def get_Y_ll_overlap_for_est(self):
        return self.Y_ll_overlap_in_for_est

    def get_Y_ll_overlap(self):
        return self.Y_ll_overlap_in

    def get_Y_nl_overlap(self):
        return self.Y_non_overlap_in

    def get_Y_nl_overlap_for_est(self):
        return self.Y_nl_overlap_in_for_est

    def prepare_local_data(self,
                           guest_ll_x,
                           ll_y,
                           guest_ul_x,
                           ul_y,
                           guest_nl_x,
                           guest_nl_y,
                           guest_all_x,
                           guest_all_y):
        X_ul_overlap = guest_ul_x
        X_ll_overlap, Y_ll_overlap = guest_ll_x, ll_y
        X_non_overlap, Y_non_overlap = guest_nl_x, guest_nl_y
        X_all_ref_block, Y_all_ref_block = guest_all_x, guest_all_y

        if self.debug:
            print("[DEBUG] Guest X_ll_overlap: {0}; {1}.".format(len(X_ll_overlap), X_ll_overlap.shape))
            print("[DEBUG] Guest X_ul_overlap: {0}; {1}.".format(get_len(X_ul_overlap), get_shape(X_ul_overlap)))
            print("[DEBUG] Guest X_non_overlap: {0}; {1}.".format(len(X_non_overlap), X_non_overlap.shape))
            print("[DEBUG] Guest X_all_ref_block: {0}; {1}.".format(len(X_all_ref_block), X_all_ref_block.shape))
            print("[DEBUG] Guest Y_ll_overlap: {0}; {1}.".format(len(Y_ll_overlap), Y_ll_overlap.shape))
            print("[DEBUG] Guest Y_all_ref_block: {0}; {1}.".format(len(Y_all_ref_block), Y_all_ref_block.shape))
            print("[DEBUG] Guest Y_non_overlap: {0}; {1}.".format(len(Y_non_overlap), Y_all_ref_block.shape))

        self.X_ul_overlap_in = X_ul_overlap if X_ul_overlap is None else X_ul_overlap.to(self.device)
        self.X_ll_overlap_in = X_ll_overlap.to(self.device)
        self.X_non_overlap_in = X_non_overlap.to(self.device)
        self.X_all_in = X_all_ref_block.to(self.device)

        # print("old Y_ll_overlap_:", Y_ll_overlap.shape)
        # print("old Y_non_overlap_:", Y_non_overlap.shape)
        # print("old Y_all_ref_block_:", Y_all_ref_block.shape)

        Y_ll_overlap_ = F.one_hot(Y_ll_overlap, num_classes=self.n_classes)
        Y_non_overlap_ = F.one_hot(Y_non_overlap, num_classes=self.n_classes)
        Y_all_ref_block_ = F.one_hot(Y_all_ref_block, num_classes=self.n_classes)

        # print("Y_ll_overlap_:", Y_ll_overlap_.shape)
        # print("Y_non_overlap_:", Y_non_overlap_.shape)
        # print("Y_all_ref_block_:", Y_all_ref_block_.shape)

        self.Y_ll_overlap_in = Y_ll_overlap_.to(self.device)
        self.Y_non_overlap_in = Y_non_overlap_.to(self.device)
        self.Y_all_in = Y_all_ref_block_.to(self.device)

        self.Y_ll_overlap_in_for_est = Y_ll_overlap_.to(self.device, dtype=torch.float32)
        self.Y_nl_overlap_in_for_est = Y_non_overlap_.to(self.device, dtype=torch.float32)
        self.Y_all_in_for_est = Y_all_ref_block_.to(self.device, dtype=torch.float32)


class VFLHost(BaseVFLParty):

    def __init__(self, party_model_param: PartyModelParam, debug=False):
        super(VFLHost, self).__init__(party_model_param)
        self.party_model_param = party_model_param
        self.keep_probability = party_model_param.keep_probability
        self.apply_dropout = party_model_param.apply_dropout
        self.device = party_model_param.device
        self.debug = debug

    def prepare_local_data(self,
                           host_ll_x,
                           ll_y,
                           host_ul_x,
                           ul_y,
                           host_nl_x,
                           host_nl_y,
                           host_all_x,
                           host_all_y):
        X_ul_overlap = host_ul_x
        X_ll_overlap = host_ll_x
        X_non_overlap = host_nl_x
        X_all_ref_block = host_all_x

        if self.debug:
            print("[DEBUG] Host X_ul_overlap: {0}; {1}".format(get_len(X_ul_overlap), get_shape(X_ul_overlap)))
            print("[DEBUG] Host X_ll_overlap: {0}; {1}".format(len(X_ll_overlap), X_ll_overlap.shape))
            print("[DEBUG] Host X_non_overlap: {0}; {1}".format(len(X_non_overlap), X_non_overlap.shape))
            print("[DEBUG] Host X_all_ref_block: {0}; {1}".format(len(X_all_ref_block), X_all_ref_block.shape))

        self.X_ul_overlap_in = X_ul_overlap if X_ul_overlap is None else X_ul_overlap.to(self.device)
        self.X_ll_overlap_in = X_ll_overlap.to(self.device)
        self.X_non_overlap_in = X_non_overlap.to(self.device)
        self.X_all_in = X_all_ref_block.to(self.device)

