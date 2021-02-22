import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from autoencoder import Autoencoder
from expanding_vertical_transfer_learning_feat_extr import RankedRepresentationLearner
from expanding_vertical_transfer_learning_ops import tf_gather_with_dynamic_partition
from expanding_vertical_transfer_learning_param import FederatedModelParam
from logistic_regression import LogisticRegression
from data_visualization import series_plot
from regularization import EarlyStoppingCheckPoint


class ExpandingVFTLParty(object):

    def __init__(self):
        self.nn = None
        self.nn_prime = None
        self.X_train = None
        self.X_test = None
        self.X_non_overlap = None
        self.overlap_indices = None
        self.non_overlap_indices = None
        self.X_non_overlap = None

    def set_model(self, nn: Autoencoder, nn_prime: Autoencoder):
        self.nn = nn
        self.nn_prime = nn_prime

    def set_session(self, sess):
        self.nn.set_session(sess)
        self.nn_prime.set_session(sess)

    def get_comm_feat_repr_dim(self):
        return self.nn_prime.get_encode_dim()

    def get_model_parameters(self):
        return self.nn.get_model_parameters(), self.nn_prime.get_model_parameters()

    def set_X_train(self, X_train):
        self.X_train = X_train

    def set_X_test(self, X_test):
        self.X_test = X_test

    def set_non_overlap_indices(self, non_overlap_indices):
        self.non_overlap_indices = non_overlap_indices

    def prepare_data(self, overlap_indices):
        self.overlap_indices = overlap_indices
        self.non_overlap_indices = np.setdiff1d(range(self.X_train.shape[0]), self.overlap_indices)
        print("non_overlap_indices length:", len(self.non_overlap_indices))
        # self.X_non_overlap = self.X[self.non_overlap_indices]

    def fetch_uniq_feat_reprs_of_all_samples(self):
        return self.nn.get_all_hidden_reprs()

    def fetch_comm_feat_reprs_of_all_samples(self):
        return self.nn_prime.get_all_hidden_reprs()

    def fetch_overlap_feat_reprs(self):
        # U_all_uniq = self.nn.get_all_hidden_reprs()
        # U_all_comm = self.nn_prime.get_all_hidden_reprs()

        U_overlap_uniq = self.nn.get_overlap_hidden_reprs()
        U_overlap_comm = self.nn_prime.get_overlap_hidden_reprs()

        # U_overlap_uniq = tf_gather_with_dynamic_partition(U_all_uniq, self.overlap_indices)
        # U_overlap_comm = tf_gather_with_dynamic_partition(U_all_comm, self.overlap_indices)

        return U_overlap_uniq, U_overlap_comm

    def fetch_feat_reprs(self):
        U_all_uniq = self.nn.get_all_hidden_reprs()
        U_all_comm = self.nn_prime.get_all_hidden_reprs()

        U_overlap_uniq = self.nn.get_overlap_hidden_reprs()
        U_overlap_comm = self.nn_prime.get_overlap_hidden_reprs()

        U_non_overlap_uniq = self.nn.get_non_overlap_hidden_reprs()
        U_non_overlap_comm = self.nn_prime.get_non_overlap_hidden_reprs()

        # U_overlap_uniq = tf_gather_with_dynamic_partition(U_all_uniq, self.overlap_indices)
        # U_overlap_comm = tf_gather_with_dynamic_partition(U_all_comm, self.overlap_indices)
        #
        # U_non_overlap_uniq = tf_gather_with_dynamic_partition(U_all_uniq, self.non_overlap_indices)
        # U_non_overlap_comm = tf_gather_with_dynamic_partition(U_all_comm, self.non_overlap_indices)

        return U_all_uniq, U_overlap_uniq, U_non_overlap_uniq, U_all_comm, U_overlap_comm, U_non_overlap_comm

    def get_non_overlap_indices(self):
        return self.non_overlap_indices

    def get_all_training_sample_size(self):
        return len(self.X_train)

    def get_non_overlapping_training_sample_size(self):
        return len(self.non_overlap_indices)

    def get_overlap_indices(self):
        return self.overlap_indices

    def get_overlap_training_sample_size(self):
        return len(self.overlap_indices)


class ExpandingVFTLGuest(ExpandingVFTLParty):

    def __init__(self):
        super(ExpandingVFTLGuest, self).__init__()
        self.Y_all_in = tf.compat.v1.placeholder(tf.float64, shape=(None, 1), name="labels_input_all")
        self.Y_overlap_in = tf.compat.v1.placeholder(tf.float64, shape=(None, 1), name="labels_input_overlap")
        self.Y_non_overlap_in = tf.compat.v1.placeholder(tf.float64, shape=(None, 1), name="labels_input_non_overlap")
        self.guest_train_labels = tf.compat.v1.placeholder(tf.float64, shape=(None, 1), name="guest_train_labels_input")

    def set_Y_train(self, Y_train):
        self.Y_train = Y_train

    def set_Y_test(self, Y_test):
        self.Y_test = Y_test

    def get_Y_test(self):
        return self.Y_test

    def get_Y_all(self):
        return self.Y_all_in

    def get_Y_overlap(self):
        return self.Y_overlap_in

    def get_Y_non_overlap(self):
        return self.Y_non_overlap_in

    def get_guest_train_labels(self):
        return self.guest_train_labels

    def get_host_alone_train_feed_dict(self, block_indices):
        X_overlap = self.X_train[self.overlap_indices]
        Y_overlap = self.Y_train[self.overlap_indices]
        X_all_ref_block = self.X_train[block_indices]
        Y_all_ref_block = self.Y_train[block_indices]
        feed_dict = {self.Y_all_in: Y_all_ref_block,
                     self.Y_overlap_in: Y_overlap,
                     self.nn_prime.X_all_in: X_all_ref_block,
                     self.nn_prime.X_overlap_in: X_overlap}
        return feed_dict

    def get_host_alone_predict_feed_dict(self, block_indices):
        # X_overlap = self.X_train[self.overlap_indices]
        X_all_ref_block = self.X_train[block_indices]
        Y_overlap = self.Y_train[self.overlap_indices]
        Y_all_ref_block = self.Y_train[block_indices]
        feed_dict = {self.Y_all_in: Y_all_ref_block,
                     self.Y_overlap_in: Y_overlap,
                     self.nn_prime.X_all_in: X_all_ref_block}
        return feed_dict

    def get_train_feed_dict(self, non_overlap_batch_range, block_indices):

        X_overlap = self.X_train[self.overlap_indices]

        non_overlap_indices_batch = self.non_overlap_indices[non_overlap_batch_range[0]: non_overlap_batch_range[1]]
        print("guest non_overlap_indices_batch shape:", len(non_overlap_indices_batch))
        X_non_overlap = self.X_train[non_overlap_indices_batch, :]
        X_all_ref_block = self.X_train[block_indices]

        Y_overlap = self.Y_train[self.overlap_indices]
        Y_non_overlap = self.Y_train[non_overlap_indices_batch, :]
        Y_all_ref_block = self.Y_train[block_indices]

        feed_dict = {self.Y_all_in: Y_all_ref_block,
                     self.Y_overlap_in: Y_overlap,
                     self.Y_non_overlap_in: Y_non_overlap,
                     self.nn.X_all_in: X_all_ref_block,
                     self.nn.X_overlap_in: X_overlap,
                     self.nn.X_non_overlap_in: X_non_overlap,
                     self.nn_prime.X_all_in: X_all_ref_block,
                     self.nn_prime.X_overlap_in: X_overlap,
                     self.nn_prime.X_non_overlap_in: X_non_overlap}
        return feed_dict

    def get_two_sides_predict_feed_dict(self, combine_axis=0):
        # if combine_axis == 0:
        #     feed_dict = {self.nn.X_all_in: self.X_test}
        # else:
        #     feed_dict = {self.nn.X_all_in: self.X_test, self.nn_prime.X_all_in: self.X_test}
        # return feed_dict

        if combine_axis == 0:
            feed_dict = {self.nn.X_overlap_in: self.X_test}
        else:
            feed_dict = {self.nn.X_overlap_in: self.X_test, self.nn_prime.X_overlap_in: self.X_test}
        return feed_dict

    def get_self_train_feed_dict(self, batch_range):

        # X_overlap = self.X_train[self.overlap_indices]
        # X_non_overlap = self.X_train[batch_range[0]: batch_range[1], :]
        # Y_non_overlap = self.Y_train[batch_range[0]: batch_range[1], :]

        X_overlap = self.X_train[self.overlap_indices]
        Y_overlap = self.Y_train[self.overlap_indices]

        # non_overlap_indices_batch = self.non_overlap_indices[non_overlap_batch_range[0]: non_overlap_batch_range[1]]
        # X_non_overlap = self.X_train[non_overlap_indices_batch, :]
        # Y_non_overlap = self.Y_train[non_overlap_indices_batch, :]

        feed_dict = {
                     self.guest_train_labels: Y_overlap,
                     self.nn.X_overlap_in: X_overlap,
                     self.nn.X_non_overlap_in: X_overlap,
                     self.nn_prime.X_non_overlap_in: X_overlap
                     }
        return feed_dict

    def get_self_predict_feed_dict(self, combine_axis=0):
        if combine_axis == 0:
            feed_dict = {self.nn.X_non_overlap_in: self.X_test}
        else:
            X_overlap = self.X_train[self.overlap_indices]
            feed_dict = {self.nn.X_overlap_in: X_overlap,
                         self.nn.X_non_overlap_in: self.X_test,
                         self.nn_prime.X_non_overlap_in: self.X_test}
        return feed_dict

    def get_host_predict_feed_dict(self, block_indices, combine_axis=0):
        if combine_axis == 0:
            feed_dict = {self.nn.X_non_overlap_in: self.X_test}
        else:
            Y_overlap = self.Y_train[self.overlap_indices]
            Y_all_ref_block = self.Y_train[block_indices]
            X_overlap = self.X_train[self.overlap_indices]
            X_all_ref_block = self.X_train[block_indices]
            feed_dict = {self.Y_all_in: Y_all_ref_block,
                         self.Y_overlap_in: Y_overlap,
                         self.nn.X_overlap_in: X_overlap,
                         self.nn_prime.X_all_in: X_all_ref_block}
        return feed_dict


class ExpandingVFTLHost(ExpandingVFTLParty):

    def __init__(self):
        super(ExpandingVFTLHost, self).__init__()

    def get_host_alone_train_feed_dict(self):
        X_overlap = self.X_train[self.overlap_indices]
        feed_dict = {self.nn_prime.X_overlap_in: X_overlap}
        return feed_dict

    def get_host_alone_predict_feed_dict(self):
        # feed_dict = {self.nn_prime.X_overlap_in: self.X_test}
        X_overlap = self.X_train[self.overlap_indices]
        feed_dict = {self.nn.X_overlap_in: X_overlap,
                     self.nn_prime.X_non_overlap_in: self.X_test,
                     self.nn.X_non_overlap_in: self.X_test}
        return feed_dict

    def get_train_feed_dict(self, non_overlap_batch_range, block_indices):
        # feed_dict = {self.nn.X_in: self.X_train, self.nn_prime.X_in: self.X_train}
        # return feed_dict

        X_overlap = self.X_train[self.overlap_indices]
        non_overlap_indices_batch = self.non_overlap_indices[non_overlap_batch_range[0]: non_overlap_batch_range[1]]
        print("host non_overlap_indices_batch shape:", len(non_overlap_indices_batch))

        X_non_overlap = self.X_train[non_overlap_indices_batch, :]
        X_all_ref_block = self.X_train[block_indices]

        feed_dict = {self.nn.X_all_in: X_all_ref_block,
                     self.nn.X_overlap_in: X_overlap,
                     self.nn.X_non_overlap_in: X_non_overlap,
                     self.nn_prime.X_all_in: X_all_ref_block,
                     self.nn_prime.X_overlap_in: X_overlap,
                     self.nn_prime.X_non_overlap_in: X_non_overlap}
        return feed_dict

    def get_two_sides_predict_feed_dict(self, combine_axis=0):
        # if combine_axis == 0:
        #     feed_dict = {self.nn.X_all_in: self.X_test}
        # else:
        #     feed_dict = {self.nn.X_all_in: self.X_test, self.nn_prime.X_all_in: self.X_test}
        # return feed_dict
        if combine_axis == 0:
            feed_dict = {self.nn.X_overlap_in: self.X_test}
        else:
            feed_dict = {self.nn.X_overlap_in: self.X_test, self.nn_prime.X_overlap_in: self.X_test}
        return feed_dict

    def get_self_predict_feed_dict(self, combine_axis=0):
        if combine_axis == 0:
            feed_dict = {self.nn.X_non_overlap_in: self.X_test}
        else:
            X_overlap = self.X_train[self.overlap_indices]
            feed_dict = {self.nn.X_overlap_in: X_overlap,
                         self.nn.X_non_overlap_in: self.X_test,
                         self.nn_prime.X_non_overlap_in: self.X_test}
        return feed_dict

    def get_guest_predict_feed_dict(self, block_indices, combine_axis=0):
        if combine_axis == 0:
            feed_dict = {self.nn.X_non_overlap_in: self.X_test}
        else:
            X_overlap = self.X_train[self.overlap_indices]
            X_all_ref_block = self.X_train[block_indices]
            feed_dict = {self.nn.X_overlap_in: X_overlap,
                         self.nn_prime.X_all_in: X_all_ref_block}
        return feed_dict

    def get_guest_self_train_feed_dict(self, block_indices):
        X_overlap = self.X_train[self.overlap_indices]
        X_all_ref_block = self.X_train[block_indices]
        feed_dict = {self.nn.X_overlap_in: X_overlap,
                     self.nn_prime.X_all_in: X_all_ref_block}
        return feed_dict


class VerticalFederatedTransferLearning(object):

    def __init__(self, vftl_guest: ExpandingVFTLGuest, vftl_host: ExpandingVFTLHost, model_param: FederatedModelParam):
        self.vftl_guest = vftl_guest
        self.vftl_host = vftl_host
        self.model_param = model_param
        self.repr_learner = RankedRepresentationLearner()
        self.stop_training = False

    def set_representation_learner(self, repr_learner):
        self.repr_learner = repr_learner
        print("using:", self.repr_learner)

    def determine_overlap_sample_indices(self):
        overlap = self.model_param.overlap_indices
        print("overlap size:", len(overlap))
        return overlap

    def prepare_data(self):
        overlap_indices = self.determine_overlap_sample_indices()
        self.vftl_guest.prepare_data(overlap_indices)
        self.vftl_host.prepare_data(overlap_indices)

    def save_model(self):
        print("TODO: save model")

    def _create_transform_matrix(self, in_dim, out_dim):
        with tf.compat.v1.variable_scope("transform_matrix"):
            Wt = tf.compat.v1.get_variable(name="Wt", initializer=tf.random.normal((in_dim, out_dim), dtype=tf.float64))
        return Wt

    def _build_feature_extraction(self):
        Ug_overlap_uniq, Ug_overlap_comm = self.vftl_guest.fetch_overlap_feat_reprs()
        Uh_overlap_uniq, Uh_overlap_comm = self.vftl_host.fetch_overlap_feat_reprs()
        Y_overlap = self.vftl_guest.get_Y_overlap()
        fed_overlap_reprs = tf.concat([Ug_overlap_uniq, Uh_overlap_uniq], axis=1)
        comm_reprs_loss = tf.nn.l2_loss(Ug_overlap_comm - Uh_overlap_comm)
        print("comm_reprs_loss shape", comm_reprs_loss.shape)
        return fed_overlap_reprs, Y_overlap, [comm_reprs_loss]

    def _build_feature_extraction_with_transfer(self):
        Ug_all_uniq, Ug_overlap_uniq, Ug_non_overlap_uniq, Ug_all_comm, Ug_overlap_comm, Ug_non_overlap_comm = self.vftl_guest.fetch_feat_reprs()
        Uh_all_uniq, Uh_overlap_uniq, Uh_non_overlap_uniq, Uh_all_comm, Uh_overlap_comm, Uh_non_overlap_comm = self.vftl_host.fetch_feat_reprs()

        self.Ug_overlap_uniq_temp = Ug_overlap_uniq
        self.Uh_overlap_uniq_temp = Uh_overlap_uniq
        self.Ug_non_overlap_uniq_temp = Ug_non_overlap_uniq
        self.Uh_non_overlap_uniq_temp = Uh_non_overlap_uniq
        self.Ug_all_comm_temp = Ug_all_comm
        self.Uh_all_comm_temp = Uh_all_comm
        self.Ug_all_uniq_temp = Ug_all_uniq
        self.Uh_all_uniq_temp = Uh_all_uniq

        combine_axis = self.model_param.combine_axis
        parallel_iterations = self.model_param.parallel_iterations
        k = self.model_param.top_k
        is_hetero_repr = self.model_param.is_hetero_repr

        W_hg = None
        if is_hetero_repr is True:
            W_hg = self._create_transform_matrix(self.vftl_host.get_comm_feat_repr_dim(),
                                                 self.vftl_guest.get_comm_feat_repr_dim())
            print("Using transform matrix:", W_hg)

        Y_overlap = self.vftl_guest.get_Y_overlap()
        Y_all = self.vftl_guest.get_Y_all()
        Y_guest_non_overlap = self.vftl_guest.get_Y_non_overlap()

        Uh_non_overlap_ested_reprs = self.repr_learner.estimate_host_representations_for_guest_party(
            Ug_non_overlap_comm,
            Ug_non_overlap_uniq,
            Ug_overlap_uniq,
            Uh_overlap_uniq,
            Uh_all_comm,
            k=k,
            combine_axis=combine_axis,
            parallel_iterations=parallel_iterations,
            W_hg=W_hg)
        print("Uh_non_overlap_ested_reprs shape", Uh_non_overlap_ested_reprs.shape)

        Ug_non_overlap_ested_reprs_w_lbls, uniq_lbls, comm_lbls = self.repr_learner.estimate_guest_representations_for_host_party(
            Uh_non_overlap_comm,
            Uh_non_overlap_uniq,
            Uh_overlap_uniq,
            Ug_overlap_uniq,
            Ug_all_comm,
            Y_overlap,
            Y_all,
            k=k,
            combine_axis=combine_axis,
            parallel_iterations=parallel_iterations,
            W_hg=W_hg)

        print("Ug_non_overlap_ested_reprs_w_lbls shape", Ug_non_overlap_ested_reprs_w_lbls.shape)
        Ug_non_overlap_estimated_reprs = Ug_non_overlap_ested_reprs_w_lbls[:, :-1]
        Ug_non_overlap_estimated_lbls = Ug_non_overlap_ested_reprs_w_lbls[:, -1]

        if combine_axis == 0:
            # using reprs, U_non_overlap_uniq and/or U_non_overlap_comm to train vertical federated model
            # fed_overlap_reprs = tf.concat([Ug_overlap_uniq, Uh_overlap_uniq], axis=1)
            # fed_guest_non_overlap_reprs = tf.concat([Ug_non_overlap_uniq, Uh_non_overlap_ested_reprs], axis=1)
            # fed_host_non_overlap_reprs = tf.concat([Ug_non_overlap_estimated_reprs, Uh_non_overlap_uniq], axis=1)
            fed_overlap_reprs = tf.concat([Ug_overlap_uniq, Uh_overlap_uniq], axis=1)
            fed_guest_non_overlap_reprs = tf.concat([Ug_non_overlap_uniq, Uh_non_overlap_ested_reprs], axis=1)
            fed_host_non_overlap_reprs = tf.concat([Ug_non_overlap_estimated_reprs, Uh_non_overlap_uniq], axis=1)
        else:
            fed_overlap_reprs = tf.concat([Ug_overlap_uniq, Ug_overlap_comm, Uh_overlap_uniq, Uh_overlap_comm], axis=1)
            fed_guest_non_overlap_reprs = tf.concat([Ug_non_overlap_uniq, Ug_non_overlap_comm, Uh_non_overlap_ested_reprs], axis=1)
            fed_host_non_overlap_reprs = tf.concat([Ug_non_overlap_estimated_reprs, Uh_non_overlap_uniq, Uh_non_overlap_comm], axis=1)

        # training_fed_reprs = tf.concat([fed_overlap_reprs, fed_guest_non_overlap_reprs], axis=0)
        fed_reprs_list = [fed_overlap_reprs, fed_guest_non_overlap_reprs, fed_host_non_overlap_reprs]
        # train_fed_reprs_list = [fed_overlap_reprs, fed_guest_non_overlap_reprs]
        train_fed_reprs_list = [fed_overlap_reprs, fed_guest_non_overlap_reprs, fed_host_non_overlap_reprs]
        training_fed_reprs = tf.concat(train_fed_reprs_list, axis=0)
        print("training_fed_reprs shape", training_fed_reprs.shape)

        Y_host_non_overlap = tf.expand_dims(Ug_non_overlap_estimated_lbls, axis=1)
        # Y_host_non_overlap = Ug_non_overlap_estimated_lbls

        print("Y_overlap shape", Y_overlap, Y_overlap.shape)
        print("Y_guest_non_overlap shape", Y_guest_non_overlap, Y_guest_non_overlap.shape)
        print("Y_host_non_overlap shape", Y_host_non_overlap, Y_host_non_overlap.shape)

        fed_Y = tf.concat([Y_overlap, Y_guest_non_overlap, Y_host_non_overlap], axis=0)
        # fed_Y = tf.concat([Y_overlap, Y_guest_non_overlap], axis=0)
        print("fed_Y shape", fed_Y.shape)

        Uh_overlap_ested_soft_lbls = self.repr_learner.estimate_lables_for_host_overlap_comm(Uh_overlap_comm,
                                                                                             Ug_all_comm,
                                                                                             Y_all,
                                                                                             W_hg=W_hg)

        self.Uh_overlap_ested_soft_lbls = self.repr_learner.estimate_lables_for_host_overlap_comm(Uh_non_overlap_comm,
                                                                                                  Ug_all_comm,
                                                                                                  Y_all,
                                                                                                  W_hg=W_hg)

        # Uh_overlap_ested_soft_lbls = self.repr_learner.estimate_lables_for_host_overlap(Uh_uniq=Uh_overlap_uniq,
        #                                                                                 Uh_overlap_uniq=Uh_overlap_uniq,
        #                                                                                 Uh_comm=Uh_overlap_comm,
        #                                                                                 Ug_all_comm=Ug_all_comm,
        #                                                                                 Yg_overlap=Y_overlap,
        #                                                                                 Yg_all=Y_all,
        #                                                                                 W_hg=W_hg)
        #
        # self.Uh_overlap_ested_soft_lbls = self.repr_learner.estimate_lables_for_host_overlap(Uh_uniq=Uh_non_overlap_uniq,
        #                                                                                      Uh_overlap_uniq=Uh_overlap_uniq,
        #                                                                                      Uh_comm=Uh_non_overlap_comm,
        #                                                                                      Ug_all_comm=Ug_all_comm,
        #                                                                                      Yg_overlap=Y_overlap,
        #                                                                                      Yg_all=Y_all,
        #                                                                                      W_hg=W_hg)

        assistant_loss_list = self.get_reprs_loss(Ug_overlap_uniq, Ug_overlap_comm, Uh_overlap_uniq, Uh_overlap_comm, W_hg)
        self.Uh_overlap_ested_lbl_loss = self.get_label_estimation_loss(Uh_overlap_ested_soft_lbls, Y_overlap)
        # label alignment for all non-overlapping samples
        label_alignment_loss = self.get_lable_alignment_loss(uniq_lbls, comm_lbls)
        assistant_loss_list.append(self.Uh_overlap_ested_lbl_loss)
        assistant_loss_list.append(label_alignment_loss)

        training_components_list = [training_fed_reprs, fed_Y, assistant_loss_list]
        return training_components_list, fed_reprs_list
        # return training_fed_reprs, fed_Y, [comm_reprs_loss]

    @staticmethod
    def get_reprs_loss(Ug_overlap_uniq, Ug_overlap_comm, Uh_overlap_uniq, Uh_overlap_comm, W_hg=None):
        guest_uniq_reprs_loss = - tf.nn.l2_loss(Ug_overlap_uniq - Ug_overlap_comm)
        host_uniq_reprs_loss = - tf.nn.l2_loss(Uh_overlap_uniq - Uh_overlap_comm)

        if W_hg is None:
            comm_reprs_loss = tf.nn.l2_loss(Ug_overlap_comm - Uh_overlap_comm)
        else:
            transformed_Uh = tf.matmul(Uh_overlap_comm, W_hg)
            comm_reprs_loss = tf.nn.l2_loss(Ug_overlap_comm - transformed_Uh)
        print("comm_reprs_loss shape", comm_reprs_loss.shape)
        return [comm_reprs_loss, guest_uniq_reprs_loss, host_uniq_reprs_loss]

    @staticmethod
    def get_label_estimation_loss(pred_soft_lbls, true_lbls):
        return tf.reduce_sum(input_tensor=true_lbls * -tf.math.log(pred_soft_lbls) + (1 - true_lbls) * -tf.math.log(1 - pred_soft_lbls))
        # return tf.nn.l2_loss(pred_soft_lbls - true_lbls)
        # loss = tf.losses.mean_squared_error(predictions=pred_soft_lbls, labels=true_lbls)
        # return tf.cast(loss, tf.float64)

    @staticmethod
    def get_lable_alignment_loss(uniq_lbls, comm_lbls):
        # num_samples = tf.cast(tf.shape(uniq_lbls)[0], tf.float64)
        # return tf.nn.l2_loss(uniq_lbls - comm_lbls) / num_samples
        return tf.nn.l2_loss(uniq_lbls - comm_lbls)

    def build(self):
        # self.fed_reprs, self.fed_Y, self.comm_reprs_loss_list = self._build_feature_extraction()
        training_components, predicting_components = self._build_feature_extraction_with_transfer()
        self.fed_reprs, self.fed_Y, self.assistant_loss_list = training_components
        fed_overlap_reprs, fed_guest_non_overlap_reprs, fed_host_non_overlap_reprs = predicting_components

        learning_rate = self.model_param.learning_rate
        input_dim = self.model_param.fed_input_dim
        reg_lambda = self.model_param.fed_reg_lambda
        loss_weight_list = self.model_param.loss_weight_list

        # reg_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float64)) for v in tf.trainable_variables()])
        # self.Uh_overlap_training_loss = 0.01 * reg_loss + 0.01 * self.assistant_loss_list[0] + self.assistant_loss_list[3]
        self.Uh_overlap_training_loss = 0.01 * self.assistant_loss_list[0] + self.assistant_loss_list[3]
        # self.Uh_overlap_training_loss = tf.cast(self.Uh_overlap_training_loss, dtype=tf.float64)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        self.host_train_op = optimizer.minimize(self.Uh_overlap_training_loss)

        self.logistic_regressor = LogisticRegression(1)
        self.logistic_regressor.set_loss_factors(self.assistant_loss_list, loss_weight_list)
        self.logistic_regressor.build(input_dim=input_dim, learning_rate=learning_rate, reg_lambda=reg_lambda,
                                      tf_X_in=self.fed_reprs, tf_labels_in=self.fed_Y)

        # self._build_two_sides_predict()
        self.logistic_regressor.set_two_sides_predict_ops(fed_overlap_reprs)
        self.logistic_regressor.set_guest_side_predict_ops(fed_guest_non_overlap_reprs)
        self.logistic_regressor.predict_lbls_for_reprs(fed_host_non_overlap_reprs)

        guest_train_labels = self.vftl_guest.get_guest_train_labels()
        self.logistic_regressor.set_guest_side_train_ops(reprs=fed_guest_non_overlap_reprs,
                                                         guest_train_labels=guest_train_labels)

    @staticmethod
    def convert_to_labels(y_prob):
        # print("y_prob:", y_prob)
        y_hat = [1 if y > 0.5 else 0 for y in y_prob]
        return np.array(y_hat)

    # @staticmethod
    # def get_block_number(sample_number, batch_size):
    #     residual = sample_number % batch_size
    #     if residual == 0:
    #         # if residual is 0, the number of samples is in multiples of batch_size.
    #         # Thus, we can directly use the real floor division operator "//" to compute
    #         # the batch_num
    #         batch_num = sample_number // batch_size
    #
    #         # if residual is 0, the last batch is a full batch.
    #         # In other words, all batches have the same number of samples
    #     else:
    #         # if residual is not 0,
    #         batch_num = sample_number // batch_size + 1
    #     return batch_num

    def f_score(self, precision, recall):
        return 2 / (1 / precision + 1 / recall)

    def two_sice_predict(self, sess, combine_axis):
        pred_feed_dict = self.vftl_guest.get_two_sides_predict_feed_dict(combine_axis=combine_axis)
        pred_host_feed_dict = self.vftl_host.get_two_sides_predict_feed_dict(combine_axis=combine_axis)
        pred_feed_dict.update(pred_host_feed_dict)
        y_prob_two_sides = sess.run(self.logistic_regressor.y_hat_two_side, feed_dict=pred_feed_dict)
        y_test = self.vftl_guest.get_Y_test()
        y_hat_two_sides = self.convert_to_labels(y_prob_two_sides)
        print("y_test:", y_test.shape)
        print("y_test sum:", sum(y_test))
        print("y_hat_two_sides:", y_hat_two_sides)
        print("y_hat_two_sides sum:", sum(y_hat_two_sides))
        res = precision_recall_fscore_support(y_test, y_hat_two_sides, average='weighted')
        all_fscore = self.f_score(res[0], res[1])
        acc = accuracy_score(y_test, y_hat_two_sides)
        auc = roc_auc_score(y_test, y_hat_two_sides)
        print("all_res:", res)
        print("all_fscore:", all_fscore)
        print("all_auc:", auc)
        print("all_acc:", acc)
        return acc, auc, all_fscore

    def guest_side_train(self, sess, batch_size, host_all_training_size, block_size, epoch=1):
        print("## ---> guest side predict")
        training_sample_size = self.vftl_guest.get_all_training_sample_size()
        batch_num = int(training_sample_size / batch_size)
        for ep in range(epoch):
            # for batch_i in range(batch_num):
            #     batch_start = batch_size * batch_i
            #     batch_end = batch_size * batch_i + batch_size

            host_block_indices = np.random.choice(host_all_training_size, block_size)
            guest_self_train_feed_dict = self.vftl_guest.get_self_train_feed_dict((1, 1))
            host_assistant_self_train_feed_dict = self.vftl_host.get_guest_self_train_feed_dict(host_block_indices)
            guest_self_train_feed_dict.update(host_assistant_self_train_feed_dict)
            _, loss = sess.run([self.logistic_regressor.guest_train_op, self.logistic_regressor.guest_train_loss],
                               feed_dict=guest_self_train_feed_dict)
            print("guest self training ep: {0}, batch: {1}, loss: {2}".format(ep, 0, loss))

    def guest_side_predict(self, sess, host_all_training_size, block_size, combine_axis, round=5):
        print("## ---> guest side predict")
        y_test = self.vftl_guest.get_Y_test()
        fscore_list = []
        acc_list = []
        auc_list = []
        for r in range(round):
            print("## round:", r)
            host_block_indices = np.random.choice(host_all_training_size, block_size)
            guest_side_pred_feed_dict = self.vftl_guest.get_self_predict_feed_dict(combine_axis=combine_axis)
            host_assistant_pred_feed_dict = self.vftl_host.get_guest_predict_feed_dict(host_block_indices,
                                                                                       combine_axis=combine_axis)
            guest_side_pred_feed_dict.update(host_assistant_pred_feed_dict)
            y_prob_guest_side = sess.run(self.logistic_regressor.y_hat_guest_side, feed_dict=guest_side_pred_feed_dict)
            y_hat_guest_side = self.convert_to_labels(y_prob_guest_side)
            print("## y_hat_guest_side:", y_hat_guest_side)
            print("## y_hat_guest_side sum:", sum(y_hat_guest_side))
            guest_res = precision_recall_fscore_support(y_test, y_hat_guest_side, average='weighted')
            fscore = self.f_score(guest_res[0], guest_res[1])
            guest_acc = accuracy_score(y_test, y_hat_guest_side)
            guest_auc = roc_auc_score(y_test, y_prob_guest_side)
            print("## guest_res:", guest_res)
            print("## guest_fscore:", fscore)
            print("## guest_auc:", guest_auc)
            print("## guest_acc:", guest_acc)
            fscore_list.append(fscore)
            acc_list.append(guest_acc)
            auc_list.append(guest_auc)
        return fscore_list, acc_list, auc_list

    def host_side_predict(self, sess, guest_all_training_size, block_size, combine_axis, round=5):
        print("## ---> host side predict")
        y_test = self.vftl_guest.get_Y_test()
        fscore_list = []
        acc_list = []
        auc_list = []
        for r in range(round):
            print("## round:", r)

            guest_block_indices = np.random.choice(guest_all_training_size, block_size)

            # host alone
            host_alone_predict_feed_dict = self.vftl_host.get_host_alone_predict_feed_dict()
            guest_for_host_predict_along_feed_dict = self.vftl_guest.get_host_alone_predict_feed_dict(
                guest_block_indices)
            host_alone_predict_feed_dict.update(guest_for_host_predict_along_feed_dict)
            distance_ested_soft_lbls = sess.run(self.Uh_overlap_ested_soft_lbls, feed_dict=host_alone_predict_feed_dict)
            # ested_hard_lbls = self.convert_to_labels(distance_ested_soft_lbls)
            print("distance_ested_soft_lbls", distance_ested_soft_lbls.shape)
            #
            host_side_pred_feed_dict = self.vftl_host.get_self_predict_feed_dict(combine_axis=combine_axis)
            guest_assistant_pred_feed_dict = self.vftl_guest.get_host_predict_feed_dict(guest_block_indices,
                                                                                        combine_axis=combine_axis)
            host_side_pred_feed_dict.update(guest_assistant_pred_feed_dict)
            fl_y_prob_host_side = sess.run(self.logistic_regressor.y_hat_host_side, feed_dict=host_side_pred_feed_dict)
            print("fl_y_prob_host_side", fl_y_prob_host_side.shape)

            test_comb_y_prob_host_side = 0.5 * fl_y_prob_host_side + 0.5 * distance_ested_soft_lbls
            # comb_y_prob_host_side = 1.0 * fl_y_prob_host_side + 0.0 * distance_ested_soft_lbls

            test_comb_y_hat_host_side = self.convert_to_labels(test_comb_y_prob_host_side)
            fl_y_hat_host_side = self.convert_to_labels(fl_y_prob_host_side)
            distance_ested_soft_lbls = self.convert_to_labels(distance_ested_soft_lbls)
            # y_hat_host_side = self.convert_to_labels(fl_y_hat_host_side)

            print("fl_y_prob_host_side:", fl_y_prob_host_side)
            print("## comb_y_hat_host_side:", fl_y_hat_host_side)
            print("## comb_y_hat_host_side sum:", sum(fl_y_hat_host_side))
            print("## test_comb_y_hat_host_side sum:", sum(test_comb_y_hat_host_side))
            test_comb_host_res = precision_recall_fscore_support(y_test, test_comb_y_hat_host_side, average='weighted')
            fl_host_res = precision_recall_fscore_support(y_test, fl_y_hat_host_side, average='weighted')
            distance_ested_soft_lbls_res = precision_recall_fscore_support(y_test, distance_ested_soft_lbls, average='weighted')
            # y_hat_host_side_res = precision_recall_fscore_support(y_test, fl_y_hat_host_side, average='weighted')
            test_comb_fscore = self.f_score(test_comb_host_res[0], test_comb_host_res[1])
            fl_fscore = self.f_score(fl_host_res[0], fl_host_res[1])
            fscore2 = self.f_score(distance_ested_soft_lbls_res[0], distance_ested_soft_lbls_res[1])
            # fscore3 = self.f_score(y_hat_host_side_res[0], y_hat_host_side_res[1])
            fl_host_acc = accuracy_score(y_test, fl_y_hat_host_side)
            fl_host_auc = roc_auc_score(y_test, fl_y_prob_host_side)
            print("## y_test:", len(y_test), sum(y_test))
            print("## * fl_host_res:", fl_host_res)
            print("## * fl_host_fscore:", fl_fscore)
            print("## fl_host_auc:", fl_host_auc)
            print("## fl_host_acc:", fl_host_acc)
            print("## * distance_ested_soft_lbls_res:", distance_ested_soft_lbls_res)
            print("## * distance_ested_soft_lbls_fscore:", fscore2)
            print("## host_test_comb_fscore:", test_comb_fscore)
            # print("## * y_hat_host_side_res_fscore:", fscore3)
            fscore_list.append(fl_fscore)
            acc_list.append(fl_host_acc)
            auc_list.append(fl_host_auc)
        return fscore_list, acc_list, auc_list

    def host_side_train(self, sess, guest_all_training_size, block_size):
        print("===> pre-train host prediction model ...")
        host_alone_fscore_list = []
        for j in range(100):
            guest_block_indices = np.random.choice(guest_all_training_size, block_size)
            # print("guest_block_indices:", len(guest_block_indices))
            host_train_feed_dict = self.vftl_host.get_host_alone_train_feed_dict()
            guest_for_host_train_feed_dict = self.vftl_guest.get_host_alone_train_feed_dict(
                guest_block_indices)
            host_train_feed_dict.update(guest_for_host_train_feed_dict)
            _, Uh_overlap_training_loss = sess.run([self.host_train_op, self.Uh_overlap_training_loss],
                                                   feed_dict=host_train_feed_dict)

            if j % 50 == 0:
                # print("host training loss:", j, Uh_overlap_training_loss)
                y_test = self.vftl_guest.get_Y_test()
                # print("y_test:", len(y_test), sum(y_test))
                guest_block_indices = np.random.choice(guest_all_training_size, block_size)
                host_alone_predict_feed_dict = self.vftl_host.get_host_alone_predict_feed_dict()
                guest_for_host_predict_along_feed_dict = self.vftl_guest.get_host_alone_predict_feed_dict(guest_block_indices)
                host_alone_predict_feed_dict.update(guest_for_host_predict_along_feed_dict)
                ested_soft_lbls = sess.run(self.Uh_overlap_ested_soft_lbls, feed_dict=host_alone_predict_feed_dict)
                ested_hard_lbls = self.convert_to_labels(ested_soft_lbls)
                host_alone_res = precision_recall_fscore_support(y_test, ested_hard_lbls, average='weighted')
                # print("host_alone_res:", host_alone_res)
                host_along_fscore = self.f_score(host_alone_res[0], host_alone_res[1])
                # print("ested_soft_lbls:", ested_soft_lbls)
                # print("ested_hard_lbls", len(ested_hard_lbls), sum(ested_hard_lbls))
                # print("host_alone_res:", host_alone_res)
                # print("host_along_fscore:", host_along_fscore)
                host_alone_fscore_list.append(host_along_fscore)
        print("|| host_alone_fscore_list:", host_alone_fscore_list)
        print("|| host_alone_fscore_mean:", np.mean(host_alone_fscore_list))
        print("<=== pre-train host prediction model finished !")

    def fit(self,
            sess,
            guest_non_overlap_batch_range,
            host_non_overlap_batch_range,
            guest_block_indices,
            host_block_indices):

        train_feed_dict = self.vftl_guest.get_train_feed_dict(guest_non_overlap_batch_range, guest_block_indices)
        train_host_feed_dict = self.vftl_host.get_train_feed_dict(host_non_overlap_batch_range, host_block_indices)
        train_feed_dict.update(train_host_feed_dict)

        # Ug_non_overlap_uniq, Uh_non_overlap_uniq, Ug_overlap_uniq_temp, Uh_overlap_uniq_temp, _, fed_reprs, fed_Y, reprs_dist_loss, reg_loss, pred_loss, ob_loss = sess.run(
        #     [self.Ug_non_overlap_uniq_temp, self.Uh_non_overlap_uniq_temp, self.Ug_overlap_uniq_temp, self.Uh_overlap_uniq_temp, self.logistic_regressor.e2e_train_op,
        #      self.fed_reprs, self.fed_Y, self.comm_reprs_loss_list[0], self.logistic_regressor.reg_loss,
        #      self.logistic_regressor.pred_loss, self.logistic_regressor.ob_loss], feed_dict=train_feed_dict)
        _, fed_reprs, fed_Y, reprs_dist_loss, reg_loss, pred_loss, ob_loss, Uh_overlap_pred_loss = sess.run(
            [self.logistic_regressor.e2e_train_op, self.fed_reprs, self.fed_Y, self.assistant_loss_list[0],
             self.logistic_regressor.regularization_loss, self.logistic_regressor.pred_loss, self.logistic_regressor.loss, self.Uh_overlap_ested_lbl_loss],
            feed_dict=train_feed_dict)

        print("Uh_overlap_pred_loss:", Uh_overlap_pred_loss)
        print("reprs_dist_loss", reprs_dist_loss)
        print("reg_loss", reg_loss)
        print("pred_loss", pred_loss)
        print("final ob_loss", ob_loss)
        return ob_loss
        # print("fed_reprs: \n", fed_reprs.shape)
        # print("fed_Y\n", fed_Y.shape)
        # print("fed_reprs: \n", fed_reprs, fed_reprs.shape)
        # print("fed_Y\n", fed_Y, fed_Y.shape)
        # print("computed_gradients \n", computed_gradients)
        # print("Ug_overlap_uniq_temp: \n", Ug_overlap_uniq_temp)
        # print("Uh_overlap_uniq_temp: \n", Uh_overlap_uniq_temp)
        # print("Ug_non_overlap_uniq: \n", Ug_non_overlap_uniq)
        # print("Uh_non_overlap_uniq: \n", Uh_non_overlap_uniq)

    def train(self):

        block_size = self.model_param.all_sample_block_size
        nol_batch_num = self.model_param.non_overlap_sample_batch_num
        ol_batch_num = self.model_param.overlap_sample_batch_num

        guest_nol_training_sample_size = self.vftl_guest.get_non_overlapping_training_sample_size()
        host_nol_training_sample_size = self.vftl_host.get_non_overlapping_training_sample_size()
        ol_training_sample_size = self.vftl_guest.get_overlap_training_sample_size()

        nol_guest_batch_size = int(guest_nol_training_sample_size / nol_batch_num)
        nol_host_batch_size = int(host_nol_training_sample_size / nol_batch_num)
        ol_batch_size = int(ol_training_sample_size / ol_batch_num)

        print("nol_batch_num:", nol_batch_num)
        print("ol_batch_num:", ol_batch_num)

        print("nol_guest_batch_size:", nol_guest_batch_size)
        print("nol_host_batch_size:", nol_host_batch_size)
        print("ol_batch_size:", ol_batch_size)

        guest_all_training_size = self.vftl_guest.get_all_training_sample_size()
        host_all_training_size = self.vftl_host.get_all_training_sample_size()

        early_stopping = EarlyStoppingCheckPoint(monitor="fscore", patience=8)
        early_stopping.set_model(self)
        early_stopping.on_train_begin()

        start_time = time.time()
        epoch = self.model_param.epoch
        init = tf.compat.v1.global_variables_initializer()
        best_f_score = 0.0
        with tf.compat.v1.Session() as sess:
            self.vftl_guest.set_session(sess)
            self.vftl_host.set_session(sess)
            self.logistic_regressor.set_session(sess)

            sess.run(init)

            loss_list = []
            auc_list = []
            acc_list = []
            fscore_list = []
            for i in range(epoch):

                # for ol_batch_i in range(ol_batch_num):
                #     ol_start = ol_batch_size * ol_batch_i
                #     ol_end = ol_batch_size * ol_batch_i + ol_batch_size

                for nol_batch_i in range(nol_batch_num):

                    # fit
                    nol_guest_start = nol_guest_batch_size * nol_batch_i
                    nol_guest_end = nol_guest_batch_size * nol_batch_i + nol_guest_batch_size
                    nol_host_start = nol_host_batch_size * nol_batch_i
                    nol_host_end = nol_host_batch_size * nol_batch_i + nol_host_batch_size

                    guest_block_indices = np.random.choice(guest_all_training_size, block_size)
                    host_block_indices = np.random.choice(host_all_training_size, block_size)

                    print("nol_guest_start:nol_guest_end ", nol_guest_start, nol_guest_end)
                    print("nol_host_start:nol_host_end ", nol_host_start, nol_host_end)
                    print("guest_block_indices:", len(guest_block_indices))
                    print("host_block_indices:", len(host_block_indices))

                    loss = self.fit(sess=sess,
                                    guest_non_overlap_batch_range=(nol_guest_start, nol_guest_end),
                                    host_non_overlap_batch_range=(nol_host_start, nol_host_end),
                                    guest_block_indices=guest_block_indices,
                                    host_block_indices=host_block_indices)
                    loss_list.append(loss)
                    print("")
                    print("------> ep", i, "batch", nol_batch_i, "loss", loss)

                    # self.host_side_train(sess, guest_all_training_size, block_size)

                    # guest self training
                    # batch_size = 512
                    # self.guest_side_train(sess, batch_size, host_all_training_size, block_size, epoch=10)

                    # two sides test
                    combine_axis = self.model_param.combine_axis
                    all_acc, all_auc, all_fscore = self.two_sice_predict(sess, combine_axis)
                    acc_list.append(all_acc)
                    auc_list.append(all_auc)
                    fscore_list.append(all_fscore)

                    # guest side test
                    g_fscore_list, g_acc_list, g_auc_list = self.guest_side_predict(sess,
                                                                                    host_all_training_size,
                                                                                    block_size,
                                                                                    combine_axis)
                    g_fscore_mean = np.mean(g_fscore_list)
                    print("%% g_fscore_list:", g_fscore_list, g_fscore_mean)
                    print("%% g_acc_list:", g_acc_list, np.mean(g_acc_list))
                    print("%% g_auc_list:", g_auc_list, np.mean(g_auc_list))

                    # host side test
                    h_fscore_list, h_acc_list, h_auc_list = self.host_side_predict(sess,
                                                                                   guest_all_training_size,
                                                                                   block_size,
                                                                                   combine_axis)
                    h_fscore_mean = np.mean(h_fscore_list)
                    print("%% h_fscore_list:", h_fscore_list, h_fscore_mean)
                    print("%% h_acc_list:", h_acc_list, np.mean(h_acc_list))
                    print("%% h_auc_list:", h_auc_list, np.mean(h_auc_list))

                    print("===> fscore: all, guest, host", all_fscore, g_fscore_mean, h_fscore_mean)
                    # ave_fscore = (all_fscore + g_fscore_mean + h_fscore_mean) / 3
                    ave_fscore = 3 / (1 / all_fscore + 1 / g_fscore_mean + 1 / h_fscore_mean)
                    print("ave_fscore:", ave_fscore)
                    log = {"fscore": h_fscore_mean, "all_fscore": all_fscore, "g_fscore": g_fscore_mean,
                           "h_fscore:": h_fscore_mean, "loss": loss}
                    # log = {"fscore": ave_fscore, "all_fscore": all_fscore, "g_fscore": g_fscore_mean,
                    #        "h_fscore:": h_fscore_mean, "loss": loss}
                    early_stopping.on_validation_end(curr_epoch=i, batch_idx=nol_batch_i, log=log)

                    if self.stop_training is True:
                        break

                if self.stop_training is True:
                    break

                    # Ug_all_uniq_temp, Ug_all_comm_temp, Ug_overlap_uniq_temp, Uh_overlap_uniq_temp = sess.run(
                    #     [self.Ug_all_uniq_temp, self.Ug_all_comm_temp, self.Ug_overlap_uniq_temp, self.Uh_overlap_uniq_temp], feed_dict=train_feed_dict)
                    # print("% Ug_all_uniq_temp: \n", Ug_all_uniq_temp, Ug_all_uniq_temp.shape)
                    # print("% Ug_all_comm_temp: \n", Ug_all_comm_temp, Ug_all_comm_temp.shape)
                    # print("% Ug_overlap_uniq_temp: \n", Ug_overlap_uniq_temp, Ug_overlap_uniq_temp.shape)
                    # print("% Uh_overlap_uniq_temp: \n", Uh_overlap_uniq_temp, Uh_overlap_uniq_temp.shape)

                    # nn_model, nn_prime_model = self.vftl_guest.get_model_parameters()
                    # nn_h_model, nn_h_prime_model = self.vftl_host.get_model_parameters()
                    # print("nn_guest_model: \n", nn_model)
                    # print("nn_guest_prime_model: \n", nn_prime_model)
                    # print("nn_host_model: \n", nn_h_model)
                    # print("nn_host_prime_model: \n", nn_h_prime_model)
                    # lr_model = self.logistic_regressor.get_model_parameters()
                    # print("lr_model: \n", lr_model)

        end_time = time.time()
        print("training time (s):", end_time - start_time)
        print("loss:", loss_list)
        print("best_f_score", best_f_score)
        print("stopped epoch, batch:", early_stopping.stopped_epoch, early_stopping.stopped_batch)
        early_stopping.print_log_of_best_result()
        series_plot(losses=loss_list, fscores=fscore_list, aucs=acc_list)



