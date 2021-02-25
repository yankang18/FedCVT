import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from expanding_vertical_transfer_learning_param import FederatedModelParam
from logistic_regression import LogisticRegression
from regularization import EarlyStoppingCheckPoint
from vertical_sstl_parties import ExpandingVFTLGuest, ExpandingVFTLHost
from vertical_sstl_representation_learner import AttentionBasedRepresentationEstimator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class VerticalFederatedTransferLearning(object):

    def __init__(self, vftl_guest: ExpandingVFTLGuest, vftl_host: ExpandingVFTLHost, model_param: FederatedModelParam):
        self.vftl_guest = vftl_guest
        self.vftl_host = vftl_host
        self.model_param = model_param
        self.n_class = self.vftl_guest.get_number_of_class()
        self.repr_estimator = AttentionBasedRepresentationEstimator()
        self.stop_training = False
        self.fed_lr = None
        # self.host_lr = None
        self.guest_lr = None

    def set_representation_estimator(self, repr_estimator):
        self.repr_estimator = repr_estimator
        print("using: {0}".format(self.repr_estimator))

    def determine_overlap_sample_indices(self):
        overlap_indices = self.model_param.overlap_indices
        print("overlap_indices size:", len(overlap_indices))
        return overlap_indices

    def save_model(self, ):
        print("TODO: save model")

    @staticmethod
    def _create_transform_matrix(in_dim, out_dim):
        with tf.compat.v1.variable_scope("transform_matrix"):
            Wt = tf.compat.v1.get_variable(name="Wt", initializer=tf.random.normal((in_dim, out_dim), dtype=tf.float64))
        return Wt

    def _build_feature_extraction_with_transfer(self):

        Ug_all, Ug_non_overlap, Ug_overlap = self.vftl_guest.fetch_feat_reprs()
        Uh_all, Uh_non_overlap, Uh_overlap = self.vftl_host.fetch_feat_reprs()

        Ug_all_uniq, Ug_all_comm = Ug_all
        Ug_non_overlap_uniq, Ug_non_overlap_comm = Ug_non_overlap
        Ug_overlap_uniq, Ug_overlap_comm = Ug_overlap

        Uh_all_uniq, Uh_all_comm = Uh_all
        Uh_non_overlap_uniq, Uh_non_overlap_comm = Uh_non_overlap
        Uh_overlap_uniq, Uh_overlap_comm = Uh_overlap

        self.Ug_all_comm_temp = Ug_all_comm
        self.Uh_all_comm_temp = Uh_all_comm
        self.Ug_all_uniq_temp = Ug_all_uniq
        self.Uh_all_uniq_temp = Uh_all_uniq

        self.Ug_non_overlap_uniq_temp = Ug_non_overlap_uniq
        self.Uh_non_overlap_uniq_temp = Uh_non_overlap_uniq
        self.Ug_non_overlap_comm_temp = Ug_non_overlap_comm
        self.Uh_non_overlap_comm_temp = Uh_non_overlap_comm

        self.Ug_overlap_uniq_temp = Ug_overlap_uniq
        self.Uh_overlap_uniq_temp = Uh_overlap_uniq
        self.Ug_overlap_comm_temp = Ug_overlap_comm
        self.Uh_overlap_comm_temp = Uh_overlap_comm

        sharpen_temp = self.model_param.sharpen_temperature
        is_hetero_repr = self.model_param.is_hetero_repr
        fed_label_prob_threshold = self.model_param.fed_label_prob_threshold
        host_label_prob_threhold = self.model_param.host_label_prob_threshold

        W_hg = None
        if is_hetero_repr is True:
            host_comm_dim = Uh_all_comm.shape[1]
            guest_comm_dim = Ug_all_comm.shape[1]
            # W_hg = self._create_transform_matrix(self.vftl_host.get_comm_feat_repr_dim(),
            #                                      self.vftl_guest.get_comm_feat_repr_dim())
            W_hg = self._create_transform_matrix(host_comm_dim, guest_comm_dim)
            print("Using transform matrix:", W_hg)

        Y_overlap = self.vftl_guest.get_Y_overlap()
        # Y_all = self.vftl_guest.get_Y_all()
        Y_guest_non_overlap = self.vftl_guest.get_Y_non_overlap()

        self.Y_overlap_for_est = self.vftl_guest.get_Y_overlap_for_est()
        Y_all_for_est = self.vftl_guest.get_Y_all_for_est()

        # estimate non-overlap feature representations on host side for guest party for main objective loss
        Uh_non_overlap_ested_reprs = self.repr_estimator.estimate_host_reprs_for_guest_party(
            Ug_non_overlap_comm,
            Ug_non_overlap_uniq,
            Ug_overlap_uniq,
            Uh_overlap_uniq,
            Uh_all_comm,
            sharpen_temperature=sharpen_temp,
            W_hg=W_hg)
        print("Uh_non_overlap_ested_reprs shape", Uh_non_overlap_ested_reprs.shape)

        # estimate overlap feature representations on host side for guest party for alignment loss
        Uh_overlap_ested_reprs = self.repr_estimator.estimate_host_reprs_for_guest_party(
            Ug_overlap_comm,
            Ug_overlap_uniq,
            Ug_overlap_uniq,
            Uh_overlap_uniq,
            Uh_all_comm,
            sharpen_temperature=sharpen_temp,
            W_hg=W_hg)

        # estimate non-overlap feature representations and labels on guest side for host party for main objective loss
        # self.Ug_non_overlap_ested_reprs_w_lbls, self.uniq_lbls, self.comm_lbls = self.repr_estimator.estimate_labeled_guest_reprs_for_host_party(
        #     Uh_non_overlap_comm,
        #     Uh_non_overlap_uniq,
        #     Uh_overlap_uniq,
        #     Ug_overlap_uniq,
        #     Ug_all_comm,
        #     self.Y_overlap_for_est,
        #     Y_all_for_est,
        #     sharpen_tempature=sharpen_temp,
        #     W_hg=W_hg)

        self.Ug_non_overlap_ested_reprs_w_lbls, _, _ = self.repr_estimator.estimate_labeled_guest_reprs_for_host_party(
            Uh_non_overlap_comm,
            Uh_non_overlap_uniq,
            Uh_overlap_uniq,
            Ug_overlap_uniq,
            Ug_all_comm,
            self.Y_overlap_for_est,
            Y_all_for_est,
            sharpen_tempature=sharpen_temp,
            W_hg=W_hg)

        self.uniq_lbls, self.comm_lbls = self.repr_estimator.estimate_unique_comm_labels_for_host_party(
            Uh_comm=Uh_all_comm,
            Uh_uniq=Uh_all_uniq,
            Uh_overlap_uniq=Uh_overlap_uniq,
            Ug_all_comm=Ug_all_comm,
            Yg_overlap=self.Y_overlap_for_est,
            Yg_all=Y_all_for_est,
            sharpen_tempature=sharpen_temp,
            W_hg=W_hg)

        print("Ug_non_overlap_ested_reprs_w_lbls shape", self.Ug_non_overlap_ested_reprs_w_lbls.shape)
        Ug_non_overlap_ested_reprs = self.Ug_non_overlap_ested_reprs_w_lbls[:, :-self.n_class]
        self.nl_ested_lbls_for_att_ested_guest_reprs = self.Ug_non_overlap_ested_reprs_w_lbls[:, -self.n_class:]

        # aggregate all guest and host feature representations for main objective loss training
        fed_ol_reprs = tf.concat([Ug_overlap_uniq, Ug_overlap_comm, Uh_overlap_uniq, Uh_overlap_comm], axis=1)
        fed_nl_host_ested_reprs = tf.concat([Ug_non_overlap_uniq, Ug_non_overlap_comm, Uh_non_overlap_ested_reprs],
                                            axis=1)
        fed_nl_guest_ested_reprs = tf.concat([Ug_non_overlap_ested_reprs, Uh_non_overlap_uniq, Uh_non_overlap_comm],
                                             axis=1)

        # estimate overlap feature representations and labels on guest side for host party for alignment loss
        result_overlap = self.repr_estimator.estimate_guest_reprs_n_lbls_for_host_party(Uh_uniq=Uh_overlap_uniq,
                                                                                        Uh_overlap_uniq=Uh_overlap_uniq,
                                                                                        Uh_comm=Uh_overlap_comm,
                                                                                        Ug_overlap_uniq=Ug_overlap_uniq,
                                                                                        Ug_all_comm=Ug_all_comm,
                                                                                        Yg_overlap=self.Y_overlap_for_est,
                                                                                        Yg_all=Y_all_for_est,
                                                                                        sharpen_tempature=sharpen_temp,
                                                                                        W_hg=W_hg)
        Ug_overlap_ested_reprs, self.Ug_overlap_ested_soft_lbls, ol_uniq_lbls, ol_comm_lbls = result_overlap

        # estimate non-overlap labels on guest side for host party for testing purpose
        result_non_overlap = self.repr_estimator.estimate_guest_reprs_n_lbls_for_host_party(Uh_uniq=Uh_non_overlap_uniq,
                                                                                            Uh_overlap_uniq=Uh_overlap_uniq,
                                                                                            Uh_comm=Uh_non_overlap_comm,
                                                                                            Ug_overlap_uniq=Ug_overlap_uniq,
                                                                                            Ug_all_comm=Ug_all_comm,
                                                                                            Yg_overlap=self.Y_overlap_for_est,
                                                                                            Yg_all=Y_all_for_est,
                                                                                            sharpen_tempature=sharpen_temp,
                                                                                            W_hg=W_hg)
        _, self.Uh_non_overlap_ested_soft_lbls, _, _ = result_non_overlap

        # predict labels for estimated host non-overlap feature representations on guest side
        self.nl_ested_lbls_for_fed_ested_guest_reprs = self.fed_lr.predict_lbls_for_reprs(fed_nl_guest_ested_reprs)
        self.nl_ested_lbls_for_lr_ested_guest_reprs = self.guest_lr.predict_lbls_for_reprs(Ug_non_overlap_ested_reprs)

        print("fed_nl_guest_ested_reprs:", fed_nl_guest_ested_reprs)
        print("Ug_non_overlap_ested_lbls:", self.nl_ested_lbls_for_att_ested_guest_reprs)
        print("nl_ested_lbls_for_fed_ested_guest_reprs:", self.nl_ested_lbls_for_fed_ested_guest_reprs)
        print("non_overlap_ested_lbls_for_ested_guest_reprs:", self.nl_ested_lbls_for_lr_ested_guest_reprs)

        # aggregate host non overlap feature representations and candidate labels
        # we will select host non overlap feature representations with corresponding labels for training based on
        # candidate labels.
        # self.host_non_overlap_reprs_w_condidate_labels = tf.concat(
        #     [fed_nl_guest_ested_reprs,
        #     self.nl_ested_lbls_for_att_ested_guest_reprs,
        #     self.nl_ested_lbls_for_fed_ested_guest_reprs], axis=1)

        self.host_non_overlap_reprs_w_condidate_labels = tf.concat(
            [fed_nl_guest_ested_reprs,
             self.nl_ested_lbls_for_lr_ested_guest_reprs,
             self.nl_ested_lbls_for_fed_ested_guest_reprs], axis=1)

        print("host_non_overlap_reprs_w_condidate_labels shape {0}".format(
            self.host_non_overlap_reprs_w_condidate_labels.shape))

        # select host non overlap feature representations with corresponding labels for training
        # selected_fed_host_non_overlap_reprs, selected_host_lbls = self.repr_estimator.select_reprs_for_multiclass(
        #     reprs_w_condidate_labels=host_non_overlap_reprs_w_condidate_labels, n_class=self.n_class, upper_bound=0.3)
        dynamic_array = self.repr_estimator.select_reprs_for_multiclass(
            reprs_w_candidate_labels=self.host_non_overlap_reprs_w_condidate_labels,
            n_class=self.n_class,
            fed_label_upper_bound=fed_label_prob_threshold,
            host_label_upper_bound=host_label_prob_threhold)

        # print("selected_fed_host_non_overlap_reprs shape {0}".format(selected_fed_host_non_overlap_reprs.shape))
        # print("selected_host_lbls shape {0}".format(selected_host_lbls.shape))

        self.num_selected_samples = dynamic_array.size()
        has_selected_samples = dynamic_array.size() > 0

        def f1():
            reprs_w_labels = dynamic_array.concat()
            reprs = reprs_w_labels[:, :-self.n_class]
            labels = reprs_w_labels[:, -self.n_class:]

            fed_reprs = tf.concat([fed_ol_reprs, fed_nl_host_ested_reprs, reprs], axis=0)
            fed_Y = tf.concat([Y_overlap, Y_guest_non_overlap, labels], axis=0)
            return fed_reprs, fed_Y

        def f2():
            fed_reprs = tf.concat([fed_ol_reprs, fed_nl_host_ested_reprs], axis=0)
            fed_Y = tf.concat([Y_overlap, Y_guest_non_overlap], axis=0)
            return fed_reprs, fed_Y

        train_fed_reprs, train_fed_Y = tf.cond(pred=has_selected_samples, true_fn=f1, false_fn=f2)
        self.training_fed_reprs_shape = tf.shape(input=train_fed_reprs)

        training_guest_ol_reprs = tf.concat(Ug_overlap, axis=1)
        training_guest_ul_reprs = tf.concat(Ug_non_overlap, axis=1)
        train_guest_reprs = tf.concat((training_guest_ol_reprs, training_guest_ul_reprs), axis=0)
        train_guest_Y = tf.concat([Y_overlap, Y_guest_non_overlap], axis=0)

        print("training_guest_ol_reprs:", training_guest_ol_reprs)
        print("training_guest_ul_reprs:", training_guest_ul_reprs)
        print("train_guest_reprs:", train_guest_reprs)
        print("train_guest_Y:", train_guest_Y)

        self.training_guest_reprs_shape = tf.shape(input=train_guest_reprs)

        guest_nl_reprs = tf.concat(Ug_non_overlap, axis=1)
        fed_reprs_list = [fed_ol_reprs, fed_nl_host_ested_reprs, fed_nl_guest_ested_reprs, guest_nl_reprs]

        print("train_fed_reprs shape", train_fed_reprs.shape)
        print("training_guest_reprs_shape shape", self.training_guest_reprs_shape.shape)

        print("Y_overlap shape", Y_overlap, Y_overlap.shape)
        print("Y_guest_non_overlap shape", Y_guest_non_overlap, Y_guest_non_overlap.shape)
        # print("Y_host_non_overlap shape", Y_host_non_overlap, Y_host_non_overlap.shape)
        # print("Y_host_non_overlap shape", selected_host_lbls, selected_host_lbls.shape)

        assistant_loss_list = list()

        # (1) loss for shared representations between host and guest
        assistant_loss_list.append(self.get_shared_reprs_loss(Ug_overlap_comm,
                                                              Uh_overlap_comm))

        # (2) (3) loss for orthogonal representation for host and guest respectively
        guest_uniq_reprs_loss, host_uniq_reprs_loss = self.get_orth_reprs_loss(Ug_all_uniq,
                                                                               Ug_all_comm,
                                                                               Uh_all_uniq,
                                                                               Uh_all_comm)
        assistant_loss_list.append(guest_uniq_reprs_loss)
        assistant_loss_list.append(host_uniq_reprs_loss)

        # (4) loss for distance between estimated host overlap labels and true labels
        # TODO
        # self.Ug_overlap_ested_lbl_loss = self.get_label_estimation_loss(self.Ug_overlap_ested_soft_lbls, Y_overlap)
        # self.Y_overlap_test_ = Y_overlap
        Ug_ol_uniq_lbl_loss = self.get_label_estimation_loss(ol_uniq_lbls, Y_overlap)
        Ug_ol_comm_lbl_loss = self.get_label_estimation_loss(ol_comm_lbls, Y_overlap)

        # assistant_loss_list.append(self.Ug_overlap_ested_lbl_loss)
        assistant_loss_list.append(Ug_ol_uniq_lbl_loss)
        assistant_loss_list.append(Ug_ol_comm_lbl_loss)

        # (5) loss for distance between estimated guest overlap representation and true guest representation
        Ug_overlap_reprs = tf.concat([Ug_overlap_uniq, Ug_overlap_comm], axis=1)
        Ug_overlap_ested_reprs_alignment_loss = self.get_alignment_loss(Ug_overlap_ested_reprs, Ug_overlap_reprs)
        assistant_loss_list.append(Ug_overlap_ested_reprs_alignment_loss)

        # (6) loss for distance between estimated host overlap representation and true host representation
        Uh_overlap_reprs = tf.concat([Uh_overlap_uniq, Uh_overlap_comm], axis=1)
        Uh_overlap_ested_reprs_alignment_loss = self.get_alignment_loss(Uh_overlap_ested_reprs, Uh_overlap_reprs)
        assistant_loss_list.append(Uh_overlap_ested_reprs_alignment_loss)

        # (7) loss for distance between shared-repr-estimated host label and uniq-repr-estimated host label
        label_alignment_loss = self.get_alignment_loss(self.uniq_lbls, self.comm_lbls)
        assistant_loss_list.append(label_alignment_loss)

        train_components_list = [train_fed_reprs, train_guest_reprs, train_fed_Y, train_guest_Y, assistant_loss_list]
        return train_components_list, fed_reprs_list

    def build(self):
        # self.fed_reprs, self.fed_Y, self.comm_reprs_loss_list = self._build_feature_extraction()
        learning_rate = self.model_param.learning_rate
        fed_input_dim = self.model_param.fed_input_dim
        fed_hidden_dim = self.model_param.fed_hidden_dim
        guest_input_dim = self.model_param.guest_input_dim
        guest_hidden_dim = self.model_param.guest_hidden_dim
        fed_reg_lambda = self.model_param.fed_reg_lambda
        guest_reg_lambda = self.model_param.guest_reg_lamba
        loss_weight_list = self.model_param.loss_weight_list
        sharpen_temp = self.model_param.sharpen_temperature
        is_hetero_repr = self.model_param.is_hetero_repr

        print("############## Hyperparameter Info #############")
        print("learning_rate: {0}".format(learning_rate))
        print("fed_input_dim: {0}".format(fed_input_dim))
        print("fed_hidden_dim: {0}".format(fed_hidden_dim))
        print("guest_input_dim: {0}".format(guest_input_dim))
        print("guest_hidden_dim: {0}".format(guest_hidden_dim))
        print("fed_reg_lambda: {0}".format(fed_reg_lambda))
        print("guest_reg_lambda: {0}".format(guest_reg_lambda))
        print("loss_weight_list: {0}".format(loss_weight_list))
        print("sharpen_temp: {0}".format(sharpen_temp))
        print("is_hetero_repr: {0}".format(is_hetero_repr))
        print("################################################")

        self.guest_lr = LogisticRegression(2)
        self.guest_lr.build(input_dim=guest_input_dim, n_class=self.n_class, hidden_dim=guest_hidden_dim)

        # self.host_lr = LogisticRegression(3)
        # self.host_lr.build(input_dim=host_input_dim, n_class=self.n_class, hidden_dim=host_hidden_dim)

        self.fed_lr = LogisticRegression(1)
        self.fed_lr.build(input_dim=fed_input_dim, n_class=self.n_class, hidden_dim=fed_hidden_dim)

        train_components, predict_components = self._build_feature_extraction_with_transfer()
        self.fed_reprs, guest_reprs, self.fed_Y, guest_Y, self.assistant_loss_list = train_components
        fed_overlap_reprs, fed_guest_non_overlap_reprs, fed_host_non_overlap_reprs, guest_nl_reprs = predict_components

        self.guest_lr.set_ops(learning_rate=learning_rate, reg_lambda=guest_reg_lambda,
                              tf_X_in=guest_reprs, tf_labels_in=guest_Y)
        self.guest_lr.set_guest_side_predict_ops(guest_nl_reprs)

        self.fed_lr.set_loss_factors(self.assistant_loss_list, loss_weight_list)
        self.fed_lr.set_ops(learning_rate=learning_rate, reg_lambda=fed_reg_lambda,
                            tf_X_in=self.fed_reprs, tf_labels_in=self.fed_Y)

        self.fed_lr.set_two_sides_predict_ops(fed_overlap_reprs)
        self.fed_lr.set_guest_side_predict_ops(fed_guest_non_overlap_reprs)

        ######################################################
        # following section defines auxiliary loss functions
        ######################################################

    @staticmethod
    def get_shared_reprs_loss(Ug_overlap_comm, Uh_overlap_comm, W_hg=None):
        num_samples = tf.cast(tf.shape(input=Ug_overlap_comm)[0], tf.float32)
        if W_hg is None:
            shared_reprs_loss = tf.nn.l2_loss(Ug_overlap_comm - Uh_overlap_comm) / num_samples
            # shared_reprs_loss = - tf.reduce_sum(tf.matmul(Ug_overlap_comm, tf.transpose(Uh_overlap_comm))) / num_samples
            # shared_reprs_loss = - tf.nn.l2_loss(tf.matmul(Ug_overlap_comm, tf.transpose(Uh_overlap_comm))) / num_samples
        else:
            transformed_Uh = tf.matmul(Uh_overlap_comm, W_hg)
            shared_reprs_loss = tf.nn.l2_loss(Ug_overlap_comm - transformed_Uh)
        print("shared_reprs_loss shape", shared_reprs_loss.shape)
        return shared_reprs_loss

    @staticmethod
    def get_orth_reprs_loss(Ug_uniq, Ug_comm, Uh_uniq, Uh_comm):
        num_samples = tf.cast(tf.shape(input=Ug_uniq)[0], tf.float32)

        # approach 1
        # guest_uniq_reprs_loss = - tf.nn.l2_loss(Ug_uniq - Ug_comm) / num_samples
        # host_uniq_reprs_loss = - tf.nn.l2_loss(Uh_uniq - Uh_comm) / num_samples

        # approach 2
        # epsilon = 0.000001
        # guest_uniq_reprs_loss = 1 / (tf.nn.l2_loss(Ug_uniq - Ug_comm) / num_samples + epsilon)
        # host_uniq_reprs_loss = 1 / (tf.nn.l2_loss(Uh_uniq - Uh_comm) / num_samples + epsilon)

        # approach 3
        # epsilon = 0.000001
        # guest_uniq_reprs_loss = 1 / (tf.reduce_sum(tf.matmul(Ug_uniq, tf.transpose(Ug_comm))) / num_samples + epsilon)
        # host_uniq_reprs_loss = 1 / (tf.reduce_sum(tf.matmul(Uh_uniq, tf.transpose(Uh_comm))) / num_samples + epsilon)

        # approach 4
        # guest_uniq_reprs_loss = tf.reduce_sum(tf.matmul(Ug_uniq, tf.transpose(Ug_comm))) / num_samples
        # host_uniq_reprs_loss = tf.reduce_sum(tf.matmul(Uh_uniq, tf.transpose(Uh_comm))) / num_samples
        guest_uniq_reprs_loss = tf.nn.l2_loss(tf.matmul(Ug_uniq, tf.transpose(a=Ug_comm))) / num_samples
        host_uniq_reprs_loss = tf.nn.l2_loss(tf.matmul(Uh_uniq, tf.transpose(a=Uh_comm))) / num_samples
        return guest_uniq_reprs_loss, host_uniq_reprs_loss

    @staticmethod
    def get_label_estimation_loss(pred_soft_lbls, true_lbls):
        """
        pred_soft_lbls is the output from a softmax layer (num_examples x num_classes)
        true_lbls is labels (num_examples x num_classes). Note that y is one-hot encoded vector.
        """
        # m = true_lbls.shape[0]
        log_likelihood = -tf.reduce_sum(input_tensor=tf.math.log(pred_soft_lbls + 1e-9) * true_lbls, axis=1)
        # print("2 log_likelihood {0}".format(log_likelihood))
        loss = tf.reduce_mean(input_tensor=log_likelihood)
        return loss

    @staticmethod
    def get_alignment_loss(ested_repr, repr):
        num_samples = tf.cast(tf.shape(input=ested_repr)[0], tf.float32)
        # return tf.nn.l2_loss(ested_repr - repr)
        return tf.nn.l2_loss(ested_repr - repr) / num_samples

    ######################################################
    ######################################################

    @staticmethod
    def convert_to_1d_labels(y_prob):
        # print("y_prob:", y_prob)
        # y_hat = [1 if y > 0.5 else 0 for y in y_prob]
        y_1d = np.argmax(y_prob, axis=1)
        # return np.array(y_hat)
        return y_1d

    def f_score(self, precision, recall):
        return 2 / (1 / precision + 1 / recall)

    def two_side_predict(self, sess, debug=True):
        # if debug:
        print("[INFO] ------> two sides predict")

        pred_feed_dict = self.vftl_guest.get_two_sides_predict_feed_dict()
        pred_host_feed_dict = self.vftl_host.get_two_sides_predict_feed_dict()
        pred_feed_dict.update(pred_host_feed_dict)

        y_prob_two_sides = sess.run(self.fed_lr.y_hat_two_side, feed_dict=pred_feed_dict)
        y_test = self.vftl_guest.get_Y_test()
        y_hat_1d = self.convert_to_1d_labels(y_prob_two_sides)
        y_test_1d = self.convert_to_1d_labels(y_test)

        debug = True
        if debug:
            print("[DEBUG] y_prob_two_sides shape {0}".format(y_prob_two_sides.shape))
            print("[DEBUG] y_prob_two_sides {0}".format(y_prob_two_sides))
            print("[DEBUG] y_test shape {0}:".format(y_test.shape))
            print("[DEBUG] y_hat_1d shape {0}:".format(y_hat_1d.shape))
            print("[DEBUG] y_test_1d shape {0}:".format(y_test_1d.shape))

        res = precision_recall_fscore_support(y_test_1d, y_hat_1d, average='weighted')
        all_fscore = self.f_score(res[0], res[1])
        acc = accuracy_score(y_test_1d, y_hat_1d)
        auc = roc_auc_score(y_test, y_prob_two_sides)

        print("[INFO] all_res:", res)
        print("[INFO] all_fscore : {0}, all_auc : {1}, all_acc : {2}".format(all_fscore, auc, acc))

        return acc, auc, all_fscore

    def guest_side_predict(self,
                           sess,
                           host_all_training_size,
                           block_size,
                           estimation_block_num=None,
                           round=5,
                           debug=True):
        # if debug:
        print("[INFO] ------> guest side predict")

        y_test = self.vftl_guest.get_Y_test()
        y_test_1d = self.convert_to_1d_labels(y_test)
        fed_fscore_list = []
        fed_acc_list = []
        fed_auc_list = []
        guest_fscore_list = []
        guest_acc_list = []
        guest_auc_list = []
        for r in range(round):

            if debug:
                print("[DEBUG] round:", r)

            if host_all_training_size is None:
                host_block_idx = np.random.choice(estimation_block_num - 1, 1)[0]
                print("host_block_idx: {0}".format(host_block_idx))
                host_assistant_pred_feed_dict = self.vftl_host.get_assist_guest_predict_feed_dict(
                    block_idx=host_block_idx)
            else:
                host_block_indices = np.random.choice(host_all_training_size, block_size)
                host_assistant_pred_feed_dict = self.vftl_host.get_assist_guest_predict_feed_dict(
                    block_indices=host_block_indices)

            guest_side_pred_feed_dict = self.vftl_guest.get_one_side_predict_feed_dict()
            guest_side_pred_feed_dict.update(host_assistant_pred_feed_dict)

            y_prob_fed_guest, y_prob_guest_side = sess.run(
                [self.fed_lr.y_hat_guest_side, self.guest_lr.y_hat_guest_side], feed_dict=guest_side_pred_feed_dict)
            y_hat_1d_fed_guest = self.convert_to_1d_labels(y_prob_fed_guest)
            y_hat_1d_only_guest = self.convert_to_1d_labels(y_prob_guest_side)
            # y_test_1d = self.convert_to_1d_labels(y_test)

            fed_guest_res = precision_recall_fscore_support(y_test_1d, y_hat_1d_fed_guest, average='weighted')
            fed_fscore = self.f_score(fed_guest_res[0], fed_guest_res[1])
            fed_guest_acc = accuracy_score(y_test_1d, y_hat_1d_fed_guest)
            fed_guest_auc = roc_auc_score(y_test, y_prob_fed_guest)

            only_guest_res = precision_recall_fscore_support(y_test_1d, y_hat_1d_only_guest, average='weighted')
            only_guest_fscore = self.f_score(only_guest_res[0], only_guest_res[1])
            only_guest_acc = accuracy_score(y_test_1d, y_hat_1d_only_guest)
            only_guest_auc = roc_auc_score(y_test, y_prob_guest_side)

            debug = True
            if debug:
                print("[DEBUG] y_hat_1d_fed_guest shape:", y_hat_1d_fed_guest.shape)
                print("[DEBUG] y_hat_1d_only_guest shape:", y_hat_1d_only_guest.shape)

            print("[INFO] fed_guest_res:", fed_guest_res)
            print("[INFO] fed_fscore : {0}, fed_guest_auc : {1}, fed_guest_acc : {2}".format(fed_fscore, fed_guest_auc, fed_guest_acc))
            print("[INFO] only_guest_res:", only_guest_res)
            print("[INFO] only_guest_fscore : {0}, only_guest_auc : {1}, only_guest_acc : {2}".format(only_guest_fscore, only_guest_auc, only_guest_acc))

            fed_fscore_list.append(fed_fscore)
            fed_acc_list.append(fed_guest_acc)
            fed_auc_list.append(fed_guest_auc)

            guest_fscore_list.append(only_guest_fscore)
            guest_acc_list.append(only_guest_acc)
            guest_auc_list.append(only_guest_auc)

        return (fed_fscore_list, fed_acc_list, fed_auc_list), (guest_fscore_list, guest_acc_list, guest_auc_list)

    def host_side_predict(self,
                          sess,
                          guest_all_training_size,
                          block_size,
                          estimation_block_num=None,
                          round=5,
                          debug=True):
        # if debug:
        print("[INFO] ------> host side predict")

        y_test = self.vftl_guest.get_Y_test()

        # y_test = y_test[:10]
        y_test_1d = self.convert_to_1d_labels(y_test)
        fscore_list = []
        acc_list = []
        auc_list = []
        for r in range(round):
            if debug:
                print("[DEBUG] round:", r)

            if guest_all_training_size is None:
                guest_block_idx = np.random.choice(estimation_block_num - 1, 1)[0]
                print("guest_block_idx: {0}".format(guest_block_idx))
                guest_for_host_predict_along_feed_dict = self.vftl_guest.get_assist_host_distance_based_predict_feed_dict(
                    block_idx=guest_block_idx)
                guest_assistant_pred_feed_dict = self.vftl_guest.get_assist_host_side_predict_feed_dict(
                    block_idx=guest_block_idx)
            else:
                print("guest_all_training_size:", guest_all_training_size)
                guest_block_indices = np.random.choice(guest_all_training_size, block_size)
                guest_for_host_predict_along_feed_dict = self.vftl_guest.get_assist_host_distance_based_predict_feed_dict(
                    block_indices=guest_block_indices)
                guest_assistant_pred_feed_dict = self.vftl_guest.get_assist_host_side_predict_feed_dict(
                    block_indices=guest_block_indices)

            # predict host labels by federated model
            host_alone_predict_feed_dict = self.vftl_host.get_one_side_distance_based_predict_feed_dict()
            host_alone_predict_feed_dict.update(guest_for_host_predict_along_feed_dict)
            att_y_prob_host_side = sess.run(self.Uh_non_overlap_ested_soft_lbls, feed_dict=host_alone_predict_feed_dict)

            # predict host labels by attention-based estimation
            host_side_pred_feed_dict = self.vftl_host.get_one_side_predict_feed_dict()
            # guest_assistant_pred_feed_dict = self.vftl_guest.get_assist_host_side_predict_feed_dict(
            #     block_indices=guest_block_indices)
            host_side_pred_feed_dict.update(guest_assistant_pred_feed_dict)
            # fl_y_prob_host_side = sess.run(self.logistic_regressor.y_hat_host_side, feed_dict=host_side_pred_feed_dict)
            fl_y_prob_host_side, fl_y_reprs = sess.run([self.fed_lr.y_hat_host_side, self.fed_lr.reprs],
                                                       feed_dict=host_side_pred_feed_dict)
            # aggregation of federated model and attention-based estimation
            agg_y_prob_host_side = 0.5 * fl_y_prob_host_side + 0.5 * att_y_prob_host_side
            # comb_y_prob_host_side = 1.0 * fl_y_prob_host_side + 0.0 * att_y_hat_host_side_1d

            # host labels predicted by federated predictive models
            fl_y_hat_host_side_1d = self.convert_to_1d_labels(fl_y_prob_host_side)
            # host labels predicted by attention based estimation
            att_y_hat_host_side_1d = self.convert_to_1d_labels(att_y_prob_host_side)
            # aggregated host labels
            agg_y_hat_host_side_1d = self.convert_to_1d_labels(agg_y_prob_host_side)

            fl_y_host_res = precision_recall_fscore_support(y_test_1d, fl_y_hat_host_side_1d, average='weighted')
            att_y_host_res = precision_recall_fscore_support(y_test_1d, att_y_hat_host_side_1d, average='weighted')
            agg_y_host_res = precision_recall_fscore_support(y_test_1d, agg_y_hat_host_side_1d, average='weighted')

            fl_y_fscore = self.f_score(fl_y_host_res[0], fl_y_host_res[1])
            att_y_fscore = self.f_score(att_y_host_res[0], att_y_host_res[1])
            agg_y_fscore = self.f_score(agg_y_host_res[0], agg_y_host_res[1])

            fl_host_acc = accuracy_score(y_test_1d, fl_y_hat_host_side_1d)
            # print("fl_y_reprs: ", fl_y_reprs, fl_y_reprs.shape)
            # model_param = self.fed_lr.get_model_parameters()
            # model_param_W = model_param["W"]
            # print("model_param:", model_param_W, model_param_W.shape)

            # for idx, fl_y_reprs_i in enumerate(fl_y_reprs):
            #     if np.isnan(fl_y_reprs_i).any():
            #         print("fl_y_reprs_i", idx, fl_y_reprs_i)

            # print("y_test_1d: ")
            # for idx, y_test_1d_i in enumerate(y_test_1d):
            #     if np.isnan(y_test_1d_i).any():
            #         print("y_test_1d_i", idx, y_test_1d_i)
            # print("att_y_prob_host_side: ")
            # for idx, att_y_prob_host_side_i in enumerate(att_y_prob_host_side):
            #     if np.isnan(att_y_prob_host_side_i).any():
            #         print("att_y_prob_host_side_i", idx, att_y_prob_host_side_i)
            # print("fl_y_prob_host_side: ")
            # for idx, fl_y_prob_host_side_i in enumerate(fl_y_prob_host_side):
            #     if np.isnan(fl_y_prob_host_side_i).any():
            #         print("fl_y_prob_host_side_i", idx, fl_y_prob_host_side_i)
            #
            # print("y_test_1d: ")
            # for idx, y_test_1d_i in enumerate(y_test_1d):
            #     # if np.isnan(y_test_1d_i).any():
            #     print("y_test_1d_i", idx, y_test_1d_i)
            #     if idx > 50:
            #         break
            # print("att_y_prob_host_side: ")
            # for idx, att_y_prob_host_side_i in enumerate(att_y_hat_host_side_1d):
            #     # if np.isnan(att_y_prob_host_side_i).any():
            #     print("att_y_prob_host_side_i", idx, att_y_prob_host_side_i)
            #     if idx > 50:
            #         break
            # print("fl_y_prob_host_side: ")
            # for idx, fl_y_prob_host_side_i in enumerate(fl_y_hat_host_side_1d):
            #     # if np.isnan(fl_y_prob_host_side_i).any():
            #     print("fl_y_prob_host_side_i", idx, fl_y_prob_host_side_i)
            #     if idx > 50:
            #         break

            fl_host_auc = roc_auc_score(y_test, fl_y_prob_host_side)
            # fl_host_auc = 1.0

            debug = True
            if debug:
                print("[DEBUG] fl_y_hat_host_side_1d shape:", fl_y_hat_host_side_1d.shape)

            print("[INFO] fl_y_host_res:", fl_y_host_res)
            print("[INFO] att_y_host_res:", att_y_host_res)
            print("[INFO] agg_y_host_res:", agg_y_host_res)
            print("[INFO] fl_y_fscore : {0}, att_y_fscore : {1}, agg_y_fscore : {2}".format(fl_y_fscore,
                                                                                             att_y_fscore,
                                                                                             agg_y_fscore))
            print("[INFO] fl_host_auc : {0}, fl_host_acc : {1}".format(fl_host_auc, fl_host_acc))

            fscore_list.append(fl_y_fscore)
            acc_list.append(fl_host_acc)
            auc_list.append(fl_host_auc)
        return fscore_list, acc_list, auc_list

    def fit(self,
            sess,
            overlap_batch_range,
            guest_non_overlap_batch_range,
            host_non_overlap_batch_range,
            guest_block_idx=None,
            host_block_idx=None,
            guest_block_indices=None,
            host_block_indices=None,
            debug=True):

        train_feed_dict = self.vftl_guest.get_train_feed_dict(overlap_batch_range=overlap_batch_range,
                                                              non_overlap_batch_range=guest_non_overlap_batch_range,
                                                              block_indices=guest_block_indices,
                                                              block_idx=guest_block_idx)
        train_host_feed_dict = self.vftl_host.get_train_feed_dict(overlap_batch_range=overlap_batch_range,
                                                                  non_overlap_batch_range=host_non_overlap_batch_range,
                                                                  block_indices=host_block_indices,
                                                                  block_idx=host_block_idx)
        train_feed_dict.update(train_host_feed_dict)

        # weights for auxiliary losses, which include:
        # (1) loss for shared representations between host and guest
        # (2) (3) loss for orthogonal representation for host and guest respectively
        # (4) loss for distance between estimated host overlap labels and true overlap labels
        # (5) loss for distance between estimated guest overlap representation and true guest representation
        # (6) loss for distance between estimated host overlap representation and true host representation
        # (7) loss for distance between shared-repr-estimated host label and uniq-repr-estimated host label

        # _, fed_reprs, fed_Y, shared_reprs_dist_loss, distinct_reprs_loss_1, distinct_reprs_loss_2, lbl_loss, \
        # guest_repr_dist, host_repr_dist, ested_lbl_dist, regularization_loss, pred_loss, mean_pred_loss, ob_loss, \
        # logits, train_lbls, ested_lbls_, true_lbls_, computed_gradients, reprswlabels, \
        # host_non_overlap_reprs_w_condidate_labels, uniqlbls, commlbls, Ug_non_overlap_ested_reprs_w_lbls, \
        # Ug_non_overlap_ested_lbls = sess.run(

        # print("=======")
        # # print("train_feed_dict:", train_feed_dict)
        # print(self.fed_lr.e2e_train_op)
        # print(self.fed_reprs)
        # print(self.fed_Y)
        # print(self.assistant_loss_list[0])
        # print(self.assistant_loss_list[1])
        # print(self.assistant_loss_list[2])
        # print(self.assistant_loss_list[3])
        # print(self.assistant_loss_list[4])
        # print(self.assistant_loss_list[5])
        # print(self.assistant_loss_list[6])
        # print("here --->")
        # print(self.fed_lr.regularization_loss)
        # print("here <----")
        # print(self.fed_lr.pred_loss)
        # print(self.fed_lr.mean_pred_loss)
        # print(self.fed_lr.loss)
        # print(self.fed_lr.logits)
        # print(self.fed_lr.tf_labels_in)
        # print(self.Ug_overlap_ested_soft_lbls)
        # print(self.Y_overlap_for_est)
        # print(self.fed_lr.computed_gradients)
        # print(self.num_selected_samples)
        # print(self.training_fed_reprs_shape)
        # print(self.host_non_overlap_reprs_w_condidate_labels)
        # print(self.uniq_lbls)
        # print(self.comm_lbls)
        # print(self.Ug_non_overlap_ested_reprs_w_lbls)
        # print(self.nl_ested_lbls_for_att_ested_guest_reprs)
        # print(self.nl_ested_lbls_for_fed_ested_guest_reprs)
        # print(self.Ug_all_comm_temp)
        # print(self.Uh_all_comm_temp)
        # print(self.Ug_all_uniq_temp)
        # print(self.Uh_all_uniq_temp)
        # print(self.Ug_non_overlap_uniq_temp)
        # print(self.Uh_non_overlap_uniq_temp)
        # print(self.Ug_non_overlap_comm_temp)
        # print(self.Uh_non_overlap_comm_temp)
        # print(self.Ug_overlap_uniq_temp)
        # print(self.Uh_overlap_uniq_temp)
        # print(self.Ug_overlap_comm_temp)
        # print(self.Uh_overlap_comm_temp)
        # print("=======")

        _, _, fed_reprs, fed_Y, shared_reprs_dist_loss, distinct_reprs_loss_1, distinct_reprs_loss_2, ol_uniq_lbl_loss, ol_comm_lbl_loss, \
        guest_repr_dist, host_repr_dist, ested_lbl_dist, regularization_loss, pred_loss, mean_pred_loss, ob_loss, \
        logits, train_lbls, true_lbls_, computed_gradients, num_selected_samples, training_fed_reprs_shape, \
        host_non_overlap_reprs_w_condidate_labels, uniqlbls, commlbls, Ug_non_overlap_ested_reprs_w_lbls, \
        Ug_non_overlap_ested_lbls, nl_ested_lbls_for_fed_ested_guest_reprs, nl_ested_lbls_for_lr_ested_guest_reprs, \
        Ug_all_comm, Uh_all_comm, Ug_all_uniq, Uh_all_uniq, \
        Ug_non_overlap_uniq, Uh_non_overlap_uniq, Ug_non_overlap_comm, Uh_non_overlap_comm, \
        Ug_overlap_uniq, Uh_overlap_uniq, Ug_overlap_comm, Uh_overlap_comm = sess.run(
            [
                # self.original_host_nonoverlap_samples_shape,
                # self.selected_host_nonoverlap_samples_shape,
                self.fed_lr.e2e_train_op,
                self.guest_lr.e2e_train_op,
                self.fed_reprs,
                self.fed_Y,
                self.assistant_loss_list[0],
                self.assistant_loss_list[1],
                self.assistant_loss_list[2],
                self.assistant_loss_list[3],
                self.assistant_loss_list[4],
                self.assistant_loss_list[5],
                self.assistant_loss_list[6],
                self.assistant_loss_list[7],
                self.fed_lr.reg_loss,
                self.fed_lr.pred_loss,
                self.fed_lr.mean_pred_loss,
                self.fed_lr.loss,
                self.fed_lr.logits,
                self.fed_lr.tf_labels_in,
                # self.Ug_overlap_ested_soft_lbls,
                # self.Y_overlap_test_,
                self.Y_overlap_for_est,
                self.fed_lr.computed_gradients,
                self.num_selected_samples,
                self.training_fed_reprs_shape,
                self.host_non_overlap_reprs_w_condidate_labels,
                self.uniq_lbls,
                self.comm_lbls,
                self.Ug_non_overlap_ested_reprs_w_lbls,
                self.nl_ested_lbls_for_att_ested_guest_reprs,
                self.nl_ested_lbls_for_fed_ested_guest_reprs,
                self.nl_ested_lbls_for_lr_ested_guest_reprs,
                self.Ug_all_comm_temp,
                self.Uh_all_comm_temp,
                self.Ug_all_uniq_temp,
                self.Uh_all_uniq_temp,
                self.Ug_non_overlap_uniq_temp,
                self.Uh_non_overlap_uniq_temp,
                self.Ug_non_overlap_comm_temp,
                self.Uh_non_overlap_comm_temp,
                self.Ug_overlap_uniq_temp,
                self.Uh_overlap_uniq_temp,
                self.Ug_overlap_comm_temp,
                self.Uh_overlap_comm_temp
            ],
            feed_dict=train_feed_dict)

        # self.Ug_all_comm_temp = Ug_all_comm
        # self.Uh_all_comm_temp = Uh_all_comm
        # self.Ug_all_uniq_temp = Ug_all_uniq
        # self.Uh_all_uniq_temp = Uh_all_uniq
        #
        # self.Ug_non_overlap_uniq_temp = Ug_non_overlap_uniq
        # self.Uh_non_overlap_uniq_temp = Uh_non_overlap_uniq
        # self.Ug_non_overlap_comm_temp = Ug_non_overlap_comm
        # self.Uh_non_overlap_comm_temp = Uh_non_overlap_comm
        #
        # self.Ug_overlap_uniq_temp = Ug_overlap_uniq
        # self.Uh_overlap_uniq_temp = Uh_overlap_uniq
        # self.Ug_overlap_comm_temp = Ug_overlap_comm
        # self.Uh_overlap_comm_temp = Uh_overlap_comm

        print("[INFO] num_selected_samples", num_selected_samples)
        print("[INFO] training_fed_reprs_shape", training_fed_reprs_shape)

        # print("[DEBUG] Ug_overlap_uniq", Ug_overlap_uniq, Ug_overlap_uniq.shape)
        # print("[DEBUG] Uh_overlap_uniq", Uh_overlap_uniq, Uh_overlap_uniq.shape)
        # print("[DEBUG] Ug_overlap_comm", Ug_overlap_comm, Ug_overlap_comm.shape)
        # print("[DEBUG] Uh_overlap_comm", Uh_overlap_comm, Uh_overlap_comm.shape)
        #
        # print("[DEBUG] Ug_non_overlap_uniq", Ug_non_overlap_uniq, Ug_non_overlap_uniq.shape)
        # print("[DEBUG] Uh_non_overlap_uniq", Uh_non_overlap_uniq, Uh_non_overlap_uniq.shape)
        # print("[DEBUG] Ug_non_overlap_comm", Ug_non_overlap_comm, Ug_non_overlap_comm.shape)
        # print("[DEBUG] Uh_non_overlap_comm", Uh_non_overlap_comm, Uh_non_overlap_comm.shape)
        #
        # print("[DEBUG] Ug_all_comm", Ug_all_comm, Ug_all_comm.shape)
        # print("[DEBUG] Uh_all_comm", Uh_all_comm, Uh_all_comm.shape)
        # print("[DEBUG] Ug_all_uniq", Ug_all_uniq, Ug_all_uniq.shape)
        # print("[DEBUG] Uh_all_uniq", Uh_all_uniq, Uh_all_uniq.shape)

        # Uh_non_overlap_comm,
        # Uh_non_overlap_uniq,
        # Uh_overlap_uniq,
        # Ug_overlap_uniq,
        # Ug_all_comm,
        # print("===> Uh_non_overlap_comm", Uh_non_overlap_comm, Uh_non_overlap_comm.shape)
        # print("===> Uh_non_overlap_uniq", Uh_non_overlap_uniq, Uh_non_overlap_uniq.shape)
        # print("===> Uh_overlap_uniq", Uh_overlap_uniq, Uh_overlap_uniq.shape)
        # print("===> Ug_overlap_uniq", Ug_overlap_uniq, Ug_overlap_uniq.shape)
        # print("===> Ug_all_comm", Ug_all_comm, Ug_all_comm.shape)

        # print("computed_gradients: ")
        # for idx, computed_gradients_i in enumerate(computed_gradients):
        #     if np.isnan(computed_gradients_i).any():
        #         print("computed_gradients_i", idx, computed_gradients_i)

        # guest_nn, guest_nn_prime = self.vftl_guest.get_model_parameters()
        # host_nn, host_nn_prime = self.vftl_host.get_model_parameters()
        # model_param = self.logistic_regressor.get_model_parameters()
        # print("model_param:", model_param["b"])
        # for idx, model_param_i in enumerate(model_param["W"]):
        #     if np.isnan(model_param_i).any():
        #         print("model_param_i", idx, model_param_i)
        # print("guest_nn", guest_nn["bo"])
        # for idx, guest_nn_i in enumerate(guest_nn["Wh"]):
        #     if np.isnan(guest_nn_i).any():
        #         print("guest_nn_i", idx, guest_nn_i)
        # print("guest_nn_prime",  guest_nn_prime["bo"])
        # for idx, guest_nn_prime_i in enumerate(guest_nn_prime["Wh"]):
        #     if np.isnan(guest_nn_prime_i).any():
        #         print("guest_nn_prime_i", idx, guest_nn_prime_i)
        # print("host_nn",  host_nn["bo"])
        # for idx, host_nn_i in enumerate(host_nn["Wh"]):
        #     if np.isnan(host_nn_i).any():
        #         print("host_nn_i", idx, host_nn_i)
        # print("host_nn_prime",  host_nn_prime["bo"])
        # for idx, host_nn_prime_i in enumerate(host_nn_prime["Wh"]):
        #     if np.isnan(host_nn_prime_i).any():
        #         print("host_nn_prime_i", idx, host_nn_prime_i)

        print("[INFO] Ug_non_overlap_ested_lbls shape:", Ug_non_overlap_ested_lbls.shape)
        print("[INFO] fed_host_non_overlap_ested_lbls shape:", nl_ested_lbls_for_fed_ested_guest_reprs.shape)
        print("[INFO] nl_ested_lbls_for_lr_ested_guest_reprs shape:", nl_ested_lbls_for_lr_ested_guest_reprs.shape)
        present_count = 0
        for att_lbl, lr_lbl, fed_lbl in zip(Ug_non_overlap_ested_lbls, nl_ested_lbls_for_lr_ested_guest_reprs,
                                            nl_ested_lbls_for_fed_ested_guest_reprs):
            fed_lbl_idx = np.argmax(fed_lbl)
            lr_lbl_idx = np.argmax(lr_lbl)
            att_lbl_idx = np.argmax(att_lbl)
            fed_lbl_prob = np.max(fed_lbl)
            lr_lbl_prob = np.max(lr_lbl)
            att_lbl_prob = np.max(att_lbl)
            if fed_lbl_idx == lr_lbl_idx:

                if present_count < 10:
                    print("att         lr       fed")
                    print("[{0}]:{1}, [{2}]:{3}, [{4}]:{5}".format(att_lbl_idx, att_lbl_prob,
                                                                   lr_lbl_idx, lr_lbl_prob,
                                                                   fed_lbl_idx, fed_lbl_prob))
                present_count += 1

        print("total number of equal predictions: {0}".format(present_count))

        debug_detail = False
        if True:
            # print("uniqlbls", uniqlbls, uniqlbls.shape)
            # print("commlbls", commlbls, commlbls.shape)
            #
            # print("Ug_non_overlap_ested_reprs_w_lbls", Ug_non_overlap_ested_reprs_w_lbls, Ug_non_overlap_ested_reprs_w_lbls.shape)
            # print("Ug_non_overlap_ested_lbls", Ug_non_overlap_ested_lbls, Ug_non_overlap_ested_lbls.shape)

            # print("host_non_overlap_reprs_w_condidate_labels:", host_non_overlap_reprs_w_condidate_labels.shape)
            # for idx, host_non_overlap_reprs_w_condidate_labels in enumerate(host_non_overlap_reprs_w_condidate_labels):
            #     if np.isnan(host_non_overlap_reprs_w_condidate_labels).any():
            #       print("host_non_overlap_reprs_w_condidate_labels", idx, host_non_overlap_reprs_w_condidate_labels)

            # print("Ug_non_overlap_ested_lbls shape:", Ug_non_overlap_ested_lbls.shape)
            # print("fed_host_non_overlap_ested_lbls shape:", nl_ested_lbls_for_fed_ested_guest_reprs.shape)
            # for Ug_host_lbl, fed_host_lbl in zip(Ug_non_overlap_ested_lbls, nl_ested_lbls_for_fed_ested_guest_reprs):
            #     print("host lbl:", np.argmax(Ug_host_lbl), np.argmax(fed_host_lbl), np.sum(Ug_host_lbl),
            #           np.sum(fed_host_lbl))

            # val = None
            # print("selected reprs with labels:", reprswlabels, reprswlabels.shape)
            # print("selected reprs with labels:", reprswlabels.shape)
            # for idx, repr_lbl in enumerate(reprswlabels):
            #     if np.isnan(repr_lbl).any():
            #         print("repr_lbl", idx, repr_lbl)
            #         val = repr_lbl[0]
            # print("val", val)

            # val = None
            # print("host_non_overlap_reprs_w_condidate_labels:", host_non_overlap_reprs_w_condidate_labels.shape)
            # for idx, host_non_overlap_reprs_w_condidate_labels in enumerate(host_non_overlap_reprs_w_condidate_labels):
            #     if host_non_overlap_reprs_w_condidate_labels[0] == val:
            #         print("host_non_overlap_reprs_w_condidate_labels", idx, host_non_overlap_reprs_w_condidate_labels)

            # print("* logits:", logits, logits.shape)
            # print("* train_lbls:", train_lbls, logits.shape)
            # # for idx in range(len(logits)):
            # #     print("l v.s. t:", idx, logits[idx], train_lbls[idx])

            # print("[DEBUG] ested_lbls_:", ested_lbls_)
            # for row in range(len(ested_lbls_)):
            #     print("[DEBUG]    row {0}: {1}".format(row, ested_lbls_[row]))
            # print("[DEBUG]ested_lbls_ shape:", ested_lbls_.shape)
            # print("[DEBUG]Y_overlap_test_:", Y_overlap_test_)
            # print("[DEBUG]Y_overlap_test_ shape:", Y_overlap_test_.shape)

            # # print("sum of true_lbls_:", np.sum(true_lbls_, axis=1))
            # # print("original_host_nonoverlap_samples_shape:", original_shape)
            # # print("selected_host_nonoverlap_samples_shape:", selected_shape)

            print("[DEBUG] (1) shared_reprs_dist_loss", shared_reprs_dist_loss)
            print("[DEBUG] (2) distinct_reprs_loss_1", distinct_reprs_loss_1)
            print("[DEBUG] (3) distinct_reprs_loss_2", distinct_reprs_loss_2)
            print("[DEBUG] (4) ol_uniq_lbl_loss", ol_uniq_lbl_loss)
            print("[DEBUG] (5) ol_comm_lbl_loss", ol_comm_lbl_loss)
            print("[DEBUG] (6) guest_repr_dist", guest_repr_dist)
            print("[DEBUG] (7) host_repr_dist", host_repr_dist)
            print("[DEBUG] (8) ested_lbl_dist", ested_lbl_dist)
            print("[DEBUG] regularization_loss", regularization_loss)
            print("[DEBUG] mean_pred_loss", mean_pred_loss)
            print("[DEBUG] total ob_loss", ob_loss)

            # print("pred_loss", pred_loss)
            # for pl in pred_loss:
            #     if pl is np.NaN:
            #         print("pl", pl)

            # print("fed_reprs: \n", fed_reprs.shape)
            # print("fed_Y\n", fed_Y.shape)
            # print("fed_reprs: \n", fed_reprs, fed_reprs.shape)
            # print("fed_Y\n", fed_Y, fed_Y.shape)
            # print("computed_gradients \n", computed_gradients)
            # print("Ug_overlap_uniq_temp: \n", Ug_overlap_uniq_temp)
            # print("Uh_overlap_uniq_temp: \n", Uh_overlap_uniq_temp)
            # print("Ug_non_overlap_uniq: \n", Ug_non_overlap_uniq)
            # print("Uh_non_overlap_uniq: \n", Uh_non_overlap_uniq)

            # print("computed_gradients:", computed_gradients)

            if debug_detail:
                guest_nn, guest_nn_prime = self.vftl_guest.get_model_parameters()
                host_nn, host_nn_prime = self.vftl_host.get_model_parameters()

                print("[DEBUG] guest_nn", guest_nn)
                print("[DEBUG] guest_nn_prime", guest_nn_prime)
                print("[DEBUG] host_nn", host_nn)
                print("[DEBUG] host_nn_prime", host_nn_prime)

        return ob_loss

    def train(self, debug=True):

        using_block_idx = self.model_param.using_block_idx
        nol_batch_size = self.model_param.non_overlap_sample_batch_size
        ol_batch_size = self.model_param.overlap_sample_batch_size

        nol_guest_batch_size = nol_batch_size
        nol_host_batch_size = nol_batch_size
        # ol_batch_size = nol_guest_batch_size

        print("[INFO] using_block_idx:", using_block_idx)
        print("[INFO] ol_batch_size:", ol_batch_size)
        print("[INFO] nol_guest_batch_size:", nol_guest_batch_size)
        print("[INFO] nol_host_batch_size:", nol_host_batch_size)

        # guest and host should have the same overlap block number
        ol_block_num = self.vftl_guest.get_ol_block_number()
        guest_nol_block_num = self.vftl_guest.get_nol_block_number()
        host_nol_block_num = self.vftl_host.get_nol_block_number()
        guest_ested_block_num = self.vftl_guest.get_ested_block_number()
        host_ested_block_num = self.vftl_host.get_ested_block_number()

        guest_all_training_size = None
        host_all_training_size = None
        if not using_block_idx:
            guest_all_training_size = self.vftl_guest.get_all_training_sample_size()
            host_all_training_size = self.vftl_host.get_all_training_sample_size()
            print("[INFO] guest_all_training_size:", guest_all_training_size)
            print("[INFO] host_all_training_size:", host_all_training_size)

        guest_block_indices = None
        host_block_indices = None
        estimation_block_size = None

        early_stopping = EarlyStoppingCheckPoint(monitor="fscore", epoch_patience=200)
        early_stopping.set_model(self)
        early_stopping.on_train_begin()

        # load validation data
        guest_val_block_size = self.vftl_guest.load_val_block(0)
        host_val_block_size = self.vftl_host.load_val_block(0)
        print("[INFO] guest_val_block_size:", guest_val_block_size)
        print("[INFO] host_val_block_size:", host_val_block_size)

        start_time = time.time()
        epoch = self.model_param.epoch
        init = tf.compat.v1.global_variables_initializer()
        gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0")
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            self.vftl_guest.set_session(sess)
            self.vftl_host.set_session(sess)
            self.fed_lr.set_session(sess)

            sess.run(init)

            loss_list = []
            all_auc_list = []
            all_acc_list = []
            all_fscore_list = []
            for i in range(epoch):
                print("[INFO] ===> start epoch:{0}".format(i))

                ol_batch_idx = 0
                nol_guest_batch_idx = 0
                nol_host_batch_idx = 0

                ol_block_idx = 0
                nol_guest_block_idx = 0
                nol_host_block_idx = 0
                ested_guest_block_idx = -1
                ested_host_block_idx = -1

                ol_end = 0
                nol_guest_end = 0
                nol_host_end = 0

                ol_guest_block_size = self.vftl_guest.load_ol_block(ol_block_idx)
                ol_host_block_size = self.vftl_host.load_ol_block(ol_block_idx)
                assert ol_guest_block_size == ol_host_block_size

                nol_guest_block_size = self.vftl_guest.load_nol_block(nol_guest_block_idx)
                nol_host_block_size = self.vftl_host.load_nol_block(nol_host_block_idx)

                iter = 0
                while True:
                    print("[INFO] ===> iter:{0} of ep: {1}".format(iter, i))
                    if ol_end >= ol_guest_block_size:
                        ol_block_idx += 1
                        if ol_block_idx == ol_block_num:
                            # if all blocks for overlapping samples have been visited,
                            # start over from the first block
                            ol_block_idx = 0
                        ol_guest_block_size = self.vftl_guest.load_ol_block(ol_block_idx)
                        ol_host_block_size = self.vftl_host.load_ol_block(ol_block_idx)
                        assert ol_guest_block_size == ol_host_block_size
                        ol_batch_idx = 0

                    ol_start = ol_batch_size * ol_batch_idx
                    ol_end = ol_batch_size * ol_batch_idx + ol_batch_size

                    print("[DEBUG] ol_block_idx:", ol_block_idx)
                    print("[DEBUG] ol_guest_block_size:", ol_guest_block_size)
                    print("[DEBUG] ol_host_block_size:", ol_host_block_size)
                    print("[DEBUG] ol batch from {0} to {1} ".format(ol_start, ol_end))

                    # nol_guest_start = nol_guest_batch_size * nol_guest_batch_idx
                    # nol_guest_end = nol_guest_batch_size * nol_guest_batch_idx + nol_guest_batch_size
                    # nol_host_start = nol_host_batch_size * nol_host_batch_idx
                    # nol_host_end = nol_host_batch_size * nol_host_batch_idx + nol_host_batch_size

                    if nol_guest_end >= nol_guest_block_size:
                        nol_guest_block_idx += 1
                        if nol_guest_block_idx == guest_nol_block_num:
                            # if all blocks for non-overlapping samples of guest have been visited,
                            # end current epoch and start a new epoch
                            print(
                                "[INFO] all blocks for non-overlapping samples of "
                                "guest have been visited, end current epoch and start a new epoch")
                            break
                        nol_guest_block_size = self.vftl_guest.load_nol_block(nol_guest_block_idx)
                        nol_guest_batch_idx = 0

                    nol_guest_start = nol_guest_batch_size * nol_guest_batch_idx
                    nol_guest_end = nol_guest_batch_size * nol_guest_batch_idx + nol_guest_batch_size

                    print("[DEBUG] nol_guest_block_idx:", nol_guest_block_idx)
                    print("[DEBUG] nol_guest_block_size:", nol_guest_block_size)
                    print("[DEBUG] nol guest batch from {0} to {1} ".format(nol_guest_start, nol_guest_end))

                    if nol_host_end >= nol_host_block_size:
                        nol_host_block_idx += 1
                        if nol_host_block_idx == host_nol_block_num:
                            # if all blocks for non-overlapping samples of host have been visited,
                            # end current epoch and start a new epoch
                            print(
                                "[INFO] all blocks for non-overlapping samples of "
                                "host have been visited, end current epoch and start a new epoch")
                            break
                        nol_host_block_size = self.vftl_host.load_nol_block(nol_host_block_idx)
                        nol_host_batch_idx = 0

                    nol_host_start = nol_host_batch_size * nol_host_batch_idx
                    nol_host_end = nol_host_batch_size * nol_host_batch_idx + nol_host_batch_size

                    print("[DEBUG] nol_host_block_idx:", nol_host_block_idx)
                    print("[DEBUG] nol_host_block_size:", nol_host_block_size)
                    print("[DEBUG] nol host batch from {0} to {1}".format(nol_host_start, nol_host_end))

                    if using_block_idx:
                        print("[DEBUG] Using block idx")

                        ested_guest_block_idx += 1
                        if ested_guest_block_idx >= guest_ested_block_num:
                            ested_guest_block_idx = 0

                        ested_host_block_idx += 1
                        if ested_host_block_idx >= host_ested_block_num:
                            ested_host_block_idx = 0

                        print("[DEBUG] ested_guest_block_idx:", ested_guest_block_idx)
                        print("[DEBUG] ested_host_block_idx:", ested_host_block_idx)
                    else:
                        print("# Using block indices")
                        estimation_block_size = self.model_param.all_sample_block_size
                        guest_block_indices = np.random.choice(guest_all_training_size, estimation_block_size)
                        host_block_indices = np.random.choice(host_all_training_size, estimation_block_size)
                        print("guest_block_indices:", len(guest_block_indices))
                        print("host_block_indices:", len(host_block_indices))

                    if debug:
                        print("ol_start:ol_end ", ol_start, ol_end)
                        print("nol_guest_start:nol_guest_end ", nol_guest_start, nol_guest_end)
                        print("nol_host_start:nol_host_end ", nol_host_start, nol_host_end)
                        if guest_block_indices is not None:
                            print("num guest_block_indices:", len(guest_block_indices))
                            print("num host_block_indices:", len(host_block_indices))

                    print("guest_block_indices: ", guest_block_indices)

                    loss = self.fit(sess=sess,
                                    overlap_batch_range=(ol_start, ol_end),
                                    guest_non_overlap_batch_range=(nol_guest_start, nol_guest_end),
                                    host_non_overlap_batch_range=(nol_host_start, nol_host_end),
                                    guest_block_idx=ested_guest_block_idx,
                                    host_block_idx=ested_host_block_idx,
                                    guest_block_indices=guest_block_indices,
                                    host_block_indices=host_block_indices,
                                    debug=debug)
                    loss_list.append(loss)
                    print("")
                    print("[INFO] ep:{0}, ol_batch_idx:{1}, nol_guest_batch_idx:{2}, nol_host_batch_idx:{3}, loss:{4}"
                        .format(i, ol_batch_idx, nol_guest_batch_idx, nol_host_batch_idx, loss))
                    print("[INFO]    ol_block_idx:{0}, nol_guest_block_idx:{1}, nol_host_block_idx:{2}, "
                          "ested_guest_block_idx:{3}, ested_host_block_idx:{4}".format(ol_block_idx,
                                                                                       nol_guest_block_idx,
                                                                                       nol_host_block_idx,
                                                                                       ested_guest_block_idx,
                                                                                       ested_host_block_idx)
                          .format(i, ol_batch_idx, nol_guest_batch_idx, nol_host_batch_idx, loss))

                    ol_batch_idx = ol_batch_idx + 1
                    nol_guest_batch_idx = nol_guest_batch_idx + 1
                    nol_host_batch_idx = nol_host_batch_idx + 1

                    # two sides test
                    all_acc, all_auc, all_fscore = self.two_side_predict(sess, debug=debug)
                    all_acc_list.append(all_acc)
                    all_auc_list.append(all_auc)
                    all_fscore_list.append(all_fscore)

                    # guest side test
                    fed_res, guest_res = self.guest_side_predict(sess,
                                                                 host_all_training_size,
                                                                 estimation_block_size,
                                                                 estimation_block_num=guest_ested_block_num,
                                                                 debug=debug)
                    g_fed_fscore_list, g_fed_acc_list, g_fed_auc_list = fed_res
                    g_only_fscore_list, g_only_acc_list, g_only_auc_list = guest_res

                    g_fed_fscore_mean = np.mean(g_fed_fscore_list)
                    g_fed_acc_mean = np.mean(g_fed_acc_list)
                    g_fed_auc_mean = np.mean(g_fed_auc_list)

                    g_only_fscore_mean = np.mean(g_only_fscore_list)
                    g_only_acc_mean = np.mean(g_only_acc_list)
                    g_only_auc_mean = np.mean(g_only_auc_list)

                    if debug:
                        print("%% g_fed_fscore_list:", g_fed_fscore_list, g_fed_fscore_mean)
                        print("%% g_fed_acc_list:", g_fed_acc_list, g_fed_acc_mean)
                        print("%% g_fed_auc_list:", g_fed_auc_list, g_fed_auc_mean)

                    # host side test
                    h_fscore_list, h_acc_list, h_auc_list = self.host_side_predict(sess,
                                                                                   guest_all_training_size,
                                                                                   estimation_block_size,
                                                                                   estimation_block_num=host_ested_block_num,
                                                                                   debug=debug)
                    h_fscore_mean = np.mean(h_fscore_list)
                    h_acc_mean = np.mean(h_acc_list)
                    h_auc_mean = np.mean(h_auc_list)

                    if debug:
                        print("%% h_fscore_list:", h_fscore_list, h_fscore_mean)
                        print("%% h_acc_list:", h_acc_list, h_acc_mean)
                        print("%% h_auc_list:", h_auc_list, h_auc_mean)

                    print("=" * 100)
                    print("[INFO] ===> fscore: all, fed_guest, only_guest, host", all_fscore, g_fed_fscore_mean,
                          g_only_fscore_mean, h_fscore_mean)
                    print("[INFO] ===> acc: all, fed_guest, only_guest, host", all_acc, g_fed_acc_mean, g_only_acc_mean,
                          h_acc_mean)
                    print("[INFO] ===> auc: all, fed_guest, only_guest, host", all_auc, g_fed_auc_mean, g_only_auc_mean,
                          h_auc_mean)
                    print("=" * 100)

                    # ave_fscore = (all_fscore + g_fed_fscore_mean + h_fscore_mean) / 3
                    ave_fscore = 3 / (1 / all_fscore + 1 / g_fed_fscore_mean + 1 / h_fscore_mean)
                    ave_accuracy = 3 / (1 / all_acc + 1 / g_fed_acc_mean + 1 / h_acc_mean)
                    ave_accuracy_2 = 2 / (1 / all_acc + 1 / g_fed_acc_mean)
                    print("[INFO] harmonic mean of fscore:", ave_fscore)
                    print("[INFO] harmonic mean of accuracy:", ave_accuracy)
                    print("[INFO] harmonic mean of accuracy 2:", ave_accuracy_2)
                    log = {"fscore": all_acc,
                           "all_fscore": all_fscore, "g_fscore": g_fed_fscore_mean, "h_fscore": h_fscore_mean,
                           "all_acc": all_acc, "g_acc": g_fed_acc_mean, "h_acc": h_acc_mean,
                           "all_auc": all_auc, "g_auc": g_fed_auc_mean, "h_auc": h_auc_mean}
                    early_stopping.on_validation_end(curr_epoch=i, batch_idx=nol_guest_batch_idx, log=log)
                    iter += 1
                    if self.stop_training is True:
                        break

                if self.stop_training is True:
                    break

        end_time = time.time()
        print("training time (s):", end_time - start_time)
        print("stopped epoch, batch:", early_stopping.stopped_epoch, early_stopping.stopped_batch)
        early_stopping.print_log_of_best_result()
        # series_plot(losses=loss_list, fscores=all_fscore_list, aucs=all_acc_list)
        return early_stopping.get_log_info(), loss_list
