import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from expanding_vertical_transfer_learning_param import FederatedModelParam
from logistic_regression import LogisticRegression
from regularization import EarlyStoppingCheckPoint
from bm_vertical_sstl_parties import ExpandingVFTLGuest, ExpandingVFTLHost
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
        self.guest_lr = None

    def set_representation_estimator(self, repr_estimator):
        self.repr_estimator = repr_estimator
        print("using: {0}".format(self.repr_estimator))

    def determine_overlap_sample_indices(self):
        overlap_indices = self.model_param.overlap_indices
        print("overlap_indices size:", len(overlap_indices))
        return overlap_indices

    def save_model(self):
        print("TODO: save model")

    @staticmethod
    def _create_transform_matrix(in_dim, out_dim):
        with tf.compat.v1.variable_scope("transform_matrix"):
            Wt = tf.compat.v1.get_variable(name="Wt", initializer=tf.random.normal((in_dim, out_dim), dtype=tf.float64))
        return Wt

    def _build_feature_extraction_with_transfer(self):

        Ug_overlap = self.vftl_guest.fetch_feat_reprs()
        Uh_overlap = self.vftl_host.fetch_feat_reprs()

        Y_overlap = self.vftl_guest.get_Y_overlap()
        fed_ol_reprs = tf.concat([Ug_overlap, Uh_overlap], axis=1)
        train_components_list = [fed_ol_reprs, Y_overlap]
        return train_components_list, fed_ol_reprs

    def build(self):
        learning_rate = self.model_param.learning_rate
        fed_input_dim = self.model_param.fed_input_dim
        fed_hidden_dim = self.model_param.fed_hidden_dim
        fed_reg_lambda = self.model_param.fed_reg_lambda

        print("############## Hyperparameter Info #############")
        print("learning_rate: {0}".format(learning_rate))
        print("fed_input_dim: {0}".format(fed_input_dim))
        print("fed_hidden_dim: {0}".format(fed_hidden_dim))
        print("fed_reg_lambda: {0}".format(fed_reg_lambda))
        print("################################################")

        self.fed_lr = LogisticRegression(1)
        self.fed_lr.build(input_dim=fed_input_dim, n_class=self.n_class, hidden_dim=fed_hidden_dim)

        train_components, fed_ol_reprs = self._build_feature_extraction_with_transfer()
        self.fed_reprs, self.fed_Y = train_components
        self.fed_lr.set_ops(learning_rate=learning_rate, reg_lambda=fed_reg_lambda,
                            tf_X_in=self.fed_reprs, tf_labels_in=self.fed_Y)

        self.fed_lr.set_two_sides_predict_ops(fed_ol_reprs)

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

    def fit(self,
            sess,
            overlap_batch_range,
            debug=True):

        train_feed_dict = self.vftl_guest.get_train_feed_dict(overlap_batch_range=overlap_batch_range)
        train_host_feed_dict = self.vftl_host.get_train_feed_dict(overlap_batch_range=overlap_batch_range)
        train_feed_dict.update(train_host_feed_dict)

        _, fed_reprs, fed_Y, ob_loss = sess.run(
            [
                self.fed_lr.e2e_train_op,
                self.fed_reprs,
                self.fed_Y,
                self.fed_lr.loss
            ],
            feed_dict=train_feed_dict)

        debug_detail = False
        if True:
            print("[DEBUG] total ob_loss", ob_loss)

            if debug_detail:
                guest_nn, guest_nn_prime = self.vftl_guest.get_model_parameters()
                host_nn, host_nn_prime = self.vftl_host.get_model_parameters()

                print("[DEBUG] guest_nn", guest_nn)
                print("[DEBUG] guest_nn_prime", guest_nn_prime)
                print("[DEBUG] host_nn", host_nn)
                print("[DEBUG] host_nn_prime", host_nn_prime)

        return ob_loss

    def train(self, debug=True):

        ol_batch_size = self.model_param.overlap_sample_batch_size
        ol_block_num = self.vftl_guest.get_ol_block_number()
        print("[INFO] ol_batch_size:", ol_batch_size)
        print("[INFO] ol_block_num:", ol_block_num)

        early_stopping = EarlyStoppingCheckPoint(monitor="fscore", patience=200)
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
                ol_block_idx = 0
                ol_end = 0

                ol_guest_block_size = self.vftl_guest.load_ol_block(ol_block_idx)
                ol_host_block_size = self.vftl_host.load_ol_block(ol_block_idx)
                assert ol_guest_block_size == ol_host_block_size

                iter = 0
                while True:
                    print("[INFO] ===> iter:{0} of ep: {1}".format(iter, i))
                    if ol_end >= ol_guest_block_size:
                        ol_block_idx += 1
                        if ol_block_idx == ol_block_num:
                            break
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

                    loss = self.fit(sess=sess,
                                    overlap_batch_range=(ol_start, ol_end),
                                    debug=debug)
                    loss_list.append(loss)
                    print("")
                    print("[INFO] ep:{0}, ol_block_idx:{1}, ol_batch_idx:{2}, loss:{3}".format(i, ol_block_idx, ol_batch_idx, loss))

                    ol_batch_idx = ol_batch_idx + 1

                    # two sides test
                    all_acc, all_auc, all_fscore = self.two_side_predict(sess, debug=debug)
                    all_acc_list.append(all_acc)
                    all_auc_list.append(all_auc)
                    all_fscore_list.append(all_fscore)

                    print("=" * 40)
                    print("[INFO] ===> fscore, acc, auc ", all_fscore, all_acc, all_auc)
                    print("=" * 40)

                    log = {"fscore": all_acc, "all_fscore": all_fscore, "all_acc": all_acc, "all_auc": all_auc, }
                    early_stopping.on_validation_end(curr_epoch=i, batch_idx=ol_batch_idx, log=log)
                    iter += 1
                #     if self.stop_training is True:
                #         break
                #
                # if self.stop_training is True:
                #     break

        end_time = time.time()
        print("training time (s):", end_time - start_time)
        print("stopped epoch, batch:", early_stopping.stopped_epoch, early_stopping.stopped_batch)
        early_stopping.print_log_of_best_result()
        return early_stopping.get_log_info(), loss_list
