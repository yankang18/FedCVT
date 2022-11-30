import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from debug_utils import analyze_estimated_labels
from fedcvt_core.fedcvt_parties import VFTLGuest, VFLHost
from fedcvt_core.fedcvt_repr_estimator import AttentionBasedRepresentationEstimator, concat_reprs, sharpen
from fedcvt_core.loss import get_shared_reprs_loss, get_orth_reprs_loss, get_alignment_loss, \
    get_label_estimation_loss
from fedcvt_core.param import FederatedModelParam
from models.mlp_models import SoftmaxRegression
from models.regularization import EarlyStoppingCheckPoint
from utils import convert_to_1d_labels, f_score, transpose


class VerticalFederatedTransferLearning(object):

    def __init__(self,
                 vftl_guest: VFTLGuest,
                 vftl_host: VFLHost,
                 fed_model_param: FederatedModelParam,
                 debug=False):
        self.vftl_guest = vftl_guest
        self.vftl_host = vftl_host
        self.fed_model_param = fed_model_param
        self.n_class = self.vftl_guest.get_number_of_class()
        self.repr_estimator = AttentionBasedRepresentationEstimator()
        self.stop_training = False
        self.fed_lr = None
        self.guest_lr = None
        self.host_lr = None
        self.debug = debug
        self.device = fed_model_param.device

        self.criteria = None
        self.fed_optimizer = None
        self.guest_optimizer = None
        self.host_optimizer = None

    def set_representation_estimator(self, repr_estimator):
        self.repr_estimator = repr_estimator
        print("using: {0}".format(self.repr_estimator))

    def determine_overlap_sample_indices(self):
        overlap_indices = self.fed_model_param.overlap_indices
        print("overlap_indices size:", len(overlap_indices))
        return overlap_indices

    def get_hyperparameters(self):
        return self.fed_model_param.get_parameters()

    def save_model(self, file_path):
        print("[INFO] TODO: save model")

    def save_info(self, file_path, eval_info):
        field_names = list(eval_info.keys())
        with open(file_path + ".csv", "a") as logfile:
            logger = csv.DictWriter(logfile, fieldnames=field_names)
            logger.writerow(eval_info)

    # @staticmethod
    # def _create_transform_matrix(in_dim, out_dim):
    #     with tf.compat.v1.variable_scope("transform_matrix"):
    #         Wt = tf.compat.v1.get_variable(name="Wt", initializer=tf.random.normal((in_dim, out_dim), dtype=tf.float32))
    #     return Wt

    def _semi_supervised_training(self):

        using_uniq = self.fed_model_param.using_uniq
        using_comm = self.fed_model_param.using_comm

        Ug_all, Ug_non_overlap_, Ug_ll_overlap_, Ug_ul_overlap_ = self.vftl_guest.fetch_feat_reprs()
        Uh_all, Uh_non_overlap_, Uh_ll_overlap_, Uh_ul_overlap_ = self.vftl_host.fetch_feat_reprs()

        Ug_all_uniq, Ug_all_comm = Ug_all
        Ug_non_overlap_uniq, Ug_non_overlap_comm = Ug_non_overlap_
        Ug_ll_overlap_uniq, Ug_ll_overlap_comm = Ug_ll_overlap_

        # TODO
        if Ug_ul_overlap_ is not None:
            Ug_ul_overlap_uniq, Ug_ul_overlap_comm = Ug_ul_overlap_
            Ug_ol_overlap_uniq = torch.cat([Ug_ll_overlap_uniq, Ug_ul_overlap_uniq], dim=0)
            Ug_ol_overlap_comm = torch.cat([Ug_ll_overlap_comm, Ug_ul_overlap_comm], dim=0)

        Uh_all_uniq, Uh_all_comm = Uh_all
        Uh_non_overlap_uniq, Uh_non_overlap_comm = Uh_non_overlap_
        Uh_ll_overlap_uniq, Uh_ll_overlap_comm = Uh_ll_overlap_

        # TODO
        if Uh_ul_overlap_ is not None:
            Uh_ul_overlap_uniq, Uh_ul_overlap_comm = Uh_ul_overlap_
            Uh_ol_overlap_uniq = torch.cat([Uh_ll_overlap_uniq, Uh_ul_overlap_uniq], dim=0)
            Uh_ol_overlap_comm = torch.cat([Uh_ll_overlap_comm, Uh_ul_overlap_comm], dim=0)

        sharpen_temp = self.fed_model_param.sharpen_temperature
        label_prob_sharpen_temperature = self.fed_model_param.label_prob_sharpen_temperature
        fed_label_prob_threshold = self.fed_model_param.fed_label_prob_threshold
        host_label_prob_threshold = self.fed_model_param.host_label_prob_threshold

        # print(f"[INFO] using_uniq:{using_uniq}")
        # print(f"[INFO] using_comm:{using_comm}")
        # print(f"[INFO] is_hetero_repr:{is_hetero_repr}")
        # print(f"[INFO] sharpen temperature:{sharpen_temp}")
        # print(f"[INFO] label_prob_sharpen_temperature:{label_prob_sharpen_temperature}")
        # print(f"[INFO] fed_label_prob_threshold:{fed_label_prob_threshold}")
        # print(f"[INFO] host_label_prob_threshold:{host_label_prob_threshold}")

        Ug_ll_overlap_reprs = concat_reprs(Ug_ll_overlap_uniq, Ug_ll_overlap_comm, using_uniq, using_comm)
        Uh_ll_overlap_reprs = concat_reprs(Uh_ll_overlap_uniq, Uh_ll_overlap_comm, using_uniq, using_comm)

        Ug_non_overlap_reprs = concat_reprs(Ug_non_overlap_uniq, Ug_non_overlap_comm, using_uniq, using_comm)
        Uh_non_overlap_reprs = concat_reprs(Uh_non_overlap_uniq, Uh_non_overlap_comm, using_uniq, using_comm)

        W_hg = None
        # if self.fed_model_param.is_hetero_repr is True:
        #     host_comm_dim = Uh_all_comm.shape[-1]
        #     guest_comm_dim = Ug_all_comm.shape[-1]
        #     W_hg = self._create_transform_matrix(host_comm_dim, guest_comm_dim)
        #     print("Using transform matrix with shape:", W_hg.shape)

        Y_ll_overlap = self.vftl_guest.get_Y_ll_overlap()
        Y_guest_non_overlap = self.vftl_guest.get_Y_non_overlap()

        Y_ll_overlap_for_estimation = self.vftl_guest.get_Y_ll_overlap_for_est()
        Y_all_for_estimation = self.vftl_guest.get_Y_all_for_est()

        # ===============================================
        # estimate representations for missing features
        # ===============================================

        # estimate feature representations of missing samples of host corresponding
        # to non-overlap samples of guest.
        Uh_non_overlap_ested_reprs = self.repr_estimator.estimate_host_reprs_for_guest_party(
            Ug_non_overlap_comm,
            Ug_non_overlap_uniq,
            Ug_ll_overlap_uniq,
            Uh_ll_overlap_uniq,
            Uh_all_comm,
            sharpen_temperature=sharpen_temp,
            W_gh=transpose(W_hg),
            using_uniq=using_uniq,
            using_comm=using_comm)

        # estimate feature representations and labels of missing samples of guest corresponding
        # to non-overlap samples of host
        Ug_non_overlap_ested_reprs_w_lbls = self.repr_estimator.estimate_labeled_guest_reprs_for_host_party(
            Uh_non_overlap_comm,
            Uh_non_overlap_uniq,
            Uh_ll_overlap_uniq,
            Ug_ll_overlap_uniq,
            Ug_all_comm,
            Y_ll_overlap_for_estimation,
            Y_all_for_estimation,
            sharpen_tempature=sharpen_temp,
            W_hg=W_hg,
            using_uniq=using_uniq,
            using_comm=using_comm)

        Ug_non_overlap_ested_reprs = Ug_non_overlap_ested_reprs_w_lbls[:, :-self.n_class]
        nl_ested_lbls_use_attention = Ug_non_overlap_ested_reprs_w_lbls[:, -self.n_class:]

        # reprs for overlapping samples
        fed_ll_reprs = torch.cat([Ug_ll_overlap_reprs, Uh_ll_overlap_reprs], dim=1)

        # reprs for non-overlapping samples of guest with host's reprs estimated.
        fed_nl_w_host_ested_reprs = torch.cat([Ug_non_overlap_reprs, Uh_non_overlap_ested_reprs], dim=1)

        # reprs for non-overlapping samples of host with guest's reprs estimated.
        fed_nl_w_guest_ested_reprs = torch.cat([Ug_non_overlap_ested_reprs, Uh_non_overlap_reprs], dim=1)

        # ===========================================================================
        # estimate (candidate) labels of the estimated representations (at the guest)
        # corresponding to non-overlap samples of host.
        # ===========================================================================

        # estimate labels using federated feature reprs in which guest's reprs estimated.
        nl_ested_lbls_use_fed_reprs = F.softmax(self.fed_lr(fed_nl_w_guest_ested_reprs), dim=1)

        # estimate labels using estimated feature reprs of missing samples of guest
        nl_ested_lbls_use_ested_guest_reprs = F.softmax(self.guest_lr(Ug_non_overlap_ested_reprs), dim=1)

        # estimate labels using feature reprs of non-overlapping samples of host
        nl_ested_lbls_use_host_reprs = F.softmax(self.host_lr(Uh_non_overlap_reprs), dim=1)

        nl_ested_lbls_use_fed_reprs = sharpen(nl_ested_lbls_use_fed_reprs, temperature=label_prob_sharpen_temperature)
        nl_ested_lbls_use_ested_guest_reprs = sharpen(nl_ested_lbls_use_ested_guest_reprs,
                                                      temperature=label_prob_sharpen_temperature)
        nl_ested_lbls_use_host_reprs = sharpen(nl_ested_lbls_use_host_reprs, temperature=label_prob_sharpen_temperature)

        if self.debug: analyze_estimated_labels(nl_ested_lbls_use_attention,
                                                nl_ested_lbls_use_ested_guest_reprs,
                                                nl_ested_lbls_use_host_reprs,
                                                nl_ested_lbls_use_fed_reprs)

        # ========================================================================================================
        # fed_nl_w_ested_guest_reprs_n_candidate_labels is the concatenation of federated feature
        # representations in which the guest's representations are estimated and three estimated candidate labels.
        # ========================================================================================================

        # fed_nl_w_ested_guest_reprs_n_candidate_labels = torch.cat(
        #     [fed_nl_w_guest_ested_reprs,
        #      nl_ested_lbls_use_host_reprs,
        #      nl_ested_lbls_use_ested_guest_reprs,
        #      nl_ested_lbls_use_fed_reprs], dim=1)

        nl_ested_lbls_use_attention = nl_ested_lbls_use_attention.detach()
        nl_ested_lbls_use_host_reprs = nl_ested_lbls_use_host_reprs.detach()
        nl_ested_lbls_use_ested_guest_reprs = nl_ested_lbls_use_ested_guest_reprs.detach()
        nl_ested_lbls_use_fed_reprs = nl_ested_lbls_use_fed_reprs.detach()

        fed_nl_w_ested_guest_reprs_n_candidate_labels = torch.cat(
            [fed_nl_w_guest_ested_reprs,
             nl_ested_lbls_use_attention,
             nl_ested_lbls_use_host_reprs,
             nl_ested_lbls_use_ested_guest_reprs,
             nl_ested_lbls_use_fed_reprs], dim=1)

        if self.debug:
            print("[DEBUG] Uh_non_overlap_ested_reprs shape", Uh_non_overlap_ested_reprs.shape)
            print("[DEBUG] Ug_non_overlap_ested_reprs_w_lbls shape", Ug_non_overlap_ested_reprs_w_lbls.shape)
            print("[DEBUG] fed_nl_w_guest_ested_reprs shape:", fed_nl_w_guest_ested_reprs.shape)
            print("[DEBUG] fed_nl_w_ested_guest_reprs_n_candidate_labels shape {0}".format(
                fed_nl_w_ested_guest_reprs_n_candidate_labels.shape))

        # ================================================================================================
        # select representations from fed_nl_w_ested_guest_reprs_n_candidate_labels based estimated labels.
        # ================================================================================================
        selected_reprs_w_lbls = self.repr_estimator.select_reprs_for_multiclass(
            reprs_w_candidate_labels=fed_nl_w_ested_guest_reprs_n_candidate_labels,
            n_class=self.n_class,
            fed_label_upper_bound=fed_label_prob_threshold,
            host_label_upper_bound=host_label_prob_threshold,
            debug=self.debug)

        has_selected_samples = True if selected_reprs_w_lbls is not None else False
        print(f"[DEBUG] has_selected_samples:{has_selected_samples}")

        # =========================================
        # prepare the expanded training set for VFL
        # =========================================
        if has_selected_samples:
            # if have selected representations
            print(f"[DEBUG] num_selected_samples shape:{selected_reprs_w_lbls.shape}")

            sel_fed_nl_w_guest_ested_reprs = selected_reprs_w_lbls[:, :-self.n_class]
            sel_labels = selected_reprs_w_lbls[:, -self.n_class:]

            fed_reprs = torch.cat([fed_ll_reprs, fed_nl_w_host_ested_reprs, sel_fed_nl_w_guest_ested_reprs], dim=0)

            # g_ll_reprs = torch.cat(Ug_ll_overlap_reprs, dim=1)
            # g_nl_reprs = torch.cat(Ug_non_overlap_reprs, dim=1)
            # g_nl_estd_reprs = sel_fed_nl_w_guest_ested_reprs[:, :g_reprs_dim]

            g_ll_reprs = Ug_ll_overlap_reprs
            g_nl_reprs = Ug_non_overlap_reprs
            g_reprs = torch.cat((g_ll_reprs, g_nl_reprs), dim=0)
            # g_reprs = tf.concat((g_ll_reprs, g_nl_reprs, g_nl_estd_reprs), axis=0)

            h_ll_reprs = Uh_ll_overlap_reprs
            # h_nl_estd_reprs = tf.concat(Uh_non_overlap_ested_reprs, axis=1)
            # h_nl = sel_fed_nl_w_guest_ested_reprs[:, g_reprs_dim:]
            # h_reprs = tf.concat((h_ll_reprs, h_nl_estd_reprs, h_nl), axis=0)

            guest_y = torch.cat([Y_ll_overlap, Y_guest_non_overlap], dim=0)
            y = torch.cat([Y_ll_overlap, Y_guest_non_overlap, sel_labels], dim=0)

            train_fed_reprs, train_guest_reprs, train_host_reprs, train_fed_y, train_g_y, train_h_y = \
                fed_reprs, g_reprs, h_ll_reprs, y, guest_y, Y_ll_overlap
            # return fed_reprs, g_reprs, h_ll_reprs, y, guest_y, Y_ll_overlap
            # return fed_reprs, g_reprs, h_reprs, y, y, y

        else:
            # if have no selected representations

            # 1
            # fed_reprs = fed_ll_reprs
            # g_ll_reprs = tf.concat(Ug_overlap, axis=1)
            # g_reprs = g_ll_reprs
            #
            # h_ll_reprs = tf.concat(Uh_overlap, axis=1)

            # 2 use representation estimation
            fed_reprs = torch.cat([fed_ll_reprs, fed_nl_w_host_ested_reprs], dim=0)

            g_ll_reprs = Ug_ll_overlap_reprs
            g_nl_reprs = Ug_non_overlap_reprs
            g_reprs = torch.cat((g_ll_reprs, g_nl_reprs), dim=0)

            h_ll_reprs = Uh_ll_overlap_reprs

            y = torch.cat([Y_ll_overlap, Y_guest_non_overlap], dim=0)
            train_fed_reprs, train_guest_reprs, train_host_reprs, train_fed_y, train_g_y, train_h_y = \
                fed_reprs, g_reprs, h_ll_reprs, y, y, Y_ll_overlap

            # 3 do not use representation estimation
            # g_ll_reprs = tf.concat(Ug_ll_overlap_reprs, axis=1)
            # h_ll_reprs = tf.concat(Uh_ll_overlap_reprs, axis=1)
            # return fed_ll_reprs, g_ll_reprs, h_ll_reprs, Y_ll_overlap, Y_ll_overlap, Y_ll_overlap

        # train_fed_reprs, train_guest_reprs, train_host_reprs, train_fed_y, train_g_y, train_h_y = tf.cond(
        #     pred=has_selected_samples,
        #     true_fn=f1,
        #     false_fn=f2)
        #####################
        if self.debug:
            training_fed_reprs_shape = train_fed_reprs.shape
            training_guest_reprs_shape = train_guest_reprs.shape
            print(f"[DEBUG] training_fed_reprs_shape:{training_fed_reprs_shape}")
            print(f"[DEBUG] training_guest_reprs_shape:{training_guest_reprs_shape}")

        guest_nl_reprs = Ug_non_overlap_reprs
        repr_list = [fed_ll_reprs, fed_nl_w_host_ested_reprs, fed_nl_w_guest_ested_reprs, guest_nl_reprs]

        # print("train_fed_reprs shape", train_fed_reprs.shape)
        # print("training_guest_reprs_shape shape", self.training_guest_reprs_shape.shape)
        # print("Y_ll_overlap shape", Y_ll_overlap, Y_ll_overlap.shape)
        # print("Y_guest_non_overlap shape", Y_guest_non_overlap, Y_guest_non_overlap.shape)
        # print("Y_host_non_overlap shape", Y_host_non_overlap, Y_host_non_overlap.shape)
        # print("Y_host_non_overlap shape", selected_host_lbls, selected_host_lbls.shape)

        # =========================
        # compute auxiliary losses
        # =========================

        # estimate labels for unique representations and shared representation of host, respectively.
        self.uniq_lbls, self.comm_lbls = self.repr_estimator.estimate_unique_comm_labels_for_host_party(
            Uh_comm=Uh_all_comm,
            Uh_uniq=Uh_all_uniq,
            Uh_overlap_uniq=Uh_ll_overlap_uniq,
            Ug_all_comm=Ug_all_comm,
            Yg_overlap=Y_ll_overlap_for_estimation,
            Yg_all=Y_all_for_estimation,
            sharpen_tempature=sharpen_temp,
            W_hg=W_hg,
            using_uniq=using_uniq,
            using_comm=using_comm)

        # estimate overlap feature representations on host side for guest party for minimizing alignment loss
        Uh_overlap_ested_reprs = self.repr_estimator.estimate_host_reprs_for_guest_party(
            Ug_ll_overlap_comm,
            Ug_ll_overlap_uniq,
            Ug_ll_overlap_uniq,
            Uh_ll_overlap_uniq,
            Uh_all_comm,
            sharpen_temperature=sharpen_temp,
            W_gh=transpose(W_hg),
            using_uniq=using_uniq,
            using_comm=using_comm)

        # estimate overlap feature representations and labels on guest side for host party for minimizing alignment loss
        result_overlap = self.repr_estimator.estimate_guest_reprs_n_lbls_for_host_party(Uh_uniq=Uh_ll_overlap_uniq,
                                                                                        Uh_overlap_uniq=Uh_ll_overlap_uniq,
                                                                                        Uh_comm=Uh_ll_overlap_comm,
                                                                                        Ug_overlap_uniq=Ug_ll_overlap_uniq,
                                                                                        Ug_all_comm=Ug_all_comm,
                                                                                        Yg_overlap=Y_ll_overlap_for_estimation,
                                                                                        Yg_all=Y_all_for_estimation,
                                                                                        sharpen_tempature=sharpen_temp,
                                                                                        W_hg=W_hg,
                                                                                        using_uniq=using_uniq,
                                                                                        using_comm=using_comm)
        Ug_overlap_ested_reprs, _, ol_uniq_lbls, ol_comm_lbls = result_overlap

        # estimate non-overlap labels on guest side for host party for testing purpose
        # estimate_labeled_guest_reprs_for_host_party
        # result_non_overlap = self.repr_estimator.estimate_guest_reprs_n_lbls_for_host_party(Uh_uniq=Uh_non_overlap_uniq,
        #                                                                                     Uh_ll_overlap_uniq=Uh_ll_overlap_uniq,
        #                                                                                     Uh_comm=Uh_non_overlap_comm,
        #                                                                                     Ug_ll_overlap_uniq=Ug_ll_overlap_uniq,
        #                                                                                     Ug_all_comm=Ug_all_comm,
        #                                                                                     Yg_overlap=Y_ll_overlap_for_estimation,
        #                                                                                     Yg_all=Y_all_for_estimation,
        #                                                                                     sharpen_tempature=sharpen_temp,
        #                                                                                     W_hg=W_hg,
        #                                                                                     using_uniq=using_uniq,
        #                                                                                     using_comm=using_comm)
        # _, self.Uh_non_overlap_ested_soft_lbls, _, _ = result_non_overlap

        assistant_loss_dict = dict()
        add_assistant_loss = True
        if add_assistant_loss:
            # (1) loss for minimizing distance between shared reprs between host and guest.
            if using_comm:
                assistant_loss_dict['lambda_dist_shared_reprs'] = get_shared_reprs_loss(Ug_ll_overlap_comm,
                                                                                        Uh_ll_overlap_comm)

            # (2) (3) loss for maximizing distance between orthogonal reprs for host and guest, respectively.
            if using_uniq and using_comm:
                guest_uniq_reprs_loss = get_orth_reprs_loss(Ug_all_uniq, Ug_all_comm)
                assistant_loss_dict['lambda_guest_sim_shared_reprs_vs_unique_repr'] = guest_uniq_reprs_loss

            if using_uniq and using_comm:
                host_uniq_reprs_loss = get_orth_reprs_loss(Uh_all_uniq, Uh_all_comm)
                assistant_loss_dict['lambda_host_sim_shared_reprs_vs_unique_repr'] = host_uniq_reprs_loss

            # (4) (5) loss for minimizing distance between estimated host overlap labels and true labels
            if using_uniq:
                Ug_ol_uniq_lbl_loss = get_label_estimation_loss(ol_uniq_lbls, Y_ll_overlap)
                assistant_loss_dict['lambda_host_dist_ested_uniq_lbl_vs_true_lbl'] = Ug_ol_uniq_lbl_loss

            if using_comm:
                Ug_ol_comm_lbl_loss = get_label_estimation_loss(ol_comm_lbls, Y_ll_overlap)
                assistant_loss_dict['lambda_host_dist_ested_comm_lbl_vs_true_lbl'] = Ug_ol_comm_lbl_loss

            # (6) loss for minimizing distance between estimated guest overlap reprs and true guest reprs
            Ug_overlap_ested_reprs_alignment_loss = get_alignment_loss(Ug_overlap_ested_reprs, Ug_ll_overlap_reprs)
            assistant_loss_dict['lambda_guest_dist_ested_repr_vs_true_repr'] = Ug_overlap_ested_reprs_alignment_loss

            # (7) loss for minimizing distance between estimated host overlap reprs and true host reprs
            Uh_overlap_ested_reprs_alignment_loss = get_alignment_loss(Uh_overlap_ested_reprs, Uh_ll_overlap_reprs)
            assistant_loss_dict['lambda_host_dist_ested_repr_vs_true_repr'] = Uh_overlap_ested_reprs_alignment_loss

            # (8) loss for distance between shared-repr-estimated host label and uniq-repr-estimated host label
            # if using_uniq and using_comm:
            #     label_alignment_loss = get_alignment_loss(self.uniq_lbls, self.comm_lbls)
            #     assistant_loss_dict['lambda_host_dist_two_ested_lbl'] = label_alignment_loss

        # print("## assistant_loss_dict:", len(assistant_loss_dict), assistant_loss_dict)
        train_component_list = [train_fed_reprs, train_guest_reprs, train_host_reprs,
                                train_fed_y, train_g_y, train_h_y,
                                assistant_loss_dict]
        return train_component_list, repr_list

    def build(self):
        learning_rate = self.fed_model_param.learning_rate
        weight_decay = self.fed_model_param.weight_decay
        fed_input_dim = self.fed_model_param.fed_input_dim
        fed_hidden_dim = self.fed_model_param.fed_hidden_dim
        guest_input_dim = self.fed_model_param.guest_input_dim
        host_input_dim = self.fed_model_param.host_input_dim
        guest_hidden_dim = self.fed_model_param.guest_hidden_dim
        host_hidden_dim = self.fed_model_param.host_hidden_dim
        fed_reg_lambda = self.fed_model_param.fed_reg_lambda
        guest_reg_lambda = self.fed_model_param.guest_reg_lamba
        loss_weight_dict = self.fed_model_param.loss_weight_dict
        sharpen_temp = self.fed_model_param.sharpen_temperature
        is_hetero_repr = self.fed_model_param.is_hetero_repr

        print("[INFO] #===> Build Federated Model.")

        print("[INFO] # ================ Hyperparameter Info ================")
        print("[INFO] learning_rate: {0}".format(learning_rate))
        print("[INFO] weight_decay: {0}".format(weight_decay))
        print("[INFO] fed_input_dim: {0}".format(fed_input_dim))
        print("[INFO] fed_hidden_dim: {0}".format(fed_hidden_dim))
        print("[INFO] guest_input_dim: {0}".format(guest_input_dim))
        print("[INFO] guest_hidden_dim: {0}".format(guest_hidden_dim))
        print("[INFO] fed_reg_lambda: {0}".format(fed_reg_lambda))
        print("[INFO] guest_reg_lambda: {0}".format(guest_reg_lambda))
        print("[INFO] sharpen_temp: {0}".format(sharpen_temp))
        print("[INFO] is_hetero_repr: {0}".format(is_hetero_repr))
        for key, val in loss_weight_dict.items():
            print("[INFO] {0}: {1}".format(key, val))
        print("[INFO] ========================================================")

        fed_hidden_dim = None
        guest_hidden_dim = None
        host_hidden_dim = None

        self.fed_lr = SoftmaxRegression(1).to(self.device)
        self.fed_lr.build(input_dim=fed_input_dim, output_dim=self.n_class, hidden_dim=fed_hidden_dim)

        self.guest_lr = SoftmaxRegression(2).to(self.device)
        self.guest_lr.build(input_dim=guest_input_dim, output_dim=self.n_class, hidden_dim=guest_hidden_dim)

        self.host_lr = SoftmaxRegression(3).to(self.device)
        self.host_lr.build(input_dim=host_input_dim, output_dim=self.n_class, hidden_dim=host_hidden_dim)

        print("[INFO] fed top model:")
        print(self.fed_lr)
        print("[INFO] guest top model:")
        print(self.guest_lr)
        print("[INFO] host top model:")
        print(self.host_lr)

        self.criteria = torch.nn.CrossEntropyLoss()

        guest_model_params = list(self.guest_lr.parameters()) + self.vftl_guest.get_model_parameters()
        self.guest_optimizer = torch.optim.Adam(params=guest_model_params, lr=learning_rate, weight_decay=weight_decay)

        host_model_params = list(self.host_lr.parameters()) + self.vftl_host.get_model_parameters()
        self.host_optimizer = torch.optim.Adam(params=host_model_params, lr=learning_rate, weight_decay=weight_decay)

        fed_model_params = list(
            self.fed_lr.parameters()) + self.vftl_host.get_model_parameters() + self.vftl_guest.get_model_parameters()
        self.fed_optimizer = torch.optim.Adam(params=fed_model_params, lr=learning_rate, weight_decay=weight_decay)

    def two_side_predict(self):
        # if debug:
        print("[INFO] ------> two sides predict")

        # pred_feed_dict = self.vftl_guest.get_two_sides_predict_feed_dict()
        # pred_host_feed_dict = self.vftl_host.get_two_sides_predict_feed_dict()
        # pred_feed_dict.update(pred_host_feed_dict)
        # y_prob_two_sides = sess.run(self.fed_lr.y_hat_two_side, feed_dict=pred_feed_dict)
        using_uniq = self.fed_model_param.using_uniq
        using_comm = self.fed_model_param.using_comm

        Ug_overlap_uniq, Ug_overlap_comm = self.vftl_guest.predict_on_overlap_data()
        Uh_overlap_uniq, Uh_overlap_comm = self.vftl_host.predict_on_overlap_data()

        Ug_overlap_reprs = concat_reprs(Ug_overlap_uniq, Ug_overlap_comm, using_uniq, using_comm)
        Uh_overlap_reprs = concat_reprs(Uh_overlap_uniq, Uh_overlap_comm, using_uniq, using_comm)

        fed_ol_reprs = torch.cat([Ug_overlap_reprs, Uh_overlap_reprs], dim=1)

        y_prob_two_sides = F.softmax(self.fed_lr(fed_ol_reprs), dim=1)
        y_prob_two_sides = y_prob_two_sides.detach().numpy()

        y_test = self.vftl_guest.get_Y_test()
        y_hat_1d = convert_to_1d_labels(y_prob_two_sides)
        y_test_1d = convert_to_1d_labels(y_test)

        # debug = True
        if self.debug:
            print("[DEBUG] y_prob_two_sides shape {0}".format(y_prob_two_sides.shape))
            print("[DEBUG] y_prob_two_sides {0}".format(y_prob_two_sides))
            print("[DEBUG] y_test shape {0}:".format(y_test.shape))
            print("[DEBUG] y_hat_1d shape {0}:".format(y_hat_1d.shape))
            print("[DEBUG] y_test_1d shape {0}:".format(y_test_1d.shape))

        # print(y_test_1d, y_test_1d.shape)
        # print(y_hat_1d, y_hat_1d.shape)
        # for y_h, y_t in zip(y_hat_1d, y_test_1d):
        #     print(y_h, y_t)

        res = precision_recall_fscore_support(y_test_1d, y_hat_1d)
        all_fscore = f_score(res[0], res[1])
        acc = accuracy_score(y_test_1d, y_hat_1d)
        auc = roc_auc_score(y_test, y_prob_two_sides)

        # print("[INFO] all_res:", res)
        # print("[INFO] all_fscore : {0}, all_auc : {1}, all_acc : {2}".format(all_fscore, auc, acc))
        print("[INFO] all_auc : {}, all_acc : {}".format(auc, acc))

        return acc, auc, all_fscore

    # def guest_side_predict(self,
    #                        sess,
    #                        host_all_training_size,
    #                        block_size,
    #                        estimation_block_num=None,
    #                        round=3,
    #                        debug=True):
    #     # if debug:
    #     print("[INFO] ------> guest side predict")
    #
    #     y_test = self.vftl_guest.get_Y_test()
    #     y_test_1d = convert_to_1d_labels(y_test)
    #     fed_fscore_list = []
    #     fed_acc_list = []
    #     fed_auc_list = []
    #     guest_fscore_list = []
    #     guest_acc_list = []
    #     guest_auc_list = []
    #     for r in range(round):
    #
    #         if debug:
    #             print("[DEBUG] round:", r)
    #
    #         if host_all_training_size is None:
    #             host_block_idx = np.random.choice(estimation_block_num - 1, 1)[0]
    #             print("host_block_idx: {0}".format(host_block_idx))
    #             host_assistant_pred_feed_dict = self.vftl_host.get_assist_guest_predict_feed_dict(
    #                 block_idx=host_block_idx)
    #         else:
    #             host_block_indices = np.random.choice(host_all_training_size, block_size)
    #             host_assistant_pred_feed_dict = self.vftl_host.get_assist_guest_predict_feed_dict(
    #                 block_indices=host_block_indices)
    #
    #         guest_side_pred_feed_dict = self.vftl_guest.get_one_side_predict_feed_dict()
    #         guest_side_pred_feed_dict.update(host_assistant_pred_feed_dict)
    #
    #         y_prob_fed_guest, y_prob_guest_side = sess.run(
    #             [self.fed_lr.y_hat_guest_side, self.guest_lr.y_hat_guest_side], feed_dict=guest_side_pred_feed_dict)
    #         y_hat_1d_fed_guest = convert_to_1d_labels(y_prob_fed_guest)
    #         y_hat_1d_only_guest = convert_to_1d_labels(y_prob_guest_side)
    #         # y_test_1d = self.convert_to_1d_labels(y_test)
    #
    #         fed_guest_res = precision_recall_fscore_support(y_test_1d, y_hat_1d_fed_guest, average='weighted')
    #         fed_fscore = f_score(fed_guest_res[0], fed_guest_res[1])
    #         fed_guest_acc = accuracy_score(y_test_1d, y_hat_1d_fed_guest)
    #         fed_guest_auc = roc_auc_score(y_test, y_prob_fed_guest)
    #
    #         only_guest_res = precision_recall_fscore_support(y_test_1d, y_hat_1d_only_guest, average='weighted')
    #         only_guest_fscore = f_score(only_guest_res[0], only_guest_res[1])
    #         only_guest_acc = accuracy_score(y_test_1d, y_hat_1d_only_guest)
    #         only_guest_auc = roc_auc_score(y_test, y_prob_guest_side)
    #
    #         debug = True
    #         if debug:
    #             print("[DEBUG] y_hat_1d_fed_guest shape:", y_hat_1d_fed_guest.shape)
    #             print("[DEBUG] y_hat_1d_only_guest shape:", y_hat_1d_only_guest.shape)
    #
    #         print("[INFO] fed_guest_res:", fed_guest_res)
    #         print("[INFO] fed_fscore : {0}, fed_guest_auc : {1}, fed_guest_acc : {2}".format(fed_fscore, fed_guest_auc,
    #                                                                                          fed_guest_acc))
    #         print("[INFO] only_guest_res:", only_guest_res)
    #         print("[INFO] only_guest_fscore : {0}, only_guest_auc : {1}, only_guest_acc : {2}".format(only_guest_fscore,
    #                                                                                                   only_guest_auc,
    #                                                                                                   only_guest_acc))
    #
    #         fed_fscore_list.append(fed_fscore)
    #         fed_acc_list.append(fed_guest_acc)
    #         fed_auc_list.append(fed_guest_auc)
    #
    #         guest_fscore_list.append(only_guest_fscore)
    #         guest_acc_list.append(only_guest_acc)
    #         guest_auc_list.append(only_guest_auc)
    #
    #     return (fed_fscore_list, fed_acc_list, fed_auc_list), (guest_fscore_list, guest_acc_list, guest_auc_list)

    # def host_side_predict(self,
    #                       sess,
    #                       guest_all_training_size,
    #                       block_size,
    #                       estimation_block_num=None,
    #                       round=3,
    #                       debug=True):
    #     # if debug:
    #     print("[INFO] ------> host side predict")
    #
    #     y_test = self.vftl_guest.get_Y_test()
    #
    #     # y_test = y_test[:10]
    #     y_test_1d = convert_to_1d_labels(y_test)
    #     fscore_list = []
    #     acc_list = []
    #     auc_list = []
    #     for r in range(round):
    #         if debug:
    #             print("[DEBUG] round:", r)
    #
    #         if guest_all_training_size is None:
    #             guest_block_idx = np.random.choice(estimation_block_num - 1, 1)[0]
    #             print("guest_block_idx: {0}".format(guest_block_idx))
    #             guest_for_host_predict_along_feed_dict = self.vftl_guest.get_assist_host_distance_based_predict_feed_dict(
    #                 block_idx=guest_block_idx)
    #             guest_assistant_pred_feed_dict = self.vftl_guest.get_assist_host_side_predict_feed_dict(
    #                 block_idx=guest_block_idx)
    #         else:
    #             print("guest_all_training_size:", guest_all_training_size)
    #             guest_block_indices = np.random.choice(guest_all_training_size, block_size)
    #             guest_for_host_predict_along_feed_dict = self.vftl_guest.get_assist_host_distance_based_predict_feed_dict(
    #                 block_indices=guest_block_indices)
    #             guest_assistant_pred_feed_dict = self.vftl_guest.get_assist_host_side_predict_feed_dict(
    #                 block_indices=guest_block_indices)
    #
    #         # predict host labels by federated model
    #         host_alone_predict_feed_dict = self.vftl_host.get_one_side_distance_based_predict_feed_dict()
    #         host_alone_predict_feed_dict.update(guest_for_host_predict_along_feed_dict)
    #         att_y_prob_host_side = sess.run(self.Uh_non_overlap_ested_soft_lbls, feed_dict=host_alone_predict_feed_dict)
    #
    #         # predict host labels by attention-based estimation
    #         host_side_pred_feed_dict = self.vftl_host.get_one_side_predict_feed_dict()
    #         host_side_pred_feed_dict.update(guest_assistant_pred_feed_dict)
    #
    #         fl_y_prob_host_side, fl_y_reprs = sess.run([self.fed_lr.y_hat_host_side, self.fed_lr.reprs],
    #                                                    feed_dict=host_side_pred_feed_dict)
    #         # aggregation of federated model and attention-based estimation
    #         agg_y_prob_host_side = 0.5 * fl_y_prob_host_side + 0.5 * att_y_prob_host_side
    #         # comb_y_prob_host_side = 1.0 * fl_y_prob_host_side + 0.0 * att_y_hat_host_side_1d
    #
    #         # host labels predicted by federated predictive models
    #         fl_y_hat_host_side_1d = convert_to_1d_labels(fl_y_prob_host_side)
    #         # host labels predicted by attention based estimation
    #         att_y_hat_host_side_1d = convert_to_1d_labels(att_y_prob_host_side)
    #         # aggregated host labels
    #         agg_y_hat_host_side_1d = convert_to_1d_labels(agg_y_prob_host_side)
    #
    #         fl_y_host_res = precision_recall_fscore_support(y_test_1d, fl_y_hat_host_side_1d, average='weighted')
    #         att_y_host_res = precision_recall_fscore_support(y_test_1d, att_y_hat_host_side_1d, average='weighted')
    #         agg_y_host_res = precision_recall_fscore_support(y_test_1d, agg_y_hat_host_side_1d, average='weighted')
    #
    #         fl_y_fscore = f_score(fl_y_host_res[0], fl_y_host_res[1])
    #         att_y_fscore = f_score(att_y_host_res[0], att_y_host_res[1])
    #         agg_y_fscore = f_score(agg_y_host_res[0], agg_y_host_res[1])
    #
    #         fl_host_acc = accuracy_score(y_test_1d, fl_y_hat_host_side_1d)
    #         fl_host_auc = roc_auc_score(y_test, fl_y_prob_host_side)
    #         # fl_host_auc = 1.0
    #
    #         debug = True
    #         if debug:
    #             print("[DEBUG] fl_y_hat_host_side_1d shape:", fl_y_hat_host_side_1d.shape)
    #
    #         print("[INFO] fl_y_host_res:", fl_y_host_res)
    #         print("[INFO] att_y_host_res:", att_y_host_res)
    #         print("[INFO] agg_y_host_res:", agg_y_host_res)
    #         print("[INFO] fl_y_fscore : {0}, att_y_fscore : {1}, agg_y_fscore : {2}".format(fl_y_fscore,
    #                                                                                         att_y_fscore,
    #                                                                                         agg_y_fscore))
    #         print("[INFO] fl_host_auc : {0}, fl_host_acc : {1}".format(fl_host_auc, fl_host_acc))
    #
    #         fscore_list.append(fl_y_fscore)
    #         acc_list.append(fl_host_acc)
    #         auc_list.append(fl_host_auc)
    #     return fscore_list, acc_list, auc_list

    def to_train_mode(self):
        self.fed_lr.train()
        self.host_lr.train()
        self.guest_lr.train()
        self.vftl_host.to_train_mode()
        self.vftl_guest.to_train_mode()

    def to_eval_mode(self):
        self.fed_lr.eval()
        self.host_lr.eval()
        self.guest_lr.eval()
        self.vftl_host.to_eval_mode()
        self.vftl_guest.to_eval_mode()

    def _train(self,
               ul_overlap_batch_range,
               ll_overlap_batch_range,
               guest_non_overlap_batch_range,
               host_non_overlap_batch_range,
               guest_block_idx=None,
               host_block_idx=None,
               guest_block_indices=None,
               host_block_indices=None):

        self.to_train_mode()
        self.guest_optimizer.zero_grad()
        self.host_optimizer.zero_grad()
        self.fed_optimizer.zero_grad()

        self.vftl_guest.prepare_local_data(ul_overlap_batch_range=ul_overlap_batch_range,
                                           ll_overlap_batch_range=ll_overlap_batch_range,
                                           non_overlap_batch_range=guest_non_overlap_batch_range,
                                           block_indices=guest_block_indices,
                                           block_idx=guest_block_idx)
        self.vftl_host.prepare_local_data(ul_overlap_batch_range=ul_overlap_batch_range,
                                          ll_overlap_batch_range=ll_overlap_batch_range,
                                          non_overlap_batch_range=host_non_overlap_batch_range,
                                          block_indices=host_block_indices,
                                          block_idx=host_block_idx)

        # ===================================================================================================
        # weights for auxiliary losses, which include:
        # (1) loss for shared representations between host and guest
        # (2) (3) loss for orthogonal representation for host and guest respectively
        # (4) loss for distance between estimated host overlap labels and true overlap labels
        # (5) loss for distance between estimated guest overlap representation and true guest representation
        # (6) loss for distance between estimated host overlap representation and true host representation
        # (7) loss for distance between shared-repr-estimated host label and uniq-repr-estimated host label
        # ===================================================================================================

        train_components, repr_list = self._semi_supervised_training()
        self.fed_reprs, guest_reprs, host_reprs, self.fed_Y, guest_Y, host_Y, self.loss_dict = train_components
        fed_overlap_reprs, fed_nl_w_host_ested_reprs, fed_nl_w_guest_ested_reprs, guest_nl_reprs = repr_list

        guest_logits = self.guest_lr.forward(guest_reprs)
        guest_loss = self.criteria(guest_logits, torch.argmax(guest_Y, dim=1))

        host_logits = self.host_lr.forward(host_reprs)
        host_loss = self.criteria(host_logits, torch.argmax(host_Y, dim=1))

        fed_logits = self.fed_lr.forward(self.fed_reprs)
        fed_objective_loss = self.criteria(fed_logits, torch.argmax(self.fed_Y, dim=1))

        loss_weight_dict = self.fed_model_param.loss_weight_dict
        if self.loss_dict is not None and loss_weight_dict is not None:
            if self.debug: print("[DEBUG] append loss factors:")
            for key, loss_fac in self.loss_dict.items():
                loss_fac_weight = loss_weight_dict[key]
                if self.debug: print(f"[DEBUG] append loss factor: {key}, [{loss_fac_weight}], {loss_fac}")
                fed_objective_loss = fed_objective_loss + loss_fac_weight * loss_fac

        all_loss = guest_loss + host_loss + fed_objective_loss

        all_loss.backward()
        # guest_loss.backward()
        # host_loss.backward()
        # fed_objective_loss.backward()

        self.guest_optimizer.step()
        self.host_optimizer.step()
        self.fed_optimizer.step()

        # debug_detail = False
        # if True:
        #     print("[DEBUG] regularization_loss", regularization_loss)
        #     print("[DEBUG] mean_pred_loss", mean_pred_loss)
        #     print("[DEBUG] total ob_loss", ob_loss)
        #
        #     if debug_detail:
        #         guest_nn, guest_nn_prime = self.vftl_guest.get_model_parameters()
        #         host_nn, host_nn_prime = self.vftl_host.get_model_parameters()
        #
        #         print("[DEBUG] guest_nn", guest_nn)
        #         print("[DEBUG] guest_nn_prime", guest_nn_prime)
        #         print("[DEBUG] host_nn", host_nn)
        #         print("[DEBUG] host_nn_prime", host_nn_prime)
        loss_dict = {"fed_loss": fed_objective_loss, "guest_loss": guest_loss, "host_loss": host_loss}
        return loss_dict

    def train(self):

        using_block_idx = self.fed_model_param.using_block_idx
        nol_batch_size = self.fed_model_param.non_overlap_sample_batch_size
        ll_batch_size = self.fed_model_param.labeled_overlap_sample_batch_size
        ul_batch_size = self.fed_model_param.unlabeled_overlap_sample_batch_size
        training_info_file_name = self.fed_model_param.training_info_file_name
        valid_iteration_interval = self.fed_model_param.valid_iteration_interval

        with open(training_info_file_name + '.json', 'w') as outfile:
            json.dump(self.fed_model_param.get_parameters(), outfile)

        nol_guest_batch_size = nol_batch_size
        nol_host_batch_size = nol_batch_size

        print("[INFO] using_block_idx:", using_block_idx)
        print("[INFO] ll_batch_size:", ll_batch_size)
        print("[INFO] ul_batch_size:", ul_batch_size)
        print("[INFO] nol_guest_batch_size:", nol_guest_batch_size)
        print("[INFO] nol_host_batch_size:", nol_host_batch_size)

        # guest and host should have the same overlap block number
        ll_block_num = self.vftl_guest.get_ll_block_number()
        ul_block_num = self.vftl_guest.get_ul_block_number()
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

        early_stopping = EarlyStoppingCheckPoint(monitor="fscore", epoch_patience=5, file_path=training_info_file_name)
        early_stopping.set_model(self)
        early_stopping.on_train_begin()

        with open(training_info_file_name + ".csv", "a", newline='') as logfile:
            logger = csv.DictWriter(logfile, fieldnames=list(early_stopping.get_log_info().keys()))
            logger.writeheader()

        # load validation data
        guest_val_block_size = self.vftl_guest.load_val_block(0)
        host_val_block_size = self.vftl_host.load_val_block(0)
        print("[INFO] guest_val_block_size:", guest_val_block_size)
        print("[INFO] host_val_block_size:", host_val_block_size)

        start_time = time.time()
        epoch = self.fed_model_param.epoch

        loss_list = list()
        all_auc_list = list()
        all_acc_list = list()
        all_fscore_list = list()
        for i in range(epoch):
            print("[INFO] ===> start epoch:{0}".format(i))

            ll_batch_idx = 0
            ul_batch_idx = 0
            nol_guest_batch_idx = 0
            nol_host_batch_idx = 0

            ll_block_idx = 0
            ul_block_idx = 0
            nol_guest_block_idx = 0
            nol_host_block_idx = 0

            ested_guest_block_idx = -1
            ested_host_block_idx = -1

            ll_end = 0
            ul_end = 0
            nol_guest_end = 0
            nol_host_end = 0

            ll_guest_block_size = self.vftl_guest.load_ll_block(ll_block_idx)
            ll_host_block_size = self.vftl_host.load_ll_block(ll_block_idx)
            assert ll_guest_block_size == ll_host_block_size

            ul_guest_block_size = self.vftl_guest.load_ul_block(ul_block_idx)
            ul_host_block_size = self.vftl_host.load_ul_block(ul_block_idx)
            assert ul_guest_block_size == ul_host_block_size

            nol_guest_block_size = self.vftl_guest.load_nol_block(nol_guest_block_idx)
            nol_host_block_size = self.vftl_host.load_nol_block(nol_host_block_idx)

            iter = 0
            while True:

                # ===================================================
                # iterate batch index for labeled overlapping samples
                # ===================================================
                # print("[INFO] => iter:{0} of ep: {1}".format(iter, i))
                if ll_end >= ll_guest_block_size:
                    ll_block_idx += 1
                    if ll_block_idx == ll_block_num:
                        # if all blocks for labeled overlapping samples have been visited,
                        # start over from the first block
                        ll_block_idx = 0
                    ll_guest_block_size = self.vftl_guest.load_ll_block(ll_block_idx)
                    ll_host_block_size = self.vftl_host.load_ll_block(ll_block_idx)
                    assert ll_guest_block_size == ll_host_block_size
                    ll_batch_idx = 0

                ll_start = ll_batch_size * ll_batch_idx
                ll_end = ll_batch_size * ll_batch_idx + ll_batch_size

                if self.debug:
                    print("[DEBUG] ll_block_idx:", ll_block_idx)
                    print("[DEBUG] ll_guest_block_size:", ll_guest_block_size)
                    print("[DEBUG] ll_host_block_size:", ll_host_block_size)
                    print("[DEBUG] ll batch from {0} to {1} ".format(ll_start, ll_end))
                    print(f"[DEBUG] ll_block_count/ll_block_num : {ll_block_idx + 1}/{ll_block_num}")
                    print(f"[DEBUG] ll_train_data/ll_guest(host)_block_size : {ll_end}/{ll_guest_block_size}")

                # ===================================================
                # iterate batch index for unlabeled overlapping samples
                # ===================================================
                if ul_guest_block_size == 0:
                    ul_start = 0
                    ul_end = 0

                    if self.debug:
                        print("[DEBUG] Unlabeled overlapping samples are NOT used.")
                else:
                    if ul_end >= ul_guest_block_size:
                        ul_block_idx += 1
                        if ul_block_idx == ul_block_num:
                            # if all blocks for unlabeled overlapping samples have been visited,
                            # start over from the first block
                            ul_block_idx = 0
                        ul_guest_block_size = self.vftl_guest.load_ul_block(ul_block_idx)
                        ul_host_block_size = self.vftl_host.load_ul_block(ul_block_idx)
                        assert ul_guest_block_size == ul_host_block_size
                        ul_batch_idx = 0

                    ul_start = ul_batch_size * ul_batch_idx
                    ul_end = ul_batch_size * ul_batch_idx + ul_batch_size

                    if self.debug:
                        print("[DEBUG] ul_block_idx:", ul_block_idx)
                        print("[DEBUG] ul_guest_block_size:", ul_guest_block_size)
                        print("[DEBUG] ul_host_block_size:", ul_host_block_size)
                        print("[DEBUG] ul batch from {0} to {1} ".format(ul_start, ul_end))
                        print(f"[DEBUG] ul_block_count/ul_block_num : {ul_block_idx + 1}/{ul_block_num}")
                        print(f"[DEBUG] ul_train_data/ul_guest(host)_block_size : {ul_end}/{ul_guest_block_size}")

                # ========================================================
                # iterate batch index for non-overlapping samples of guest
                # ========================================================
                if nol_guest_end >= nol_guest_block_size:
                    nol_guest_block_idx += 1
                    if nol_guest_block_idx == guest_nol_block_num:
                        # if all blocks for non-overlapping samples of guest have been visited,
                        # end current epoch and start a new epoch
                        print("[INFO] all blocks for non-overlapping samples of "
                              "guest have been visited, end current epoch and start a new epoch")
                        break
                    nol_guest_block_size = self.vftl_guest.load_nol_block(nol_guest_block_idx)
                    nol_guest_batch_idx = 0

                nol_guest_start = nol_guest_batch_size * nol_guest_batch_idx
                nol_guest_end = nol_guest_batch_size * nol_guest_batch_idx + nol_guest_batch_size

                if self.debug:
                    print("[DEBUG] nol_guest_block_idx:", nol_guest_block_idx)
                    print("[DEBUG] nol_guest_block_size:", nol_guest_block_size)
                    print("[DEBUG] nol guest batch from {0} to {1} ".format(nol_guest_start, nol_guest_end))
                    print(f"[DEBUG] nol_guest_block_count/nol_guest_block_num : "
                          f"{nol_guest_block_idx + 1}/{guest_nol_block_num}")
                    print(f"[DEBUG] nol_guest_train_data/nol_guest_block_size : {nol_guest_end}/{nol_guest_block_size}")

                # =======================================================
                # iterate batch index for non-overlapping samples of host
                # =======================================================
                if nol_host_end >= nol_host_block_size:
                    nol_host_block_idx += 1
                    if nol_host_block_idx == host_nol_block_num:
                        # if all blocks for non-overlapping samples of host have been visited,
                        # end current epoch and start a new epoch
                        print("[INFO] all blocks for non-overlapping samples of "
                              "host have been visited, end current epoch and start a new epoch")
                        break
                    nol_host_block_size = self.vftl_host.load_nol_block(nol_host_block_idx)
                    nol_host_batch_idx = 0

                nol_host_start = nol_host_batch_size * nol_host_batch_idx
                nol_host_end = nol_host_batch_size * nol_host_batch_idx + nol_host_batch_size

                if self.debug:
                    print("[DEBUG] nol_host_block_idx:", nol_host_block_idx)
                    print("[DEBUG] nol_host_block_size:", nol_host_block_size)
                    print("[DEBUG] nol host batch from {0} to {1}".format(nol_host_start, nol_host_end))
                    print(f"[DEBUG] nol_host_block_count/nol_host_block_num : "
                          f"{nol_host_block_idx + 1}/{host_nol_block_num}")
                    print(f"[DEBUG] nol_host_train_data/nol_host_block_size : {nol_host_end}/{nol_host_block_size}")

                # ===========================================================
                # iterate estimation block index or estimation sample indices.
                # ===========================================================
                if using_block_idx:
                    ested_guest_block_idx += 1
                    if ested_guest_block_idx >= guest_ested_block_num:
                        ested_guest_block_idx = 0

                    ested_host_block_idx += 1
                    if ested_host_block_idx >= host_ested_block_num:
                        ested_host_block_idx = 0

                    if self.debug:
                        print("[DEBUG] Using block idx")
                        print("[DEBUG] ested_guest_block_idx:", ested_guest_block_idx)
                        print("[DEBUG] ested_host_block_idx:", ested_host_block_idx)
                else:
                    estimation_block_size = self.fed_model_param.all_sample_block_size
                    guest_block_indices = np.random.choice(guest_all_training_size, estimation_block_size)
                    host_block_indices = np.random.choice(host_all_training_size, estimation_block_size)

                    if self.debug:
                        print("[DEBUG] Using block indices")
                        print("[DEBUG] guest_block_indices: ", guest_block_indices)
                        print("[DEBUG] guest_block_indices:", len(guest_block_indices))
                        print("[DEBUG] host_block_indices:", len(host_block_indices))

                # ==========
                #   train
                # ==========
                loss_dict = self._train(ul_overlap_batch_range=(ul_start, ul_end),
                                        ll_overlap_batch_range=(ll_start, ll_end),
                                        guest_non_overlap_batch_range=(nol_guest_start, nol_guest_end),
                                        host_non_overlap_batch_range=(nol_host_start, nol_host_end),
                                        guest_block_idx=ested_guest_block_idx,
                                        host_block_idx=ested_host_block_idx,
                                        guest_block_indices=guest_block_indices,
                                        host_block_indices=host_block_indices)

                fed_loss = loss_dict['fed_loss']
                loss_list.append(fed_loss)

                print("[INFO] ==> ep:{0}, iter:{1}, ll_batch_idx:{2}, nol_guest_batch_idx:{3}, nol_host_batch_idx:{4}, "
                      "fed_loss:{5}".format(i, iter, ll_batch_idx, nol_guest_batch_idx, nol_host_batch_idx, fed_loss))
                print("[INFO] ll_block_idx:{0}, nol_guest_block_idx:{1}, nol_host_block_idx:{2}, "
                      "ested_guest_block_idx:{3}, ested_host_block_idx:{4}".format(ll_block_idx,
                                                                                   nol_guest_block_idx,
                                                                                   nol_host_block_idx,
                                                                                   ested_guest_block_idx,
                                                                                   ested_host_block_idx))

                ll_batch_idx = ll_batch_idx + 1
                nol_guest_batch_idx = nol_guest_batch_idx + 1
                nol_host_batch_idx = nol_host_batch_idx + 1

                # ==========
                # validation
                # ==========
                if (iter + 1) % valid_iteration_interval == 0:
                    self.to_eval_mode()

                    # ====================
                    # two sides validation
                    # ====================
                    all_acc, all_auc, all_fscore = self.two_side_predict()
                    all_acc_list.append(all_acc)
                    all_auc_list.append(all_auc)
                    all_fscore_list.append(all_fscore)

                    # # =====================
                    # # guest side validation
                    # # =====================
                    # fed_res, guest_res = self.guest_side_predict(host_all_training_size,
                    #                                              estimation_block_size,
                    #                                              estimation_block_num=guest_ested_block_num)
                    # g_fed_fscore_list, g_fed_acc_list, g_fed_auc_list = fed_res
                    # g_only_fscore_list, g_only_acc_list, g_only_auc_list = guest_res
                    #
                    # g_fed_fscore_mean = np.mean(g_fed_fscore_list)
                    # g_fed_acc_mean = np.mean(g_fed_acc_list)
                    # g_fed_auc_mean = np.mean(g_fed_auc_list)
                    #
                    # g_only_fscore_mean = np.mean(g_only_fscore_list)
                    # g_only_acc_mean = np.mean(g_only_acc_list)
                    # g_only_auc_mean = np.mean(g_only_auc_list)
                    #
                    # if debug:
                    #     print("%% g_fed_fscore_list:", g_fed_fscore_list, g_fed_fscore_mean)
                    #     print("%% g_fed_acc_list:", g_fed_acc_list, g_fed_acc_mean)
                    #     print("%% g_fed_auc_list:", g_fed_auc_list, g_fed_auc_mean)
                    #
                    # # ====================
                    # # host side validation
                    # # ====================
                    # h_fscore_list, h_acc_list, h_auc_list = self.host_side_predict(guest_all_training_size,
                    #                                                                estimation_block_size,
                    #                                                                estimation_block_num=host_ested_block_num)
                    # h_fscore_mean = np.mean(h_fscore_list)
                    # h_acc_mean = np.mean(h_acc_list)
                    # h_auc_mean = np.mean(h_auc_list)
                    #
                    # if debug:
                    #     print("%% h_fscore_list:", h_fscore_list, h_fscore_mean)
                    #     print("%% h_acc_list:", h_acc_list, h_acc_mean)
                    #     print("%% h_auc_list:", h_auc_list, h_auc_mean)
                    #
                    # print("=" * 100)
                    # print(f"epoch:{i}, iter:{iter}")
                    # print("[INFO] fscore: all, fed_guest, only_guest, host - {0:0.4f},{1:0.4f},{2:0.4f},{3:0.4f}"
                    #       .format(all_fscore, g_fed_fscore_mean, g_only_fscore_mean, h_fscore_mean))
                    # print("[INFO] acc: all, fed_guest, only_guest, host - {0:0.4f},{1:0.4f},{2:0.4f},{3:0.4f}"
                    #       .format(all_acc, g_fed_acc_mean, g_only_acc_mean, h_acc_mean))
                    # print("[INFO] auc: all, fed_guest, only_guest, host - {0:0.4f},{1:0.4f},{2:0.4f},{3:0.4f}"
                    #       .format(all_auc, g_fed_auc_mean, g_only_auc_mean, h_auc_mean))
                    # print("=" * 100)

                    # ave_fscore = (all_fscore + g_fed_fscore_mean + h_fscore_mean) / 3
                    # ave_fscore = 3 / (1 / all_fscore + 1 / g_fed_fscore_mean + 1 / h_fscore_mean)
                    # ave_accuracy = 3 / (1 / all_acc + 1 / g_fed_acc_mean + 1 / h_acc_mean)
                    # ave_accuracy_2 = 2 / (1 / all_acc + 1 / g_fed_acc_mean)
                    # print("[INFO] harmonic mean of fscore:", ave_fscore)
                    # print("[INFO] harmonic mean of accuracy:", ave_accuracy)
                    # print("[INFO] harmonic mean of accuracy_2:", ave_accuracy_2)

                    # log = {"fscore": all_acc,
                    #        "all_fscore": all_fscore, "g_fscore": g_fed_fscore_mean, "h_fscore": h_fscore_mean,
                    #        "all_acc": all_acc, "g_acc": g_fed_acc_mean, "h_acc": h_acc_mean,
                    #        "all_auc": all_auc, "g_auc": g_fed_auc_mean, "h_auc": h_auc_mean}
                    log = {}

                    early_stopping.on_iteration_end(epoch=i, batch=nol_guest_batch_idx, log=log)
                    if self.stop_training is True:
                        break
                    # end validation

                iter += 1
            if self.stop_training is True:
                break

        end_time = time.time()
        print("[INFO] training time (s):", end_time - start_time)
        print("[INFO] stopped epoch, batch:", early_stopping.stopped_epoch, early_stopping.stopped_batch)
        early_stopping.print_log_of_best_result()
        # series_plot(losses=loss_list, fscores=all_fscore_list, aucs=all_acc_list)
        return early_stopping.get_log_info(), loss_list
