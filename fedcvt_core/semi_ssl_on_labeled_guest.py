import torch
import torch.nn.functional as F

from debug_utils import analyze_estimated_labels
from fedcvt_core.fedcvt_repr_estimator import AttentionBasedRepresentationEstimator, concat_reprs
from fedcvt_core.loss import get_shared_reprs_loss, get_orth_reprs_loss, get_alignment_loss, \
    get_label_estimation_loss
from fedcvt_core.param import FederatedTrainingParam
from utils import transpose


class SemiSupervisedLearningBase(object):

    def __init__(self,
                 fed_predictor,
                 guest_predictor,
                 host_predictor,
                 fed_training_param: FederatedTrainingParam,
                 n_class,
                 debug=False):
        self.fed_training_param = fed_training_param
        self.n_class = n_class
        self.repr_estimator = AttentionBasedRepresentationEstimator()
        self.fed_predictor = fed_predictor
        self.guest_predictor = guest_predictor
        self.host_predictor = host_predictor
        self.debug = debug

    def forward(self, input_dict):

        using_uniq = self.fed_training_param.using_uniq
        using_comm = self.fed_training_param.using_comm

        Ug_all = input_dict["Ug_all"]
        Ug_nl_overlap = input_dict["Ug_non_overlap"]
        Ug_ll_overlap = input_dict["Ug_ll_overlap"]
        Ug_ul_overlap = input_dict["Ug_ul_overlap"]

        Uh_all = input_dict["Uh_all"]
        Uh_nl_overlap = input_dict["Uh_non_overlap"]
        Uh_ll_overlap = input_dict["Uh_ll_overlap"]
        Uh_ul_overlap = input_dict["Uh_ul_overlap"]

        Y_ll_overlap = input_dict["Y_ll_overlap"]
        Y_ll_for_estimation = input_dict["Y_ll_for_estimation"]

        #
        Y_nl_overlap = input_dict["Y_guest_non_overlap"]
        Y_nl_for_estimation = input_dict["Y_nl_for_estimation"]
        # Y_guest_all = input_dict["Y_guest_all"]

        # print("Ug_ll_overlap:", Ug_ll_overlap)
        # print("Uh_ll_overlap_:", Uh_ll_overlap_)
        #
        # print("Ug_ul_overlap:", Ug_ul_overlap)
        # print("Uh_ul_overlap_:", Uh_ul_overlap_)
        #
        # print("Ug_nl_overlap:", Ug_nl_overlap)
        # print("Uh_nl_overlap:", Uh_nl_overlap)
        #
        # print("Ug_all:", Ug_all)
        # print("Uh_all:", Uh_all)

        has_unlabeled_overlapping_samples = True if Ug_ul_overlap is not None and Uh_ul_overlap is not None else False

        Ug_all_uniq, Ug_all_comm = Ug_all
        Ug_nl_overlap_uniq, Ug_nl_overlap_comm = Ug_nl_overlap
        Ug_ll_overlap_uniq, Ug_ll_overlap_comm = Ug_ll_overlap

        Uh_all_uniq, Uh_all_comm = Uh_all
        Uh_nl_overlap_uniq, Uh_nl_overlap_comm = Uh_nl_overlap
        Uh_ll_overlap_uniq, Uh_ll_overlap_comm = Uh_ll_overlap

        if has_unlabeled_overlapping_samples:
            # This is the case where we have unlabeled overlapping samples for training,
            # we prepare overlapping reprs (by concatenating labeled overlapping reprs
            # and unlabeled overlapping reprs) for later use.

            Ug_ul_overlap_uniq, Ug_ul_overlap_comm = Ug_ul_overlap
            Ug_ol_overlap_uniq = torch.cat([Ug_ll_overlap_uniq, Ug_ul_overlap_uniq], dim=0)
            Ug_ol_overlap_comm = torch.cat([Ug_ll_overlap_comm, Ug_ul_overlap_comm], dim=0)

            Uh_ul_overlap_uniq, Uh_ul_overlap_comm = Uh_ul_overlap
            Uh_ol_overlap_uniq = torch.cat([Uh_ll_overlap_uniq, Uh_ul_overlap_uniq], dim=0)
            Uh_ol_overlap_comm = torch.cat([Uh_ll_overlap_comm, Uh_ul_overlap_comm], dim=0)
        else:
            # This is the case where we have no unlabeled overlapping samples for training.
            # In other words, all overlapping reprs are labeled ones.

            # Ug_ul_overlap_uniq, Ug_ul_overlap_comm = None, None
            Uh_ul_overlap_uniq, Uh_ul_overlap_comm = None, None
            Ug_ol_overlap_uniq = Ug_ll_overlap_uniq
            Ug_ol_overlap_comm = Ug_ll_overlap_comm
            Uh_ol_overlap_uniq = Uh_ll_overlap_uniq
            Uh_ol_overlap_comm = Uh_ll_overlap_comm

        sharpen_temp = self.fed_training_param.sharpen_temperature
        # label_prob_sharpen_temperature = self.fed_training_param.label_prob_sharpen_temperature
        fed_label_prob_threshold = self.fed_training_param.fed_label_prob_threshold
        guest_label_prob_threshold = self.fed_training_param.guest_label_prob_threshold
        host_label_prob_threshold = self.fed_training_param.host_label_prob_threshold

        # print(f"[INFO] using_uniq:{using_uniq}")
        # print(f"[INFO] using_comm:{using_comm}")
        # print(f"[INFO] is_hetero_repr:{is_hetero_repr}")
        # print(f"[INFO] sharpen temperature:{sharpen_temp}")
        # print(f"[INFO] label_prob_sharpen_temperature:{label_prob_sharpen_temperature}")
        # print(f"[INFO] fed_label_prob_threshold:{fed_label_prob_threshold}")
        # print(f"[INFO] host_label_prob_threshold:{host_label_prob_threshold}")

        # labeled overlap reprs.
        Ug_ll_reprs = concat_reprs(Ug_ll_overlap_uniq, Ug_ll_overlap_comm, using_uniq, using_comm)
        Uh_ll_reprs = concat_reprs(Uh_ll_overlap_uniq, Uh_ll_overlap_comm, using_uniq, using_comm)

        # overlap reprs, including both labeled and unlabeled ones
        Ug_ol_reprs = concat_reprs(Ug_ol_overlap_uniq, Ug_ol_overlap_comm, using_uniq, using_comm)
        Uh_ol_reprs = concat_reprs(Uh_ol_overlap_uniq, Uh_ol_overlap_comm, using_uniq, using_comm)

        # non-overlap reprs.
        Ug_nl_reprs = concat_reprs(Ug_nl_overlap_uniq, Ug_nl_overlap_comm, using_uniq, using_comm)
        Uh_nl_reprs = concat_reprs(Uh_nl_overlap_uniq, Uh_nl_overlap_comm, using_uniq, using_comm)

        W_hg = None

        # =====================================================
        # estimate feature representations for missing features
        # =====================================================

        # estimate feature reprs of host's missing samples corresponding to guest's non-overlap samples.
        Uh_nl_ested_reprs = self.repr_estimator.estimate_host_reprs_for_guest_party(
            Ug_comm=Ug_nl_overlap_comm,
            Ug_uniq=Ug_nl_overlap_uniq,
            Ug_overlap_uniq=Ug_ol_overlap_uniq,
            Uh_overlap_uniq=Uh_ol_overlap_uniq,
            Uh_all_comm=Uh_all_comm,
            sharpen_temperature=sharpen_temp,
            W_gh=transpose(W_hg),
            using_uniq=using_uniq,
            using_comm=using_comm)

        # estimate feature reprs and labels of guest's missing samples corresponding to host's non-overlap samples.
        Ug_labeled_reprs_comm = torch.cat([Ug_ll_overlap_comm, Ug_nl_overlap_comm], dim=0)
        # Ug_y = torch.cat([Y_ll_overlap, Y_nl_overlap], dim=0)
        Ug_y = torch.cat([Y_ll_for_estimation, Y_nl_for_estimation], dim=0)

        Ug_nl_ested_reprs_w_lbls = self.repr_estimator.estimate_labeled_guest_reprs_for_host_party(
            Uh_comm=Uh_nl_overlap_comm,
            Uh_uniq=Uh_nl_overlap_uniq,
            Uh_overlap_uniq=Uh_ll_overlap_uniq,
            Ug_overlap_uniq=Ug_ll_overlap_uniq,
            Yg_overlap=Y_ll_for_estimation,
            Ug_all_comm=Ug_labeled_reprs_comm,
            Yg_all=Ug_y,
            sharpen_tempature=sharpen_temp,
            W_hg=W_hg,
            using_uniq=using_uniq,
            using_comm=using_comm)

        Ug_nl_ested_reprs = Ug_nl_ested_reprs_w_lbls[:, :-self.n_class]

        # reprs for labeled overlapping samples
        fed_ll_reprs = torch.cat([Ug_ll_reprs, Uh_ll_reprs], dim=1)

        # reprs for unlabeled overlapping samples
        fed_nl_reprs = torch.cat([Ug_nl_ested_reprs, Uh_nl_reprs], dim=1)

        host_dim = Uh_nl_reprs.shape[-1]

        # ===========================================================================
        # estimate (candidate) labels of the estimated representations (at the guest)
        # corresponding to both guest's and host's non-overlap samples.
        # ===========================================================================

        # estimate labels using federated feature reprs in which guest's reprs estimated.
        ested_lbls_use_fed_nl_reprs = F.softmax(self.fed_predictor(fed_nl_reprs), dim=1)

        # estimate labels using estimated feature reprs of missing samples of guest
        ested_lbls_use_guest_ested_reprs = F.softmax(self.guest_predictor(Ug_nl_ested_reprs), dim=1)

        # estimate labels using feature reprs of non-overlapping samples of host
        ested_lbls_use_host_nl_reprs = F.softmax(self.host_predictor(Uh_nl_reprs), dim=1)

        # ested_lbls_use_fed_nl_reprs = sharpen(ested_lbls_use_fed_nl_reprs,
        #                                       temperature=label_prob_sharpen_temperature)
        # ested_lbls_use_guest_ested_reprs = sharpen(ested_lbls_use_guest_ested_reprs,
        #                                               temperature=label_prob_sharpen_temperature)
        # ested_lbls_use_host_nl_reprs = sharpen(ested_lbls_use_host_nl_reprs,
        #                                        temperature=label_prob_sharpen_temperature)

        if self.debug:
            nl_ested_lbls_use_attention = Ug_nl_ested_reprs_w_lbls[:, -self.n_class:]
            analyze_estimated_labels(nl_ested_lbls_use_attention,
                                     ested_lbls_use_guest_ested_reprs,
                                     ested_lbls_use_host_nl_reprs,
                                     ested_lbls_use_fed_nl_reprs)

        # ========================================================================================================
        # fed_nl_w_ested_guest_reprs_n_candidate_labels is the concatenation of federated feature
        # representations in which the guest's representations are estimated and three estimated candidate labels.
        # ========================================================================================================
        ested_lbls_use_fed_nl_reprs = ested_lbls_use_fed_nl_reprs.detach()
        ested_lbls_use_guest_ested_reprs = ested_lbls_use_guest_ested_reprs.detach()
        ested_lbls_use_host_nl_reprs = ested_lbls_use_host_nl_reprs.detach()
        # nl_ested_lbls_use_attention = nl_ested_lbls_use_attention.detach()

        fed_nl_w_ested_guest_reprs_n_candidate_labels = torch.cat(
            [fed_nl_reprs,
             # nl_ested_lbls_use_attention,
             ested_lbls_use_host_nl_reprs,
             ested_lbls_use_guest_ested_reprs,
             ested_lbls_use_fed_nl_reprs], dim=1)

        if self.debug:
            # print("[DEBUG] Uh_nl_ested_reprs shape", Uh_nl_ested_reprs.shape)
            print("[DEBUG] Ug_nl_ested_reprs_w_lbls shape", Ug_nl_ested_reprs_w_lbls.shape)
            # print("[DEBUG] fed_nl_w_guest_ested_reprs shape:", fed_nl_w_guest_ested_reprs.shape)
            print("[DEBUG] fed_nl_w_ested_guest_reprs_n_candidate_labels shape {0}".format(
                fed_nl_w_ested_guest_reprs_n_candidate_labels.shape))

        # ================================================================================================
        # select representations from fed_nl_w_ested_guest_reprs_n_candidate_labels based estimated labels.
        # ================================================================================================
        selected_reprs_w_lbls = self.repr_estimator.select_reprs_for_multiclass(
            reprs_w_candidate_labels=fed_nl_w_ested_guest_reprs_n_candidate_labels,
            n_class=self.n_class,
            fed_label_upper_bound=fed_label_prob_threshold,
            guest_label_upper_bound=guest_label_prob_threshold,
            host_label_upper_bound=host_label_prob_threshold,
            debug=self.debug)

        # =========================================
        # prepare the expanded training set for VFL
        # =========================================
        num_candidate_samples = fed_nl_w_ested_guest_reprs_n_candidate_labels.shape[0]
        mv_train_data_dict = self.get_multiview_train_data(
            fed_ll_reprs=fed_ll_reprs,
            Ug_ll_overlap_reprs=Ug_ll_reprs,
            Uh_ll_overlap_reprs=Uh_ll_reprs,
            Ug_nl_reprs=Ug_nl_reprs,
            Uh_nl_ested_reprs=Uh_nl_ested_reprs,
            Y_ll_overlap=Y_ll_overlap,
            Y_nl_overlap=Y_nl_overlap,
            selected_reprs_w_lbls=selected_reprs_w_lbls,
            num_candidate_samples=num_candidate_samples)

        # =========================
        # compute auxiliary losses
        # =========================
        # # estimate labels for non-overlapping representations of host.
        # uniq_lbls, comm_lbls = self.repr_estimator.estimate_unique_comm_labels_for_host_party(
        #     Uh_comm=Uh_all_comm,
        #     Uh_uniq=Uh_all_uniq,
        #     Uh_overlap_uniq=Uh_ll_overlap_uniq,
        #     Ug_all_comm=Ug_labeled_reprs_comm,
        #     Yg_overlap=Y_ll_for_estimation,
        #     Yg_all=Ug_y,
        #     sharpen_tempature=sharpen_temp,
        #     W_hg=W_hg,
        #     using_uniq=using_uniq,
        #     using_comm=using_comm)

        # estimate overlap feature representations on host side for guest party for minimizing alignment loss
        Uh_ol_overlap_ested_reprs = self.repr_estimator.estimate_host_reprs_for_guest_party(
            Ug_comm=Ug_ol_overlap_comm,
            Ug_uniq=Ug_ol_overlap_uniq,
            Ug_overlap_uniq=Ug_ll_overlap_uniq,
            Uh_overlap_uniq=Uh_ll_overlap_uniq,
            Uh_all_comm=Uh_all_comm,
            sharpen_temperature=sharpen_temp,
            W_gh=transpose(W_hg),
            using_uniq=using_uniq,
            using_comm=using_comm)

        # estimate overlap feature representations and labels on guest side for host party for minimizing alignment loss
        result_ll_overlap = self.repr_estimator.estimate_guest_reprs_n_lbls_for_host_party(
            Uh_comm=Uh_ll_overlap_comm,
            Uh_uniq=Uh_ll_overlap_uniq,
            Uh_overlap_uniq=Uh_ll_overlap_uniq,
            Ug_overlap_uniq=Ug_ll_overlap_uniq,
            Ug_all_comm=Ug_ll_overlap_comm,
            Yg_overlap=Y_ll_for_estimation,
            Yg_all=Y_ll_for_estimation,
            sharpen_tempature=sharpen_temp,
            W_hg=W_hg,
            using_uniq=using_uniq,
            using_comm=using_comm)

        Ug_ll_overlap_ested_reprs, _, ll_uniq_lbls, ll_comm_lbls = result_ll_overlap

        if has_unlabeled_overlapping_samples:
            Ug_ul_overlap_ested_reprs = self.repr_estimator.estimate_guest_reprs_for_host_party(
                Uh_comm=Uh_ul_overlap_comm,
                Uh_uniq=Uh_ul_overlap_uniq,
                Uh_overlap_uniq=Uh_ll_overlap_uniq,
                Ug_overlap_uniq=Ug_ll_overlap_uniq,
                Ug_all_comm=Ug_all_comm,
                sharpen_tempature=sharpen_temp,
                W_hg=W_hg,
                using_uniq=using_uniq,
                using_comm=using_comm)
            Ug_ol_overlap_ested_reprs = torch.cat([Ug_ll_overlap_ested_reprs, Ug_ul_overlap_ested_reprs], dim=0)
        else:
            Ug_ol_overlap_ested_reprs = Ug_ll_overlap_ested_reprs

        # # estimate non-overlap labels on guest side for host party for testing purpose
        # # estimate_labeled_guest_reprs_for_host_party
        # result_non_overlap = self.repr_estimator.estimate_guest_reprs_n_lbls_for_host_party(
        #     Uh_uniq=Uh_nl_overlap_uniq,
        #     Uh_ll_overlap_uniq=Uh_ll_overlap_uniq,
        #     Uh_comm=Uh_nl_overlap_comm,
        #     Ug_ll_overlap_uniq=Ug_ll_overlap_uniq,
        #     Ug_all_comm=Ug_all_comm,
        #     Yg_overlap=Y_ll_overlap_for_estimation,
        #     Yg_all=Y_all_for_estimation,
        #     sharpen_tempature=sharpen_temp,
        #     W_hg=W_hg,
        #     using_uniq=using_uniq,
        #     using_comm=using_comm)
        # # _, self.Uh_non_overlap_ested_soft_lbls, _, _ = result_non_overlap

        auxiliary_loss_dict = dict()
        add_assistant_loss = True
        if add_assistant_loss:
            # ========================================================================
            # The losses (1), (2), and (3) correspond to the formula (1), (2), and (3)
            # defined in the original paper.
            # ========================================================================

            # (1) loss for minimizing distance between shared reprs between host and guest.
            if using_comm:
                auxiliary_loss_dict['lambda_dist_shared_reprs'] = get_shared_reprs_loss(Ug_ll_overlap_comm,
                                                                                        Uh_ll_overlap_comm)

            # (2) (3) loss for maximizing distance between orthogonal reprs for host and guest, respectively.
            if using_uniq and using_comm:
                guest_uniq_reprs_loss = get_orth_reprs_loss(Ug_all_uniq, Ug_all_comm)
                auxiliary_loss_dict['lambda_guest_sim_shared_reprs_vs_unique_repr'] = guest_uniq_reprs_loss

            if using_uniq and using_comm:
                host_uniq_reprs_loss = get_orth_reprs_loss(Uh_all_uniq, Uh_all_comm)
                auxiliary_loss_dict['lambda_host_sim_shared_reprs_vs_unique_repr'] = host_uniq_reprs_loss

            # ========================================================================
            # The losses (4) and (5) correspond to the two losses defined in formula
            # (7) and (8) in the original paper.
            # ========================================================================

            # (4) loss for minimizing distance between estimated guest overlap reprs and true guest reprs
            Ug_overlap_ested_reprs_alignment_loss = get_alignment_loss(Ug_ol_overlap_ested_reprs, Ug_ol_reprs)
            auxiliary_loss_dict['lambda_guest_dist_ested_repr_vs_true_repr'] = Ug_overlap_ested_reprs_alignment_loss

            # (5) loss for minimizing distance between estimated host overlap reprs and true host reprs
            Uh_overlap_ested_reprs_alignment_loss = get_alignment_loss(Uh_ol_overlap_ested_reprs, Uh_ol_reprs)
            auxiliary_loss_dict['lambda_host_dist_ested_repr_vs_true_repr'] = Uh_overlap_ested_reprs_alignment_loss

            # ========================================================================
            # the losses (6), (7), and (8) are newly added losses that are not covered
            # in the original paper
            # ========================================================================

            # (6) (7) loss for minimizing distance between estimated host overlap labels and true labels
            if using_uniq:
                Ug_uniq_lbl_loss = get_label_estimation_loss(ll_uniq_lbls, Y_ll_overlap)
                auxiliary_loss_dict['lambda_host_dist_ested_uniq_lbl_vs_true_lbl'] = Ug_uniq_lbl_loss

            if using_comm:
                Ug_comm_lbl_loss = get_label_estimation_loss(ll_comm_lbls, Y_ll_overlap)
                auxiliary_loss_dict['lambda_host_dist_ested_comm_lbl_vs_true_lbl'] = Ug_comm_lbl_loss

            # # (8) loss for distance between shared-repr-estimated host label and uniq-repr-estimated host label
            # if using_uniq and using_comm:
            #     label_alignment_loss = get_alignment_loss(self.uniq_lbls, self.comm_lbls)
            #     auxiliary_loss_dict['lambda_host_dist_two_ested_lbl'] = label_alignment_loss
        return mv_train_data_dict, auxiliary_loss_dict

    # def get_multiview_train_data(self,
    #                              fed_ll_reprs,
    #                              Ug_ll_overlap_reprs,
    #                              Uh_ll_overlap_reprs,
    #                              Ug_nl_reprs,
    #                              Uh_nl_ested_reprs,
    #                              Y_ll_overlap,
    #                              Y_nl_overlap,
    #                              selected_reprs_w_lbls,
    #                              num_candidate_samples):
    #     has_selected_samples = True if selected_reprs_w_lbls is not None else False
    #     if has_selected_samples:
    #         # if have selected representations
    #         num_selected_samples = selected_reprs_w_lbls.shape[0]
    #         print("[DEBUG] has_selected_samples:{}; ratio of selected samples:{}/{}".format(has_selected_samples,
    #                                                                                         num_selected_samples,
    #                                                                                         num_candidate_samples))
    #
    #         sel_fed_nl_w_guest_ested_reprs = selected_reprs_w_lbls[:, :-self.n_class]
    #         sel_labels = selected_reprs_w_lbls[:, -self.n_class:]
    #
    #         # print("fed_ll_reprs:", fed_ll_reprs.shape)
    #         # print("sel_fed_nl_w_guest_ested_reprs:", sel_fed_nl_w_guest_ested_reprs.shape)
    #
    #         # fed_reprs = torch.cat([fed_ll_reprs, sel_fed_nl_w_guest_ested_reprs], dim=0)
    #         # fed_y = torch.cat([Y_ll_overlap, sel_labels], dim=0)
    #
    #         # use fed_reprs with host reprs estimated.
    #         fed_nl_w_host_ested_reprs = torch.cat((Ug_nl_reprs, Uh_nl_ested_reprs), dim=0)
    #         fed_reprs = torch.cat([fed_ll_reprs, sel_fed_nl_w_guest_ested_reprs, fed_nl_w_host_ested_reprs], dim=0)
    #         fed_y = torch.cat([Y_ll_overlap, sel_labels, Y_nl_overlap], dim=0)
    #
    #         g_reprs = torch.cat((Ug_ll_overlap_reprs, Ug_nl_reprs), dim=0)
    #         g_y = torch.cat((Y_ll_overlap, Y_nl_overlap), dim=0)
    #
    #         # h_nl_reprs = sel_fed_nl_w_guest_ested_reprs[:, :-host_dim]
    #         # h_reprs = torch.cat((Uh_ll_overlap_reprs, h_nl_reprs), dim=0)
    #         # h_y = torch.cat((Y_ll_overlap, sel_labels), dim=0)
    #
    #         h_reprs = Uh_ll_overlap_reprs
    #
    #         # train_fed_reprs, train_guest_reprs, train_host_reprs, train_fed_y, train_g_y, train_h_y = \
    #         #     fed_reprs, g_reprs, h_reprs, fed_y, g_y, h_y
    #         train_fed_reprs, train_guest_reprs, train_host_reprs, train_fed_y, train_g_y, train_h_y = \
    #             fed_reprs, g_reprs, h_reprs, fed_y, g_y, Y_ll_overlap
    #
    #     else:
    #         # if have no selected representations
    #         print("[DEBUG] has_selected_samples:{}.".format(has_selected_samples))
    #
    #         # 1
    #         # fed_reprs = fed_ll_reprs
    #         # g_ll_reprs = tf.concat(Ug_overlap, axis=1)
    #         # g_reprs = g_ll_reprs
    #         #
    #         # h_ll_reprs = tf.concat(Uh_overlap, axis=1)
    #
    #         # 2 use representation estimation
    #         # fed_reprs = torch.cat([fed_ll_reprs, fed_nl_w_host_ested_reprs], dim=0)
    #
    #         g_reprs = torch.cat((Ug_ll_overlap_reprs, Ug_nl_reprs), dim=0)
    #         g_y = torch.cat((Y_ll_overlap, Y_nl_overlap), dim=0)
    #         # g_nl_reprs = Ug_nl_overlap_reprs
    #         # g_reprs = torch.cat((g_ll_reprs, g_nl_reprs), dim=0)
    #
    #         h_ll_reprs = Uh_ll_overlap_reprs
    #
    #         # y = torch.cat([Y_ll_overlap, Y_guest_non_overlap], dim=0)
    #         train_fed_reprs, train_guest_reprs, train_host_reprs, train_fed_y, train_g_y, train_h_y = \
    #             fed_ll_reprs, g_reprs, h_ll_reprs, Y_ll_overlap, g_y, Y_ll_overlap
    #
    #         # 3 do not use representation estimation
    #         # g_ll_reprs = tf.concat(Ug_ll_overlap_reprs, axis=1)
    #         # h_ll_reprs = tf.concat(Uh_ll_overlap_reprs, axis=1)
    #         # return fed_ll_reprs, g_ll_reprs, h_ll_reprs, Y_ll_overlap, Y_ll_overlap, Y_ll_overlap
    #
    #     result = dict()
    #     result["train_fed_reprs"] = train_fed_reprs
    #     result["train_guest_reprs"] = train_guest_reprs
    #     result["train_host_reprs"] = train_host_reprs
    #     result["train_fed_y"] = train_fed_y
    #     result["train_guest_y"] = train_g_y
    #     result["train_host_y"] = train_h_y
    #
    #     if self.debug:
    #         training_fed_reprs_shape = train_fed_reprs.shape
    #         training_guest_reprs_shape = train_guest_reprs.shape
    #         print("[DEBUG] training_fed_reprs_shape:{}".format(training_fed_reprs_shape))
    #         print("[DEBUG] training_guest_reprs_shape:{}".format(training_guest_reprs_shape))
    #
    #     return result

    def get_multiview_train_data(self, fed_ll_reprs, Ug_ll_overlap_reprs, Uh_ll_overlap_reprs, Ug_nl_reprs,
                                 Uh_nl_ested_reprs, Y_ll_overlap, Y_nl_overlap, selected_reprs_w_lbls,
                                 num_candidate_samples) -> dict:
        pass


class SemiSupervisedLearningV2(SemiSupervisedLearningBase):

    def __init__(self, fed_predictor, guest_predictor, host_predictor, fed_training_param: FederatedTrainingParam,
                 n_class, debug=False):
        super(SemiSupervisedLearningV2, self).__init__(fed_predictor,
                                                       guest_predictor,
                                                       host_predictor,
                                                       fed_training_param,
                                                       n_class, debug)
        self.debug = debug

    def get_multiview_train_data(self,
                                 fed_ll_reprs,
                                 Ug_ll_overlap_reprs,
                                 Uh_ll_overlap_reprs,
                                 Ug_nl_reprs,
                                 Uh_nl_ested_reprs,
                                 Y_ll_overlap,
                                 Y_nl_overlap,
                                 selected_reprs_w_lbls,
                                 num_candidate_samples):
        has_selected_samples = True if selected_reprs_w_lbls is not None else False
        if has_selected_samples:
            # if have selected representations
            num_selected_samples = selected_reprs_w_lbls.shape[0]
            print("[DEBUG] has_selected_samples:{}; ratio of selected samples:{}/{}".format(has_selected_samples,
                                                                                            num_selected_samples,
                                                                                            num_candidate_samples))

            sel_fed_nl_w_guest_ested_reprs = selected_reprs_w_lbls[:, :-self.n_class]
            sel_labels = selected_reprs_w_lbls[:, -self.n_class:]

            # # use fed_reprs with host reprs estimated.
            fed_nl_w_host_ested_reprs = torch.cat((Ug_nl_reprs, Uh_nl_ested_reprs), dim=1)
            fed_reprs = torch.cat([fed_ll_reprs, sel_fed_nl_w_guest_ested_reprs, fed_nl_w_host_ested_reprs], dim=0)
            fed_y = torch.cat([Y_ll_overlap, sel_labels, Y_nl_overlap], dim=0)
            # fed_reprs = torch.cat([fed_ll_reprs, sel_fed_nl_w_guest_ested_reprs], dim=0)
            # fed_y = torch.cat([Y_ll_overlap, sel_labels], dim=0)

            g_reprs = torch.cat((Ug_ll_overlap_reprs, Ug_nl_reprs), dim=0)

            g_y = torch.cat((Y_ll_overlap, Y_nl_overlap), dim=0)

            h_reprs = Uh_ll_overlap_reprs

            train_fed_reprs, train_guest_reprs, train_host_reprs, train_fed_y, train_g_y, train_h_y = \
                fed_reprs, g_reprs, h_reprs, fed_y, g_y, Y_ll_overlap

        else:
            # if have no selected representations
            print("[DEBUG] has_selected_samples:{}.".format(has_selected_samples))

            g_reprs = torch.cat((Ug_ll_overlap_reprs, Ug_nl_reprs), dim=0)

            g_y = torch.cat((Y_ll_overlap, Y_nl_overlap), dim=0)

            h_ll_reprs = Uh_ll_overlap_reprs

            train_fed_reprs, train_guest_reprs, train_host_reprs, train_fed_y, train_g_y, train_h_y = \
                fed_ll_reprs, g_reprs, h_ll_reprs, Y_ll_overlap, g_y, Y_ll_overlap

        result = dict()
        result["train_fed_reprs"] = train_fed_reprs
        result["train_guest_reprs"] = train_guest_reprs
        result["train_host_reprs"] = train_host_reprs
        result["train_fed_y"] = train_fed_y
        result["train_guest_y"] = train_g_y
        result["train_host_y"] = train_h_y

        if self.debug:
            training_fed_reprs_shape = train_fed_reprs.shape
            training_guest_reprs_shape = train_guest_reprs.shape
            print("[DEBUG] training_fed_reprs_shape:{}".format(training_fed_reprs_shape))
            print("[DEBUG] training_guest_reprs_shape:{}".format(training_guest_reprs_shape))

        return result

