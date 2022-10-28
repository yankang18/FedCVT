import torch
import torch.nn.functional as F


# def sharpen(p, temperature=0.1, axis=1):
#     u = tf.math.pow(p, 1 / temperature)
#     return u / tf.math.reduce_sum(input_tensor=u, axis=axis, keepdims=True)

def sharpen(p, temperature=0.1):
    u = torch.pow(p, 1 / temperature)
    return u / torch.sum(u, dim=1, keepdim=True)


def compute_queries_keys_sim(queries, keys, Wqk=None):
    if Wqk is None:
        # print("keys[1].shape:", keys[1].shape[0])
        return torch.matmul(queries, torch.transpose(keys, 0, 1)) / torch.sqrt(torch.tensor(keys[1].shape[0]))
    else:
        trans_queries = torch.matmul(queries, Wqk)
        return torch.matmul(trans_queries, torch.transpose(keys, 0, 1)) / torch.sqrt(torch.tensor(keys[1].shape[0]))


def softmax_with_sharpening(matrix, sharpen_temperature=None):
    softmax_matrix = F.softmax(matrix, dim=1, _stacklevel=5)
    if sharpen_temperature is not None:
        softmax_matrix = sharpen(softmax_matrix, temperature=sharpen_temperature)
    return softmax_matrix


def compute_attention_using_similarity_matrix(similarity_matrix, values):
    return torch.matmul(similarity_matrix, values)


def compute_softmax_matrix(queries, keys, temperature=0.1, Wqk=None):
    similarity_matrix = compute_queries_keys_sim(queries, keys, Wqk)
    return softmax_with_sharpening(similarity_matrix, temperature)


def compute_attention(queries, keys, values, sharpen_temperature=0.1, Wqk=None):
    softmax_matrix = compute_softmax_matrix(queries, keys, sharpen_temperature, Wqk)
    return compute_attention_using_similarity_matrix(softmax_matrix, values)


def concat_reprs(uniq_reprs, comm_reprs, using_uniq, using_comm):
    if using_uniq and using_comm:
        return torch.cat([uniq_reprs, comm_reprs], dim=1)

    return uniq_reprs if using_uniq else comm_reprs


class AttentionBasedRepresentationEstimator(object):

    @staticmethod
    def estimate_host_reprs_for_guest_party(Ug_comm,
                                            Ug_uniq,
                                            Ug_overlap_uniq,
                                            Uh_overlap_uniq,
                                            Uh_all_comm,
                                            sharpen_temperature=0.1,
                                            W_gh=None,
                                            using_uniq=True,
                                            using_comm=True):
        """
        Estimate feature representations on the host side for guest party.

        :param Ug_comm:
        :param Ug_uniq:
        :param Ug_overlap_uniq:
        :param Uh_overlap_uniq:
        :param Uh_all_comm:
        :param sharpen_temperature:
        :param W_gh:
        :return:
        """

        uniq_reprs = None if not using_uniq else compute_attention(queries=Ug_uniq,
                                                                   keys=Ug_overlap_uniq,
                                                                   values=Uh_overlap_uniq,
                                                                   sharpen_temperature=sharpen_temperature,
                                                                   Wqk=None)
        comm_reprs = None if not using_comm else compute_attention(queries=Ug_comm,
                                                                   keys=Uh_all_comm,
                                                                   values=Uh_all_comm,
                                                                   sharpen_temperature=sharpen_temperature,
                                                                   Wqk=W_gh)

        if using_uniq and using_comm:
            return torch.cat([uniq_reprs, comm_reprs], dim=1)
        return uniq_reprs if using_uniq else comm_reprs

    @staticmethod
    def _estimate_guest_reprs_w_lbls_for_host_party(Uh_comm,
                                                    Uh_uniq,
                                                    Uh_overlap_uniq,
                                                    Ug_overlap_uniq,
                                                    Ug_all_comm,
                                                    Yg_overlap,
                                                    Yg_all,
                                                    sharpen_tempature=0.1,
                                                    W_hg=None,
                                                    using_uniq=True,
                                                    using_comm=True):

        softmax_matrix_uniq = None if not using_uniq else compute_softmax_matrix(queries=Uh_uniq,
                                                                                 keys=Uh_overlap_uniq,
                                                                                 temperature=sharpen_tempature,
                                                                                 Wqk=None)
        softmax_matrix_comm = None if not using_comm else compute_softmax_matrix(queries=Uh_comm,
                                                                                 keys=Ug_all_comm,
                                                                                 temperature=sharpen_tempature,
                                                                                 Wqk=W_hg)
        uniq_reprs = None
        uniq_lbls = None
        comm_reprs = None
        comm_lbls = None
        if using_uniq:
            uniq_reprs = torch.matmul(softmax_matrix_uniq, Ug_overlap_uniq)
            uniq_lbls = torch.matmul(softmax_matrix_uniq, Yg_overlap)

        if using_comm:
            comm_reprs = torch.matmul(softmax_matrix_comm, Ug_all_comm)
            comm_lbls = torch.matmul(softmax_matrix_comm, Yg_all)

        return uniq_reprs, uniq_lbls, comm_reprs, comm_lbls

    @staticmethod
    def estimate_unique_comm_labels_for_host_party(Uh_comm,
                                                   Uh_uniq,
                                                   Uh_overlap_uniq,
                                                   Ug_all_comm,
                                                   Yg_overlap,
                                                   Yg_all,
                                                   sharpen_tempature=0.1,
                                                   W_hg=None,
                                                   using_uniq=True,
                                                   using_comm=True):
        softmax_matrix_uniq = None if not using_uniq else compute_softmax_matrix(queries=Uh_uniq,
                                                                                 keys=Uh_overlap_uniq,
                                                                                 temperature=sharpen_tempature,
                                                                                 Wqk=None)
        softmax_matrix_comm = None if not using_comm else compute_softmax_matrix(queries=Uh_comm,
                                                                                 keys=Ug_all_comm,
                                                                                 temperature=sharpen_tempature,
                                                                                 Wqk=W_hg)
        uniq_lbls = None
        comm_lbls = None
        if using_uniq:
            uniq_lbls = torch.matmul(softmax_matrix_uniq, Yg_overlap)
        if using_comm:
            comm_lbls = torch.matmul(softmax_matrix_comm, Yg_all)

        return uniq_lbls, comm_lbls

    def estimate_labeled_guest_reprs_for_host_party(self,
                                                    Uh_comm,
                                                    Uh_uniq,
                                                    Uh_overlap_uniq,
                                                    Ug_overlap_uniq,
                                                    Ug_all_comm,
                                                    Yg_overlap,
                                                    Yg_all,
                                                    sharpen_tempature=0.1,
                                                    W_hg=None,
                                                    using_uniq=True,
                                                    using_comm=True):
        result = self._estimate_guest_reprs_w_lbls_for_host_party(Uh_comm=Uh_comm,
                                                                  Uh_uniq=Uh_uniq,
                                                                  Uh_overlap_uniq=Uh_overlap_uniq,
                                                                  Ug_overlap_uniq=Ug_overlap_uniq,
                                                                  Ug_all_comm=Ug_all_comm,
                                                                  Yg_overlap=Yg_overlap,
                                                                  Yg_all=Yg_all,
                                                                  sharpen_tempature=sharpen_tempature,
                                                                  W_hg=W_hg,
                                                                  using_uniq=using_uniq,
                                                                  using_comm=using_comm)

        uniq_reprs, uniq_lbls, comm_reprs, comm_lbls = result
        if using_uniq and using_comm:
            combine_lbls = 0.5 * uniq_lbls + 0.5 * comm_lbls
            labeled_reprs = torch.cat([uniq_reprs, comm_reprs, combine_lbls], dim=1)
        elif using_uniq:
            labeled_reprs = torch.cat([uniq_reprs, uniq_lbls], dim=1)
        else:
            labeled_reprs = torch.cat([comm_reprs, comm_lbls], dim=1)

        return labeled_reprs

    def estimate_guest_reprs_n_lbls_for_host_party(self,
                                                   Uh_comm,
                                                   Uh_uniq,
                                                   Uh_overlap_uniq,
                                                   Ug_overlap_uniq,
                                                   Ug_all_comm,
                                                   Yg_overlap,
                                                   Yg_all,
                                                   sharpen_tempature=0.1,
                                                   W_hg=None,
                                                   using_uniq=True,
                                                   using_comm=True):
        result = self._estimate_guest_reprs_w_lbls_for_host_party(Uh_comm=Uh_comm,
                                                                  Uh_uniq=Uh_uniq,
                                                                  Uh_overlap_uniq=Uh_overlap_uniq,
                                                                  Ug_overlap_uniq=Ug_overlap_uniq,
                                                                  Ug_all_comm=Ug_all_comm,
                                                                  Yg_overlap=Yg_overlap,
                                                                  Yg_all=Yg_all,
                                                                  sharpen_tempature=sharpen_tempature,
                                                                  W_hg=W_hg,
                                                                  using_uniq=using_uniq,
                                                                  using_comm=using_comm)

        uniq_reprs, uniq_lbls, comm_reprs, comm_lbls = result
        if using_uniq and using_comm:
            combine_lbls = 0.5 * uniq_lbls + 0.5 * comm_lbls
            reprs = torch.cat([uniq_reprs, comm_reprs], dim=1)
            return reprs, combine_lbls, uniq_lbls, comm_lbls
        elif using_uniq:
            return uniq_reprs, uniq_lbls, uniq_lbls, comm_lbls
        else:
            return comm_reprs, comm_lbls, uniq_lbls, comm_lbls

    def select_reprs_for_multiclass(self,
                                    reprs_w_candidate_labels,
                                    n_class,
                                    fed_label_upper_bound=0.5,
                                    host_label_upper_bound=0.5):

        sel_reprs = list()
        for repr_w_candidate_lbl in reprs_w_candidate_labels:

            reprs = repr_w_candidate_lbl[:-3 * n_class]
            # fetch fed labels
            candidate_lbl_fed = repr_w_candidate_lbl[-n_class:]
            # fetch guest_lr labels
            candidate_lbl_guest = repr_w_candidate_lbl[-2 * n_class:-n_class]
            # fetch host_lr labels
            candidate_lbl_host = repr_w_candidate_lbl[-3 * n_class:- 2 * n_class]

            print("[DEBUG] repr", reprs.shape)
            print("[DEBUG] candidate_lbl_fed", candidate_lbl_fed.shape)
            print("[DEBUG] candidate_lbl_guest", candidate_lbl_guest.shape)
            print("[DEBUG] candidate_lbl_host", candidate_lbl_host.shape)

            index_1 = torch.argmax(input=candidate_lbl_fed)
            index_2 = torch.argmax(input=candidate_lbl_guest)
            index_3 = torch.argmax(input=candidate_lbl_host)

            is_same_class_1 = torch.eq(index_1, index_2)
            is_same_class_2 = torch.eq(index_2, index_3)

            prob_1 = candidate_lbl_fed[index_1]
            prob_2 = candidate_lbl_guest[index_2]
            prob_3 = candidate_lbl_host[index_3]

            is_beyond_threshold = torch.logical_and(torch.greater(prob_1, fed_label_upper_bound),
                                                    torch.greater(prob_3, host_label_upper_bound))
            is_same_class = torch.logical_and(is_same_class_1, is_same_class_2)
            to_gather = torch.logical_and(is_beyond_threshold, is_same_class)

            if to_gather:
                # candidate_lbl_fed = candidate_lbl_fed.unsqueeze(dim=0)
                # reprs = reprs.unsqueeze(dim=0)
                candidate_lbl_fed.unsqueeze_(dim=0)
                reprs.unsqueeze_(dim=0)
                s_condidate_lbl_1 = sharpen(candidate_lbl_fed, temperature=0.1)
                print("[DEBUG] s_condidate_lbl_1:", s_condidate_lbl_1.shape)
                concate_reprs_w_lbls = torch.cat((reprs, s_condidate_lbl_1), dim=1)
                print("[DEBUG] concate_reprs_w_lbls:", concate_reprs_w_lbls.shape)
                sel_reprs.append(concate_reprs_w_lbls)
            print(f"[DEBUG] len(sel_reprs):{len(sel_reprs)}")
            sel_reprs_tensor = torch.stack(sel_reprs) if len(sel_reprs) > 0 else None
            return sel_reprs_tensor
