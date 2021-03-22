import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops


def sharpen(p, temperature=0.1, axis=1):
    u = tf.math.pow(p, 1 / temperature)
    return u / tf.math.reduce_sum(input_tensor=u, axis=axis, keepdims=True)


def compute_queries_keys_sim(queries, keys, Wqk=None):
    if Wqk is None:
        return tf.matmul(queries, tf.transpose(a=keys)) / tf.sqrt(tf.cast(tf.shape(input=keys)[1], dtype=tf.float32))
    else:
        trans_queries = tf.matmul(queries, Wqk)
        return tf.matmul(trans_queries, tf.transpose(a=keys)) / tf.sqrt(
            tf.cast(tf.shape(input=keys)[1], dtype=tf.float32))


def softmax_with_sharpening(matrix, sharpen_temperature=None):
    softmax_matrix = tf.nn.softmax(matrix, axis=1)
    if sharpen_temperature is not None:
        softmax_matrix = sharpen(softmax_matrix, temperature=sharpen_temperature)
    return softmax_matrix


def compute_attention_using_similarity_matrix(similarity_matrix, values):
    return tf.matmul(similarity_matrix, values)


def compute_softmax_matrix(queries, keys, sharpen_temperature=0.1, Wqk=None):
    similarity_matrix = compute_queries_keys_sim(queries, keys, Wqk)
    return softmax_with_sharpening(similarity_matrix, sharpen_temperature)


def compute_attention(queries, keys, values, sharpen_temperature=0.1, Wqk=None):
    softmax_matrix = compute_softmax_matrix(queries, keys, sharpen_temperature, Wqk)
    return compute_attention_using_similarity_matrix(softmax_matrix, values)


class AttentionBasedRepresentationEstimator(object):

    @staticmethod
    def estimate_host_reprs_for_guest_party(Ug_comm,
                                            Ug_uniq,
                                            Ug_overlap_uniq,
                                            Uh_overlap_uniq,
                                            Uh_all_comm,
                                            sharpen_temperature=0.1,
                                            W_gh=None):
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
        uniq_reprs = compute_attention(queries=Ug_uniq,
                                       keys=Ug_overlap_uniq,
                                       values=Uh_overlap_uniq,
                                       sharpen_temperature=sharpen_temperature,
                                       Wqk=None)
        comm_reprs = compute_attention(queries=Ug_comm,
                                       keys=Uh_all_comm,
                                       values=Uh_all_comm,
                                       sharpen_temperature=sharpen_temperature,
                                       Wqk=W_gh)
        reprs = tf.concat([uniq_reprs, comm_reprs], axis=1)
        return reprs

    @staticmethod
    def _estimate_guest_reprs_w_lbls_for_host_party(Uh_comm,
                                                    Uh_uniq,
                                                    Uh_overlap_uniq,
                                                    Ug_overlap_uniq,
                                                    Ug_all_comm,
                                                    Yg_overlap,
                                                    Yg_all,
                                                    sharpen_tempature=0.1,
                                                    W_hg=None):
        softmax_matrix_uniq = compute_softmax_matrix(queries=Uh_uniq,
                                                     keys=Uh_overlap_uniq,
                                                     sharpen_temperature=sharpen_tempature,
                                                     Wqk=None)
        softmax_matrix_comm = compute_softmax_matrix(queries=Uh_comm,
                                                     keys=Ug_all_comm,
                                                     sharpen_temperature=sharpen_tempature,
                                                     Wqk=W_hg)

        uniq_reprs = tf.matmul(softmax_matrix_uniq, Ug_overlap_uniq)
        uniq_lbls = tf.matmul(softmax_matrix_uniq, Yg_overlap)

        comm_reprs = tf.matmul(softmax_matrix_comm, Ug_all_comm)
        comm_lbls = tf.matmul(softmax_matrix_comm, Yg_all)
        return uniq_reprs, uniq_lbls, comm_reprs, comm_lbls

    @staticmethod
    def estimate_unique_comm_labels_for_host_party(Uh_comm,
                                                   Uh_uniq,
                                                   Uh_overlap_uniq,
                                                   Ug_all_comm,
                                                   Yg_overlap,
                                                   Yg_all,
                                                   sharpen_tempature=0.1,
                                                   W_hg=None):
        softmax_matrix_uniq = compute_softmax_matrix(queries=Uh_uniq,
                                                     keys=Uh_overlap_uniq,
                                                     sharpen_temperature=sharpen_tempature,
                                                     Wqk=None)
        softmax_matrix_comm = compute_softmax_matrix(queries=Uh_comm,
                                                     keys=Ug_all_comm,
                                                     sharpen_temperature=sharpen_tempature,
                                                     Wqk=W_hg)

        uniq_lbls = tf.matmul(softmax_matrix_uniq, Yg_overlap)
        comm_lbls = tf.matmul(softmax_matrix_comm, Yg_all)
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
                                                    W_hg=None):
        result = self._estimate_guest_reprs_w_lbls_for_host_party(Uh_comm=Uh_comm,
                                                                  Uh_uniq=Uh_uniq,
                                                                  Uh_overlap_uniq=Uh_overlap_uniq,
                                                                  Ug_overlap_uniq=Ug_overlap_uniq,
                                                                  Ug_all_comm=Ug_all_comm,
                                                                  Yg_overlap=Yg_overlap,
                                                                  Yg_all=Yg_all,
                                                                  sharpen_tempature=sharpen_tempature,
                                                                  W_hg=W_hg)

        uniq_reprs, uniq_lbls, comm_reprs, comm_lbls = result
        combine_lbls = 0.5 * uniq_lbls + 0.5 * comm_lbls
        labeled_reprs = tf.concat([uniq_reprs, comm_reprs, combine_lbls], axis=1)
        return labeled_reprs, uniq_lbls, comm_lbls

    def estimate_guest_reprs_n_lbls_for_host_party(self,
                                                   Uh_comm,
                                                   Uh_uniq,
                                                   Uh_overlap_uniq,
                                                   Ug_overlap_uniq,
                                                   Ug_all_comm,
                                                   Yg_overlap,
                                                   Yg_all,
                                                   sharpen_tempature=0.1,
                                                   W_hg=None):
        result = self._estimate_guest_reprs_w_lbls_for_host_party(Uh_comm=Uh_comm,
                                                                  Uh_uniq=Uh_uniq,
                                                                  Uh_overlap_uniq=Uh_overlap_uniq,
                                                                  Ug_overlap_uniq=Ug_overlap_uniq,
                                                                  Ug_all_comm=Ug_all_comm,
                                                                  Yg_overlap=Yg_overlap,
                                                                  Yg_all=Yg_all,
                                                                  sharpen_tempature=sharpen_tempature,
                                                                  W_hg=W_hg)

        uniq_reprs, uniq_lbls, comm_reprs, comm_lbls = result
        combine_soft_lbls = 0.5 * uniq_lbls + 0.5 * comm_lbls
        reprs = tf.concat([uniq_reprs, comm_reprs], axis=1)
        return reprs, combine_soft_lbls, uniq_lbls, comm_lbls

    def select_reprs_for_multiclass(self,
                                    reprs_w_candidate_labels,
                                    n_class,
                                    fed_label_upper_bound=0.5,
                                    host_label_upper_bound=0.5):
        dynamic_array = tensor_array_ops.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False)

        def cond(i, j, row):
            return j < tf.shape(input=reprs_w_candidate_labels)[0]

        def body(i, j, row):
            print("-------> iter {0}".format(j))
            reprs = reprs_w_candidate_labels[j, :-3 * n_class]

            # fetch fed labels
            condidate_lbl_1 = reprs_w_candidate_labels[j, -n_class:]
            # fetch guest_lr labels
            condidate_lbl_2 = reprs_w_candidate_labels[j, -2 * n_class:-n_class]
            # fetch host_lr labels
            condidate_lbl_3 = reprs_w_candidate_labels[j, -3 * n_class:- 2 * n_class]

            # print("reprs", reprs)
            # print("condidate_lbl_1", condidate_lbl_1)
            # print("condidate_lbl_2", condidate_lbl_2)

            index_1 = tf.argmax(input=condidate_lbl_1)
            index_2 = tf.argmax(input=condidate_lbl_2)
            index_3 = tf.argmax(input=condidate_lbl_3)

            is_same_class_1 = tf.math.equal(index_1, index_2)
            is_same_class_2 = tf.math.equal(index_2, index_3)

            prob_1 = condidate_lbl_1[index_1]
            # prob_2 = condidate_lbl_2[index_2]
            prob_3 = condidate_lbl_3[index_3]
            is_beyond_threshold = tf.math.logical_and(tf.math.greater(prob_1, fed_label_upper_bound),
                                                      tf.math.greater(prob_3, host_label_upper_bound))
            is_same_class = tf.math.logical_and(is_same_class_1, is_same_class_2)
            to_gather = tf.math.logical_and(is_beyond_threshold, is_same_class)

            def f1():
                # selected
                print("---> f1:selected")

                a_reprs = tf.expand_dims(reprs, axis=0)
                fed_condidate_lbl = reprs_w_candidate_labels[j, -n_class:]
                a_condidate_lbl_1 = tf.expand_dims(fed_condidate_lbl, axis=0)
                a_condidate_lbl_1 = sharpen(a_condidate_lbl_1, temperature=0.1)
                concate_reprs_w_lbls = tf.concat((a_reprs, a_condidate_lbl_1), axis=1)
                # print("concate_reprs_w_lbls:", concate_reprs_w_lbls)
                row_update = row.write(i, concate_reprs_w_lbls)
                # row_update = row.write(i, temp)
                return i + 1, j + 1, row_update

            def f2():
                print("---> f2:unselected")
                return i, j + 1, row

            i, j, row_update = tf.cond(pred=to_gather, true_fn=f1, false_fn=f2)
            return [i, j, row_update]

        _, _, list_vals = tf.while_loop(cond=cond, body=body, loop_vars=[0, 0, dynamic_array])
        return list_vals

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
