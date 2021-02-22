import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops


def sharpen(p, tempature=0.1):
    u = tf.math.pow(p, 1 / tempature)
    return u / tf.math.reduce_sum(input_tensor=u, axis=1, keepdims=True)


def compute_queries_keys_sim(queries, keys, Wqk=None):
    if Wqk is None:
        return tf.matmul(queries, tf.transpose(a=keys)) / tf.sqrt(tf.cast(tf.shape(input=keys)[1], dtype=tf.float64))
    else:
        trans_queries = tf.matmul(queries, Wqk)
        return tf.matmul(trans_queries, tf.transpose(a=keys)) / tf.sqrt(tf.cast(tf.shape(input=keys)[1], dtype=tf.float64))


def softmax_with_sharpenning(matrix, sharpen_tempature=None):
    softmax_matrix = tf.nn.softmax(matrix, axis=1)
    if sharpen_tempature is not None:
        softmax_matrix = sharpen(softmax_matrix, tempature=sharpen_tempature)
    return softmax_matrix


def compute_attention_using_similarity_matrix(similarity_matrix, values):
    return tf.matmul(similarity_matrix, values)


def compute_attention(queries, keys, values, sharpen_tempature=0.1, Wqk=None):
    similarity_matrix = compute_queries_keys_sim(queries, keys, Wqk)
    softmax_matrix = softmax_with_sharpenning(similarity_matrix, sharpen_tempature)
    return compute_attention_using_similarity_matrix(softmax_matrix, values)


class AttentionBasedRepresentationLearner(object):

    @staticmethod
    def estimate_host_representations_for_guest_party(Ug_non_overlap_comm,
                                                      Ug_non_overlap_uniq,
                                                      Ug_overlap_uniq,
                                                      Uh_overlap_uniq,
                                                      Uh_all_comm,
                                                      combine_axis=0,
                                                      k=2,
                                                      parallel_iterations=10,
                                                      W_hg=None):
        sims_u = tf.matmul(Ug_non_overlap_uniq, tf.transpose(a=Ug_overlap_uniq)) / tf.sqrt(
            tf.cast(tf.shape(input=Ug_non_overlap_uniq)[1], dtype=tf.float64))

        if W_hg is None:
            sims_c = tf.matmul(Ug_non_overlap_comm, tf.transpose(a=Uh_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Ug_non_overlap_comm)[1], dtype=tf.float64))
        else:
            transformed_Ug = tf.matmul(Ug_non_overlap_comm, tf.transpose(a=W_hg))
            sims_c = tf.matmul(transformed_Ug, tf.transpose(a=Uh_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Ug_non_overlap_comm)[1], dtype=tf.float64))

        if combine_axis == 0:
            concat_sim = tf.concat([sims_u, sims_c], axis=1)
            norm_concat_sim = tf.nn.softmax(concat_sim, axis=1)
            sharpened_concat_sim = sharpen(norm_concat_sim)
            concat_org = tf.concat([Uh_overlap_uniq, Uh_all_comm], axis=0)
            reprs = tf.matmul(sharpened_concat_sim, concat_org)
            return reprs

        else:
            norm_sims_u = tf.nn.softmax(sims_u, axis=1)
            sharpened_sims_u = sharpen(norm_sims_u)
            uniq_reprs = tf.matmul(sharpened_sims_u, Uh_overlap_uniq)
            norm_sims_c = tf.nn.softmax(sims_c, axis=1)
            sharpened_sims_c = sharpen(norm_sims_c)
            comm_reprs = tf.matmul(sharpened_sims_c, Uh_all_comm)
            reprs = tf.concat([uniq_reprs, comm_reprs], axis=1)
            return reprs

    @staticmethod
    def estimate_guest_representations_for_host_party(Uh_non_overlap_comm,
                                                      Uh_non_overlap_uniq,
                                                      Uh_overlap_uniq,
                                                      Ug_overlap_uniq,
                                                      Ug_all_comm,
                                                      Yg_overlap,
                                                      Yg_all,
                                                      combine_axis=0,
                                                      k=2,
                                                      parallel_iterations=10,
                                                      W_hg=None,
                                                      sharpen_tempature=0.1):

        sims_u = tf.matmul(Uh_non_overlap_uniq, tf.transpose(a=Uh_overlap_uniq)) / tf.sqrt(
            tf.cast(tf.shape(input=Uh_non_overlap_uniq)[1], dtype=tf.float64))

        if W_hg is None:
            sims_c = tf.matmul(Uh_non_overlap_comm, tf.transpose(a=Ug_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Uh_non_overlap_comm)[1], dtype=tf.float64))
        else:
            transformed_Uh = tf.matmul(Uh_non_overlap_comm, W_hg)
            sims_c = tf.matmul(transformed_Uh, tf.transpose(a=Ug_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Uh_non_overlap_comm)[1], dtype=tf.float64))

        pos_label = tf.constant(1, dtype=tf.float64)
        neg_label = tf.constant(0, dtype=tf.float64)
        if combine_axis == 0:

            concat_sim = tf.concat([sims_u, sims_c], axis=1)
            norm_concat_sim = tf.nn.softmax(concat_sim, axis=1)
            print("norm_concat_sim", norm_concat_sim)
            sharpened_concat_sim = sharpen(norm_concat_sim)
            print("sharpened_concat_sim", sharpened_concat_sim)
            concat_lbls = tf.concat([Yg_overlap, Yg_all], axis=0)
            soft_lbls = tf.matmul(sharpened_concat_sim, concat_lbls)
            print("soft_lbls:", soft_lbls)
            hard_labels = tf.map_fn(lambda lbl: tf.cond(pred=lbl[0] > 0.5, true_fn=lambda: pos_label, false_fn=lambda: neg_label), soft_lbls,
                                    parallel_iterations=parallel_iterations)

            hard_labels = tf.expand_dims(hard_labels, axis=1)
            print("hard_labels:", hard_labels)

            concat_org = tf.concat([Ug_overlap_uniq, Ug_all_comm], axis=0)
            compressed_reprs = tf.matmul(sharpened_concat_sim, concat_org)
            labeled_compressed_reprs = tf.concat([compressed_reprs, hard_labels], axis=1)
            return labeled_compressed_reprs

        else:
            norm_sims_u = tf.nn.softmax(sims_u, axis=1)
            if sharpen_tempature is not None:
                norm_sims_u = sharpen(norm_sims_u, tempature=sharpen_tempature)
            uniq_reprs = tf.matmul(norm_sims_u, Ug_overlap_uniq)
            uniq_lbls = tf.matmul(norm_sims_u, Yg_overlap)

            norm_sims_c = tf.nn.softmax(sims_c, axis=1)
            if sharpen_tempature is not None:
                norm_sims_c = sharpen(norm_sims_c, tempature=sharpen_tempature)
            comm_reprs = tf.matmul(norm_sims_c, Ug_all_comm)
            comm_lbls = tf.matmul(norm_sims_c, Yg_all)

            combine_soft_lbls = 0.5 * uniq_lbls + 0.5 * comm_lbls

            # hard_labels = tf.map_fn(lambda lbl: tf.cond(lbl[0] > 0.5, lambda: pos_label, lambda: neg_label),
            #                         combine_soft_lbls, parallel_iterations=parallel_iterations)
            # labels = tf.expand_dims(hard_labels, axis=1)
            labels = combine_soft_lbls
            print("labels:", labels)
            # labeled_compressed_reprs = tf.concat([uniq_reprs, comm_reprs, hard_labels], axis=1)
            labeled_compressed_reprs = tf.concat([uniq_reprs, comm_reprs, labels], axis=1)

            return labeled_compressed_reprs, uniq_lbls, comm_lbls

    @staticmethod
    def estimate_lables_for_host_overlap_comm(Uh_overlap_comm, Ug_all_comm, Yg_all, W_hg=None):
        if W_hg is None:
            sims_overlap_c = tf.matmul(Uh_overlap_comm, tf.transpose(a=Ug_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Uh_overlap_comm)[1], dtype=tf.float64))
        else:
            transformed_Uh_2 = tf.matmul(Uh_overlap_comm, W_hg)
            sims_overlap_c = tf.matmul(transformed_Uh_2, tf.transpose(a=Ug_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Uh_overlap_comm)[1], dtype=tf.float64))

        norm_sims_overlap_c = tf.nn.softmax(sims_overlap_c, axis=1)
        sharpened_sims_overlap_c = sharpen(norm_sims_overlap_c, tempature=0.1)
        print("# Yg_all:", Yg_all)
        overlap_comm_lbls = tf.matmul(sharpened_sims_overlap_c, Yg_all)
        # overlap_comm_lbls = tf.matmul(sims_overlap_c, Yg_all) / tf.cast(tf.shape(Yg_all)[0], tf.float64)
        return overlap_comm_lbls

    @staticmethod
    def estimate_lbls_for_host_overlap(Uh_comm,
                                       Uh_uniq,
                                       Uh_overlap_uniq,
                                       Ug_overlap_uniq,
                                       Ug_all_comm,
                                       Yg_overlap,
                                       Yg_all,
                                       W_hg=None,
                                       sharpen_tempature=0.1):

        sims_u = tf.matmul(Uh_uniq, tf.transpose(a=Uh_overlap_uniq)) / tf.sqrt(
            tf.cast(tf.shape(input=Uh_uniq)[1], dtype=tf.float64))
        if W_hg is None:
            sims_c = tf.matmul(Uh_comm, tf.transpose(a=Ug_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Uh_comm)[1], dtype=tf.float64))
        else:
            transformed_Uh = tf.matmul(Uh_comm, W_hg)
            sims_c = tf.matmul(transformed_Uh, tf.transpose(a=Ug_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Uh_comm)[1], dtype=tf.float64))

        norm_sims_c = tf.nn.softmax(sims_c, axis=1)
        if sharpen_tempature is not None:
            norm_sims_c = sharpen(norm_sims_c, tempature=sharpen_tempature)
        comm_reprs = tf.matmul(norm_sims_c, Ug_all_comm)
        comm_lbls = tf.matmul(norm_sims_c, Yg_all)

        norm_sims_u = tf.nn.softmax(sims_u, axis=1)
        if sharpen_tempature is not None:
            norm_sims_u = sharpen(norm_sims_u, tempature=sharpen_tempature)
        uniq_reprs = tf.matmul(norm_sims_u, Ug_overlap_uniq)
        uniq_lbls = tf.matmul(norm_sims_u, Yg_overlap)

        reprs = tf.concat([uniq_reprs, comm_reprs], axis=1)
        combine_soft_lbls = 0.5 * uniq_lbls + 0.5 * comm_lbls
        return combine_soft_lbls, reprs

    def select_reprs(self, reprs_w_condidate_labels):

        pos_label = tf.constant(1, dtype=tf.float64)
        neg_label = tf.constant(0, dtype=tf.float64)
        # repr_list = []

        dynamic_array = tensor_array_ops.TensorArray(
            dtype=tf.float64,
            size=0,
            dynamic_size=True,
            clear_after_read=False)

        # i = tf.constant(0)
        # repr_list = tf.Variable([])
        # repr_list = tf.Variable(initial_value=[], dtype=tf.float64)

        def cond(i, j, row):
            return j < tf.shape(input=reprs_w_condidate_labels)[0]

        def body(i, j, row):
            condidate_lbl_1 = reprs_w_condidate_labels[j, -1]
            condidate_lbl_2 = reprs_w_condidate_labels[j, -2]

            comb_lbl = (condidate_lbl_1 + condidate_lbl_2) / 2

            to_gather = tf.math.logical_or(tf.math.logical_and(tf.math.greater(condidate_lbl_1, 0.85),
                                                               tf.math.greater(condidate_lbl_2, 0.85)),
                                           tf.math.logical_and(tf.math.less_equal(condidate_lbl_1, 0.15),
                                                               tf.math.less_equal(condidate_lbl_2, 0.15)))

            print("comb_lbl:", comb_lbl)
            print("to_gather:", to_gather)

            def f1():
                temp = tf.expand_dims(reprs_w_condidate_labels[i, :], axis=0)
                print("temp:", temp)
                print("reprs", row)
                row_update = row.write(i, temp)
                return i + 1, j+1, row_update

            def f2():
                return i, j+1, row

            i, j, row_update = tf.cond(pred=to_gather, true_fn=f1, false_fn=f2)
            return [i, j, row_update]

        index, index_2, list_vals = tf.while_loop(cond=cond, body=body, loop_vars=[0, 0, dynamic_array])

        reprs_w_labels = list_vals.concat()
        reprs = reprs_w_labels[:, :-2]
        ave_labels = (reprs_w_labels[:, -1] + reprs_w_labels[:, -2]) / 2
        hard_labels = tf.map_fn(lambda lbl: tf.cond(pred=lbl > 0.5, true_fn=lambda: pos_label, false_fn=lambda: neg_label),
                                ave_labels, parallel_iterations=300)
        return reprs, tf.expand_dims(hard_labels, axis=1)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
