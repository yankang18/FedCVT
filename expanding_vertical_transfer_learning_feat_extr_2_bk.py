import tensorflow as tf


def sharpen(p, T=0.1):
    u = tf.math.pow(p, 1 / T)
    return u / tf.math.reduce_sum(input_tensor=u, axis=1, keepdims=True)


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
            compressed_reprs = tf.matmul(sharpened_concat_sim, concat_org)
            return compressed_reprs

        else:
            norm_sims_u = tf.nn.softmax(sims_u, axis=1)
            sharpened_sims_u = sharpen(norm_sims_u)
            uniq_reprs = tf.matmul(sharpened_sims_u, Uh_overlap_uniq)
            norm_sims_c = tf.nn.softmax(sims_c, axis=1)
            sharpened_sims_c = sharpen(norm_sims_c)
            comm_reprs = tf.matmul(sharpened_sims_c, Uh_all_comm)
            compressed_reprs = tf.concat([uniq_reprs, comm_reprs], axis=1)
            return compressed_reprs

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
            # transformed_Ug = tf.matmul(Ug_all_comm, tf.transpose(W_hg))
            # sims_c = tf.matmul(Uh_non_overlap_comm, tf.transpose(transformed_Ug)) / tf.sqrt(
            #     tf.cast(tf.shape(Uh_non_overlap_comm)[1], dtype=tf.float64))

        # concat_sim = tf.concat([sims_u, sims_c], axis=1)
        # norm_concat_sim = tf.nn.softmax(concat_sim, axis=1)
        # print("norm_concat_sim", norm_concat_sim)
        # sharpened_concat_sim = sharpen(norm_concat_sim)
        # print("sharpened_concat_sim", sharpened_concat_sim)
        # concat_lbls = tf.concat([Yg_overlap, Yg_all], axis=0)
        # soft_lbls = tf.matmul(sharpened_concat_sim, concat_lbls)
        #
        # print("soft_lbls:", soft_lbls)
        # pos_label = tf.constant(1, dtype=tf.float64)
        # neg_label = tf.constant(0, dtype=tf.float64)
        # hard_labels = tf.map_fn(lambda lbl: tf.cond(lbl[0] > 0.5, lambda: pos_label, lambda: neg_label), soft_lbls,
        #                        parallel_iterations=parallel_iterations)
        #
        # hard_labels = tf.expand_dims(hard_labels, axis=1)
        # print("hard_labels:", hard_labels)

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
                norm_sims_u = sharpen(norm_sims_u, T=sharpen_tempature)
            uniq_reprs = tf.matmul(norm_sims_u, Ug_overlap_uniq)
            uniq_lbls = tf.matmul(norm_sims_u, Yg_overlap)

            norm_sims_c = tf.nn.softmax(sims_c, axis=1)
            if sharpen_tempature is not None:
                norm_sims_c = sharpen(norm_sims_c, T=sharpen_tempature)
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
        sharpened_sims_overlap_c = sharpen(norm_sims_overlap_c, T=0.1)
        print("# Yg_all:", Yg_all)
        overlap_comm_lbls = tf.matmul(sharpened_sims_overlap_c, Yg_all)
        # overlap_comm_lbls = tf.matmul(sims_overlap_c, Yg_all) / tf.cast(tf.shape(Yg_all)[0], tf.float64)
        return overlap_comm_lbls

    @staticmethod
    def estimate_lables_for_host_overlap(Uh_uniq,
                                         Uh_overlap_uniq,
                                         Uh_comm,
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
            transformed_Uh_2 = tf.matmul(Uh_comm, W_hg)
            sims_c = tf.matmul(transformed_Uh_2, tf.transpose(a=Ug_all_comm)) / tf.sqrt(
                tf.cast(tf.shape(input=Uh_comm)[1], dtype=tf.float64))

        norm_sims_c = tf.nn.softmax(sims_c, axis=1)
        if sharpen_tempature is not None:
            norm_sims_c = sharpen(norm_sims_c, T=sharpen_tempature)
        comm_lbls = tf.matmul(norm_sims_c, Yg_all)

        norm_sims_u = tf.nn.softmax(sims_u, axis=1)
        if sharpen_tempature is not None:
            norm_sims_u = sharpen(norm_sims_u, T=sharpen_tempature)
        uniq_lbls = tf.matmul(norm_sims_u, Yg_overlap)

        combine_soft_lbls = 0.5 * uniq_lbls + 0.5 * comm_lbls
        return combine_soft_lbls

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
