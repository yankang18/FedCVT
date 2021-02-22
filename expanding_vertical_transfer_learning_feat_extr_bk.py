import tensorflow as tf

from expanding_vertical_transfer_learning_ops import tf_euclidean_distance, tf_gather_with_dynamic_partition


def convert_to_similarty(distance):
    return 1 / (1 + distance)
    # return 1 / tf.math.exp(distance)
    # return - distance


def compute_similarity(query_vector, values_matrix):
    return tf.reduce_sum(input_tensor=tf.multiply(query_vector, values_matrix), axis=1) / tf.cast(tf.shape(input=values_matrix)[1],
                                                                                     dtype=tf.float64)


def do_pick_top_k_samples(sample_uniq, sample_comm, U_overlap_uniq, U_the_other_all_comm, top_k):
    # # this distance can be computed privately at local
    # distances_u = tf_euclidean_distance(sample_uniq, U_overlap_uniq)
    # # this distance should be computed through secret sharing
    # distances_c = tf_euclidean_distance(sample_comm, U_the_other_all_comm)
    #
    # sim_u = convert_to_similarty(distances_u)
    # sim_c = convert_to_similarty(distances_c)
    sim_u = compute_similarity(sample_uniq, U_overlap_uniq)
    sim_c = compute_similarity(sample_comm, U_the_other_all_comm)

    scores_u, indices_u = tf.nn.top_k(input=sim_u, k=top_k)
    scores_c, indices_c = tf.nn.top_k(input=sim_c, k=top_k)

    return scores_u, indices_u, scores_c, indices_c


def estimate_host_representations_for_guest_party(Ug_non_overlap_comm,
                                                  Ug_non_overlap_uniq,
                                                  Ug_overlap_uniq,
                                                  Uh_overlap_uniq,
                                                  Uh_all_comm,
                                                  k=2,
                                                  combine_axis=0,
                                                  parallel_iterations=10):
    def do_estimate_representation(Us):
        print("---- iter do estimate host ested_repr ----")
        Us = tf.expand_dims(Us, axis=0)
        print("Us shape", Us.shape)

        sample_u = Us[:, :Ug_non_overlap_uniq.shape[1]]
        sample_c = Us[:, Ug_non_overlap_uniq.shape[1]:]

        print("sample_u shape", sample_u)
        print("sample_c shape", sample_c)

        scores_u, indices_u, scores_c, indices_c = do_pick_top_k_samples(sample_uniq=sample_u,
                                                                         sample_comm=sample_c,
                                                                         U_overlap_uniq=Ug_overlap_uniq,
                                                                         U_the_other_all_comm=Uh_all_comm,
                                                                         top_k=k)

        print("indices_u shape", indices_u)
        print("indices_c shape", indices_c)

        if k > 1:
            print("k > 1")

            indices_u_ordered = tf.argsort(indices_u)
            indices_c_ordered = tf.argsort(indices_c)

            scores_u = tf.gather(scores_u, indices_u_ordered)
            scores_c = tf.gather(scores_c, indices_c_ordered)

            gathered_u = tf_gather_with_dynamic_partition(Uh_overlap_uniq, indices_u)
            gathered_c = tf_gather_with_dynamic_partition(Uh_all_comm, indices_c)

        elif k == 1:
            print("k = 1")
            print("Uh_overlap_uniq:", Uh_overlap_uniq)
            print("indices_u:", indices_u)
            print("indices_u[0]", indices_u[0])
            gathered_u = tf.expand_dims(Uh_overlap_uniq[indices_u[0]], axis=0)
            gathered_c = tf.expand_dims(Uh_all_comm[indices_c[0]], axis=0)
        else:
            raise ValueError("k should be above 1")

        print("gathered_u shape", gathered_u)
        print("gathered_c shape", gathered_c)

        print(" host combine_axis", combine_axis)
        if combine_axis == 0:
            # t1 = [[1, 2, 3], [4, 5, 6]]
            # t2 = [[7, 8, 9], [10, 11, 12]]
            # tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
            # tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
            selected_reprs = tf.concat([gathered_u, gathered_c], axis=0)
            score_vals = tf.concat([scores_u, scores_c], axis=0)
            # score_vals = tf.concat([scores_u_ordered, scores_c_ordered], axis=0)
            # print("selected_reprs shape", selected_reprs.shape)
            # print("score_vals shape", score_vals.shape)

            print("selected_reprs", selected_reprs)
            print("score_vals", score_vals)

            norm_score_vals = tf.nn.softmax(score_vals)
            # norm_score_vals = score_vals
            print("norm_score_vals", norm_score_vals)

            # selected_reprs should have shape (k, repr_dim)
            # norm_score_vals should have shape (k,)
            ested_repr = tf.reduce_sum(input_tensor=tf.multiply(selected_reprs, tf.expand_dims(norm_score_vals, axis=1)), axis=0)

        else:
            norm_scores_u = tf.nn.softmax(scores_u)
            norm_scores_c = tf.nn.softmax(scores_c)
            # norm_scores_u = tf.nn.softmax(scores_u_ordered)
            # norm_scores_c = tf.nn.softmax(scores_c_ordered)
            print("norm_scores_u:", norm_scores_u)
            print("norm_scores_c:", norm_scores_c)
            repr_u = tf.reduce_sum(input_tensor=tf.multiply(gathered_u, tf.expand_dims(norm_scores_u, axis=1)), axis=0)
            repr_c = tf.reduce_sum(input_tensor=tf.multiply(gathered_c, tf.expand_dims(norm_scores_c, axis=1)), axis=0)

            ested_repr = tf.concat([repr_u, repr_c], axis=0)

            print("ested_repr:", ested_repr)

        return ested_repr

    print("U_non_overlap_uniq shape", Ug_non_overlap_uniq.shape)
    print("U_non_overlap_comm shape", Ug_non_overlap_comm.shape)

    stacked = tf.concat([Ug_non_overlap_uniq, Ug_non_overlap_comm], axis=1)
    print("stacked shape", stacked.shape)
    return tf.map_fn(fn=do_estimate_representation, elems=stacked, parallel_iterations=parallel_iterations)


def estimate_guest_representations_for_host_party(Uh_non_overlap_comm,
                                                  Uh_non_overlap_uniq,
                                                  Uh_overlap_uniq,
                                                  Ug_overlap_uniq,
                                                  Ug_all_comm,
                                                  Yg_overlap,
                                                  Yg_all,
                                                  k=2,
                                                  combine_axis=0,
                                                  parallel_iterations=10):
    pos_label = tf.constant(1, dtype=tf.float64)
    neg_label = tf.constant(0, dtype=tf.float64)

    def do_estimate_representation(Us):
        print("---- iter do estimate guest repr ----")
        Us = tf.expand_dims(Us, axis=0)
        print("Us shape", Us.shape)

        sample_u = Us[:, :Uh_non_overlap_uniq.shape[1]]
        sample_c = Us[:, Uh_non_overlap_uniq.shape[1]:]
        # distances_u = compute_distances(sample_u, U_overlap_uniq)
        # distances_c = compute_distances(sample_c, U_the_other_all_comm)
        print("sample_u shape", sample_u)
        print("sample_c shape", sample_c)

        scores_u, indices_u, scores_c, indices_c = do_pick_top_k_samples(sample_u, sample_c, Uh_overlap_uniq,
                                                                         Ug_all_comm, k)

        if k > 1:
            indices_u_ordered = tf.argsort(indices_u)
            indices_c_ordered = tf.argsort(indices_c)

            scores_u = tf.gather(scores_u, indices_u_ordered)
            scores_c = tf.gather(scores_c, indices_c_ordered)

            print("scores_u shape", scores_u)
            print("scores_c shape", scores_c)
            # print("scores_u_ordered shape", scores_u_ordered)
            # print("scores_c_ordered shape", scores_c_ordered)
            print("indices_u shape", indices_u)
            print("indices_c shape", indices_c)

            gathered_u = tf_gather_with_dynamic_partition(Ug_overlap_uniq, indices_u)
            gathered_c = tf_gather_with_dynamic_partition(Ug_all_comm, indices_c)
            gathered_u_lbls = tf_gather_with_dynamic_partition(Yg_overlap, indices_u)
            gathered_c_lbls = tf_gather_with_dynamic_partition(Yg_all, indices_c)

        elif k == 1:
            print("guest k = 1")
            print("indices_u[0]:", indices_u[0])
            gathered_u = tf.expand_dims(Ug_overlap_uniq[indices_u[0]], axis=0)
            gathered_c = tf.expand_dims(Ug_all_comm[indices_c[0]], axis=0)
            gathered_u_lbls = tf.expand_dims(Yg_overlap[indices_u[0]], axis=0)
            gathered_c_lbls = tf.expand_dims(Yg_all[indices_c[0]], axis=0)
        else:
            raise ValueError("k should be above 1")

        print("Yg_overlap:", Yg_overlap)
        print("Yg_all:", Yg_all)

        print("gathered_u shape", gathered_u)
        print("gathered_c shape", gathered_c)

        print("gathered_u_lbls shape", gathered_u_lbls.shape)
        print("gathered_c_lbls shape", gathered_c_lbls.shape)

        selected_reprs_lbls = tf.concat([gathered_u_lbls, gathered_c_lbls], axis=0)
        score_vals = tf.concat([scores_u, scores_c], axis=0)
        # score_vals = tf.concat([scores_u_ordered, scores_c_ordered], axis=0)
        print("selected_reprs_lbls", selected_reprs_lbls)
        print("score_vals", score_vals)

        norm_score_vals = tf.nn.softmax(score_vals)
        print("norm_score_vals", norm_score_vals)
        norm_score_vals_exp = tf.expand_dims(norm_score_vals, axis=1)
        print("norm_score_vals_exp", norm_score_vals_exp)

        ested_soft_lbl = tf.reduce_sum(input_tensor=tf.multiply(selected_reprs_lbls, norm_score_vals_exp))
        print("ested_soft_lbl", ested_soft_lbl)

        ested_lbl = tf.expand_dims(tf.cond(pred=ested_soft_lbl > 0.5, true_fn=lambda: pos_label, false_fn=lambda: neg_label), axis=0)
        print("ested_lbl", ested_lbl)

        print(" guest combine_axis", combine_axis)
        if combine_axis == 0:

            selected_reprs = tf.concat([gathered_u, gathered_c], axis=0)
            print("selected_reprs", selected_reprs)

            ested_repr = tf.reduce_sum(input_tensor=tf.multiply(selected_reprs, norm_score_vals_exp), axis=0)
            print("ested_repr", ested_repr)

            stacked = tf.concat([ested_repr, ested_lbl], axis=0)
            print("stacked", stacked)

        else:

            norm_scores_u = tf.nn.softmax(scores_u)
            norm_scores_c = tf.nn.softmax(scores_c)
            # norm_scores_u = tf.nn.softmax(scores_u_ordered)
            # norm_scores_c = tf.nn.softmax(scores_c_ordered)
            print("norm_scores_u:", norm_scores_u)
            print("norm_scores_c:", norm_scores_c)

            repr_u = tf.reduce_sum(input_tensor=tf.multiply(gathered_u, tf.expand_dims(norm_scores_u, axis=1)), axis=0)
            repr_c = tf.reduce_sum(input_tensor=tf.multiply(gathered_c, tf.expand_dims(norm_scores_c, axis=1)), axis=0)

            ested_repr = tf.concat([repr_u, repr_c], axis=0)
            print("ested_repr", ested_repr)

            stacked = tf.concat([ested_repr, ested_lbl], axis=0)
            print("stacked", stacked)

        return stacked

    print("U_non_overlap_uniq shape", Uh_non_overlap_uniq.shape)
    print("U_non_overlap_comm shape", Uh_non_overlap_comm.shape)

    stacked = tf.concat([Uh_non_overlap_uniq, Uh_non_overlap_comm], axis=1)
    print("stacked shape", stacked.shape)
    return tf.map_fn(fn=do_estimate_representation, elems=stacked, parallel_iterations=parallel_iterations)
