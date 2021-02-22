import unittest

import numpy as np
import tensorflow as tf

from vertical_sstl_representation_learner import AttentionBasedRepresentationEstimator
from expanding_vertical_transfer_learning_ops_whitebox_test import assert_arrays


def softmax(x):
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    return e_x / np.expand_dims(e_x.sum(axis=1), axis=1)


class Test_Feature_Extraction(unittest.TestCase):

    def test_estimate_host_representations_with_combine_axis_1_for_guest(self):
        print("------ test_estimate_host_representations_with_combine_axis_1_for_guest ------")

        Ug_non_overlap_uniq = np.array([[0.1, 0.2, 0.3, 0.4],
                                        [0.3, 0.5, 0.6, 0.6]])

        Ug_non_overlap_comm = np.array([[0.1, 0.2, 0.3, 1.4],
                                        [1.1, 1.2, 0.3, 1.4]])

        Ug_overlap_uniq = np.array([[0.4, 0.6, 0.7, 0.8],
                                    [0.2, 0.3, 0.4, 0.4],
                                    [0.2, 0.2, 0.3, 0.4]])

        Uh_overlap_uniq = np.array([[1.4, 1.6, 1.7, 1.8],
                                    [1.2, 1.3, 1.4, 1.4],
                                    [0.2, 0.2, 0.3, 0.4]])

        Uh_all_comm = np.array([[1.0, 1.2, 1.3, 1.4],
                                [0.2, 0.2, 0.3, 0.4],
                                [0.2, 0.2, 0.3, 0.5],
                                [0.1, 0.2, 0.3, 1.4]])

        sim_u = np.matmul(Ug_non_overlap_uniq, np.transpose(Ug_overlap_uniq)) / np.sqrt(Ug_non_overlap_uniq.shape[1])
        sim_c = np.matmul(Ug_non_overlap_comm, np.transpose(Uh_all_comm)) / np.sqrt(Ug_non_overlap_comm.shape[1])

        print("sim_u:", sim_u)
        print("sim_c:", sim_c)

        # calculate expected reprs
        norm_sim_u = softmax(sim_u)
        norm_sim_c = softmax(sim_c)

        print("norm_sim_u:", norm_sim_u)
        print("norm_sim_c:", norm_sim_c)

        uniq_reprs = np.matmul(norm_sim_u, Uh_overlap_uniq)
        comm_reprs = np.matmul(norm_sim_c, Uh_all_comm)
        expected_compressed_reprs = np.concatenate([uniq_reprs, comm_reprs], axis=1)

        # calculate actual reprs
        tf_Ug_non_overlap_uniq = tf.convert_to_tensor(value=Ug_non_overlap_uniq, dtype=tf.float64)
        tf_Ug_non_overlap_comm = tf.convert_to_tensor(value=Ug_non_overlap_comm, dtype=tf.float64)
        tf_Ug_overlap_uniq = tf.convert_to_tensor(value=Ug_overlap_uniq, dtype=tf.float64)
        tf_Uh_overlap_uniq = tf.convert_to_tensor(value=Uh_overlap_uniq, dtype=tf.float64)
        tf_Uh_all_comm = tf.convert_to_tensor(value=Uh_all_comm, dtype=tf.float64)

        repr_learner = AttentionBasedRepresentationEstimator()
        reprs = repr_learner.estimate_host_reprs_for_guest_party(tf_Ug_non_overlap_comm,
                                                                 tf_Ug_non_overlap_uniq,
                                                                 tf_Ug_overlap_uniq,
                                                                 tf_Uh_overlap_uniq,
                                                                 tf_Uh_all_comm,
                                                                 sharpen_temperature=None)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            result_vals = sess.run(reprs)
            print("result_vals:\n", result_vals)
            print("expected_compressed_reprs:\n", expected_compressed_reprs)
            assert_arrays(expected_compressed_reprs, result_vals)

    def test_estimate_guest_representations_with_combine_axis_1_for_host(self):
        print("------ test_estimate_guest_representations_with_combine_axis_1_for_host ------")

        # tf.enable_eager_execution()
        Uh_non_overlap_uniq = np.array([[0.1, 0.2, 0.3, 0.4],
                                        [0.3, 0.5, 0.6, 0.6]])

        Uh_non_overlap_comm = np.array([[0.1, 0.2, 0.3, 1.4],
                                        [1.1, 1.2, 0.3, 1.4]])

        Uh_overlap_uniq = np.array([[0.4, 0.6, 0.7, 0.8],
                                    [0.2, 0.3, 0.4, 0.4],
                                    [0.2, 0.2, 0.3, 0.4]])

        Ug_overlap_uniq = np.array([[1.4, 1.6, 1.7, 1.8],
                                    [1.2, 1.3, 1.4, 1.4],
                                    [0.2, 0.2, 0.3, 0.4]])

        Ug_all_comm = np.array([[1.0, 1.2, 1.3, 1.4],
                                [0.2, 0.2, 0.3, 0.4],
                                [0.2, 0.2, 0.3, 0.5],
                                [0.1, 0.2, 0.3, 1.4]])

        Yg_overlap = np.array([[0],
                               [0],
                               [1]])

        Yg_all = np.array([[1],
                           [0],
                           [0],
                           [1]])

        sim_u = np.matmul(Uh_non_overlap_uniq, np.transpose(Uh_overlap_uniq)) / np.sqrt(Uh_non_overlap_uniq.shape[1])
        sim_c = np.matmul(Uh_non_overlap_comm, np.transpose(Ug_all_comm)) / np.sqrt(Uh_non_overlap_comm.shape[1])

        print("sim_u:", sim_u)
        print("sim_c:", sim_c)

        # calculate expected reprs
        norm_sim_u = softmax(sim_u)
        norm_sim_c = softmax(sim_c)

        print("norm_sim_u:", norm_sim_u)
        print("norm_sim_c:", norm_sim_c)

        uniq_reprs = np.matmul(norm_sim_u, Ug_overlap_uniq)
        comm_reprs = np.matmul(norm_sim_c, Ug_all_comm)
        expected_compressed_reprs = np.concatenate([uniq_reprs, comm_reprs], axis=1)

        uniq_lbls = np.matmul(norm_sim_u, Yg_overlap)
        comm_lbls = np.matmul(norm_sim_c, Yg_all)

        expected_combine_soft_lbls = 0.5 * uniq_lbls + 0.5 * comm_lbls

        # calculate actual reprs
        tf_Uh_non_overlap_uniq = tf.convert_to_tensor(value=Uh_non_overlap_uniq, dtype=tf.float64)
        tf_Uh_non_overlap_comm = tf.convert_to_tensor(value=Uh_non_overlap_comm, dtype=tf.float64)
        tf_Uh_overlap_uniq = tf.convert_to_tensor(value=Uh_overlap_uniq, dtype=tf.float64)
        tf_Ug_overlap_uniq = tf.convert_to_tensor(value=Ug_overlap_uniq, dtype=tf.float64)
        tf_Ug_all_comm = tf.convert_to_tensor(value=Ug_all_comm, dtype=tf.float64)
        tf_Yg_overlap = tf.convert_to_tensor(value=Yg_overlap, dtype=tf.float64)
        tf_Yg_all = tf.convert_to_tensor(value=Yg_all, dtype=tf.float64)

        repr_learner = AttentionBasedRepresentationEstimator()
        reprs_with_lbls, _, _ = repr_learner.estimate_labeled_guest_reprs_for_host_party(tf_Uh_non_overlap_comm,
                                                                                         tf_Uh_non_overlap_uniq,
                                                                                         tf_Uh_overlap_uniq,
                                                                                         tf_Ug_overlap_uniq,
                                                                                         tf_Ug_all_comm,
                                                                                         tf_Yg_overlap,
                                                                                         tf_Yg_all,
                                                                                         sharpen_tempature=None)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            reprs_with_lbls = sess.run(reprs_with_lbls)
            print("reprs_with_lbls:\n", reprs_with_lbls)
            print("expected_compressed_reprs:\n", expected_compressed_reprs)
            print("expected_combine_soft_lbls:\n", expected_combine_soft_lbls)
            assert_arrays(reprs_with_lbls[:, :-1], expected_compressed_reprs)
            assert_arrays(reprs_with_lbls[:, -1], np.squeeze(expected_combine_soft_lbls))

    def test_select_reprs_w_biclass_for_host(self):
        # tf.enable_eager_execution()
        repr_learner = AttentionBasedRepresentationEstimator()

        Ug_overlap_uniq = np.array([[1.4, 1.6, 1.7, 1.8, 0.6, 0.7],
                                    [0.2, 0.2, 0.3, 0.4, 0.1, 0.9],
                                    [0.9, 0.2, 0.7, 0.4, 0.4, 0.7],
                                    [1.2, 1.3, 1.4, 1.4, 0.4, 0.2]])
        tf_reprs_w_labels = tf.convert_to_tensor(value=Ug_overlap_uniq, dtype=tf.float64)
        reprs, labels = repr_learner.select_reprs_for_biclass(tf_reprs_w_labels, upper_bound=0.5, lower_bound=0.5)

        print("reprs:", reprs)
        print("labels:", labels)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            reprs = sess.run(reprs)
            labels = sess.run(labels)
            print("reprs:\n", reprs, reprs.shape)
            print("labels:\n", labels, labels.shape)

    def test_select_reprs_w_multiclass_for_host(self):
        tf.compat.v1.enable_eager_execution()
        repr_learner = AttentionBasedRepresentationEstimator()
        n_class = 3

        # Ug_overlap_uniq = np.array([[1.4, 1.6, 1.7, 1.8, 0.1, 0.7, 0.2, 0.3, 0.6, 0.1],
        #                             [0.2, 0.2, 0.3, 0.4, 0.1, 0.8, 0.1, 0.7, 0.2, 0.1],
        #                             [0.9, 0.2, 0.7, 0.4, 0.2, 0.6, 0.2, 0.1, 0.3, 0.6],
        #                             [1.2, 1.3, 1.4, 1.4, 0.6, 0.1, 0.3, 0.7, 0.2, 0.1],
        #                             [1.5, 1.1, 1.2, 1.1, 0.1, 0.1, 0.8, 0.2, 0.2, 0.6]])

        # Ug_overlap_uniq = np.array([[5.7445389e-01, 9.9998730e-01,
        #                             5.1113117e-01, 4.3606196e-04, 4.8843288e-01, 4.1152945e-01,
        #                             2.8101736e-01, 3.0745322e-01]])

        Ug_overlap_uniq = np.array([[-0.31627315, -0.98935145, -0.7872633, 1., 0.35779756, 0.10964867,
                                     0.5325539, 0.39274016, 0.21056582, 0.39669403]])

        tf_reprs_w_labels = tf.convert_to_tensor(value=Ug_overlap_uniq, dtype=tf.float32)
        dynamic_array = repr_learner.select_reprs_for_multiclass(reprs_w_condidate_labels=tf_reprs_w_labels,
                                                                 n_class=n_class,
                                                                 fed_label_upper_bound=0.3)

        reprs_w_labels = dynamic_array.concat()
        reprs = reprs_w_labels[:, :-n_class]
        labels = reprs_w_labels[:, -n_class:]

        print("reprs:", reprs)
        print("labels:", labels)

        # init = tf.global_variables_initializer()
        # with tf.Session() as sess:
        #     sess.run(init)
        #     reprs = sess.run(reprs)
        #     labels = sess.run(labels)
        #     print("reprs:\n", reprs, reprs.shape)
        #     print("labels:\n", labels, labels.shape)


if __name__ == '__main__':
    unittest.main()
