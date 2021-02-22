import unittest

import numpy as np
import tensorflow as tf

from expanding_vertical_transfer_learning_feat_extr import RankedRepresentationLearner, convert_to_similarty
from expanding_vertical_transfer_learning_ops_whitebox_test import softmax, assert_arrays


class Test_Feature_Extraction(unittest.TestCase):

    def test_estimate_host_representations_for_guest(self):
        print("------ test_estimate_host_representations_for_guest ------")

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

        # distances_u = np.linalg.norm(Ug_non_overlap_uniq[0] - Ug_overlap_uniq, axis=1)
        # distances_c = np.linalg.norm(Ug_non_overlap_comm[0] - Uh_all_comm, axis=1)
        # print("distances_u", distances_u)
        # print("distances_c", distances_c)

        # calculate expected reprs
        distances_u = np.linalg.norm(Ug_non_overlap_uniq[0] - Ug_overlap_uniq[1:3], axis=1)
        distances_c = np.linalg.norm(Ug_non_overlap_comm[0] - Uh_all_comm[2:4], axis=1)
        dist_list = np.concatenate([distances_u, distances_c], axis=0)
        sim_list = np.expand_dims(softmax(convert_to_similarty(dist_list)), 1)

        reprs = np.concatenate([Uh_overlap_uniq[1:3], Uh_all_comm[2:4]], axis=0)
        expected_repr_0 = np.sum(reprs * sim_list, axis=0)
        print("expected_repr_0", expected_repr_0)

        distances_u = np.linalg.norm(Ug_non_overlap_uniq[1] - Ug_overlap_uniq[0:2], axis=1)
        distances_c = np.linalg.norm(Ug_non_overlap_comm[1] - Uh_all_comm[[0, 3]], axis=1)
        dist_list = np.concatenate([distances_u, distances_c], axis=0)
        sim_list = np.expand_dims(softmax(convert_to_similarty(dist_list)), 1)

        reprs = np.concatenate([Uh_overlap_uniq[0:2], Uh_all_comm[[0, 3]]], axis=0)
        expected_repr_1 = np.sum(reprs * sim_list, axis=0)
        print("expected_repr_1", expected_repr_1)

        # calculate actual reprs
        tf_Ug_non_overlap_uniq = tf.convert_to_tensor(value=Ug_non_overlap_uniq, dtype=tf.float64)
        tf_Ug_non_overlap_comm = tf.convert_to_tensor(value=Ug_non_overlap_comm, dtype=tf.float64)
        tf_Ug_overlap_uniq = tf.convert_to_tensor(value=Ug_overlap_uniq, dtype=tf.float64)
        tf_Uh_overlap_uniq = tf.convert_to_tensor(value=Uh_overlap_uniq, dtype=tf.float64)
        tf_Uh_all_comm = tf.convert_to_tensor(value=Uh_all_comm, dtype=tf.float64)

        repr_learner = RankedRepresentationLearner()
        reprs = repr_learner.estimate_host_representations_for_guest_party(tf_Ug_non_overlap_comm,
                                                                           tf_Ug_non_overlap_uniq,
                                                                           tf_Ug_overlap_uniq,
                                                                           tf_Uh_overlap_uniq,
                                                                           tf_Uh_all_comm,
                                                                           k=2)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            result_vals = sess.run(reprs)
            print(result_vals[0], expected_repr_0)
            print(result_vals[1], expected_repr_1)
            assert_arrays(expected_repr_0, result_vals[0])
            assert_arrays(expected_repr_1, result_vals[1])

    def test_estimate_guest_representations_for_host(self):
        print("------ estimate_guest_representations_for_host ------")

        # tf.enable_eager_execution()
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

        # calculate expected reprs
        distances_u_all = np.linalg.norm(Ug_non_overlap_uniq[0] - Ug_overlap_uniq, axis=1)
        distances_c_all = np.linalg.norm(Ug_non_overlap_comm[0] - Uh_all_comm, axis=1)

        print("distances_u_all[0]:", distances_u_all)
        print("distances_c_all[0]:", distances_c_all)

        distances_u = np.linalg.norm(Ug_non_overlap_uniq[0] - Ug_overlap_uniq[1:3], axis=1)
        distances_c = np.linalg.norm(Ug_non_overlap_comm[0] - Uh_all_comm[2:4], axis=1)
        dist_list = np.concatenate([distances_u, distances_c], axis=0)
        sim_list_0 = np.expand_dims(softmax(convert_to_similarty(dist_list)), 1)
        print("sim_list_0:", sim_list_0)

        reprs = np.concatenate([Uh_overlap_uniq[1:3], Uh_all_comm[2:4]], axis=0)
        expected_repr_0 = np.sum(reprs * sim_list_0, axis=0)
        print("expected_repr_0", expected_repr_0)

        distances_u_all = np.linalg.norm(Ug_non_overlap_uniq[1] - Ug_overlap_uniq, axis=1)
        distances_c_all = np.linalg.norm(Ug_non_overlap_comm[1] - Uh_all_comm, axis=1)

        print("distances_u_all[1]:", distances_u_all)
        print("distances_c_all[1]:", distances_c_all)

        distances_u = np.linalg.norm(Ug_non_overlap_uniq[1] - Ug_overlap_uniq[0:2], axis=1)
        distances_c = np.linalg.norm(Ug_non_overlap_comm[1] - Uh_all_comm[[0, 3]], axis=1)
        dist_list = np.concatenate([distances_u, distances_c], axis=0)
        sim_list_1 = np.expand_dims(softmax(convert_to_similarty(dist_list)), 1)
        print("sim_list_1:", sim_list_1)

        reprs = np.concatenate([Uh_overlap_uniq[0:2], Uh_all_comm[[0, 3]]], axis=0)
        expected_repr_1 = np.sum(reprs * sim_list_1, axis=0)
        print("expected_repr_1", expected_repr_1)

        # calculate actual reprs
        tf_Ug_non_overlap_uniq = tf.convert_to_tensor(value=Ug_non_overlap_uniq, dtype=tf.float64)
        tf_Ug_non_overlap_comm = tf.convert_to_tensor(value=Ug_non_overlap_comm, dtype=tf.float64)
        tf_Ug_overlap_uniq = tf.convert_to_tensor(value=Ug_overlap_uniq, dtype=tf.float64)
        tf_Uh_overlap_uniq = tf.convert_to_tensor(value=Uh_overlap_uniq, dtype=tf.float64)
        tf_Uh_all_comm = tf.convert_to_tensor(value=Uh_all_comm, dtype=tf.float64)

        Yg_overlap = np.array([[0],
                               [0],
                               [1]])

        Yg_all = np.array([[1],
                           [0],
                           [0],
                           [1]])

        tf_Yg_overlap = tf.convert_to_tensor(value=Yg_overlap, dtype=tf.float64)
        tf_Yg_all = tf.convert_to_tensor(value=Yg_all, dtype=tf.float64)

        lbls = np.concatenate([Yg_overlap[1:3], Yg_all[2:4]], axis=0)
        expected_soft_lbl_0 = np.sum(lbls * sim_list_0)
        expected_lbl_0 = 1 if expected_soft_lbl_0 > 0.5 else 0
        lbls = np.concatenate([Yg_overlap[0:2], Yg_all[[0, 3]]], axis=0)
        expected_soft_lbl_1 = np.sum(lbls * sim_list_1)
        expected_lbl_1 = 1 if expected_soft_lbl_1 > 0.5 else 0

        # reprs_with_lbls = estimate_guest_representations_for_host_party(Uh_non_overlap_comm,
        #                                                                                   Uh_non_overlap_uniq,
        #                                                                                   Uh_overlap_uniq,
        #                                                                                   Ug_overlap_uniq,
        #                                                                                   Ug_all_comm,
        #                                                                                   Y_overlap,
        #                                                                                   Y_all)

        repr_learner = RankedRepresentationLearner()
        reprs_with_lbls = repr_learner.estimate_guest_representations_for_host_party(tf_Ug_non_overlap_comm,
                                                                                     tf_Ug_non_overlap_uniq,
                                                                                     tf_Ug_overlap_uniq,
                                                                                     tf_Uh_overlap_uniq,
                                                                                     tf_Uh_all_comm,
                                                                                     tf_Yg_overlap,
                                                                                     tf_Yg_all,
                                                                                     k=2)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            reprs_with_lbls = sess.run(reprs_with_lbls)
            print(reprs_with_lbls[0][-1], expected_lbl_0)
            print(reprs_with_lbls[1][-1], expected_lbl_1)
            print(reprs_with_lbls[0][:-1], expected_repr_0)
            print(reprs_with_lbls[1][:-1], expected_repr_1)
            assert reprs_with_lbls[0][-1] == expected_lbl_0
            assert reprs_with_lbls[1][-1] == expected_lbl_1
            assert_arrays(reprs_with_lbls[0][:-1], expected_repr_0)
            assert_arrays(reprs_with_lbls[1][:-1], expected_repr_1)

    # def test_tf_argsort(self):
    #
    #     tf.enable_eager_execution()
    #     # tf.convert_to_tensor(
    #     # array1 = np.array([3, 0])
    #     # array2 = np.array([1, 2])
    #
    #     array1 = np.array([3, 0, 1])
    #     array2 = np.array([1, 3, 2])
    #
    #     array1_v = tf.argsort(array1)
    #     array2_v = tf.argsort(array2)
    #
    #     print(array1_v)
    #     print(array2_v)
    #
    #     print(tf.gather(array1, indices=array1_v))
    #     print(tf.gather(array2, indices=array2_v))





if __name__ == '__main__':
    unittest.main()
