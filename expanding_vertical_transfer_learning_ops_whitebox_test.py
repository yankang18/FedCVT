import unittest

import numpy as np
import tensorflow as tf

from expanding_vertical_transfer_learning_ops import tf_gather_with_dynamic_partition
from expanding_vertical_transfer_learning_feat_extr import do_pick_top_k_samples


def assert_arrays(a1, a2):
    assert a1.shape == a2.shape
    a1 = a1.flatten()
    a2 = a2.flatten()
    for i, j in zip(a1, a2):
        # print(i, j)
        assert round(i, 8) == round(j, 8)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Test_Expanding_VFTL_OPS(unittest.TestCase):

    # def test_tf_cosine_sim(self):
    #     matrix_0 = tf.convert_to_tensor([[0.1, 0.2, 0.3, 0.4],
    #                                      [0.1, 0.6, 0.3, 0.4],
    #                                      [0.1, -0.6, 0.3, 0.4],
    #                                      [0.1, 0.2, 0.3, 0.4]], dtype=tf.float64)
    #
    #     matrix_1 = tf.convert_to_tensor([[0.1, 0.2, 0.3, 0.4],
    #                                      [0.1, 0.6, 0.3, 0.4],
    #                                      [0.1, -0.6, 0.3, 0.4],
    #                                      [0.1, 0.2, 0.3, 0.4]], dtype=tf.float64)
    #
    #     expected = np.array([1., 1., 1.])
    #     cosine_sim = tf_cosine_sim(matrix_0, matrix_1)
    #     with tf.Session() as sess:
    #         result = sess.run(cosine_sim)
    #         print("cosine_sim result", result)
    #         for expected, actual in zip(expected, result):
    #             assert expected == round(actual)
    #
    # def test_tf_normalize(self):
    #     vector_1 = tf.convert_to_tensor([4.2, 2.4, 1.6, 2.8], dtype=tf.float64)
    #     vector_2 = tf.convert_to_tensor([-0.32, -0.4], dtype=tf.float64)
    #     normalize_1 = tf_1_normalize(vector_1)
    #     normalize_2 = tf.nn.softmax(vector_2)
    #     with tf.Session() as sess:
    #         result_1 = sess.run(normalize_1)
    #         result_2 = sess.run(normalize_2)
    #         print("normalize_1 result", result_1)
    #         print("normalize_2 result", result_2)
    #         assert sum(result_1) == 1
    #         assert sum(result_2) == 1

    def test_do_pick_samples(self):
        print("------ test_do_pick_samples ------")

        exptected_uniq_top_k_index = np.array([1, 2])
        exptected_comm_top_k_index = np.array([2, 3])

        sample_u = tf.convert_to_tensor(value=[[0.1, 0.2, 0.3, 0.4]], dtype=tf.float64)
        Ug_overlap_uniq = tf.convert_to_tensor(value=[[0.4, 0.6, 0.7, 0.8],
                                                [0.2, 0.2, 0.3, 0.4],
                                                [0.1, 0.3, 0.4, 0.4]], dtype=tf.float64)

        sample_c = tf.convert_to_tensor(value=[[0.1, 0.2, 0.3, 1.4]], dtype=tf.float64)
        Uh_all_comm = tf.convert_to_tensor(value=[[1.0, 1.2, 1.3, 1.4],
                                            [0.2, 0.2, 0.3, 0.4],
                                            [0.2, 0.2, 0.4, 0.7],
                                            [0.1, 0.2, 0.3, 1.4]], dtype=tf.float64)

        result = do_pick_top_k_samples(sample_u, sample_c, Ug_overlap_uniq, Uh_all_comm, top_k=2)

        values_u, indices_u, values_c, indices_c = result
        indices_u_ordered = tf.argsort(indices_u)
        indices_c_ordered = tf.argsort(indices_c)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            result_vals, indices_u_ordered_val, indices_c_ordered_val = sess.run([result, indices_u_ordered, indices_c_ordered])
            values_u, indices_u, values_c, indices_c = result_vals
            print("values_u", values_u)
            print("indices_u", indices_u)
            print("values_c", values_c)
            print("indices_c", indices_c)
            print("indices_u_ordered_val", indices_u_ordered_val)
            print("indices_c_ordered_val", indices_c_ordered_val)
            assert_arrays(exptected_uniq_top_k_index, np.sort(indices_u))
            assert_arrays(exptected_comm_top_k_index, np.sort(indices_c))

    def test_tf_gather(self):
        num_samples = 20
        # un_sorted_overlap_indices = np.random.choice(num_samples, size=5, replace=False)
        un_sorted_overlap_indices = [7, 2, 1, 9, 6, 4, 19]
        sorted_overlap_indices = np.sort(un_sorted_overlap_indices)
        print("un_sorted_overlap_indices:", un_sorted_overlap_indices)
        print("sorted_overlap_indices:", sorted_overlap_indices)
        samples = np.random.rand(num_samples, 3)
        overlap_samples = tf_gather_with_dynamic_partition(samples, un_sorted_overlap_indices)
        print("samples: \n", samples)
        expected_overlap_samples = samples[sorted_overlap_indices]
        with tf.compat.v1.Session() as sess:
            actual_overlap_samples = sess.run(overlap_samples)
            print("expected_overlap_samples: \n", expected_overlap_samples)
            print("actual_overlap_samples: \n", actual_overlap_samples)
            assert_arrays(expected_overlap_samples, actual_overlap_samples)


if __name__ == '__main__':
    unittest.main()
