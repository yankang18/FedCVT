import numpy as np
import tensorflow as tf


def tf_gather(samples, overlap_indices):
    depth = samples.shape[0]
    partitions = tf.reduce_sum(input_tensor=tf.one_hot(overlap_indices, depth, dtype='int32'), axis=0)
    parted_samples = tf.dynamic_partition(samples, partitions, 2)
    return parted_samples[1]


if __name__ == '__main__':
    overlap_indices = [i for i in range(10)]
    samples = np.random.rand(20, 3)
    overlap_samples = tf_gather(samples, overlap_indices)

    expected_overlap_samples = samples[overlap_indices]
    with tf.compat.v1.Session() as sess:
        actual_overlap_samples = sess.run(overlap_samples)
        print("expected_overlap_samples: \n", expected_overlap_samples)
        print("actual_overlap_samples: \n", actual_overlap_samples)
        np.allclose(expected_overlap_samples, actual_overlap_samples)

