import tensorflow as tf


def tf_cosine_sim(matrix_1, matrix_2, name='cosine_sim'):
    with tf.compat.v1.name_scope(name):
        x1_val = tf.sqrt(tf.reduce_sum(input_tensor=tf.multiply(matrix_1, matrix_1), axis=1))
        x2_val = tf.sqrt(tf.reduce_sum(input_tensor=tf.multiply(matrix_2, matrix_2), axis=1))
        denom = tf.multiply(x1_val, x2_val)
        num = tf.reduce_sum(input_tensor=tf.multiply(matrix_1, matrix_2), axis=1)
        return tf.compat.v1.div(num, denom)


def tf_dot_sim(matrix_1, matrix_2, name='dot_sim'):
    with tf.compat.v1.name_scope(name):
        # x1_val = tf.sqrt(tf.reduce_sum(tf.multiply(matrix_1, matrix_1), axis=1))
        # x2_val = tf.sqrt(tf.reduce_sum(tf.multiply(matrix_2, matrix_2), axis=1))
        # denom = tf.multiply(x1_val, x2_val)
        # num = tf.reduce_sum(tf.multiply(matrix_1, matrix_2), axis=1)
        # return tf.div(num, denom)
        return tf.reduce_sum(input_tensor=tf.multiply(matrix_1, matrix_2), axis=1)


def tf_euclidean_distance(matrix_1, matrix_2, name='euclidean_distance'):
    with tf.compat.v1.name_scope(name):
        return tf.linalg.norm(tensor=(matrix_1 - matrix_2), axis=1)

# def tf_compute_distance_loss(input_a, input_b):
#     return tf.losses.cosine_distance(input_a, input_b, axis=1, reduction="none")


def tf_compute_distances(sample, samples):
    return tf_dot_sim(tf.tile(tf.expand_dims(sample, axis=0), (samples.shape[0], 1)), samples)


def tf_1_normalize(x, name='1-norm'):
    with tf.compat.v1.name_scope(name):
        return tf.compat.v1.div(x, tf.linalg.norm(tensor=x, ord=1))


def tf_gather_with_dynamic_partition(samples, overlap_indices):
    depth = tf.shape(input=samples)[0]
    # print("depth:", depth)
    partitions = tf.reduce_sum(input_tensor=tf.one_hot(overlap_indices, depth, dtype='int32'), axis=0)
    parted_samples = tf.dynamic_partition(samples, partitions, 2)
    return parted_samples[1]
