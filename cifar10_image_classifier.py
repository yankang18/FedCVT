import os
import pickle
import time

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(data_folder, batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = data_folder + 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def construct_model(x, keep_prob, is_train):
    conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

    # 1, 2
    conv1 = tf.nn.conv2d(input=x, filters=conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool2d(input=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.compat.v1.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(input=conv1_bn, filters=conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool2d(input=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_bn = tf.compat.v1.layers.batch_normalization(conv2_pool)

    # 5, 6
    conv3 = tf.nn.conv2d(input=conv2_bn, filters=conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool2d(input=conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3_bn = tf.compat.v1.layers.batch_normalization(conv3_pool)

    # 7, 8
    conv4 = tf.nn.conv2d(input=conv3_bn, filters=conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool2d(input=conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4_bn = tf.compat.v1.layers.batch_normalization(conv4_pool)

    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)

    # 10
    out = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=10, activation_fn=None)
    return out


def construct_model_2(x, keep_prob, is_train):

    conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.08))
    conv5_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv6_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 128, 128], mean=0, stddev=0.08))

    drop_rate = 1 - keep_prob

    # activation_func = tf.nn.relu
    activation_func = tf.nn.leaky_relu

    # 1, 2
    conv1 = tf.nn.conv2d(input=x, filters=conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = activation_func(conv1)

    conv2 = tf.nn.conv2d(input=conv1, filters=conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = activation_func(conv2)
    conv2 = tf.compat.v1.layers.batch_normalization(conv2)
    conv2 = tf.nn.max_pool2d(input=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.compat.v1.layers.dropout(inputs=conv2, rate=drop_rate, training=is_train)

    # 3, 4
    conv3 = tf.nn.conv2d(input=conv2, filters=conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = activation_func(conv3)

    conv4 = tf.nn.conv2d(input=conv3, filters=conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv4 = activation_func(conv4)
    conv4 = tf.compat.v1.layers.batch_normalization(conv4)
    conv4 = tf.nn.max_pool2d(input=conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv4 = tf.compat.v1.layers.dropout(inputs=conv4, rate=drop_rate, training=is_train)

    # 5, 6
    conv5 = tf.nn.conv2d(input=conv4, filters=conv5_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv5 = activation_func(conv5)
    conv6 = tf.nn.conv2d(input=conv5, filters=conv6_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv6 = activation_func(conv6)
    conv6 = tf.compat.v1.layers.batch_normalization(conv6)
    conv6 = tf.nn.max_pool2d(input=conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv6 = tf.compat.v1.layers.dropout(inputs=conv6, rate=drop_rate, training=is_train)

    # 9
    flat = tf.contrib.layers.flatten(conv6)

    print("{0} flatten shape: {1}".format("[INFO]:", flat))

    # 10
    dense1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=activation_func)
    dense1 = tf.compat.v1.layers.batch_normalization(dense1)
    out = tf.contrib.layers.fully_connected(inputs=dense1, num_outputs=10, activation_fn=None)
    return out


def print_stats(sess, feature_batch, label_batch, valid_features, valid_labels, cost, accuracy):
    loss = sess.run(cost,
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.,
                        is_train: False
                    })
    valid_acc = sess.run(accuracy,
                         feed_dict={
                             x: valid_features,
                             y: valid_labels,
                             keep_prob: 1.,
                             is_train: False
                         })

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))


if __name__ == "__main__":
    dataset_folder_path = "../data/cifar-10-batches-py/"

    epochs = 20
    batch_size = 128
    keep_probability = 0.75
    learning_rate = 0.001
    n_batches = 5

    valid_features, valid_labels = pickle.load(open(dataset_folder_path + 'preprocess_validation.p', mode='rb'))

    # Remove previous weights, bias, inputs, etc..
    tf.compat.v1.reset_default_graph()

    # Inputs
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name='output_y')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

    logits = construct_model_2(x, keep_prob, is_train)
    model = tf.identity(logits, name='logits')  # Name logits Tensor, so that can be loaded from disk after training

    # Loss and Optimizer
    cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(y)))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=y, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32), name='accuracy')

    save_model_path = dataset_folder_path + 'image_classification'

    print('Training...')
    gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0")
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        # Training cycle
        start_time = time.time()
        for epoch in range(epochs):
            # Loop over all batches
            # n_batches = 5
            for batch_i in range(1, n_batches + 1):
                min_batch_index = 0
                for batch_features, batch_labels in load_preprocess_training_batch(dataset_folder_path,
                                                                                   batch_i, batch_size):
                    # train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                    sess.run(optimizer,
                             feed_dict={
                                 x: batch_features,
                                 y: batch_labels,
                                 keep_prob: keep_probability,
                                 is_train: True
                             })

                    last_batch_features = batch_features
                    last_batch_labels = batch_labels

                    if min_batch_index % 10 == 0:
                        print('Epoch {:>2}, CIFAR-10 Batch {}, min-batch {}:  '.format(epoch + 1, batch_i,
                                                                                       min_batch_index), end='')
                        print_stats(sess, last_batch_features, last_batch_labels, valid_features, valid_labels, cost,
                                    accuracy)
                    min_batch_index += 1

        end_time = time.time()
        running_time = end_time - start_time
        print("total running time {0}:".format(running_time))

        # Save Model
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, save_model_path)
