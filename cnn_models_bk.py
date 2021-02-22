import os
import pickle
import time

import tensorflow as tf
from autoencoder import FeatureExtractor

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_minibatch(data_folder, batch_id, batch_size, file_name='preprocess_batch_'):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    # filename = data_folder + 'preprocess_batch_' + str(batch_id) + '.p'
    # features, labels = pickle.load(open(filename, mode='rb'))

    features, labels = load_training_batch(data_folder, batch_id, file_name=file_name)

    # half_feature_dim = int(features.shape[2] / 2)
    # features = features[:, :, :half_feature_dim]
    print("loaded features with shape {0}".format(features.shape))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def load_training_batch(data_folder, batch_id, file_name):
    filename = data_folder + file_name + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


class CNNFeatureExtractor(FeatureExtractor):

    def __init__(self, an_id):
        super(CNNFeatureExtractor, self).__init__()
        self.id = str(an_id)
        self._sess = None
        self._input_shape = None
        self.learning_rate = None

    def set_session(self, sess):
        self._sess = sess

    def get_session(self):
        return self._sess

    def build(self, input_shape, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._input_shape = input_shape
        self._set_variable_initializer()
        self._add_input_placeholder()
        self._add_forward_ops()
        self._add_loss_op()

    def _add_input_placeholder(self):
        input_dim = len(self._input_shape)
        print("input dim : {0}".format(input_dim))
        if input_dim == 3:
            self.X_all_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,
                                                                    self._input_shape[0],
                                                                    self._input_shape[1],
                                                                    self._input_shape[2]), name="X_input_all")
            self.X_overlap_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,
                                                                        self._input_shape[0],
                                                                        self._input_shape[1],
                                                                        self._input_shape[2]), name="X_input_overlap")
            self.X_non_overlap_in = tf.compat.v1.placeholder(dtype=tf.float32,
                                                   shape=(None,
                                                          self._input_shape[0],
                                                          self._input_shape[1],
                                                          self._input_shape[2]), name="X_input_non_overlap")
        else:
            raise Exception("input dim mush be 3, but is {0}".format(input_dim))

        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
        self.y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name='output_y')

    def get_all_samples(self):
        return self.X_all_in

    def get_overlap_samples(self):
        return self.X_overlap_in

    def get_non_overlap_samples(self):
        return self.X_non_overlap_in

    def get_is_train(self):
        return self.is_train

    def get_keep_probability(self):
        return self.keep_prob

    def _set_variable_initializer(self):
        # self.conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.08))
        # self.conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 32], mean=0, stddev=0.08))
        # self.conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.08))
        # self.conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], mean=0, stddev=0.08))
        # self.conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], mean=0, stddev=0.08))
        # self.conv5_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 128], mean=0, stddev=0.08))
        # self.conv6_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 128], mean=0, stddev=0.08))

        # self.conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 32], mean=0, stddev=0.08))
        # self.conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.08))
        # self.conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.08))
        # self.conv4_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.08))
        # self.conv5_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
        # self.conv6_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], mean=0, stddev=0.08))

        # self.conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 32], mean=0, stddev=0.08))
        # self.conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.08))
        # self.conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.08))
        # self.conv4_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
        # self.conv5_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], mean=0, stddev=0.08))
        # self.conv6_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], mean=0, stddev=0.08))

        self.conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 1, 16], mean=0, stddev=0.1))
        self.conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 16], mean=0, stddev=0.1))
        self.conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 32], mean=0, stddev=0.1))
        self.conv4_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1))
        self.conv5_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.1))
        self.conv6_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1))

        self.conv_1x1_filter_1 = tf.Variable(tf.random.truncated_normal(shape=[1, 1, 16, 16], mean=0, stddev=0.1))
        self.conv_1x1_filter_2 = tf.Variable(tf.random.truncated_normal(shape=[1, 1, 32, 16], mean=0, stddev=0.1))

    def _add_forward_ops(self):
        self.all_hidden_reprs = self._forward_hidden(self.X_all_in)
        self.overlap_hidden_reprs = self._forward_hidden(self.X_overlap_in)
        self.non_overlap_hidden_reprs = self._forward_hidden(self.X_non_overlap_in)

    def _add_loss_op(self):
        representation = self._forward_hidden(self.X_all_in)
        logits = tf.contrib.layers.fully_connected(inputs=representation, num_outputs=10, activation_fn=None)
        model = tf.identity(logits, name='logits')  # Name logits Tensor, so that can be loaded from disk after training

        # Loss and Optimizer
        self.cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(self.y)))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=self.y, axis=1))
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32), name='accuracy')

    # def _forward_hidden(self, x):
    #     conv1 = tf.nn.conv2d(x, self.conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv1 = tf.nn.relu(conv1)
    #     conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #
    #     flat1 = tf.contrib.layers.flatten(conv1_pool)
    #
    #     conv2 = tf.nn.conv2d(conv1_pool, self.conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv2 = tf.nn.relu(conv2)
    #     conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #
    #     flat2 = tf.contrib.layers.flatten(conv2_pool)
    #     flat = tf.concat(values=[flat1, flat2], axis=1)
    #     print("flat shape:", flat.shape)
    #     out = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=16, activation_fn=tf.nn.tanh)
    #     return out

    # def _forward_hidden(self, x):
    #
    #     drop_rate = 1 - self.keep_prob
    #
    #     # 1, 2
    #     conv1 = tf.nn.conv2d(x, self.conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv1 = tf.nn.leaky_relu(conv1)
    #
    #     conv2 = tf.nn.conv2d(conv1, self.conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv2 = tf.layers.batch_normalization(conv2)
    #     conv2 = tf.nn.leaky_relu(conv2)
    #
    #     # 3, 4
    #     conv3 = tf.nn.conv2d(conv2, self.conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv3 = tf.nn.leaky_relu(conv3)
    #
    #     conv4 = tf.nn.conv2d(conv3, self.conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv4 = tf.layers.batch_normalization(conv4)
    #     conv4 = tf.nn.leaky_relu(conv4)
    #     conv4_pool = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #     # 5, 6
    #     conv5 = tf.nn.conv2d(conv4_pool, self.conv5_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv5 = tf.nn.leaky_relu(conv5)
    #
    #     conv6 = tf.nn.conv2d(conv5, self.conv6_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv6 = tf.layers.batch_normalization(conv6)
    #     conv6 = tf.nn.leaky_relu(conv6)
    #     conv6_pool = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #     # conv6_dp = tf.layers.dropout(inputs=conv6_pool, rate=drop_rate, training=self.is_train)
    #     flat = tf.contrib.layers.flatten(conv6_pool)
    #     print("flat shape : {0}".format(flat.shape))
    #     out = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=64, activation_fn=tf.nn.leaky_relu)
    #     return out

    # def _forward_hidden(self, x):
    #
    #     conv1 = tf.nn.conv2d(x, self.conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv1 = tf.nn.leaky_relu(conv1)
    #     conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #
    #     conv2 = tf.nn.conv2d(conv1, self.conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv2 = tf.layers.batch_normalization(conv2)
    #     conv2 = tf.nn.leaky_relu(conv2)
    #     conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #
    #     flat1 = tf.contrib.layers.flatten(conv2)
    #
    #     conv3 = tf.nn.conv2d(conv2, self.conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
    #     conv3 = tf.layers.batch_normalization(conv3)
    #     conv3 = tf.nn.leaky_relu(conv3)
    #
    #     flat2 = tf.contrib.layers.flatten(conv3)
    #     flat = tf.concat(values=[flat1, flat2], axis=1)
    #     print("flat shape:", flat.shape)
    #     out = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=64, activation_fn=tf.nn.leaky_relu)
    #     return out

    def _forward_hidden(self, x):

        conv1 = tf.nn.conv2d(input=x, filters=self.conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.leaky_relu(conv1)
        conv1 = tf.nn.max_pool2d(input=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = tf.nn.conv2d(input=conv1, filters=self.conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.compat.v1.layers.batch_normalization(conv2)
        conv2 = tf.nn.leaky_relu(conv2)
        conv2 = tf.nn.max_pool2d(input=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        side_conv_1 = tf.nn.conv2d(input=conv2, filters=self.conv_1x1_filter_1, strides=[1, 1, 1, 1], padding='SAME')
        side_flat_1 = tf.contrib.layers.flatten(side_conv_1)

        conv3 = tf.nn.conv2d(input=conv2, filters=self.conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.compat.v1.layers.batch_normalization(conv3)
        conv3 = tf.nn.leaky_relu(conv3)

        side_conv_2 = tf.nn.conv2d(input=conv3, filters=self.conv_1x1_filter_2, strides=[1, 1, 1, 1], padding='SAME')
        side_flat_2 = tf.contrib.layers.flatten(side_conv_2)

        print("side_flat_1 shape:", side_flat_1.shape)
        print("side_flat_2 shape:", side_flat_2.shape)

        flat = tf.concat(values=[side_flat_1, side_flat_2], axis=1)

        print("flat shape:", flat.shape)

        out = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=64, activation_fn=tf.nn.leaky_relu)
        return out

    def get_all_hidden_reprs(self):
        return self.all_hidden_reprs

    def get_overlap_hidden_reprs(self):
        return self.overlap_hidden_reprs

    def get_non_overlap_hidden_reprs(self):
        return self.non_overlap_hidden_reprs

    # def get_encode_dim(self):
    #     return self.hidden_dim_list[-1]


def print_stats(sess, extractor, feature_batch, label_batch, valid_features, valid_labels):
    loss = sess.run(extractor.cost,
                    feed_dict={
                        extractor.X_all_in: feature_batch,
                        extractor.y: label_batch,
                        extractor.keep_prob: 1.,
                        extractor.is_train: False
                    })
    valid_acc = sess.run(extractor.accuracy,
                         feed_dict={
                             extractor.X_all_in: valid_features,
                             extractor.y: valid_labels,
                             extractor.keep_prob: 1.,
                             extractor.is_train: False
                         })

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))


if __name__ == "__main__":

    dataset_folder_path = "../data/fashionmnist/"
    # dataset_folder_path = "../data/cifar-10-batches-py/"

    # n_batches = 15
    # preprocess_training_file_name = 'preprocess_batch_'
    # preprocess_validation_file_name = 'preprocess_validation.p'

    n_batches = 1
    training_dataset_folder_path = "../data/fashionmnist_250/"
    preprocess_training_file_name = 'all_overlap_500_block_'
    preprocess_validation_file_name = 'preprocess_validation.p'

    epochs = 40
    batch_size = 128
    keep_probability = 0.75
    learning_rate = 0.01

    valid_features, valid_labels = pickle.load(open(dataset_folder_path + preprocess_validation_file_name, mode='rb'))
    print("valid features shape:", valid_features.shape)
    # half_feature_dim = int(valid_features.shape[2] / 2)
    # valid_features = valid_features[:, :, :half_feature_dim]
    # print("valid_features shape {0}".format(valid_features.shape))

    # Remove previous weights, bias, inputs, etc..
    tf.compat.v1.reset_default_graph()

    extractor = CNNFeatureExtractor("cnn_1")
    extractor.build(input_shape=(28, 28, 1))
    # extractor.build(input_shape=(32, 32, 3))

    save_model_path = dataset_folder_path + 'image_classification'

    print('Training...')
    gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0")
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # Training cycle
        start_time = time.time()
        for epoch in range(epochs):
            # Loop over all batches
            for batch_i in range(n_batches):
                min_batch_index = 0
                for batch_features, batch_labels in load_preprocess_training_minibatch(training_dataset_folder_path,
                                                                                       batch_i, batch_size,
                                                                                       file_name=preprocess_training_file_name):
                    # train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                    sess.run(extractor.optimizer,
                             feed_dict={
                                 extractor.X_all_in: batch_features,
                                 extractor.y: batch_labels,
                                 extractor.keep_prob: keep_probability,
                                 extractor.is_train: True
                             })

                    last_batch_features = batch_features
                    last_batch_labels = batch_labels

                    if min_batch_index % 10 == 0:
                        print('Epoch {:>2}, CIFAR-10 Block {}, min-batch-idx {}:  '.format(epoch + 1, batch_i,
                                                                                       min_batch_index), end='')
                        print_stats(sess, extractor, last_batch_features, last_batch_labels, valid_features, valid_labels)
                    min_batch_index += 1

        end_time = time.time()
        running_time = end_time - start_time
        print("total running time {0}:".format(running_time))

        # Save Model
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, save_model_path)
