import os
import pickle
import time

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tag_INFO = "[INFO]"


class FileDataLoader(object):

    def __init__(self, batch_size, num_batch_blocks, data_folder, file_name='preprocess_batch_',
                 half_image_features=False):
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.file_name = file_name
        self.half_image_features = half_image_features
        self.num_batch_blocks = num_batch_blocks
        self.half_image_features = half_image_features

    def load_minibatch(self):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """

        for batch_block_idx in range(self.num_batch_blocks):
            features, labels = load_training_batch(self.data_folder, batch_block_idx, file_name=self.file_name)

            if self.half_image_features:
                half_feature_dim = int(features.shape[2] / 2)
                features = features[:, :, :half_feature_dim]
            print("loaded features with shape {0}".format(features.shape))

            # Return the training data in batches of size <batch_size> or less
            return batch_features_labels(features, labels, batch_size)


class SimpleDataLoader(object):

    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size

    def load_minibatch(self):
        return batch_features_labels(self.features, self.labels, batch_size)


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_training_batch(data_folder, batch_id, file_name):
    filename = data_folder + file_name + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


def print_training_stats(sess, extractor, feature_batch, label_batch, valid_features, valid_labels):
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
    return valid_acc


def print_testing_stats(sess, extractor, test_features, test_labels):
    test_acc = sess.run(extractor.accuracy,
                        feed_dict={
                            extractor.X_all_in: test_features,
                            extractor.y: test_labels,
                            extractor.keep_prob: 1.,
                            extractor.is_train: False
                        })

    print('Test Accuracy: {:.6f}'.format(test_acc))
    return test_acc


def train(train_data_loader, val_features, val_labels, test_features, test_labels, extractor, training_meta_info):
    print('{0} Start training...'.format(tag_INFO))

    keep_probability = training_meta_info['keep_probability']
    save_model_path = training_meta_info['save_model_path']
    epochs = training_meta_info['epochs']

    gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0")
    best_test_acc = 0
    best_valid_acc = 0
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        start_time = time.time()
        for epoch in range(epochs):

            min_batch_index = 0
            for batch_features, batch_labels in train_data_loader.load_minibatch():
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
                    print('Epoch {:>2}, min-batch-idx {}:'.format(epoch + 1, min_batch_index), end='')
                    valid_acc = print_training_stats(sess,
                                                     extractor,
                                                     last_batch_features,
                                                     last_batch_labels,
                                                     val_features,
                                                     val_labels)

                    test_acc = print_testing_stats(sess,
                                                   extractor,
                                                   test_features,
                                                   test_labels)

                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        best_test_acc = test_acc

                min_batch_index += 1

        end_time = time.time()
        running_time = end_time - start_time
        print("{0} total running time {1}:".format(tag_INFO, running_time))

        # Save Model
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, save_model_path)
        return save_path, best_test_acc
