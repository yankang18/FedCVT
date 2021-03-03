import os
import pickle
import time
import argparse
import tensorflow as tf
from cnn_models import BenchmarkFullImageCNNFeatureExtractor, BenchmarkDeeperHalfImageCNNFeatureExtractor, \
    BenchmarkHalfImageCNNFeatureExtractor, ClientMiniVGG, ClientVGG8, ClientMiniGoogLeNet
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tag_INFO = "[INFO]"


class FileDataLoader(object):

    def __init__(self, batch_size, data_folder, file_name='preprocess_batch_', half_image_features=False):
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.file_name = file_name
        self.half_image_features = half_image_features

    def load_minibatch(self, batch_id):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """

        features, labels = load_training_batch(self.data_folder, batch_id, file_name=self.file_name)

        if half_image_features:
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


def run_experiment(meta_info):
    half_image_features = meta_info["half_image_features"]
    image_shape = meta_info["image_shape"]

    n_batches = meta_info["n_batches"]
    epochs = meta_info["epochs"]
    batch_size = meta_info["batch_size"]
    keep_probability = meta_info["keep_probability"]
    learning_rate = meta_info["learning_rate"]

    validation_dataset_folder_path = meta_info["validation_dataset_folder_path"]
    preprocess_validation_file_name = meta_info["preprocess_validation_file_name"]
    training_dataset_folder_path = meta_info["training_dataset_folder_path"]
    preprocess_training_file_name = meta_info["preprocess_training_file_name"]
    save_model_path = meta_info["save_model_path"]

    print("{0} number of blocks: {1}".format(tag_INFO, n_batches))
    print("{0} validation_dataset_folder_path: {1}".format(tag_INFO, validation_dataset_folder_path))
    print("{0} preprocess_validation_file_name: {1}".format(tag_INFO, preprocess_validation_file_name))
    print("{0} training_dataset_folder_path: {1}".format(tag_INFO, training_dataset_folder_path))
    print("{0} preprocess_training_file_name: {1}".format(tag_INFO, preprocess_training_file_name))

    val_features, val_labels = pickle.load(
        open(validation_dataset_folder_path + preprocess_validation_file_name, mode='rb'))
    print("{0} original validation features shape {1}:".format(tag_INFO, val_features.shape))

    if half_image_features:
        half_feature_dim = int(val_features.shape[2] / 2)
        val_features = val_features[:, :, :half_feature_dim]
        # val_features = val_features[:, :, half_feature_dim:]
        print("{0} half validation features shape {1}".format(tag_INFO, val_features.shape))

    # prepare validation samples and test samples
    half_val_sample_number = int(len(val_features) / 2)
    test_features, test_labels = val_features[half_val_sample_number:], val_labels[half_val_sample_number:]
    val_features, val_labels = val_features[:half_val_sample_number], val_labels[:half_val_sample_number]

    print("{0} val_features shape: {1}".format(tag_INFO, val_features.shape))
    print("{0} val_labels shape: {1}".format(tag_INFO, val_labels.shape))
    print("{0} test_features shape: {1}".format(tag_INFO, test_features.shape))
    print("{0} test_labels shape: {1}".format(tag_INFO, test_labels.shape))

    # Remove previous weights, bias, inputs, etc..
    tf.compat.v1.reset_default_graph()

    if not half_image_features:
        # extractor = BenchmarkFullImageCNNFeatureExtractor("cnn_1")
        extractor = ClientVGG8("cnn_1")
        extractor.build(input_shape=image_shape, learning_rate=learning_rate)
    else:
        extractor = ClientVGG8("cnn_1")
        extractor.build(input_shape=image_shape, learning_rate=learning_rate)

    print('{0} Start training...'.format(tag_INFO))
    gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0")
    best_test_acc = 0
    best_valid_acc = 0
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # Training cycle
        start_time = time.time()
        for epoch in range(epochs):
            # Loop over all blocks
            for batch_i in range(n_batches):
                min_batch_index = 0
                # Loop over all mini-batches
                for batch_features, batch_labels in load_minibatch(training_dataset_folder_path,
                                                                   batch_i, batch_size,
                                                                   file_name=preprocess_training_file_name,
                                                                   half_image_features=half_image_features):
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


if __name__ == "__main__":

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-c", "--complete_image", type=bool)
    # args = ap.parse_args()

    # print("args:", args)
    # whether using the whole image
    # complete_image = args.complete_image
    half_image_features = True
    print("{0} half image features: {1} ".format(tag_INFO, half_image_features))

    # If using half image, whether just using overlapping training samples
    is_half_image_test_using_only_overlapping_samples = True

    # hyper-parameters
    epochs = 60
    batch_size = 128
    keep_probability = 0.75
    learning_rate = 0.001

    # the list of number of overlapping samples for training
    overlapping_sample_list = [250, 500, 1000, 2000, 4000]
    result_map = dict()
    result_map["250"] = list()
    result_map["500"] = list()
    result_map["1000"] = list()
    result_map["2000"] = list()
    result_map["4000"] = list()

    num_try = 5
    for try_idx in range(num_try):
        for overlapping_sample in overlapping_sample_list:
            print("[INFO] {0} try for testing with overlapping samples: {1}".format(try_idx, overlapping_sample))

            # using all data
            # n_batches = 15
            n_batches = 5
            # validation_dataset_folder_path = "../data/fashionmnist/"
            validation_dataset_folder_path = "../data/cifar-10-batches-py/"
            preprocess_validation_file_name = 'preprocess_validation.p'
            training_dataset_folder_path = validation_dataset_folder_path
            preprocess_training_file_name = 'preprocess_batch_'

            if not half_image_features:
                print("{0} Run benchmark test for complete image:".format(tag_INFO))
                # image_shape = (28, 28, 1)
                image_shape = (32, 32, 3)
            else:
                print("{0} Run benchmark test for half image:".format(tag_INFO))
                # image_shape = (28, 14, 1)
                image_shape = (32, 16, 3)

                if is_half_image_test_using_only_overlapping_samples:
                    print("{0} Run benchmark test for half image, using only overlapping samples:".format(tag_INFO))
                    n_batches = 1
                    validation_dataset_folder_path = "../data/cifar-10-batches-py/"
                    preprocess_validation_file_name = 'preprocess_validation.p'
                    training_dataset_folder_path = "../data/cifar-10-batches-py_250/"
                    preprocess_training_file_name = 'all_overlap_' + str(overlapping_sample) + '_block_'

            save_model_path = training_dataset_folder_path + 'image_classification'

            training_meta_info = dict()
            training_meta_info["half_image_features"] = half_image_features
            training_meta_info["image_shape"] = image_shape
            training_meta_info["n_batches"] = n_batches
            training_meta_info["epochs"] = epochs
            training_meta_info["batch_size"] = batch_size
            training_meta_info["keep_probability"] = keep_probability
            training_meta_info["learning_rate"] = learning_rate
            training_meta_info["validation_dataset_folder_path"] = validation_dataset_folder_path
            training_meta_info["preprocess_validation_file_name"] = preprocess_validation_file_name
            training_meta_info["training_dataset_folder_path"] = training_dataset_folder_path
            training_meta_info["preprocess_training_file_name"] = preprocess_training_file_name
            training_meta_info["save_model_path"] = save_model_path

            _, best_test_acc = run_experiment(meta_info=training_meta_info)
            result_map[str(overlapping_sample)].append(best_test_acc)

    for overlapping_sample in overlapping_sample_list:
        all_try_acc = result_map[str(overlapping_sample)]
        print("overlapping sample {0} has mean acc: {1}".format(overlapping_sample,
                                                                np.mean(all_try_acc)))
        print("all test tries:", all_try_acc)
