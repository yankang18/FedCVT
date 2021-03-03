import os

import numpy as np
import tensorflow as tf

from benchmark_cnn_train import batch_features_labels, train
from cnn_models import ClientVGG8
from data_util.modelnet_data_loader import get_two_party_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tag_INFO = "[INFO]"


class SimpleDataLoader(object):

    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size

    def load_minibatch(self):
        return batch_features_labels(self.features, self.labels, self.batch_size)


def run_experiment(train_features, train_labels,
                   val_features, val_labels,
                   test_features, test_labels,
                   training_meta_info):
    print("############# Data Info ##############")
    print("X_guest_train shape", train_features.shape)
    print("y_train shape", train_labels.shape)
    print("X_guest_val shape", val_features.shape)
    print("y_val shape", val_labels.shape)
    print("X_guest_test shape", test_features.shape)
    print("y_test shape", test_labels.shape)

    dense_units = training_meta_info['dense_units']
    image_shape = training_meta_info['image_shape']
    learning_rate = training_meta_info['learning_rate']
    batch_size = training_meta_info['batch_size']

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    extractor = ClientVGG8("cnn_0", dense_units)
    extractor.build(input_shape=image_shape, learning_rate=learning_rate)

    data_loader = SimpleDataLoader(features=train_features, labels=train_labels, batch_size=batch_size)
    return train(data_loader, val_features, val_labels, test_features, test_labels, extractor, training_meta_info)


if __name__ == "__main__":

    data_dir = "../../Data/modelnet40v1png/"

    num_classes = 10
    X_guest_train, X_host_train, Y_train = get_two_party_data(data_dir=data_dir, data_type="train", k=2, c=num_classes)
    X_guest_test, X_host_test, Y_test = get_two_party_data(data_dir=data_dir, data_type="test", k=2, c=num_classes)

    print(X_guest_train[0].shape)
    print(X_host_train[0].shape)
    print(Y_train[0].shape)

    # hyper-parameters
    epochs = 60
    batch_size = 128
    keep_probability = 0.75
    learning_rate = 0.001
    image_shape = (32, 32, 3)
    dense_units = 128
    overlapping_sample_list = [500]
    # # the list of number of overlapping samples for training
    # overlapping_sample_list = [250, 500, 1000, 2000, 4000]

    save_model_path = file_folder = "benchmark_info/"

    training_meta_info = dict()
    training_meta_info["image_shape"] = image_shape
    training_meta_info["dense_units"] = dense_units
    training_meta_info["epochs"] = epochs
    training_meta_info["batch_size"] = batch_size
    training_meta_info["keep_probability"] = keep_probability
    training_meta_info["learning_rate"] = learning_rate
    training_meta_info["save_model_path"] = save_model_path

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

            num_val = int(X_guest_test.shape[0] / 2)
            X_guest_val = X_guest_test[:num_val]
            Y_val = Y_test[:num_val]

            X_guest_test = X_guest_test[num_val:]
            Y_test = Y_test[num_val:]

            _, best_test_acc = run_experiment(train_features=X_guest_train, train_labels=Y_train,
                                              test_features=X_guest_test, test_labels=Y_test,
                                              val_features=X_guest_val, val_labels=Y_val,
                                              training_meta_info=training_meta_info)
            result_map[str(overlapping_sample)].append(best_test_acc)

    for overlapping_sample in overlapping_sample_list:
        all_try_acc = result_map[str(overlapping_sample)]
        print("overlapping sample {0} has mean acc: {1}".format(overlapping_sample, np.mean(all_try_acc)))
        print("all test tries:", all_try_acc)
