import pickle
import numpy as np
from data_util.data_loader import TwoPartyDataLoader


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_minibatch(data_folder, batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    # filename = data_folder + 'preprocess_batch_' + str(batch_id) + '.p'
    # features, labels = pickle.load(open(filename, mode='rb'))

    features, labels = load_training_batch(data_folder, batch_id)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def load_training_batch(data_folder, batch_id):
    filename = data_folder + 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


class TwoPartyCifar10DataLoader(TwoPartyDataLoader):

    def __init__(self, infile):
        self.infile = infile
        self.labels = None

    def get_training_data(self, num_samples=None):
        features_list = []
        labels_list = []
        for i in range(5):
            batch_features_i, batch_labels_i = load_training_batch(self.infile, i + 1)
            features_list.append(batch_features_i)
            labels_list.append(batch_labels_i)

        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)

        feature_dim = features_array.shape[2]
        # print("feature_dim: {0}".format(feature_dim))
        half_feature_dim = int(feature_dim / 2)
        # print("half_feature_dim: {0}".format(half_feature_dim))
        return features_array[:, :, :half_feature_dim], features_array[:, :, half_feature_dim:], labels_array

    def get_test_data(self, num_samples=None):
        valid_features, valid_labels = pickle.load(open(self.infile + 'preprocess_validation.p', mode='rb'))
        feature_dim = valid_features.shape[2]
        half_feature_dim = int(feature_dim / 2)
        return valid_features[:, :, :half_feature_dim], valid_features[:, :, half_feature_dim:], valid_labels


if __name__ == "__main__":
    cifar10_dataset_folder_path = "../../data/cifar-10-batches-py/"

    cifar10_data_loader = TwoPartyCifar10DataLoader(cifar10_dataset_folder_path)

    train_features_A, train_features_B, train_labels = cifar10_data_loader.get_training_data()
    test_features_A, test_features_B, test_labels = cifar10_data_loader.get_test_data()

    print("train_features_A shape: {0}".format(train_features_A.shape))
    print("train_features_B shape: {0}".format(train_features_B.shape))
    print("train_labels shape: {0}".format(train_labels.shape))

    print("test_features_A shape: {0}".format(test_features_A.shape))
    print("test_features_B shape: {0}".format(test_features_B.shape))
    print("test_labels shape: {0}".format(test_labels.shape))
