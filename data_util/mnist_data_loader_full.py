import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def normalize_w_min_max(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded


def get_batch_num(all_sample_size, batch_size):
    residual = all_sample_size % batch_size
    if residual == 0:
        return int(all_sample_size / batch_size)
    else:
        return int(all_sample_size / batch_size) + 1


def load_data_block(block_file_full_path, block_id):
    filename = block_file_full_path + str(block_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


def preprocess_and_save_data_v2(from_dataset_folder_path,
                                to_dataset_folder_path,
                                load_data,
                                num_overlap,
                                guest_nonoverlap_block_size,
                                guest_estimation_block_size,
                                host_nonoverlap_block_size,
                                host_estimation_block_size):

    train_features, train_labels, val_features, val_labels = load_data(dataset_folder_path=from_dataset_folder_path)

    print("from_dataset_folder_path: {0}".format(from_dataset_folder_path))
    print("to_dataset_folder_path: {0}".format(to_dataset_folder_path))

    print("##############################")
    print("## preprocess and save data ##")
    print("##############################")

    print("---> normalize images")

    min_val = np.min(train_features)
    max_val = np.max(train_features)

    print("max_val: {0}".format(max_val))
    print("min_val: {0}".format(min_val))

    train_features = normalize_w_min_max(train_features, min_val=min_val, max_val=max_val)
    val_features = normalize_w_min_max(val_features, min_val=min_val, max_val=max_val)

    print("---> one-hot encode labels")

    train_labels = one_hot_encode(train_labels)
    val_labels = one_hot_encode(val_labels)

    print("---> prepare guest and host data")

    shape_dim = len(train_features.shape)
    print("shape_dim: {0}".format(shape_dim))
    feature_dim = train_features.shape[shape_dim - 2]
    half_feature_dim = int(feature_dim / 2)

    # prepare validation data for guest and host
    # guest_val_features = val_features[:, :, :half_feature_dim, :]
    # host_val_features = val_features[:, :, half_feature_dim:, :]
    guest_val_features = val_features[:, :, :half_feature_dim]
    host_val_features = val_features[:, :, half_feature_dim:]
    num_val = len(guest_val_features)

    # prepare overlapping data for guest and host
    overlap_train_features, overlap_train_labels = train_features[:num_overlap], train_labels[:num_overlap]
    # overlap_guest_train_features = overlap_train_features[:, :, :half_feature_dim, :]
    # overlap_host_train_features = overlap_train_features[:, :, half_feature_dim:, :]
    overlap_guest_train_features = overlap_train_features[:, :, :half_feature_dim]
    overlap_host_train_features = overlap_train_features[:, :, half_feature_dim:]

    # prepare non-overlapping data for guest and host
    num_nonoverlap = len(train_features) - num_overlap
    half_num_nonoverlap = int(num_nonoverlap / 2)

    print("num train_features", len(train_features))
    print("number of overlapping samples", num_overlap)
    print("number of nonoverlapping samples:", num_nonoverlap)
    print("number of half nonoverlapping samples:", half_num_nonoverlap)

    # split data into guest and host
    # nonoverlap_guest_train_features = train_features[num_overlap:num_overlap + half_num_nonoverlap, :,
    #                                   :half_feature_dim, :]
    # nonoverlap_host_train_features = train_features[num_overlap + half_num_nonoverlap:, :, half_feature_dim:, :]
    nonoverlap_guest_train_features = train_features[num_overlap:num_overlap + half_num_nonoverlap, :, :half_feature_dim]
    nonoverlap_guest_train_labels = train_labels[num_overlap:num_overlap + half_num_nonoverlap]

    nonoverlap_host_train_features = train_features[num_overlap + half_num_nonoverlap:, :, half_feature_dim:]
    nonoverlap_host_train_labels = train_labels[num_overlap + half_num_nonoverlap:]

    # collect all samples for guest and host, respectively
    guest_train_features = np.concatenate((overlap_guest_train_features, nonoverlap_guest_train_features), axis=0)
    guest_train_labels = np.concatenate((overlap_train_labels, nonoverlap_guest_train_labels), axis=0)
    host_train_features = np.concatenate((overlap_host_train_features, nonoverlap_host_train_features), axis=0)
    host_train_labels = np.concatenate((overlap_train_labels, nonoverlap_host_train_labels), axis=0)

    num_overlap = len(overlap_train_features)
    num_guest_estimation = len(guest_train_features)
    num_host_estimation = len(host_train_features)
    num_guest_nonoverlap = len(nonoverlap_guest_train_features)
    num_host_nonoverlap = len(nonoverlap_host_train_features)
    print("overlap_train_features shape: {0}", overlap_train_features.shape)
    print("overlap_train_labels shape: {0}", overlap_train_labels.shape)
    print("overlap_guest_train_features shape: {0}", overlap_guest_train_features.shape)
    print("overlap_host_train_features shape: {0}", overlap_host_train_features.shape)
    print("nonoverlap_guest_train_features shape: {0}", nonoverlap_guest_train_features.shape)
    print("nonoverlap_guest_train_labels shape: {0}", nonoverlap_guest_train_labels.shape)
    print("nonoverlap_host_train_features shape: {0}", nonoverlap_host_train_features.shape)
    print("nonoverlap_host_train_labels shape: {0}", nonoverlap_host_train_labels.shape)
    print("all_guest_train_features shape: {0}", guest_train_features.shape)
    print("all_guest_train_labels shape: {0}", guest_train_labels.shape)
    print("all_host_train_features shape: {0}", host_train_features.shape)
    print("all_host_train_labels shape: {0}", host_train_labels.shape)

    # overlap_block_size = 1
    # guest_nonoverlap_block_size = 4000
    # guest_ested_block_size = 5000
    # host_nonoverlap_block_size = 4000
    # host_ested_block_size = 5000

    val_block_num = get_batch_num(num_val, num_val)
    # all_overlap_block_num = get_batch_num(num_overlap, 4000)
    overlap_block_num = get_batch_num(num_overlap, num_overlap)
    guest_nonoverlap_block_num = get_batch_num(num_guest_nonoverlap, guest_nonoverlap_block_size)
    guest_ested_block_num = get_batch_num(num_guest_estimation, guest_estimation_block_size)
    host_nonoverlap_block_num = get_batch_num(num_host_nonoverlap, host_nonoverlap_block_size)
    host_ested_block_num = get_batch_num(num_host_estimation, host_estimation_block_size)

    meta_data = dict()
    meta_data["guest_val_block_num"] = val_block_num
    meta_data["host_val_block_num"] = val_block_num
    meta_data["guest_overlap_block_num"] = overlap_block_num
    meta_data["host_overlap_block_num"] = overlap_block_num
    meta_data["guest_nonoverlap_block_num"] = guest_nonoverlap_block_num
    meta_data["host_nonoverlap_block_num"] = host_nonoverlap_block_num
    meta_data["guest_estimation_block_num"] = guest_ested_block_num
    meta_data["host_estimation_block_num"] = host_ested_block_num

    print("---> save meta data: {0}".format(meta_data))
    with open(to_dataset_folder_path + "meta_data.json", "w") as write_file:
        json.dump(meta_data, write_file)

    print("---> save guest and host data")

    print("validation block size {0}, num {1}.".format(num_val, val_block_num))
    print("overlap block size {0}, num {1}.".format(num_overlap, overlap_block_num))
    print("guest nonoverlap block size {0}, num {1}.".format(guest_nonoverlap_block_size, guest_nonoverlap_block_num))
    print("guest estimation block size {0}, num {1}.".format(guest_estimation_block_size, guest_ested_block_num))
    print("host nonoverlap block size {0}, num {1}.".format(host_nonoverlap_block_size, host_nonoverlap_block_num))
    print("host estimation block size {0}, num {1}.".format(host_estimation_block_size, host_ested_block_num))

    # file_full_path, block_num, block_size, features, labels, normalize_func, one_hot_encode_func
    print("preprocessing all overlap sample block ...")
    all_overlap_sample_num_list = [250, 500, 1000, 2000, 4000]
    for all_overlap_sample_num in all_overlap_sample_num_list:
        print("preprocessing all overlap sample {0} blocks ...".format(all_overlap_sample_num))
        block_file_full_path = to_dataset_folder_path + 'all_overlap_' + str(all_overlap_sample_num) + '_block_'
        _preprocess_and_save_batches(block_file_full_path,
                                     1,
                                     all_overlap_sample_num,
                                     train_features,
                                     train_labels)

    print("preprocessing guest validation sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_val_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 val_block_num,
                                 num_val,
                                 guest_val_features,
                                 val_labels)

    print("preprocessing host validation sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_val_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 val_block_num,
                                 num_val,
                                 host_val_features,
                                 val_labels)

    print("preprocessing guest overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_overlap_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 overlap_block_num,
                                 num_overlap,
                                 overlap_guest_train_features,
                                 overlap_train_labels)

    print("preprocessing host overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_overlap_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 overlap_block_num,
                                 num_overlap,
                                 overlap_host_train_features,
                                 overlap_train_labels)

    print("preprocessing guest non-overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_nonoverlap_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 guest_nonoverlap_block_num,
                                 guest_nonoverlap_block_size,
                                 nonoverlap_guest_train_features,
                                 nonoverlap_guest_train_labels)

    print("preprocessing host non-overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_nonoverlap_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 host_nonoverlap_block_num,
                                 host_nonoverlap_block_size,
                                 nonoverlap_host_train_features,
                                 nonoverlap_host_train_labels)

    print("preprocessing guest estimated sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_ested_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 guest_ested_block_num,
                                 guest_estimation_block_size,
                                 guest_train_features,
                                 guest_train_labels)

    print("preprocessing host estimated sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_ested_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 host_ested_block_num,
                                 host_estimation_block_size,
                                 host_train_features,
                                 host_train_labels)


def _preprocess_and_save_batches(file_full_path,
                                 block_num,
                                 block_size,
                                 features,
                                 labels,
                                 normalize_func=None,
                                 one_hot_encode_func=None):
    for batch_idx in range(block_num):
        start_idx = batch_idx * block_size
        end_idx = start_idx + block_size
        print("processing samples [{0}:{1}]".format(start_idx, end_idx))
        _preprocess_and_save(normalize_func, one_hot_encode_func,
                             features[start_idx: end_idx], labels[start_idx: end_idx],
                             file_full_path + str(batch_idx) + '.p')


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    if normalize is not None:
        features = normalize(features)
    if one_hot_encode is not None:
        labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))


def load_fashionMNIST_data(dataset_folder_path):
    # MNIST datasets:
    # column 0 is labels
    # column 1-785 is datasets, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)

    train = pd.read_csv(dataset_folder_path + 'fashion-mnist_train.csv')
    test = pd.read_csv(dataset_folder_path + 'fashion-mnist_test.csv')
    train = train.values
    test = test.values
    print("train shape:{0}".format(train.shape))
    print("test shape:{0}".format(test.shape))
    train = shuffle(train)

    Xtrain = train[:, 1:]
    Xtrain = Xtrain.reshape(len(Xtrain), 28, 28, 1)
    Ytrain = train[:, 0].astype(np.int32)
    Xtest = test[:, 1:]
    Xtest = Xtest.reshape(len(Xtest), 28, 28, 1)
    Ytest = test[:, 0].astype(np.int32)

    print("Xtrain shape:{0}".format(Xtrain.shape))
    print("Ytrain shape:{0}".format(Ytrain.shape))
    print("Xtest shape:{0}".format(Xtest.shape))
    print("Ytest shape:{0}".format(Ytest.shape))

    return Xtrain, Ytrain, Xtest, Ytest


def preprocess_and_save_data(dataset_folder_path,
                             load_data,
                             block_size=4000):
    train_features, train_labels, val_features, val_labels = load_data(dataset_folder_path=dataset_folder_path)

    print("##############################")
    print("## preprocess and save data ##")
    print("##############################")

    print("---> normalize images")

    min_val = np.min(train_features)
    max_val = np.max(train_features)

    print("max_val: {0}".format(max_val))
    print("min_val: {0}".format(min_val))

    train_features = normalize_w_min_max(train_features, min_val=min_val, max_val=max_val)
    val_features = normalize_w_min_max(val_features, min_val=min_val, max_val=max_val)

    print("---> one-hot encode labels")

    train_labels = one_hot_encode(train_labels)
    val_labels = one_hot_encode(val_labels)

    num_train = len(train_features)
    block_num = get_batch_num(num_train, block_size)
    block_file_full_path = dataset_folder_path + 'preprocess_batch_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=block_num,
                                 block_size=block_size,
                                 features=train_features,
                                 labels=train_labels)

    block_file_full_path = dataset_folder_path + 'preprocess_validation'
    _preprocess_and_save(None, None,
                         val_features, val_labels,
                         block_file_full_path + '.p')


def show_data_block(data_full_path, block_num):
    for block_idx in range(block_num):
        features, labels = load_data_block(data_full_path, block_idx)
        print("block idx {0}: with features {1}, labels {2}".format(block_idx, features.shape, labels.shape))


def show_data(data_folder):
    print("####################")
    print("## show data info ##")
    print("####################")

    with open(data_folder + "meta_data.json", "r") as read_file:
        meta_data = json.load(read_file)

    guest_val_block_num = meta_data["guest_val_block_num"]
    guest_overlap_block_num = meta_data["guest_overlap_block_num"]
    guest_nonoverlap_block_num = meta_data["guest_nonoverlap_block_num"]
    guest_ested_block_num = meta_data["guest_estimation_block_num"]

    host_val_block_num = meta_data["host_val_block_num"]
    host_overlap_block_num = meta_data["host_overlap_block_num"]
    host_nonoverlap_block_num = meta_data["host_nonoverlap_block_num"]
    host_ested_block_num = meta_data["host_estimation_block_num"]

    print("guest_val_block_num {0}".format(guest_val_block_num))
    print("guest_overlap_block_num {0}".format(guest_overlap_block_num))
    print("guest_nonoverlap_block_num {0}".format(guest_nonoverlap_block_num))
    print("guest_ested_block_num {0}".format(guest_ested_block_num))
    print("host_val_block_num {0}".format(host_val_block_num))
    print("host_overlap_block_num {0}".format(host_overlap_block_num))
    print("host_nonoverlap_block_num {0}".format(host_nonoverlap_block_num))
    print("host_ested_block_num {0}".format(host_ested_block_num))

    # file_full_path, block_num, block_size, features, labels, normalize_func, one_hot_encode_func
    print("### show guest validation sample blocks ...")
    overlap_block_file_full_path = dataset_folder_path + 'guest_val_block_'
    show_data_block(overlap_block_file_full_path, guest_val_block_num)

    print("### show host validation sample blocks ...")
    overlap_block_file_full_path = dataset_folder_path + 'host_val_block_'
    show_data_block(overlap_block_file_full_path, host_val_block_num)

    print("### show guest overlap sample blocks ...")
    overlap_block_file_full_path = dataset_folder_path + 'guest_overlap_block_'
    show_data_block(overlap_block_file_full_path, guest_overlap_block_num)

    print("### show host overlap sample blocks ...")
    overlap_block_file_full_path = dataset_folder_path + 'host_overlap_block_'
    show_data_block(overlap_block_file_full_path, host_overlap_block_num)

    print("### show guest non-overlap sample blocks ...")
    overlap_block_file_full_path = dataset_folder_path + 'guest_nonoverlap_block_'
    show_data_block(overlap_block_file_full_path, guest_nonoverlap_block_num)

    print("### show host non-overlap sample blocks ...")
    overlap_block_file_full_path = dataset_folder_path + 'host_nonoverlap_block_'
    show_data_block(overlap_block_file_full_path, host_nonoverlap_block_num)

    print("### show guest estimated sample blocks ...")
    overlap_block_file_full_path = dataset_folder_path + 'guest_ested_block_'
    show_data_block(overlap_block_file_full_path, guest_ested_block_num)

    print("### show host estimated sample blocks ...")
    overlap_block_file_full_path = dataset_folder_path + 'host_ested_block_'
    show_data_block(overlap_block_file_full_path, host_ested_block_num)


if __name__ == "__main__":

    # data_path = "../../data/"
    # # data_path = "../data/"
    # dataset_folder_path = data_path + "fashionmnist/"
    # num_overlap_samples = 250
    # to_processed_data_folder_path = data_path + "fashionmnist" + "_" + str(num_overlap_samples) + "/"
    #
    # if not os.path.exists(to_processed_data_folder_path):
    #     print("{0} does not exist, create one".format(to_processed_data_folder_path))
    #     os.makedirs(to_processed_data_folder_path)
    #
    # preprocess_and_save_data_v2(from_dataset_folder_path=dataset_folder_path,
    #                             to_dataset_folder_path=to_processed_data_folder_path,
    #                             load_data=load_fashionMNIST_data,
    #                             num_overlap=num_overlap_samples,
    #                             guest_nonoverlap_block_size=5000,
    #                             guest_estimation_block_size=5000,
    #                             host_nonoverlap_block_size=5000,
    #                             host_estimation_block_size=5000)

    # preprocess_and_save_data(dataset_folder_path=dataset_folder_path,
    #                          load_data=load_fashionMNIST_data)

    # batch_id = 0
    # sample_id = 77
    # display_stats_v2(dataset_folder_path + "guest_overlap_block_", batch_id, sample_id)
    # display_stats_v2(dataset_folder_path + "host_overlap_block_", batch_id, sample_id)

    # show_data(dataset_folder_path)

    data_path = "../../data/"
    # data_path = "../data/"
    dataset_folder_path = data_path + "fashionmnist"
    num_overlap_samples = 250
    to_processed_data_folder_path = dataset_folder_path + "_" + str(num_overlap_samples)

    if not os.path.exists(to_processed_data_folder_path):
        print("{0} does not exist, create one.".format(to_processed_data_folder_path))
        os.makedirs(to_processed_data_folder_path)

    preprocess_and_save_data_v2(from_dataset_folder_path=dataset_folder_path + "/",
                                to_dataset_folder_path=to_processed_data_folder_path + "/",
                                load_data=load_fashionMNIST_data,
                                num_overlap=num_overlap_samples,
                                guest_nonoverlap_block_size=4000,
                                guest_estimation_block_size=5000,
                                host_nonoverlap_block_size=4000,
                                host_estimation_block_size=5000)

