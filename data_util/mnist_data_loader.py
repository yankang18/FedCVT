import json
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from data_util.cifar_data_loader import preprocess_and_save_data_v2, load_data_block, get_batch_num, \
    _preprocess_and_save_batches, _preprocess_and_save
from data_util.common_data_util import one_hot_encode, normalize_w_min_max


def load_label_names():
    return ['T-shirt', 'Trouser', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def display_stats_v2(cifar10_dataset_folder_path, block_id, sample_id):
    features, labels = load_data_block(cifar10_dataset_folder_path, block_id)

    print("features shape:", features.shape)
    print("labels shape:", labels.shape)

    if not (0 <= sample_id < len(features)):
        print('{} samples in block {}.  {} is out of range.'.format(len(features), block_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(block_id))
    print('# of Samples: {}\n'.format(len(features)))

    label_names = load_label_names()
    # label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    # for key, value in label_counts.items():
    #     print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = np.argmax(labels[sample_id])

    print("sample_label:", sample_label)

    channel_size = sample_image.shape[2]

    if channel_size == 1:
        sample_image = np.squeeze(sample_image)

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    plt.imshow(sample_image)
    plt.show()


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

    print("train label before shuffle", train[:100, 0])
    print("test label before shuffle", test[:100, 0])

    train = shuffle(train)
    test = shuffle(test)

    Xtrain = train[:, 1:]
    Xtrain = Xtrain.reshape(len(Xtrain), 28, 28, 1)
    Ytrain = train[:, 0].astype(np.int32)
    Xtest = test[:, 1:]
    Xtest = Xtest.reshape(len(Xtest), 28, 28, 1)
    Ytest = test[:, 0].astype(np.int32)

    print("Ytrain label after shuffle", Ytrain[:100])
    print("Ytest label after shuffle", Ytest[:100])

    print("Xtrain shape:{0}".format(Xtrain.shape))
    print("Ytrain shape:{0}".format(Ytrain.shape))
    print("Xtest shape:{0}".format(Xtest.shape))
    print("Ytest shape:{0}".format(Ytest.shape))

    return Xtrain, Ytrain, Xtest, Ytest


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


if __name__ == "__main__":

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

    # batch_id = 0
    # sample_id = 99
    # display_stats_v2(to_processed_data_folder_path + "/" + "guest_overlap_block_", batch_id, sample_id)
    # display_stats_v2(to_processed_data_folder_path + "/" + "host_overlap_block_", batch_id, sample_id)

    # show_data(dataset_folder_path)
