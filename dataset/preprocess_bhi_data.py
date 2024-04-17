# from data_util.common_data_util import normalize, one_hot_encode, normalize_w_min_max
import json
import os
import pickle
# import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset.bhi_dataset import BHIDataset2Party

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    cifar10_data_block_full_path = cifar10_dataset_folder_path + '/data_batch_' + str(batch_id)
    print("load cifar10 block {0} from {1}".format(batch_id, cifar10_data_block_full_path))
    with open(cifar10_data_block_full_path, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


def load_data_block(block_file_full_path, block_id):
    filename = block_file_full_path + str(block_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))

    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    # plt.imshow(sample_image)


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

    print("sample_label", sample_label)

    channel_size = sample_image.shape[2]

    if channel_size == 1:
        sample_image = np.squeeze(sample_image)

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    print("sample_label 0:", sample_image[:, :, 0])
    print("sample_label 1:", sample_image[:, :, 1])
    print("sample_label 2:", sample_image[:, :, 2])

    plt.imshow(sample_image[:, :, 0])
    plt.show()
    plt.imshow(sample_image[:, :, 1])
    plt.show()
    plt.imshow(sample_image[:, :, 2])
    plt.show()
    plt.imshow(sample_image)
    plt.show()


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    if normalize is not None:
        features = normalize(features)
    if one_hot_encode is not None:
        labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))


# def preprocess_and_save_data(dataset_folder_path, normalize_func, one_hot_encode_func):
#     n_batches = 5
#
#     valid_features = []
#     valid_labels = []
#
#     for batch_i in range(1, n_batches + 1):
#         features, labels = load_cfar10_batch(dataset_folder_path, batch_i)
#
#         print("features shape {0}".format(features.shape))
#         print("labels shape {0}".format(len(labels)))
#
#         # find index to be the point as validation data in the whole dataset of the batch (10%)
#         index_of_validation = int(len(features) * 0.1)
#
#         # preprocess the 90% of the whole dataset of the batch
#         # - normalize the features
#         # - one_hot_encode the lables
#         # - save in a new file named, "preprocess_batch_" + batch_number
#         # - each file for each batch
#         train_file_full_path = dataset_folder_path + 'preprocess_batch_' + str(batch_i) + '.p'
#         _preprocess_and_save(normalize_func, one_hot_encode_func,
#                              features[:-index_of_validation], labels[:-index_of_validation],
#                              train_file_full_path)
#
#         # unlike the training dataset, validation dataset will be added through all batch dataset
#         # - take 10% of the whold dataset of the batch
#         # - add them into a list of
#         #   - valid_features
#         #   - valid_labels
#         valid_features.extend(features[-index_of_validation:])
#         valid_labels.extend(labels[-index_of_validation:])
#
#     # preprocess the all stacked validation dataset
#     val_file_full_path = dataset_folder_path + 'preprocess_validation.p'
#     _preprocess_and_save(normalize_func, one_hot_encode_func,
#                          np.array(valid_features), np.array(valid_labels),
#                          val_file_full_path)
#
#     # load the test dataset
#     with open(dataset_folder_path + '/test_batch', mode='rb') as file:
#         batch = pickle.load(file, encoding='latin1')
#
#     # preprocess the testing data
#     test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
#     test_labels = batch['labels']
#
#     # Preprocess and Save all testing data
#     test_file_full_path = dataset_folder_path + 'preprocess_test.p'
#     _preprocess_and_save(normalize_func, one_hot_encode_func,
#                          np.array(test_features), np.array(test_labels),
#                          test_file_full_path)


def get_batch_num(all_sample_size, batch_size):
    residual = all_sample_size % batch_size
    if residual == 0:
        return int(all_sample_size / batch_size)
    else:
        return int(all_sample_size / batch_size) + 1


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
        print("[INFO] processing samples [{0}:{1}]".format(start_idx, end_idx))
        _preprocess_and_save(normalize_func, one_hot_encode_func,
                             features[start_idx: end_idx], labels[start_idx: end_idx],
                             file_full_path + str(batch_idx) + '.p')


def load_cifar_data(dataset_folder_path):
    n_batches = 5
    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []

    # load cifar image data
    for batch_i in range(1, n_batches + 1):
        print("load block {0}".format(batch_i))
        features, labels = load_cfar10_batch(dataset_folder_path, batch_i)

        print("features shape {0}".format(features.shape))
        print("labels shape {0}".format(len(labels)))

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        print("index_of_validation", index_of_validation)

        train_features.append(features[:-index_of_validation])
        train_labels.append(labels[:-index_of_validation])

        valid_features.append(features[-index_of_validation:])
        valid_labels.append(labels[-index_of_validation:])

    train_features_array = np.concatenate(train_features, axis=0)
    train_labels_array = np.concatenate(train_labels, axis=0)

    print("[INFO] train_features_array shape {0}".format(train_features_array.shape))
    print("[INFO] train_labels_array shape {0}".format(len(train_labels_array)))

    val_features_array = np.concatenate(valid_features, axis=0)
    val_labels_array = np.concatenate(valid_labels, axis=0)

    print("[INFO] val_features_array shape {0}".format(val_features_array.shape))
    print("[INFO] val_labels_array shape {0}".format(len(val_labels_array)))

    return train_features_array, train_labels_array, val_features_array, val_labels_array


def preprocess_and_save_data_v2(from_dataset_folder_path,
                                to_dataset_folder_path,
                                load_data,
                                num_overlap,
                                num_labeled_overlap,
                                guest_nonoverlap_block_size,
                                guest_estimation_block_size,
                                host_nonoverlap_block_size,
                                host_estimation_block_size):

    print("from_dataset_folder_path:", from_dataset_folder_path)
    print("to_dataset_folder_path:", to_dataset_folder_path)
    guest_x_train, host_x_train, y_train = load_data(data_dir=from_dataset_folder_path, data_type="train")
    guest_x_val, host_x_val, y_val = load_data(data_dir=from_dataset_folder_path, data_type="test")

    print("[INFO] from_dataset_folder_path: {0}".format(from_dataset_folder_path))
    print("[INFO] to_dataset_folder_path: {0}".format(to_dataset_folder_path))

    print("##############################")
    print("## preprocess and save data ##")
    print("##############################")

    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val)

    print("[INFO] ===> prepare guest and host data")

    num_val_samples = len(guest_x_val)

    num_unlabeled_overlap = num_overlap - num_labeled_overlap

    guest_ll_x_train = guest_x_train[0:num_labeled_overlap]
    host_ll_x_train = host_x_train[0:num_labeled_overlap]
    ll_y_train = y_train[0:num_labeled_overlap]

    guest_ul_x_train = guest_x_train[num_labeled_overlap:num_overlap]
    host_ul_x_train = host_x_train[num_labeled_overlap:num_overlap]
    ul_y_train = y_train[num_labeled_overlap:num_overlap]

    num_train = guest_x_train.shape[0]
    num_non_overlap = num_train - num_overlap
    num_half_non_overlap = int(num_non_overlap / 2)

    guest_nl_x_train = guest_x_train[num_overlap:num_half_non_overlap]
    guest_nl_y_train = y_train[num_overlap:num_half_non_overlap]

    host_nl_x_train = host_x_train[
                      num_overlap + num_half_non_overlap:num_overlap + num_half_non_overlap + num_half_non_overlap]
    host_nl_y_train = y_train[
                      num_overlap + num_half_non_overlap:num_overlap + num_half_non_overlap + num_half_non_overlap]

    # #
    # # prepare non-overlapping data for guest and host
    # #

    # collect all samples for guest and host, respectively
    guest_all_x_train = np.concatenate((guest_ll_x_train, guest_ul_x_train, guest_nl_x_train), axis=0)
    guest_all_y_train = np.concatenate((ll_y_train, ul_y_train, guest_nl_y_train), axis=0)
    host_all_x_train = np.concatenate((host_ll_x_train, host_ul_x_train, host_nl_x_train), axis=0)
    host_all_y_train = np.concatenate((ll_y_train, ul_y_train, host_nl_y_train), axis=0)

    num_guest_estimation = len(guest_all_x_train)
    num_host_estimation = len(host_all_x_train)

    print("[INFO] guest_x_train shape: {0}".format(guest_x_train.shape))
    print("[INFO] host_x_train shape: {0}".format(host_x_train.shape))

    print("[INFO] guest_ll_x_train shape: {0}".format(guest_ll_x_train.shape))
    print("[INFO] guest_ul_x_train shape: {0}".format(guest_ul_x_train.shape))
    print("[INFO] non-guest_nl_x_train shape: {0}".format(guest_nl_x_train.shape))

    print("[INFO] non-host_ll_x_train shape: {0}".format(host_ll_x_train.shape))
    print("[INFO] non-host_ul_x_train shape: {0}".format(host_ul_x_train.shape))
    print("[INFO] non-host_nl_x_train shape: {0}".format(host_nl_x_train.shape))

    print("[INFO] guest_all_x_train shape: {0}".format(guest_all_x_train.shape))
    print("[INFO] guest_all_y_train shape: {0}".format(guest_all_y_train.shape))
    print("[INFO] host_all_x_train shape: {0}".format(host_all_x_train.shape))
    print("[INFO] num_guest_estimation shape: {0}".format(num_guest_estimation))
    print("[INFO] num_host_estimation shape: {0}".format(num_host_estimation))

    # overlap_block_size = 1
    # guest_nonoverlap_block_size = 4000
    # guest_ested_block_size = 5000
    # host_nonoverlap_block_size = 4000
    # host_ested_block_size = 5000

    val_block_num = get_batch_num(num_val_samples, num_val_samples)
    ll_block_num = get_batch_num(num_labeled_overlap, num_labeled_overlap)
    ul_block_num = get_batch_num(num_unlabeled_overlap, num_unlabeled_overlap)
    guest_nl_block_num = get_batch_num(guest_nl_x_train.shape[0], guest_nonoverlap_block_size)
    guest_ested_block_num = get_batch_num(num_guest_estimation, guest_estimation_block_size)
    host_nl_block_num = get_batch_num(host_nl_x_train.shape[0], host_nonoverlap_block_size)
    host_ested_block_num = get_batch_num(num_host_estimation, host_estimation_block_size)

    meta_data = dict()
    meta_data["guest_val_block_num"] = val_block_num
    meta_data["host_val_block_num"] = val_block_num
    meta_data["guest_ll_block_num"] = ll_block_num
    meta_data["host_ll_block_num"] = ll_block_num
    meta_data["guest_ul_block_num"] = ul_block_num
    meta_data["host_ul_block_num"] = ul_block_num
    meta_data["guest_nl_block_num"] = guest_nl_block_num
    meta_data["host_nl_block_num"] = host_nl_block_num
    meta_data["guest_estimation_block_num"] = guest_ested_block_num
    meta_data["host_estimation_block_num"] = host_ested_block_num

    print("[INFO] ---> save meta data: {0}".format(meta_data))
    with open(to_dataset_folder_path + "meta_data.json", "w") as write_file:
        json.dump(meta_data, write_file)

    print("[INFO] ---> save guest and host data")

    print("[INFO] validation block size {0}, num {1}.".format(num_val_samples, val_block_num))
    print("[INFO] overlap block size {0}, num {1}.".format(num_overlap, ll_block_num))
    print("[INFO] guest non-overlap block size {0}, num {1}.".format(guest_nonoverlap_block_size,
                                                                     guest_nl_block_num))
    print("[INFO] host non-overlap block size {0}, num {1}.".format(host_nonoverlap_block_size,
                                                                    host_nl_block_num))
    print("[INFO] guest estimation block size {0}, num {1}.".format(guest_estimation_block_size,
                                                                    guest_ested_block_num))
    print("[INFO] host estimation block size {0}, num {1}.".format(host_estimation_block_size,
                                                                   host_ested_block_num))

    # # file_full_path, block_num, block_size, features, labels, normalize_func, one_hot_encode_func
    # print("[INFO] processing all overlap sample block ...")
    # ll_sample_num_list = [200, 400, 600, 800, 1000]
    # for ll_sample_num in ll_sample_num_list:
    #     print("[INFO] processing ll samples {0} ...".format(ll_sample_num))
    #     block_file_full_path = to_dataset_folder_path + 'll_' + str(ll_sample_num) + '_block_'
    #     _preprocess_and_save_batches(file_full_path=block_file_full_path,
    #                                  block_num=1,
    #                                  block_size=ll_sample_num,
    #                                  features=x_train,
    #                                  labels=y_train,
    #                                  normalize_func=None,
    #                                  one_hot_encode_func=None)

    print("[INFO] processing guest validation sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_val_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=val_block_num,
                                 block_size=num_val_samples,
                                 features=guest_x_val,
                                 labels=y_val,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing host validation sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_val_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=val_block_num,
                                 block_size=num_val_samples,
                                 features=host_x_val,
                                 labels=y_val,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing guest labeled aligned sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_ll_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=ll_block_num,
                                 block_size=num_overlap,
                                 features=guest_ll_x_train,
                                 labels=ll_y_train,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing host labeled aligned sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_ll_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=ll_block_num,
                                 block_size=num_overlap,
                                 features=host_ll_x_train,
                                 labels=ll_y_train,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing guest unlabeled aligned sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_ul_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=ul_block_num,
                                 block_size=num_unlabeled_overlap,
                                 features=guest_ul_x_train,
                                 labels=ul_y_train,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing host unlabeled aligned sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_ul_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=ul_block_num,
                                 block_size=num_unlabeled_overlap,
                                 features=host_ul_x_train,
                                 labels=ul_y_train,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing guest non-overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_nl_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=guest_nl_block_num,
                                 block_size=guest_nonoverlap_block_size,
                                 features=guest_nl_x_train,
                                 labels=guest_nl_y_train,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing host non-overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_nl_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=host_nl_block_num,
                                 block_size=host_nonoverlap_block_size,
                                 features=host_nl_x_train,
                                 labels=host_nl_y_train,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing guest estimated sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_ested_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=guest_ested_block_num,
                                 block_size=guest_estimation_block_size,
                                 features=guest_all_x_train,
                                 labels=guest_all_y_train,
                                 normalize_func=None,
                                 one_hot_encode_func=None)

    print("[INFO] processing host estimated sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_ested_block_'
    _preprocess_and_save_batches(file_full_path=block_file_full_path,
                                 block_num=host_ested_block_num,
                                 block_size=host_estimation_block_size,
                                 features=host_all_x_train,
                                 labels=host_all_y_train,
                                 normalize_func=None,
                                 one_hot_encode_func=None)


def load_bhi_data(data_dir, data_type="train"):
    train_dataset = BHIDataset2Party(data_dir, data_type, 32, 32, 2)
    # X_train, Y_train = train_dataset.get_data()

    # X_guest_train = np.array(X_train[0])
    # X_host_train = np.array(X_train[1])
    #
    # print("### original data shape")
    # # print("X_image_train shape", X_image_train.shape)
    # # print("X_text_train shape", X_text_train.shape)
    # print("X_guest_train shape", X_train[0])
    # # print("X_image_test shape", X_image_test.shape)
    # # print("X_text_test shape", X_text_test.shape)
    # print("X_host_train shape", X_train[1].shape)

    train_dataloader = DataLoader(train_dataset, batch_size=100000, pin_memory=False, drop_last=False)

    guest = None
    host = None
    label = None
    for x, y in train_dataloader:
        guest = x[0]
        host = x[1]
        label = y
    return guest, host, label


if __name__ == "__main__":
    data_path = "../../../dataset/bhi/"
    to_data_path = "../../../dataset/bhi_proc/"

    num_overlap_samples = 4000
    # to_processed_data_folder_path = cifar10_dataset_folder_path + "_" + str(num_overlap_samples)
    # if not os.path.exists(to_processed_data_folder_path):
    #     print("{0} does not exist, create one".format(to_processed_data_folder_path))
    #     os.makedirs(to_processed_data_folder_path)

    preprocess_and_save_data_v2(from_dataset_folder_path=data_path,
                                to_dataset_folder_path=to_data_path,
                                load_data=load_bhi_data,
                                num_overlap=num_overlap_samples,
                                num_labeled_overlap=200,
                                guest_nonoverlap_block_size=4000,
                                guest_estimation_block_size=5000,
                                host_nonoverlap_block_size=4000,
                                host_estimation_block_size=5000)

    # batch_id = 1
    # sample_id = 10
    # display_stats_v2(cifar10_dataset_folder_path + "/preprocess_batch_", batch_id, sample_id)
    # display_stats_v2(cifar10_dataset_folder_path + "/preprocess_batch_", batch_id, sample_id)
