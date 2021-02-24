import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
# from data_util.common_data_util import normalize, one_hot_encode, normalize_w_min_max
import json

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
                                guest_nonoverlap_block_size,
                                guest_estimation_block_size,
                                host_nonoverlap_block_size,
                                host_estimation_block_size):
    train_features, train_labels, val_features, val_labels = load_data(dataset_folder_path=from_dataset_folder_path)

    print("[INFO] from_dataset_folder_path: {0}".format(from_dataset_folder_path))
    print("[INFO] to_dataset_folder_path: {0}".format(to_dataset_folder_path))

    print("##############################")
    print("## preprocess and save data ##")
    print("##############################")

    print("[INFO] ===> normalize images")

    min_val = np.min(train_features)
    max_val = np.max(train_features)

    print("[INFO] max_val: {0}".format(max_val))
    print("[INFO] min_val: {0}".format(min_val))

    train_features = normalize_w_min_max(train_features, min_val=min_val, max_val=max_val)
    val_features = normalize_w_min_max(val_features, min_val=min_val, max_val=max_val)

    print("[INFO] ===> one-hot encode labels")

    train_labels = one_hot_encode(train_labels)
    val_labels = one_hot_encode(val_labels)

    print("[INFO] ===> prepare guest and host data")

    shape_dim = len(train_features.shape)
    feat_idx = shape_dim - 2
    feature_dim = train_features.shape[feat_idx]
    half_feature_dim = int(feature_dim / 2)

    guest_feature_start_idx = 0
    guest_feature_end_idx = half_feature_dim
    host_feature_start_idx = half_feature_dim
    host_feature_end_idx = feature_dim
    print("[INFO] train_features shape: {0}".format(shape_dim))
    print("[INFO] index of feature to be cut half: {0}".format(feat_idx))
    print("[INFO] features dim: {0}".format(feature_dim))
    print("[INFO] guest features start from {0} to {1}".format(guest_feature_start_idx, guest_feature_end_idx))
    print("[INFO] host features start from {0} to {1}".format(host_feature_start_idx, host_feature_end_idx))

    # prepare validation data for guest and host
    # guest_val_features = val_features[:, :, :half_feature_dim, :]
    # host_val_features = val_features[:, :, half_feature_dim:, :]
    guest_val_features = val_features[:, :, guest_feature_start_idx:guest_feature_end_idx]
    host_val_features = val_features[:, :, host_feature_start_idx:host_feature_end_idx]
    num_val_samples = len(guest_val_features)

    # prepare overlapping data for guest and host
    print("[INFO] number of total train samples:", len(train_features))
    print("[INFO] number of overlapping samples:", num_overlap)
    print("[INFO] overlapping train sample split: from {0} to {1}".format(0, num_overlap))
    overlap_train_features, overlap_train_labels = train_features[0:num_overlap], train_labels[0:num_overlap]
    overlap_guest_train_features = overlap_train_features[:, :, guest_feature_start_idx:guest_feature_end_idx]
    overlap_host_train_features = overlap_train_features[:, :, host_feature_start_idx:host_feature_end_idx]

    # prepare non-overlapping data for guest and host
    num_nonoverlap = len(train_features) - num_overlap
    half_num_nonoverlap = int(num_nonoverlap / 2)

    print("[INFO] number of non-overlapping samples:", num_nonoverlap)
    print("[INFO] number of half non-overlapping samples:", half_num_nonoverlap)

    # split data into guest and host
    # nonoverlap_guest_train_features = train_features[num_overlap:num_overlap + half_num_nonoverlap, :,
    #                                   :half_feature_dim, :]
    # nonoverlap_host_train_features = train_features[num_overlap + half_num_nonoverlap:, :, half_feature_dim:, :]

    nl_guest_train_samples_from_idx = num_overlap
    nl_guest_train_samples_to_idx = num_overlap + half_num_nonoverlap
    print("[INFO] guest non-overlapping train sample split: from {0} to {1}".format(nl_guest_train_samples_from_idx,
                                                                                    nl_guest_train_samples_to_idx))
    nonoverlap_guest_train_features = train_features[nl_guest_train_samples_from_idx:nl_guest_train_samples_to_idx, :,
                                      guest_feature_start_idx:guest_feature_end_idx]
    nonoverlap_guest_train_labels = train_labels[nl_guest_train_samples_from_idx:nl_guest_train_samples_to_idx]

    nl_host_train_samples_from_idx = num_overlap + half_num_nonoverlap
    nl_host_train_samples_to_idx = len(train_features)
    print("[INFO] host non-overlapping train sample split: from {0} to {1}".format(nl_host_train_samples_from_idx,
                                                                                   nl_host_train_samples_to_idx))
    nonoverlap_host_train_features = train_features[nl_host_train_samples_from_idx:nl_host_train_samples_to_idx, :,
                                     host_feature_start_idx:host_feature_end_idx]
    nonoverlap_host_train_labels = train_labels[nl_host_train_samples_from_idx:nl_host_train_samples_to_idx]

    # collect all samples for guest and host, respectively
    guest_train_features = np.concatenate((overlap_guest_train_features, nonoverlap_guest_train_features), axis=0)
    guest_train_labels = np.concatenate((overlap_train_labels, nonoverlap_guest_train_labels), axis=0)
    host_train_features = np.concatenate((overlap_host_train_features, nonoverlap_host_train_features), axis=0)
    host_train_labels = np.concatenate((overlap_train_labels, nonoverlap_host_train_labels), axis=0)

    num_overlap = len(overlap_train_features)
    num_guest_nonoverlap = len(nonoverlap_guest_train_features)
    num_host_nonoverlap = len(nonoverlap_host_train_features)
    num_guest_estimation = len(guest_train_features)
    num_host_estimation = len(host_train_features)
    print("[INFO] overlap_train_features shape: {0}".format(overlap_train_features.shape))
    print("[INFO] overlap_train_labels shape: {0}".format(overlap_train_labels.shape))
    print("[INFO] overlap_guest_train_features shape: {0}".format(overlap_guest_train_features.shape))
    print("[INFO] overlap_host_train_features shape: {0}".format(overlap_host_train_features.shape))
    print("[INFO] non-overlap_guest_train_features shape: {0}".format(nonoverlap_guest_train_features.shape))
    print("[INFO] non-overlap_guest_train_labels shape: {0}".format(nonoverlap_guest_train_labels.shape))
    print("[INFO] non-overlap_host_train_features shape: {0}".format(nonoverlap_host_train_features.shape))
    print("[INFO] non-overlap_host_train_labels shape: {0}".format(nonoverlap_host_train_labels.shape))
    print("[INFO] all_guest_train_features shape: {0}".format(guest_train_features.shape))
    print("[INFO] all_guest_train_labels shape: {0}".format(guest_train_labels.shape))
    print("[INFO] all_host_train_features shape: {0}".format(host_train_features.shape))
    print("[INFO] all_host_train_labels shape: {0}".format(host_train_labels.shape))

    # overlap_block_size = 1
    # guest_nonoverlap_block_size = 4000
    # guest_ested_block_size = 5000
    # host_nonoverlap_block_size = 4000
    # host_ested_block_size = 5000

    val_block_num = get_batch_num(num_val_samples, num_val_samples)
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
    meta_data["guest_non-overlap_block_num"] = guest_nonoverlap_block_num
    meta_data["host_non-overlap_block_num"] = host_nonoverlap_block_num
    meta_data["guest_estimation_block_num"] = guest_ested_block_num
    meta_data["host_estimation_block_num"] = host_ested_block_num

    print("[INFO] ---> save meta data: {0}".format(meta_data))
    with open(to_dataset_folder_path + "meta_data.json", "w") as write_file:
        json.dump(meta_data, write_file)

    print("[INFO] ---> save guest and host data")

    print("[INFO] validation block size {0}, num {1}.".format(num_val_samples, val_block_num))
    print("[INFO] overlap block size {0}, num {1}.".format(num_overlap, overlap_block_num))
    print("[INFO] guest non-overlap block size {0}, num {1}.".format(guest_nonoverlap_block_size,
                                                                     guest_nonoverlap_block_num))
    print("[INFO] host non-overlap block size {0}, num {1}.".format(host_nonoverlap_block_size,
                                                                    host_nonoverlap_block_num))
    print("[INFO] guest estimation block size {0}, num {1}.".format(guest_estimation_block_size,
                                                                    guest_ested_block_num))
    print("[INFO] host estimation block size {0}, num {1}.".format(host_estimation_block_size,
                                                                   host_ested_block_num))

    # file_full_path, block_num, block_size, features, labels, normalize_func, one_hot_encode_func
    print("[INFO] processing all overlap sample block ...")
    all_overlap_sample_num_list = [250, 500, 1000, 2000, 4000]
    for all_overlap_sample_num in all_overlap_sample_num_list:
        print("[INFO] processing all overlap {0} samples ...".format(all_overlap_sample_num))
        block_file_full_path = to_dataset_folder_path + 'all_overlap_' + str(all_overlap_sample_num) + '_block_'
        _preprocess_and_save_batches(block_file_full_path,
                                     1,
                                     all_overlap_sample_num,
                                     train_features,
                                     train_labels)

    print("[INFO] processing guest validation sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_val_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 val_block_num,
                                 num_val_samples,
                                 guest_val_features,
                                 val_labels)

    print("[INFO] processing host validation sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_val_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 val_block_num,
                                 num_val_samples,
                                 host_val_features,
                                 val_labels)

    print("[INFO] processing guest overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_overlap_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 overlap_block_num,
                                 num_overlap,
                                 overlap_guest_train_features,
                                 overlap_train_labels)

    print("[INFO] processing host overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_overlap_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 overlap_block_num,
                                 num_overlap,
                                 overlap_host_train_features,
                                 overlap_train_labels)

    print("[INFO] processing guest non-overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_nonoverlap_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 guest_nonoverlap_block_num,
                                 guest_nonoverlap_block_size,
                                 nonoverlap_guest_train_features,
                                 nonoverlap_guest_train_labels)

    print("[INFO] processing host non-overlap sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_nonoverlap_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 host_nonoverlap_block_num,
                                 host_nonoverlap_block_size,
                                 nonoverlap_host_train_features,
                                 nonoverlap_host_train_labels)

    print("[INFO] processing guest estimated sample block ...")
    block_file_full_path = to_dataset_folder_path + 'guest_ested_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 guest_ested_block_num,
                                 guest_estimation_block_size,
                                 guest_train_features,
                                 guest_train_labels)

    print("[INFO] processing host estimated sample block ...")
    block_file_full_path = to_dataset_folder_path + 'host_ested_block_'
    _preprocess_and_save_batches(block_file_full_path,
                                 host_ested_block_num,
                                 host_estimation_block_size,
                                 host_train_features,
                                 host_train_labels)


if __name__ == "__main__":
    data_path = "../../../data/"
    cifar10_dataset_folder_path = data_path + "cifar-10-batches-py"

    num_overlap_samples = 500
    to_processed_data_folder_path = cifar10_dataset_folder_path + "_" + str(num_overlap_samples)

    if not os.path.exists(to_processed_data_folder_path):
        print("{0} does not exist, create one".format(to_processed_data_folder_path))
        os.makedirs(to_processed_data_folder_path)

    preprocess_and_save_data_v2(from_dataset_folder_path=cifar10_dataset_folder_path + "/",
                                to_dataset_folder_path=to_processed_data_folder_path + "/",
                                load_data=load_cifar_data,
                                num_overlap=num_overlap_samples,
                                guest_nonoverlap_block_size=4000,
                                guest_estimation_block_size=5000,
                                host_nonoverlap_block_size=4000,
                                host_estimation_block_size=5000)

    # batch_id = 1
    # sample_id = 10
    # display_stats_v2(cifar10_dataset_folder_path + "/preprocess_batch_", batch_id, sample_id)
    # display_stats_v2(cifar10_dataset_folder_path + "/preprocess_batch_", batch_id, sample_id)
