import pickle


def load_preprocess_training_minibatch(block_file_full_path, block_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """

    features, labels = load_training_batch(block_file_full_path, block_id)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def load_training_batch(block_file_full_path, block_id):
    filename = block_file_full_path + str(block_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


def load_data_block(block_file_full_path, block_id):
    filename = block_file_full_path + str(block_id) + '.p'
    print("load data block: {0}".format(filename))
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]
