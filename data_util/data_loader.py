import csv

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle


def balance_X_y(X, y, bal_factor=1, seed=5):
    np.random.seed(seed)
    num_pos = np.sum(y == 1)
    num_neg = np.sum(y == 0)
    pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0]
    neg_indexes = [i for (i, _y) in enumerate(y) if _y <= 0]

    print("num_pos < num_neg", num_pos, num_neg)
    print("bal_factor", bal_factor)
    if bal_factor * num_pos < num_neg:
        np.random.shuffle(neg_indexes)
        # randomly pick negative samples of the size equal to bal_factor times of positive samples
        rand_indexes = neg_indexes[:bal_factor * int(num_pos)]
        indexes = pos_indexes + rand_indexes
        y = [y[i] for i in indexes]
        X = [X[i] for i in indexes]
    return np.array(X), np.array(y)


def shuffle_X_y(X, y, seed=5):
    np.random.seed(seed)
    data_size = X.shape[0]
    shuffle_index = list(range(data_size))
    np.random.shuffle(shuffle_index)
    X = X[shuffle_index, :]
    y = y[shuffle_index]
    return X, y


class TwoPartyDataLoader(object):

    def get_training_data(self, *args, **kwargs):
        pass

    def get_test_data(self, *args, **kwargs):
        pass


def load_UCI_Credit_Card_data(infile=None, balanced=True, seed=5):
    print("balanced", balanced)

    X = []
    y = []
    sids = []

    with open(infile, "r") as fi:
        fi.readline()
        reader = csv.reader(fi)
        for row in reader:
            sids.append(row[0])
            X.append(row[1:-1])
            y0 = int(row[-1])
            # if y0 == 0:
            #     y0 = -1
            y.append(y0)
    y = np.array(y)

    if balanced:
        X, y = balance_X_y(X, y, seed)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X, y = shuffle_X_y(X, y, seed)

    encoder = OneHotEncoder(categorical_features=[1, 2, 3])
    encoder.fit(X)
    X = encoder.transform(X).toarray()

    # X, y = shuffle_X_y(X, y, seed)

    scale_model = StandardScaler()
    X = scale_model.fit_transform(X)

    # return X, np.expand_dims(y, axis=1)
    return X, y


class TwoPartyUCICreditCardDataLoader(TwoPartyDataLoader):

    def __init__(self, infile, split_index=5, balanced=True, seed=None):
        self.infile = infile
        self.balanced = balanced
        self.split_index = split_index
        self.seed = seed

    def set_balance(self, balanced):
        self.balanced = balanced

    def get_training_data(self):
        return self.load_and_split_UCI_Credit_Card_data(self.infile,
                                                        balanced=self.balanced,
                                                        split_index=self.split_index,
                                                        seed=self.seed)

    def load_and_split_UCI_Credit_Card_data(self, infile=None, balanced=True, split_index=5, seed=5):
        print("balanced", balanced)

        X = []
        y = []
        sids = []

        with open(infile, "r") as fi:
            fi.readline()
            reader = csv.reader(fi)
            for row in reader:
                sids.append(row[0])
                X.append(row[1:-1])
                y0 = int(row[-1])
                # if y0 == 0:
                #     y0 = -1
                y.append(y0)
        y = np.array(y)

        if balanced:
            X, y = balance_X_y(X=X, y=y, seed=seed)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        X, y = shuffle_X_y(X=X, y=y, seed=seed)

        X_A = X[:, :split_index]
        X_B = X[:, split_index:]

        encoder = OneHotEncoder(categorical_features=[1, 2, 3])
        encoder.fit(X_A)
        X_A = encoder.transform(X_A).toarray()

        # X, y = shuffle_X_y(X, y, seed)

        scale_model = StandardScaler()
        X_A = scale_model.fit_transform(X_A)

        scale_model = StandardScaler()
        X_B = scale_model.fit_transform(X_B)

        # return X, np.expand_dims(y, axis=1)
        return X_A, X_B, y


class TwoPartyBreastDataLoader(TwoPartyDataLoader):

    def __init__(self, infile):
        self.infile = infile
        self.seed = 5

    def get_training_data(self):
        return self.load_breast_data(self.infile)

    def load_breast_data(self, infile=None, seed=5):

        X_A = []
        X_B = []
        y = []
        sids = []

        breast_a_file = infile + "breast_a.csv"
        breast_b_file = infile + "breast_b.csv"

        with open(breast_a_file, "r") as fi:
            fi.readline()
            reader = csv.reader(fi)
            for row in reader:
                sids.append(row[0])
                X_B.append(row[1:])

        with open(breast_b_file, "r") as fi:
            fi.readline()
            reader = csv.reader(fi)
            for row in reader:
                sids.append(row[0])
                X_A.append(row[2:])
                y0 = int(row[1])
                # if y0 == 0:
                #     y0 = -1
                y.append(y0)
        y = np.array(y)

        X_A = np.array(X_A, dtype=np.float32)
        X_B = np.array(X_B, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return shuffle(X_A, X_B, y)
