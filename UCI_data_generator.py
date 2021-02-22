import numpy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing.data import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from autoencoder import Autoencoder
import tensorflow as tf
import numpy as np
import json
import pickle
import time
import csv
from sklearn.utils import shuffle

def balance_X_y(X, y, seed=5):
    np.random.seed(seed)
    num_pos = np.sum(y == 1)
    num_neg = np.sum(y == 0)
    pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0]
    neg_indexes = [i for (i, _y) in enumerate(y) if _y <= 0]

    print("num_pos < num_neg", num_pos, num_neg)
    if num_pos < num_neg:
        np.random.shuffle(neg_indexes)
        # randomly pick negative samples of the size equal to that of positive samples
        rand_indexes = neg_indexes[:int(num_pos)]
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

    def get_data(self):
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

    def set_balance(self, balance):
        self.balance = balance

    def get_data(self):
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
            X, y = balance_X_y(X, y, seed)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        X, y = shuffle_X_y(X, y, seed)

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

    def get_data(self):
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


def generate_time_stamp():
    local_time = time.localtime(time.time())
    return time.strftime("%Y%m%d%H%M%S", local_time)


def euclidean_distances(point1, point2):
    return np.sqrt(np.sum(np.square((point1 - point2))))


def generate_two_party_data_by_two_clusters(data_X, data_Y):

    kmeans = KMeans(n_clusters=2).fit(data_X)
    centers = np.array(kmeans.cluster_centers_)
    # print(centers, centers.shape)
    # print(kmeans.labels_)

    partyA_X = []
    partyB_X = []
    partyA_Y = []
    partyB_Y = []
    for point, label in zip(data_X, data_Y):
        dist_0 = euclidean_distances(point, centers[0])
        dist_1 = euclidean_distances(point, centers[1])
        if dist_0 < dist_1:
            partyA_X.append(point)
            partyA_Y.append(label)
        else:
            partyB_X.append(point)
            partyB_Y.append(label)

    print("A cluster", len(partyA_X), " with pos labels:", np.sum(partyA_Y))
    print("B cluster", len(partyB_X), " with pos labels:", np.sum(partyB_Y))
    print("------------")

    partyA_X, partyA_Y = np.array(partyA_X), np.array(partyA_Y)
    partyB_X, partyB_Y = np.array(partyB_X), np.array(partyB_Y)

    partyA_Y = partyA_Y.reshape((len(partyA_Y), 1))
    partyB_Y = partyB_Y.reshape((len(partyB_Y), 1))

    return partyA_X, partyA_Y, partyB_X, partyB_Y


def generate_two_party_data_by_four_clusters(data_X, data_Y):

    kmeans = KMeans(n_clusters=4).fit(data_X)
    centers = np.array(kmeans.cluster_centers_)
    # print(centers, centers.shape)
    # print(kmeans.labels_)

    partyA_X = []
    partyB_X = []
    partyC_X = []
    partyD_X = []
    partyA_Y = []
    partyB_Y = []
    partyC_Y = []
    partyD_Y = []
    for point, label in zip(data_X, data_Y):
        dist_0 = euclidean_distances(point, centers[0])
        dist_1 = euclidean_distances(point, centers[1])
        dist_2 = euclidean_distances(point, centers[2])
        dist_3 = euclidean_distances(point, centers[3])
        # dist_4 = euclidean_distances(point, centers[4])
        if dist_0 < dist_1 and dist_0 < dist_2 and dist_0 < dist_3:
            partyA_X.append(point)
            partyA_Y.append(label)
        elif dist_1 < dist_0 and dist_1 < dist_2 and dist_1 < dist_3:
            partyB_X.append(point)
            partyB_Y.append(label)
        elif dist_2 < dist_0 and dist_2 < dist_1 and dist_2 < dist_3:
            partyC_X.append(point)
            partyC_Y.append(label)
        else:
            partyD_X.append(point)
            partyD_Y.append(label)

    print("A cluster", len(partyA_X), " with pos labels:", np.sum(partyA_Y))
    print("B cluster", len(partyB_X), " with pos labels:", np.sum(partyB_Y))
    print("C cluster", len(partyC_X), " with pos labels:", np.sum(partyC_Y))
    print("D cluster", len(partyD_X), " with pos labels:", np.sum(partyD_Y))
    print("------------")

    partyA_merged_X = np.concatenate((partyA_X, partyB_X), axis=0)
    partyA_merged_Y = np.concatenate((partyA_Y, partyB_Y), axis=0)

    partyB_merged_X = np.concatenate((partyC_X, partyD_X), axis=0)
    partyB_merged_Y = np.concatenate((partyC_Y, partyD_Y), axis=0)

    partyA_merged_Y = partyA_merged_Y.reshape((len(partyA_merged_Y), 1))
    partyB_merged_Y = partyB_merged_Y.reshape((len(partyB_merged_Y), 1))

    print("merged A_X", partyA_merged_X.shape, " with pos labels:", np.sum(partyA_merged_Y))
    print("merged B_X", partyB_merged_X.shape, " with pos labels:", np.sum(partyB_merged_Y))
    return partyA_merged_X, partyA_merged_Y, partyB_merged_X, partyB_merged_Y


# if __name__ == "__main__":

    # data_X, data_Y = load_UCI_Credit_Card_data(infile, balanced=False)
    # partyA_X, partyA_Y, partyB_X, partyB_Y = generate_two_party_data_by_two_clusters(data_X, data_Y)

    # if partyA_X.shape[0] > partyB_X.shape[0]:
    #     partyA_X, partyA_Y = balance_X_y(partyA_X, partyA_Y, seed=5)

    # partyA = np.concatenate((partyA_X, partyA_Y), axis=1)
    # partyB = np.concatenate((partyB_X, partyB_Y), axis=1)
    #
    # print("partyA.shape", partyA.shape)
    # print("partyB.shape", partyB.shape)
    #
    # a_ratio = float(sum(partyA_Y) / len(partyA_X))
    # b_ratio = float(sum(partyB_Y) / len(partyB_X))
    # print("a_ratio", a_ratio)
    # print("b_ratio Y", b_ratio)
    #
    # # while True:
    # #     partyA_X, partyA_Y, partyB_X, partyB_Y = generate_two_party_data_by_four_clusters(data_X, data_Y)
    # #     print("A_X", partyA_X.shape, partyA_Y.shape)
    # #     print("B_X", partyB_X.shape, partyB_Y.shape)
    # #
    # #     a_ratio = float(sum(partyA_Y) / len(partyA_X))
    # #     b_ratio = float(sum(partyB_Y) / len(partyB_X))
    # #     if 3/4 >= a_ratio >= 1/3 and 3/4 >= b_ratio >= 1/4:
    # #         print("a_ratio", a_ratio)
    # #         print("b_ratio Y", b_ratio)
    # #         break
    # #
    # # partyA = np.concatenate((partyA_X, partyA_Y), axis=1)
    # # partyB = np.concatenate((partyB_X, partyB_Y), axis=1)
    # #
    # # print("partyA.shape", partyA.shape)
    # # print("partyB.shape", partyB.shape)
    # #
    # time_stamp = generate_time_stamp()
    # np.savetxt(fname="./datasets/UCI_Credit_Card/" + time_stamp + "_partyA_" + str(len(partyA)) + "_" + str(np.sum(partyA_Y)) + ".datasets", X=partyA, delimiter=",")
    # np.savetxt(fname="./datasets/UCI_Credit_Card/" + time_stamp + "_partyB_" + str(len(partyB)) + "_" + str(np.sum(partyB_Y)) + ".datasets", X=partyB, delimiter=",")

