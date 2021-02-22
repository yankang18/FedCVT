import time

import numpy
# from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_visualization import projection_with_tsne, projection_with_autoencoder


def generate_time_stamp():
    local_time = time.localtime(time.time())
    return time.strftime("%Y%m%d%H%M%S", local_time)


def euclidean_distances(point1, point2):
    return np.sqrt(np.sum(np.square((point1 - point2))))


def split_into_4_clusters(data_X, data_Y):
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

    partyA_X = np.array(partyA_X)
    partyB_X = np.array(partyB_X)
    partyC_X = np.array(partyC_X)
    partyD_X = np.array(partyD_X)

    partyA_Y = np.array(partyA_Y)
    partyB_Y = np.array(partyB_Y)
    partyC_Y = np.array(partyC_Y)
    partyD_Y = np.array(partyD_Y)

    # partyA_Y = partyA_Y.reshape((len(partyA_Y), 1))
    # partyB_Y = partyB_Y.reshape((len(partyB_Y), 1))
    # partyC_Y = partyC_Y.reshape((len(partyC_Y), 1))
    # partyD_Y = partyD_Y.reshape((len(partyD_Y), 1))

    print(partyA_X.shape, partyA_Y.shape)
    print(partyB_X.shape, partyB_Y.shape)
    print(partyC_X.shape, partyC_Y.shape)
    print(partyD_X.shape, partyD_Y.shape)

    clusters = [(partyA_X, partyA_Y), (partyB_X, partyB_Y), (partyC_X, partyC_Y), (partyD_X, partyD_Y)]
    return clusters, centers


def generate_two_party_data(clusters):
    partyA_merged_X = np.concatenate((clusters[0][0], clusters[1][0]), axis=0)
    partyA_merged_Y = np.concatenate((clusters[0][1], clusters[1][1]), axis=0)

    partyB_merged_X = np.concatenate((clusters[2][0], clusters[3][0]), axis=0)
    partyB_merged_Y = np.concatenate((clusters[2][1], clusters[3][1]), axis=0)

    partyA_merged_Y = partyA_merged_Y.reshape((len(partyA_merged_Y), 1))
    partyB_merged_Y = partyB_merged_Y.reshape((len(partyB_merged_Y), 1))

    print("merged A_X", partyA_merged_X.shape, " with pos labels:", np.sum(partyA_merged_Y))
    print("merged B_X", partyB_merged_X.shape, " with pos labels:", np.sum(partyB_merged_Y))
    return partyA_merged_X, partyA_merged_Y, partyB_merged_X, partyB_merged_Y


if __name__ == "__main__":

    data_X, data_Y = [], []
    # with open("./spambase/spambase.datasets") as fin:
    with open("./datasets/credit_data.csv") as fin:
        # with open("./credit_g/credit_g_num_data") as fin:
        for line in fin:
            data = line.split(",")
            data_X.append([float(e) for e in data[:-1]])
            data_Y.append(int(data[-1]))

    data_X = numpy.array(data_X)
    data_Y = numpy.array(data_Y)
    print(data_X.shape, data_Y.shape)
    print("sum of Y:", np.sum(data_Y))

    scaler = StandardScaler()
    data_X = scaler.fit_transform(data_X)

    clusters, _ = split_into_4_clusters(data_X, data_Y)
    # projection_with_autoencoder(clusters)

    while True:
        partyA_X, partyA_Y, partyB_X, partyB_Y = generate_two_party_data(clusters)
        print("A_X", partyA_X.shape, partyA_Y.shape)
        print("B_X", partyB_X.shape, partyB_Y.shape)

        a_ratio = float(sum(partyA_Y) / len(partyA_X))
        b_ratio = float(sum(partyB_Y) / len(partyB_X))
        if 3/4 >= a_ratio >= 1/4 and 3/4 >= b_ratio >= 1/4:
            print("a_ratio", a_ratio)
            print("b_ratio Y", b_ratio)
            break

    partyA = np.concatenate((partyA_X, partyA_Y), axis=1)
    partyB = np.concatenate((partyB_X, partyB_Y), axis=1)

    print("partyA.shape", partyA.shape)
    print("partyB.shape", partyB.shape)

    time_stamp = generate_time_stamp()
    np.savetxt("./datasets/" + time_stamp + "_partyA_" + str(len(partyA)) + "_" + str(np.sum(partyA_Y)) + ".datasets", partyA, delimiter=",")
    np.savetxt("./datasets/" + time_stamp + "_partyB_" + str(len(partyB)) + "_" + str(np.sum(partyB_Y)) + ".datasets", partyB, delimiter=",")












