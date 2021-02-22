import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import shuffle

from autoencoder_bk import Autoencoder
from instance_weighting import calculate_weights_based_on_autoencoder_loss, calculate_weights_based_on_pdf_ratio


def train_autoencoder_for_dimension_reduction(samples_X, hidden_dim=2):
    tf.compat.v1.reset_default_graph()
    autoencoder = Autoencoder(0)
    autoencoder.build(input_dim=samples_X.shape[1], hidden_dim=hidden_dim, learning_rate=0.01)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        autoencoder.set_session(session)
        session.run(init_op)
        autoencoder.fit(samples_X, batch_size=512, epoch=1000, show_fig=False)

        model_parameters = autoencoder.get_model_parameters()
        # print("model_parameters", model_parameters)

        with open('./autoencoder_model_parameters', 'wb') as outfile:
            # json.dump(model_parameters, outfile)
            pickle.dump(model_parameters, outfile)

        output = autoencoder.transform(samples_X)

    return output


def projection_with_autoencoder(party_data_list, hidden_dim=2):
    party_re_list = []
    for party_data in party_data_list:
        party_X, party_Y = party_data[0], party_data[1]
        party_X_re = train_autoencoder_for_dimension_reduction(party_X, hidden_dim=hidden_dim)
        party_re_list.append((party_X_re, party_Y))

    # party_data_plot(party_re_list)
    (fig, subplots) = plt.subplots(1, len(party_data_list), figsize=(10, 6), squeeze=False)
    for i in range(len(party_data_list)):
        ax = subplots[0][i]
        party_i_X = party_re_list[i][0]
        party_i_Y = party_re_list[i][1]
        print("party_i_X, party_i_Y", party_i_X.shape, party_i_Y.shape)
        ax.scatter(party_i_X[party_i_Y == 0, 0], party_i_X[party_i_Y == 0, 1], c="r")
        ax.scatter(party_i_X[party_i_Y == 1, 0], party_i_X[party_i_Y == 1, 1], c="g")
        # ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
    plt.show()


def projection_with_tsne(party_data_list, n_components=2):
    tsne = manifold.TSNE(n_components=n_components, perplexity=50, init='random', random_state=0)

    party_re_list = []
    for party_data in party_data_list:
        party_X, party_Y = party_data[0], party_data[1]
        party_X_re = tsne.fit_transform(party_X)
        party_re_list.append((party_X_re, party_Y))

    party_data_plot(party_re_list)


def party_data_plot(party_re_list):
    (fig, subplots) = plt.subplots(1, len(party_re_list), figsize=(12, 6), squeeze=False)
    for i in range(len(party_re_list)):
        ax = subplots[0][i]
        party_i_X = party_re_list[i][0]
        party_i_Y = party_re_list[i][1]
        ax.scatter(party_i_X[party_i_Y == 0, 0], party_i_X[party_i_Y == 0, 1], c="r")
        ax.scatter(party_i_X[party_i_Y == 1, 0], party_i_X[party_i_Y == 1, 1], c="g")
        # ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        ax.set_xlim((-100, 100))
        ax.set_ylim((-100, 100))
    plt.show()


def series_plot(losses, fscores, aucs):
    fig = plt.figure(figsize=(20, 40))

    plt.subplot(311)
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('values')
    plt.title("loss")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(fscores)
    plt.xlabel('epoch')
    plt.ylabel('values')
    plt.title("fscore")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(aucs)
    plt.xlabel('epoch')
    plt.ylabel('values')
    plt.title("auc")
    plt.grid(True)

    # plt.subplot(414)
    # plt.plot(vs3)
    # plt.xlabel('epoch')
    # plt.ylabel('values')
    # plt.title("auc")
    # plt.grid(True)

    plt.show()

if __name__ == "__main__":
    # partyA = np.loadtxt("./datasets/20190224170051_partyA_3084_800.datasets", delimiter=",")
    # partyB = np.loadtxt("./datasets/20190224170051_partyB_1517_1013.datasets", delimiter=",")

    partyA = np.loadtxt("./datasets/UCI_Credit_Card/20190225141443_partyA_4913_2686.0.datasets", delimiter=",")
    partyB = np.loadtxt("./datasets/UCI_Credit_Card/20190225141443_partyB_8359_3950.0.datasets", delimiter=",")

    # projection for party A and party B separately
    projection_with_autoencoder([(partyA[:, :-1], partyA[:, -1]), (partyB[:, :-1], partyB[:, -1])])

    partA_X = partyA[:, 25:45]
    partA_Y = partyA[:, -1]
    partB_X = partyB[:, 25:45]
    partB_Y = partyB[:, -1]

    # projection_with_tsne([(partA_X, partA_Y), (partB_X, partB_Y)])
    projection_with_autoencoder([(partA_X, partA_Y), (partA_X, partA_Y)])

    A_sample_weight_mean, filtered_partA_X, filtered_partA_Y = calculate_weights_based_on_pdf_ratio(
        partB_X,
        partA_X,
        partA_X,
        partA_Y)
        # hidden_dim=hidden_dim,
        # exclude_ratio=0.1)

    projection_with_autoencoder([(filtered_partA_X, filtered_partA_Y), (partB_X, partB_Y)])

    # projection for all datasets
    # partyA_X, partyA_Y, partyB_X, partyB_Y = partyA[:, :-1], partyA[:, -1], partyB[:, :-1], partyB[:, -1]
    # partyAll_X = np.concatenate((partyA_X, partyB_X), axis=0)
    # partyAll_Y = np.concatenate((partyA_Y, partyB_Y), axis=0)
    #
    # partyAll_X, partyAll_Y = shuffle(partyAll_X, partyAll_Y, random_state=0)
    #
    # projection_with_autoencoder([(partyAll_X, partyAll_Y)])
