import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from logistic_regression import LogisticRegression

from instance_weighting_based_transfer import transfer_label_predict
from instance_weighting import calculate_weights_based_on_autoencoder_loss, calculate_weights_based_on_pdf_ratio


def train(partA_X, partA_Y, partB_X, partB_Y, hidden_dim, A_sample_weight_mean=None):
    print("partA_X shape", partA_X.shape)
    print("partA_Y shape", partA_Y.shape)
    print("partB_X shape", partB_X.shape)
    print("partB_Y shape", partB_Y.shape)

    # partA_X_half = partA_X[:1542]
    # print("partA_X_half shape", partA_X_half.shape)
    part_X = np.concatenate((partA_X, partB_X), axis=0)
    np.random.shuffle(part_X)

    # hidden_dim = int(partA_X.shape[1] * 0.8)
    # print("hidden_dim", hidden_dim)

    tf.compat.v1.reset_default_graph()

    autoencoder = Autoencoder(0)
    autoencoder.build(input_dim=partA_X.shape[1], hidden_dim_list=hidden_dim, learning_rate=0.01)

    logistic_regressor = LogisticRegression(1)
    logistic_regressor.build(input_dim=hidden_dim[-1], learning_rate=0.01,
                             representation=autoencoder.get_all_hidden_reprs())
    # logistic_regressor.build(input_dim=hidden_dim[-1], learning_rate=0.01)

    ae_batch_size = 512
    ae_epoch = 1000
    ae_N, ae_D = part_X.shape
    ae_n_batches = ae_N // ae_batch_size

    clf_batch_size = 512
    clf_epoch = 1000
    clf_N, clf_D = partA_X.shape
    clf_n_batches = clf_N // clf_batch_size
    print("clf_n_batches", clf_n_batches)

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        autoencoder.set_session(sess)
        logistic_regressor.set_session(sess)
        sess.run(init)

        for epo in range(1):
            costs = []
            print("--> autoencoder training")
            for ep in range(ae_epoch):
                for i in range(ae_n_batches + 1):
                    batch = part_X[i * ae_batch_size: i * ae_batch_size + ae_batch_size]
                    _, c = sess.run([autoencoder.e2e_train_op, autoencoder.loss], feed_dict={autoencoder.X_all_in: batch})
                    costs.append(c)

                    if ep % 100 == 0:
                        print(ep, i, "/", ae_n_batches, "cost:", c)

                if len(costs) > 1 and costs[-1] < 0.001 and abs(costs[-1] - costs[len(costs) - 2]) < 0.001:
                    # print(costs)
                    print("converged")
                    break

            # plt.title("autoencoder")
            # plt.plot(costs)
            # plt.show()

            costs = []
            print("--> classifier training")
            for ep in range(clf_epoch):
                for i in range(clf_n_batches + 1):
                    partA_X_batch = partA_X[i * clf_batch_size: i * clf_batch_size + clf_batch_size]
                    partA_Y_batch = partA_Y[i * clf_batch_size: i * clf_batch_size + clf_batch_size]
                    partA_Y_batch = partA_Y_batch.reshape(partA_Y_batch.shape[0], 1)
                    # weights_batch = A_sample_weight_mean[i * clf_batch_size: i * clf_batch_size + clf_batch_size]
                    # print("partA_X_batch partA_Y_batch shape", partA_X_batch.shape, partA_Y_batch.shape)

                    # partA_X_repr_batch = autoencoder.transform(partA_X_batch)
                    # feed_dictionary = {logistic_regressor.X_in: partA_X_repr_batch, logistic_regressor.labels: partA_Y_batch}

                    feed_dictionary = {autoencoder.X_all_in: partA_X_batch, logistic_regressor.labels: partA_Y_batch}
                    _, c = sess.run([logistic_regressor.e2e_train_op, logistic_regressor.loss],
                                    feed_dict=feed_dictionary)
                    costs.append(c)

                    if ep % 100 == 0:
                        print(ep, i, "/", clf_n_batches, "cost:", c)

                if len(costs) > 1 and costs[-1] < 0.01 and abs(costs[-1] - costs[len(costs) - 2]) < 0.01:
                    print(costs, costs[-1])
                    print("converged")
                    break

            # plt.title("classification")
            # plt.plot(costs)
            # plt.show()

        # print("partyB[-1]", partyB[:, -1])
        # print("partB_X", partB_X.shape)

        # partB_Y = partB_Y.reshape(partB_Y.shape[0], 1)
        # partB_X_repr_batch = autoencoder.transform(partB_X)
        # eval_result = sess.run([logistic_regressor.accuracy],
        #                        feed_dict={logistic_regressor.X_in: partB_X_repr_batch, logistic_regressor.labels: partB_Y})

        partB_Y = partB_Y.reshape(partB_Y.shape[0], 1)
        eval_result = sess.run([logistic_regressor.accuracy],
                               feed_dict={autoencoder.X_all_in: partB_X, logistic_regressor.labels: partB_Y})

        print("eval_result", eval_result)
        return eval_result


if __name__ == "__main__":

    partyA = np.loadtxt("./datasets/20190224170051_partyA_3084_800.datasets", delimiter=",")
    partyB = np.loadtxt("./datasets/20190224170051_partyB_1517_1013.datasets", delimiter=",")
    np.random.shuffle(partyA)
    np.random.shuffle(partyB)

    # using overlap
    partA_X_all = partyA[:, :-1]
    partB_X_all = partyB[:, :-1]
    partA_Y = partyA[:, -1]
    partB_Y = partyB[:, -1]

    overlap = 20
    start_index_list = [0, 5, 15, 25, 35, 45]
    # start_index_list = [30]
    results = {}
    for index in start_index_list:
        print("index:index + overlap:", index, index + overlap)
        partA_X_ = partA_X_all[:, index:index + overlap]
        partB_X_ = partB_X_all[:, index:index + overlap]
        print("partA_X_.shape, partB_X_.shape", partA_X_.shape, partB_X_.shape)
        score = transfer_label_predict(partA_X_, partA_Y, partB_X_, partB_Y)

        partA_X = partA_X_all[:, index:index + overlap]
        partB_X = partB_X_all[:, index:index + overlap]

        result_list = []
        for r in range(10):
            # A_sample_weight_mean, filtered_partA_X, filtered_partA_Y = calculate_weights_based_on_pdf_ratio(
            #     partB_X,
            #     partA_X,
            #     partA_X,
            #     partA_Y)
            #     # hidden_dim=hidden_dim,
            #     # exclude_ratio=0.1)

            result = train(partA_X, partA_Y, partB_X, partB_Y, [16, 12])
            # result = train(filtered_partA_X, filtered_partA_Y, partB_X, partB_Y, [16, 12])
            result_list.append(result)
            print("[using overlapping features of part A] score", score)

        results[str(index)] = {"benchmark": score, "tl": result_list}

    print("results\n", results)
