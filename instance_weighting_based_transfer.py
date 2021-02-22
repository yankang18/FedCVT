from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from autoencoder import Autoencoder
import tensorflow as tf
import numpy as np
import pickle

from instance_weighting import calculate_weights_based_on_autoencoder_loss


def transfer_label_predict(x1, y1, x2, y2):
    # print(x1.shape, x2.shape)
    clf = LogisticRegression(solver="lbfgs").fit(x1, y1)
    return clf.score(x2, y2)


def transfer_feature_A_2_B(partA_X, partB_X, A_start, A_end, B_start, B_end, A_sample_weight=None):
    model = DecisionTreeRegressor(max_depth=4).fit(partA_X[:, B_start - A_start:A_end - A_start],
                                                   partA_X[:, A_start - A_start:B_start - A_start],
                                                   sample_weight=A_sample_weight)
    output = model.predict(partB_X[:, 0:(A_end - B_start)])
    return output


def transfer_feature_A_2_B_linear(partA_X, partB_X, A_start, A_end, B_start, B_end, A_sample_weight=None):
    model = Ridge().fit(partA_X[:, B_start - A_start:A_end - A_start],
                        partA_X[:, A_start - A_start:B_start - A_start],
                        sample_weight=A_sample_weight)
    output = model.predict(partB_X[:, 0:(A_end - B_start)])
    return output


def transfer_feature_B_2_A(x, test_x, A_start, A_end, B_start, B_end):
    # poly_reg =PolynomialFeatures(degree=1)
    # x_ploy =poly_reg.fit_transform(x[:, 0:28])
    reg = Ridge(solver="saga").fit(x[:, 0:(A_end - B_start)], x[:, A_end - B_start:(B_end - B_start)])
    print("B2A", test_x.shape)
    # reg = DecisionTreeRegressor(max_depth=4).fit(x[:, 0:(A_end - B_start)], x[:, (A_end - B_start):(B_end - B_start)])
    output = reg.predict(test_x[:, B_start - A_start:])
    return output


def get_weights(partA_X):
    # test whether autoencoder can be restored from stored model parameters

    with open('./autoencoder_model_parameters', 'rb') as file:
        model_parameters = pickle.load(file)

    tf.compat.v1.reset_default_graph()

    autoencoder = Autoencoder(0)
    autoencoder.restore_model(model_parameters)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        autoencoder.set_session(session)
        session.run(init_op)

        output = autoencoder.compute_loss(partA_X)
        weights = np.mean(output, axis=1)
    return 1 / weights


def normalize_weights(weights):
    # weights = 1 / weights
    weights_1 = weights / np.sum(weights)
    weights_2 = softmax(weights)
    # print("weights_1", weights_1, weights_1.shape)
    # print("weights_2", weights_2, weights_2.shape)
    return weights_1, weights_2


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


def train(partA_X, partA_Y, partB_X, partB_Y, num_party_A_train_data, num_party_B_train_data):

    # over_lap_list = [5, 10, 15]
    # fore_num_list = [5, 10, 15]
    # back_num_list = [5, 10, 15]
    # start_index_list = [0, 5, 10]

    over_lap_list = [10, 15, 20]
    fore_num_list = [10, 15, 20]
    back_num_list = [10, 15, 20]
    start_index_list = [0, 5, 10]

    for over_lap in over_lap_list:
        for fore_num in fore_num_list:
            for back_num in back_num_list:
                for start_index in start_index_list:
                    A_start = start_index
                    A_end = start_index + fore_num + over_lap
                    B_start = start_index + fore_num
                    B_end = start_index + fore_num + over_lap + back_num
                    if A_end >= 57 or B_end >= 57:
                        continue

                    partA_sub_X = partA_X[:, A_start:A_end]
                    partB_sub_X = partB_X[:, B_start:B_end]

                    do_train(partA_sub_X, partA_Y, partB_sub_X, partB_Y, num_party_A_train_data, num_party_B_train_data,
                             A_start, A_end, B_start, B_end, over_lap)


# def train(data_X, data_Y, idx, A_start, A_end, B_start, B_end, over_lap):
def do_train(partA_X, partA_Y, partB_X, partB_Y, num_party_A_train_data, num_party_B_train_data,
             A_start, A_end, B_start, B_end, over_lap):
    print("info@: A_start, A_end, B_start, B_end:", A_start, A_end, B_start, B_end)
    print("info@: partA_X, partA_Y, partB_X, partB_Y:", partA_X.shape, partA_Y.shape, partB_X.shape, partB_Y.shape)

    partA_train_X = partA_X[:num_party_A_train_data]
    partA_train_Y = partA_Y[:num_party_A_train_data]
    partA_test_X = partA_X[num_party_A_train_data:]
    partA_test_Y = partA_Y[num_party_A_train_data:]

    partA_overlap_X = partA_X[:, B_start - A_start:]

    partB_train_X = partB_X[:num_party_B_train_data]
    partB_train_Y = partB_Y[:num_party_B_train_data]
    partB_test_X = partB_X[num_party_B_train_data:]
    partB_test_Y = partB_Y[num_party_B_train_data:]

    partB_overlap_X = partB_X[:, 0:(A_end - B_start)]

    # score = local_train(train_X, train_Y, test_X, test_Y)
    # print("all model:", score)
    # score = local_train(partA_train_X, partA_train_Y, partA_test_X, partA_test_Y)
    # print("part A model score:", score)
    # overlap_alone_score = local_train(partB_train_X, partB_train_Y, partB_test_X, partB_test_Y)
    # print("part B model score:", overlap_alone_score, partB_train_X.shape)

    partB_side_score = transfer_label_predict(partB_train_X, partB_train_Y, partB_test_X, partB_test_Y)

    overlap_alone_score = transfer_label_predict(partA_overlap_X, partA_Y, partB_overlap_X, partB_Y)

    output_nonlinear = transfer_feature_A_2_B(partA_X, partB_X, A_start, A_end, B_start, B_end)
    new_partB_nonlinear_X = np.concatenate((output_nonlinear, partB_overlap_X), axis=1)
    TrPartB_score = transfer_label_predict(partA_X, partA_Y, new_partB_nonlinear_X, partB_Y)

    output_linear = transfer_feature_A_2_B_linear(partA_X, partB_X, A_start, A_end, B_start, B_end)
    new_partB_linear_X = np.concatenate((output_linear, partB_overlap_X), axis=1)
    TrPartB_score_2 = transfer_label_predict(partA_X, partA_Y, new_partB_linear_X, partB_Y)

    print("[benchmark] partB_side_only_score", partB_side_score)
    print("overlap_alone_score", overlap_alone_score)
    print("TrPartB model score nonlinear", TrPartB_score)
    print("TrPartB model score linear", TrPartB_score_2)

    nonlinear_score_list = []
    linear_score_list = []
    iterations = 5
    for i in range(iterations):
        # A_sample_weight, _ = train_autoencoder(partB_overlap_X, partA_overlap_X, hidden_dim=int(over_lap*0.8))
        A_sample_weight_mean, filtered_partA_X, filtered_partA_Y = calculate_weights_based_on_autoencoder_loss(partB_overlap_X,
                                                                                                               partA_overlap_X,
                                                                                                               partA_X,
                                                                                                               partA_Y,
                                                                                                               hidden_dim=int(over_lap * 0.8),
                                                                                                               exclude_ratio=0.2)
        # A_sample_weight = get_weights(partA_X[:, B_start - A_start:])
        # A_sample_weight_mean, _ = normalize_weights(A_sample_weight)
        print("filtered partA_X, partA_Y", filtered_partA_X.shape, filtered_partA_Y.shape)
        nonlinear_output_with_sample_weight = transfer_feature_A_2_B(filtered_partA_X, partB_X, A_start, A_end, B_start, B_end,
                                                                     A_sample_weight_mean)
        linear_output_with_sample_weight = transfer_feature_A_2_B_linear(filtered_partA_X, partB_X, A_start, A_end, B_start, B_end,
                                                                         A_sample_weight_mean)

        # print(nonlinear_output_with_sample_weight.shape)
        # print(output_with_sample_weight_soft.shape)

        new_partB_nonlinear_X_with_sample_w = np.concatenate((nonlinear_output_with_sample_weight, partB_overlap_X), axis=1)
        new_partB_linear_X_with_sample_w = np.concatenate((linear_output_with_sample_weight, partB_overlap_X), axis=1)

        # print("new_partB_nonlinear_X_with_sample_w shape", new_partB_nonlinear_X_with_sample_w.shape)
        # print("new_partB_linear_X_with_sample_w shape", new_partB_linear_X_with_sample_w.shape)

        # new_partB_train_X_with_w = new_partB_nonlinear_X_with_sample_w[:num_party_B_train_data]
        # new_partB_test_X_with_w = new_partB_nonlinear_X_with_sample_w[num_party_B_train_data:]
        TrPartyB_nonlinear_score_with_weight = transfer_label_predict(filtered_partA_X, filtered_partA_Y, new_partB_nonlinear_X_with_sample_w, partB_Y)
        nonlinear_score_list.append(TrPartyB_nonlinear_score_with_weight)
        # print("TrPartB model score /w w", TrPartyB_nonlinear_score_with_weight)

        # new_partB_train_X_with_w_soft = new_partB_linear_X_with_sample_w[:num_party_B_train_data]
        # new_partB_test_X_with_w_soft = new_partB_linear_X_with_sample_w[num_party_B_train_data:]
        TrPartyB_linear_score_with_weight = transfer_label_predict(filtered_partA_X, filtered_partA_Y, new_partB_linear_X_with_sample_w, partB_Y)
        linear_score_list.append(TrPartyB_linear_score_with_weight)
    # print("TrPartB model score /w w_soft", TrPartyB_linear_score_with_weight)

    print("TrPartB TrPartyB_nonlinear_score_with_weight mean", np.mean(nonlinear_score_list))
    print("TrPartB TrPartyB_linear_score_with_weight mean", np.mean(linear_score_list))
    print("-" * 20)


# output = transfer_feature_B_2_A(partB_X, partA_X, A_start, A_end, B_start, B_end)
# new_partA_X = numpy.concatenate((partA_X, output), axis=1)
#
# new_partA_train_X = new_partA_X[:num_party_A_train_data]
# new_partA_test_X = new_partA_X[num_party_A_train_data:]
# score = local_train(new_partA_train_X, partA_train_Y, new_partA_test_X, partA_test_Y)
# print("TrPartA model score", score)


if __name__ == "__main__":

    # data_X, data_Y = [], []
    # with open("./spambase/spambase.datasets") as fin:
    #     # with open("./credit_g/credit_g_num_data") as fin:
    #     for line in fin:
    #         datasets = line.split(",")
    #         data_X.append([float(e) for e in datasets[:-1]])
    #         data_Y.append(int(datasets[-1]))
    # # print(data_X[0])
    #
    # data_X = numpy.array(data_X)
    # data_Y = numpy.array(data_Y)
    #
    # # print(data_X.shape, data_Y.shape)
    # # print("datasets Y", data_Y)
    #
    # scaler = StandardScaler()
    # data_X = scaler.fit_transform(data_X)
    # # print(data_X[0])
    #
    # idx = numpy.arange(data_X.shape[0])
    # numpy.random.shuffle(idx)
    #
    # num_train_data = 4000
    # num_part_A_data = 1000
    # num_party_A_train_data = 800
    # num_party_B_train_data = 2400
    #
    # train_X = data_X[idx[0:num_train_data]]
    # train_Y = data_Y[idx[0:num_train_data]]
    # test_X = data_X[idx[num_train_data:]]
    # test_Y = data_Y[idx[num_train_data:]]
    #
    # partA_X = data_X[idx[0:num_part_A_data]]
    # partA_Y = data_Y[idx[0:num_part_A_data]]
    #
    # partB_X = data_X[idx[num_part_A_data:]]
    # partB_Y = data_Y[idx[num_part_A_data:]]

    # partA_X, partA_Y = [], []
    # with open("./partyA_3208_[919].datasets") as fin:
    #     for line in fin:
    #         datasets = line.split(",")
    #         partA_X.append([float(e) for e in datasets[:-1]])
    #         partA_Y.append(np.int32(datasets[-1]))
    # num_party_A_train_data = 2600
    #
    # partB_X, partB_Y = [], []
    # with open("./partyB_1393_[894].datasets") as fin:
    #     for line in fin:
    #         datasets = line.split(",")
    #         partB_X.append([float(e) for e in datasets[:-1]])
    #         partB_Y.append(np.int32(datasets[-1]))

    # partyA = np.loadtxt("./datasets/UCI_Credit_Card/20190225141443_partyA_4913_2686.0.datasets", delimiter=",")
    # partyB = np.loadtxt("./datasets/UCI_Credit_Card/20190225141443_partyB_8359_3950.0.datasets", delimiter=",")
    partyA = np.loadtxt("./datasets/20190224170051_partyA_3084_800.datasets", delimiter=",")
    partyB = np.loadtxt("./datasets/20190224170051_partyB_1517_1013.datasets", delimiter=",")

    np.random.shuffle(partyA)
    np.random.shuffle(partyB)

    print("partyA.shape", partyA.shape)
    print("partyB.shape", partyB.shape)

    partA_X = partyA[:, :-1]
    partA_Y = partyA[:, -1]
    partB_X = partyB[:, :-1]
    partB_Y = partyB[:, -1]
    print(partA_Y)
    print(sum(partB_Y))
    num_party_A_train_data = 600
    num_party_B_train_data = 300

    # partA_X = np.array(partA_X)
    # partA_Y = np.array(partA_Y)
    # partB_X = np.array(partB_X)
    # partB_Y = np.array(partB_Y)

    train(partA_X, partA_Y, partB_X, partB_Y, num_party_A_train_data, num_party_B_train_data)








