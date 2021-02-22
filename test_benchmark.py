import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.utils import shuffle
from data_util.nus_wide_processed_data_util import TwoPartyNusWideDataLoader
from data_visualization import series_plot


def get_classifier():
    cl = LogisticRegression(solver='lbfgs')
    # cl = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5)
    # cl = MLPClassifier()
    return cl


def train(X_A_train, y_A_train, X_A_test, y_A_test,
          X_B_train, y_B_train, X_B_test, y_B_test,
          X_train, y_train, X_test, y_test):

    lr = get_classifier()
    lr.fit(X_A_train, y_A_train)
    acc1 = lr.score(X_A_test, y_A_test)
    y_A_pred = lr.predict(X_A_test)
    res1 = precision_recall_fscore_support(y_A_test, y_A_pred, average='weighted')
    auc1 = roc_auc_score(y_A_test, y_A_pred)
    fscore1 = f_score(res1[0], res1[1])
    print("A acc1:", acc1)
    print("A auc1:", auc1)
    print("A res1:", res1)
    print("A fscore1:", fscore1)

    lr = get_classifier()
    lr.fit(X_B_train, y_B_train)
    acc2 = lr.score(X_B_test, y_B_test)
    y_B_pred = lr.predict(X_B_test)
    res2 = precision_recall_fscore_support(y_B_test, y_B_pred, average='weighted')
    auc2 = roc_auc_score(y_B_test, y_B_pred)
    fscore2 = f_score(res2[0], res2[1])
    print("B acc2:", acc2)
    print("B auc2:", auc2)
    print("B res2:", res2)
    print("B fscore2:", fscore2)

    lr = get_classifier()
    lr.fit(X_train, y_train)
    acc3 = lr.score(X_test, y_test)
    y_pred = lr.predict(X_test)
    res3 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # auc3 = roc_auc_score(y_test, y_pred, average="weighted")
    auc3 = roc_auc_score(y_test, y_pred)
    fscore3 = f_score(res3[0], res3[1])
    print("AnB acc3:", acc3)
    print("AnB auc3:", auc3)
    print("AnB res3:", res3)
    print("AnB fscore3:", fscore3)
    return auc1, auc2, auc3, acc1, acc2, acc3, fscore1, fscore2, fscore3


def f_score(precision, recall):
    return 2 / (1 / precision + 1 / recall)


if __name__ == "__main__":

    # infile = "./datasets/UCI_Credit_Cardd/UCI_Credit_Card.csv"
    # split_index = 5, 11
    # data_loader = TwoPartyUCICreditCardDataLoader(infile, split_index=5, balanced=False, seed=None)

    file_dir = "/datasets/app/fate/yankang/"
    data_loader = TwoPartyNusWideDataLoader(file_dir)
    X_A_r, X_B_r, y_r = data_loader.get_train_data(target_labels=["person"])

    # num_train = int(0.80 * X_A.shape[0])
    # test_start_index = num_train
    # num_labeled_train_samples = 200
    test_start_index = 40000
    print("test_start_index", test_start_index)

    times = 10
    # num_labeled_train_samples_list = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
    num_labeled_train_samples_list = [20, 50, 100, 200, 500, 1000, 2000]
    auc1_mean_list = []
    auc2_mean_list = []
    auc3_mean_list = []
    acc1_mean_list = []
    acc2_mean_list = []
    acc3_mean_list = []
    fscore1_mean_list = []
    fscore2_mean_list = []
    fscore3_mean_list = []
    for num_labeled_train_samples in num_labeled_train_samples_list:
        print("num_labeled_train_samples:", num_labeled_train_samples)
        auc1_list = []
        auc2_list = []
        auc3_list = []
        acc1_list = []
        acc2_list = []
        acc3_list = []
        fscore1_list = []
        fscore2_list = []
        fscore3_list = []
        for tm in range(times):
            print("times:", tm)

            X_A, X_B, y = shuffle(X_A_r, X_B_r, y_r)
            # infile = "./datasets/"
            # data_loader = TwoPartyBreastDataLoader(infile)
            # X_A, X_B, y = data_loader.get_data()

            print("X_A", X_A.shape)
            print("X_B", X_B.shape)
            print("y", y.shape)
            print("y", sum(y))

            X_A_train, y_A_train = X_A[:num_labeled_train_samples], y[:num_labeled_train_samples]
            X_A_test, y_A_test = X_A[test_start_index:], y[test_start_index:]

            print("X_A_train", X_A_train.shape)
            print("y_A_train", y_A_train.shape)
            print("X_A_test", X_A_test.shape)
            print("y_A_test", y_A_test.shape)

            X_B_train, y_B_train = X_B[:num_labeled_train_samples], y[:num_labeled_train_samples]
            X_B_test, y_B_test = X_B[test_start_index:], y[test_start_index:]

            print("X_B_train", X_B_train.shape)
            print("y_B_train", y_B_train.shape)
            print("X_B_test", X_B_test.shape)
            print("y_B_test", y_B_test.shape)

            X = np.concatenate([X_A, X_B], axis=1)
            X_train, y_train = X[:num_labeled_train_samples], y[:num_labeled_train_samples]
            X_test, y_test = X[test_start_index:], y[test_start_index:]
            print("X_train:", X_train.shape)
            print("y_train:", y_train.shape)
            print("X_test:", X_test.shape)
            print("y_test:", y_test.shape)

            auc1, auc2, auc3, acc1, acc2, acc3, fscore1, fscore2, fscore3 = train(X_A_train, y_A_train, X_A_test, y_A_test,
                                                                                  X_B_train, y_B_train, X_B_test, y_B_test,
                                                                                  X_train, y_train, X_test, y_test)

            auc1_list.append(auc1)
            auc2_list.append(auc2)
            auc3_list.append(auc3)
            acc1_list.append(acc1)
            acc2_list.append(acc2)
            acc3_list.append(acc3)
            fscore1_list.append(fscore1)
            fscore2_list.append(fscore2)
            fscore3_list.append(fscore3)

        auc1_mean, auc1_std = np.mean(auc1_list), np.std(auc1_list)
        acc1_mean, acc1_std = np.mean(acc1_list), np.std(acc1_list)
        fscore1_mean, fscore1_std = np.mean(fscore1_list), np.std(fscore1_list)

        auc2_mean, auc2_std = np.mean(auc2_list), np.std(auc2_list)
        acc2_mean, acc2_std = np.mean(acc2_list), np.std(acc2_list)
        fscore2_mean, fscore2_std = np.mean(fscore2_list), np.std(fscore2_list)

        auc3_mean, auc3_std = np.mean(auc3_list), np.std(auc3_list)
        acc3_mean, acc3_std = np.mean(acc3_list), np.std(acc3_list)
        fscore3_mean, fscore3_std = np.mean(fscore3_list), np.std(fscore3_list)

        auc1_mean_list.append(auc1_mean)
        auc2_mean_list.append(auc2_mean)
        auc3_mean_list.append(auc3_mean)
        acc1_mean_list.append(acc1_mean)
        acc2_mean_list.append(acc2_mean)
        acc3_mean_list.append(acc3_mean)
        fscore1_mean_list.append(fscore1_mean)
        fscore2_mean_list.append(fscore2_mean)
        fscore3_mean_list.append(fscore3_mean)

        print("----------------------------------------------")
        print("auc1 mean, std:", auc1_mean, auc1_std)
        print("acc1 mean, std:", acc1_mean, acc1_std)
        print("fscore1 mean, std:", fscore1_mean, fscore1_std)
        print("###")
        print("auc2 mean, std:", auc2_mean, auc2_std)
        print("acc2 mean, std:", acc2_mean, acc2_std)
        print("fscore2 mean, std:", fscore2_mean, fscore2_std)
        print("###")
        print("auc3 mean, std:", auc3_mean, auc3_std)
        print("acc3 mean, std:", acc3_mean, acc3_std)
        print("fscore3 mean, std:", fscore3_mean, fscore3_std)

    print("auc1_mean_list:", auc1_mean_list)
    print("auc2_mean_list:", auc2_mean_list)
    print("auc3_mean_list:", auc3_mean_list)
    print("acc1_mean_list:", acc1_mean_list)
    print("acc2_mean_list:", acc2_mean_list)
    print("acc3_mean_list:", acc3_mean_list)
    print("fscore1_mean_list:", fscore1_mean_list)
    print("fscore2_mean_list:", fscore2_mean_list)
    print("fscore3_mean_list:", fscore3_mean_list)

    series_plot(losses=fscore1_mean_list, fscores=fscore2_mean_list, aucs=fscore3_mean_list)

