import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# try:
#     from sklearn.preprocessing.data import StandardScaler
# except:
#     from sklearn.preprocessing._data import StandardScaler


def load_two_party_data(data_dir, selected_labels, data_type, neg_label=-1, n_samples=-1):
    print("# load_two_party_data")

    Xa, Xb, y = get_labeled_data_with_2_party(data_dir=data_dir,
                                              selected_labels=selected_labels,
                                              n_samples=n_samples, dtype=data_type)

    scale_model = StandardScaler()
    Xa = scale_model.fit_transform(Xa)
    Xb = scale_model.fit_transform(Xb)

    y_ = []
    pos_count = 0
    neg_count = 0
    count = {}
    for i in range(y.shape[0]):
        # the first label in y as the first class while the other labels as the second class
        label = np.nonzero(y[i, :])[0][0]
        y_.append(label)
        if label not in count:
            count[label] = 1
        else:
            count[label] = count[label] + 1
    print(count)

    # y = np.expand_dims(y_, axis=1)

    # print("yyy:", y_)

    return [Xa, Xb, y_]


def get_labeled_data_with_2_party(data_dir, selected_labels, n_samples, dtype="Train"):
    # get labels
    data_path = "NUS_WIDE/Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_labels:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    # print(data_labels)
    if len(selected_labels) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    # print(selected.shape)

    # get XA, which are image low level features
    features_path = "NUS_WIDE/Low_Level_Features"
    # features_path = "NUS_WIDE/NUS_WID_Low_Level_Features/Low_Level_Features"
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        # for file in ['{}_Normalized_EDH.dat'.format(dtype), '{}_Normalized_CM55.dat'.format(dtype), '{}_Normalized_CH.dat'.format(dtype),\
        #             '{}_Normalized_CORR.dat'.format(dtype), '{}_Normalized_WT.dat'.format(dtype)]:
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            print("{0} datasets features {1}".format(file, len(df.columns)))
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_XA_selected = data_XA.loc[selected.index]
    # print("XA shape:", data_XA_selected.shape)  # 634 columns

    # get XB, which are tags
    tag_path = "NUS_WIDE/NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_XB_selected = tagsdf.loc[selected.index]
    # print("XB shape:", data_XB_selected.shape)
    if n_samples != -1:
        return data_XA_selected.values[:n_samples], data_XB_selected.values[:n_samples], selected.values[:n_samples]
    else:
        # load all data
        return data_XA_selected.values, data_XB_selected.values, selected.values
