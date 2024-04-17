# encoding: utf-8

import os
import platform

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from models.ctr_models_utils import build_input_features, SparseFeat, DenseFeat, get_feature_names


class Criteo2party(object):

    def __init__(self, data_dir, data_type, k, input_size, split_mode='equal'):
        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.k = k
        col_train = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
                     'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
                     'C16',
                     'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
        col_test = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
                    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
                    'C16',
                    'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
        train_ratio = 0.2
        self.data_dir = data_dir

        # split features
        self.feature_list = []

        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]

        sparse_features_list = []
        dense_features_list = []
        if split_mode == 'col_type':
            sparse_features_list = [sparse_features, []]
            dense_features_list = [[], dense_features]
        elif split_mode == 'col_type_reverse':
            sparse_features_list = [[], sparse_features]
            dense_features_list = [dense_features, []]
        elif split_mode == 'equal':
            sparse_features_list = [sparse_features[:14], sparse_features[14:]]
            dense_features_list = [dense_features[:8], dense_features[8:]]
        elif split_mode == 'equal_reverse':
            sparse_features_list = [sparse_features[14:], sparse_features[:14]]
            dense_features_list = [dense_features[8:], dense_features[:8]]

        # if platform.system() == 'Windows':
        #     data = pd.read_csv(os.path.join(self.data_dir, 'train_sample.txt'))
        # else:
        #     train = pd.read_csv(os.path.join(self.data_dir, 'train_20W.txt'))  # , sep='\t', header=None)
        #     test = pd.read_csv(os.path.join(self.data_dir, 'test_4W.txt'))  # , sep='\t', header=None)
        #     train = train.sample(frac=0.5)
        #     test = test.sample(frac=0.5)
        #     data = pd.concat([train, test], axis=0)

        train = pd.read_csv(os.path.join(self.data_dir, 'train_10W.txt'))  # , sep='\t', header=None)
        test = pd.read_csv(os.path.join(self.data_dir, 'test_2W.txt'))  # , sep='\t', header=None)
        # train = train.sample(frac=0.5)
        # test = test.sample(frac=0.5)
        data = pd.concat([train, test], axis=0)

        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
        if platform.system() == 'Windows':
            train, test = train_test_split(data, test_size=0.2, shuffle=False)
        else:
            train = data.iloc[:100000]
            test = data.iloc[100000:]

        print("[DEBUG] data:", data.shape)
        print("[DEBUG] train:", train.shape)
        print("[DEBUG] test:", test.shape)

        if data_type.lower() == 'train':
            labels = train['label']
        else:
            labels = test['label']
        self.y = labels.values

        self.x = []
        self.feature_dim = []
        # 2.count #unique features for each sparse field,and record dense feature field name
        for i in range(len(sparse_features_list)):
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), input_size)
                                      for feat in sparse_features_list[i]] + [DenseFeat(feat, 1, )
                                                                              for feat in dense_features_list[i]]

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns

            self.feature_list.append(fixlen_feature_columns)

            feature_names = get_feature_names(
                linear_feature_columns + dnn_feature_columns)

            # 3.generate input data for model

            if data_type.lower() == 'train':
                x = {name: train[name] for name in feature_names}
            else:
                x = {name: test[name] for name in feature_names}

            feature_index = build_input_features(
                linear_feature_columns + dnn_feature_columns)

            if isinstance(x, dict):
                x = [x[feature] for feature in feature_index]

            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            x = np.concatenate(x, axis=-1)
            x = torch.tensor(x, dtype=torch.float32)
            self.x.append(x)
        print("x[0], x[1]:", self.x[0].shape, self.x[1].shape)
        del data, train, test

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def get_data(self):
        return self.x, self.y

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, indexx):  # this is single_indexx

        # total_features = self.x[indexx]
        # print(total_features.shape)
        # data = []
        labels = []

        # acoustic = total_features[:20]
        # seismic = total_features[20:]
        data = [self.x[0][indexx], self.x[1][indexx]]

        # labels.append(self.y[indexx])

        # return data, np.array(labels).ravel()
        return data, np.array(self.y[indexx])


class Avazu2party():

    def __init__(self, data_dir, data_type, k, input_size, split_mode='equal'):
        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.k = k
        train_ratio = 0.2
        self.data_dir = data_dir

        # split features
        self.feature_list = []

        sparse_features = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                           'device_id', 'device_ip', 'device_model', 'C14', 'C17', 'C19', 'C20', 'C21']
        dense_features = ['C1', 'banner_pos', 'device_type', 'device_conn_type', 'C15', 'C16', 'C18']

        sparse_features_list = []
        dense_features_list = []
        if split_mode == 'col_type':
            sparse_features_list = [sparse_features, []]
            dense_features_list = [[], dense_features]
        elif split_mode == 'col_type_reverse':
            sparse_features_list = [[], sparse_features]
            dense_features_list = [dense_features, []]
        elif split_mode == 'equal':
            sparse_features_list = [sparse_features[7:], sparse_features[:7]]
            dense_features_list = [dense_features[4:], dense_features[:4]]
        elif split_mode == 'equal_reverse':
            sparse_features_list = [sparse_features[:7], sparse_features[7:]]
            dense_features_list = [dense_features[:4], dense_features[4:]]

        # sub_data_dir = "avazu"
        # if platform.system() == 'Windows':
        #     data = pd.read_csv(os.path.join(self.data_dir, sub_data_dir, 'train_sample.txt'))  # , header=None)
        # else:
        #     train = pd.read_csv(os.path.join(self.data_dir, sub_data_dir, 'train_20W.txt'))  # , sep='\t', header=None)
        #     test = pd.read_csv(os.path.join(self.data_dir, sub_data_dir, 'test_4W.txt'))  # , sep='\t', header=None)
        #     # train = train.sample(frac=0.5)
        #     # test = test.sample(frac=0.5)
        #     data = pd.concat([train, test], axis=0)

        train_data_file = 'train_10W.txt'
        test_data_file = 'test_2W.txt'
        # train_data_file = 'train_20W.txt'
        # test_data_file = 'test_4W.txt'
        train = pd.read_csv(os.path.join(self.data_dir, train_data_file))  # , sep='\t', header=None)
        test = pd.read_csv(os.path.join(self.data_dir, test_data_file))  # , sep='\t', header=None)
        data = pd.concat([train, test], axis=0)
        print("[DEBUG] avazu data shape : {}".format(data.shape))

        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        if platform.system() == 'Windows':
            train, test = train_test_split(data, test_size=0.2, shuffle=False)
        else:
            # train = data.iloc[:200000]
            # test = data.iloc[200000:]
            train = data.iloc[:100000]
            test = data.iloc[100000:]

        print(train.shape, test.shape)

        if data_type.lower() == 'train':
            labels = train['click']
        else:
            labels = test['click']
        self.y = labels.values

        self.x = []
        self.feature_dim = []
        # 2.count #unique features for each sparse field,and record dense feature field name
        for i in range(len(sparse_features_list)):
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), input_size)
                                      for feat in sparse_features_list[i]] + [DenseFeat(feat, 1, )
                                                                              for feat in dense_features_list[i]]
            print("[DEBUG] fixlen_feature_columns:", fixlen_feature_columns)

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns
            # self.dnn_feature_columns = dnn_feature_columns
            # self.linear_feature_columns = linear_feature_columns
            # self.feature_dim.append(self.compute_input_dim(dnn_feature_columns))

            self.feature_list.append(fixlen_feature_columns)

            feature_names = get_feature_names(
                linear_feature_columns + dnn_feature_columns)

            # 3.generate input data for model

            if data_type.lower() == 'train':
                x = {name: train[name] for name in feature_names}
            else:
                x = {name: test[name] for name in feature_names}
            # self.features = features

            feature_index = build_input_features(
                linear_feature_columns + dnn_feature_columns)

            # self.x = features

            if isinstance(x, dict):
                x = [x[feature] for feature in feature_index]

            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            x = np.concatenate(x, axis=-1)
            x = torch.tensor(x, dtype=torch.float32)
            self.x.append(x)

        print("[DEBUG] party A data shape:{}, party B data shape:{}".format(self.x[0].shape, self.x[1].shape))
        print("x[0], x[1]:", self.x[0].shape, self.x[1].shape)
        del data, train, test

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def get_data(self):
        return self.x, self.y

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, indexx):  # this is single_indexx

        # total_features = self.x[indexx]
        # print(total_features.shape)
        # data = []
        # labels = []

        # acoustic = total_features[:20]
        # seismic = total_features[20:]
        data = [self.x[0][indexx], self.x[1][indexx]]

        # labels.append(self.y[indexx])
        # print("np.array(labels).ravel() shape:", np.array(labels).ravel().shape)
        # return data, np.array(labels).ravel()
        return data, np.array(self.y[indexx])

# def test_dataset_v2():
#     DATA_DIR = '../../../dataset/avazu'
#     train_dataset = Criteo2party(DATA_DIR, 'Train', 2)
#     valid_dataset = Criteo2party(DATA_DIR, 'test', 2)
#     n_train = len(train_dataset)
#     n_valid = len(valid_dataset)
#     print(n_train)
#     print(n_valid)
#     train_indices = list(range(n_train))
#     valid_indices = list(range(n_valid))
#     #print(len(train_indices))
#     #print(len(valid_indices))
#
#     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
#     valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=32,
#                                                num_workers=0,
#                                                sampler=train_sampler,
#                                                pin_memory=False)
#     valid_loader = torch.utils.data.DataLoader(valid_dataset,
#                                                batch_size=32,
#                                                num_workers=0,
#                                                pin_memory=True)
#     print(len(train_loader))
#     for i, (x1, y) in enumerate(train_loader):
#         print(y, len(x1))
#         print(x1[0].shape, x1[1].shape, y.shape)
#         print(x1)
#         #print(x1.shape)
#
#         break
#
#
#     # print(train_dataset.features.shape)
#     # print(train_dataset.y.shape)
#
#     from deepctr_torch.models import DNNFM
#     # model = DNNFM(linear_feature_columns=train_dataset.linear_feature_columns,
#     #                dnn_feature_columns=train_dataset.dnn_feature_columns,
#     #                use_fm = False,
#     #                task='binary',
#     #                l2_reg_embedding=1e-5, device='cpu')
#     #
#     # criterion = nn.BCELoss()
#     # optimizer =torch.optim.Adagrad(model.parameters())
#     #
#     # epochs = 10
#     # losses = []
#     # for epoch in range(epochs):
#     #     for step, (trn_X, trn_y) in enumerate(train_loader):
#     #         #print(trn_X.shape, trn_y.shape)
#     #         N = trn_y.size(0)
#     #         pred = model(trn_X.float())
#     #         loss = criterion(pred, trn_y.float())
#     #         optimizer.zero_grad()
#     #         loss.backward()
#     #         optimizer.step()
#     #         losses.append(loss.item()/N)
#     #     losses_avg = sum(losses)/len(losses)
#     #     print(epoch, losses_avg)
#
#
#     # model.compile("adagrad", "binary_crossentropy",
#     #               metrics=["binary_crossentropy", "auc"], )
#     #
#     # history = model.fit(train_dataset.features, train_dataset.y, batch_size=32, epochs=10, verbose=2,
#     #                     validation_split=0.2)
#     # pred_ans = model.predict(valid_dataset.features, 32)
#     # print("")
#     # print("test LogLoss", round(log_loss(valid_dataset.y, pred_ans), 4))
#     # print("test AUC", round(roc_auc_score(valid_dataset.y, pred_ans), 4))
#
# def test_dataset_v3():
#     DATA_DIR = '../../../dataset/avazu'
#     train_dataset = Avazu2party(DATA_DIR, 'Train', 2)
#     valid_dataset = Avazu2party(DATA_DIR, 'test', 2)
#     n_train = len(train_dataset)
#     n_valid = len(valid_dataset)
#     print(n_train)
#     print(n_valid)
#     train_indices = list(range(n_train))
#     valid_indices = list(range(n_valid))
#     #print(len(train_indices))
#     #print(len(valid_indices))
#
#     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
#     valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=32,
#                                                num_workers=0,
#                                                sampler=train_sampler,
#                                                pin_memory=False)
#     valid_loader = torch.utils.data.DataLoader(valid_dataset,
#                                                batch_size=32,
#                                                num_workers=0,
#                                                pin_memory=True)
#     print(len(train_loader))
#     for i, (x1, y) in enumerate(train_loader):
#         print(y, len(x1))
#         print(x1[0].shape, x1[1].shape, y.shape)
#         print(x1)
#         #print(x1.shape)
#
#         break
#
#
#
# if __name__ == "__main__":
#     test_dataset_v3()
