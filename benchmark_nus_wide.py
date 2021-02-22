import time

import numpy as np
# import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.utils import shuffle

# from autoencoder import Autoencoder
from data_util.nus_wide_processed_data_util import TwoPartyNusWideDataLoader
# from logistic_regression import LogisticRegression
# from regularization import EarlyStoppingCheckPoint


def f_score(precision, recall):
    return 2 / (1 / precision + 1 / recall)


# class BaseModel(object):
#
#     def _set_session(self, session):
#         pass
#
#     def _get_input_placeholder(self):
#         pass
#
#     def _get_label_placeholder(self):
#         pass
#
#     def _get_loss_op(self):
#         pass
#
#     def _get_train_op(self):
#         pass
#
#     def _get_predict_op(self):
#         pass
#
#     def _should_stopping_training(self):
#         pass
#
#     def fit(self, X_train, y_train, epochs, batch_size, validation_data, early_stopping):
#
#         early_stopping.set_model(self)
#
#         num_labeled_train_samples = X_train.shape[0]
#
#         residual = num_labeled_train_samples % batch_size
#         if residual == 0:
#             # if residual is 0, the number of samples is multiples of batch_size.
#             # Thus, we can directly use the real floor division operator "//" to compute
#             # the num_batch
#             batch_num = num_labeled_train_samples // batch_size
#         else:
#             # if residual is not 0,
#             batch_num = num_labeled_train_samples // batch_size + 1
#
#         print("batch_num: {0}".format(batch_num))
#
#         X_val = validation_data[0]
#         y_val = validation_data[1]
#
#         early_stopping.on_train_begin()
#         start_time = time.time()
#         init = tf.global_variables_initializer()
#         with tf.Session() as sess:
#
#             # self.logistic_regressor.set_session(sess)
#             self._set_session(sess)
#
#             sess.run(init)
#
#             loss_list = []
#             auc_list = []
#             acc_list = []
#             fscore_list = []
#             for i in range(epochs):
#                 for batch_i in range(batch_num):
#                     batch_i_start = batch_size * batch_i
#                     batch_i_end = batch_size * batch_i + batch_size
#                     # print("batch from {0} to {1}".format(batch_i_start, batch_i_end))
#
#                     X_train_batch_i = X_train[batch_i_start:batch_i_end]
#                     y_train_batch_i = y_train[batch_i_start:batch_i_end]
#
#                     feed_dictionary = {self._get_input_placeholder(): X_train_batch_i,
#                                        self._get_label_placeholder(): y_train_batch_i}
#                     loss, _ = sess.run(fetches=[self._get_loss_op(), self._get_train_op()],
#                                        feed_dict=feed_dictionary)
#
#                     loss_list.append(loss)
#                     print("")
#                     print("------> ep", i, "batch", batch_i, "loss", loss)
#
#                     # validation
#                     if batch_i % 20 == 0:
#                         feed_dictionary_test = {self._get_input_placeholder(): X_val}
#                         y_pred_one_hot = sess.run(self._get_predict_op(), feed_dict=feed_dictionary_test)
#
#                         y_pred_1d = np.argmax(y_pred_one_hot, axis=1)
#                         y_test_1d = np.argmax(y_val, axis=1)
#
#                         # print("y_pred_one_hot:", y_pred_one_hot.shape)
#                         # print("y_pred_1d:", y_pred_1d.shape)
#                         # print("y_test:", y_test.shape)
#                         # print("y_test_1d:", y_test_1d.shape)
#
#                         res = precision_recall_fscore_support(y_test_1d, y_pred_1d, average='weighted')
#                         fscore = f_score(res[0], res[1])
#                         acc = accuracy_score(y_test_1d, y_pred_1d)
#                         auc = roc_auc_score(y_val, y_pred_one_hot)
#
#                         print("res: {0}".format(res))
#                         print("fscore: {0}, acc: {1}, auc:{2}".format(fscore, acc, auc))
#
#                         fscore_list.append(fscore)
#                         acc_list.append(acc)
#                         auc_list.append(auc)
#
#                         log = {"fscore": fscore, "all_acc": acc, "all_auc": auc, "loss": loss}
#                         early_stopping.on_validation_end(curr_epoch=i, batch_idx=batch_i, log=log)
#
#                     if self._should_stopping_training():
#                         break
#
#                 if self._should_stopping_training():
#                     break
#
#         end_time = time.time()
#         print("training time (s):", end_time - start_time)
#         print("loss:", loss_list)
#         print("stopped epoch, batch:", early_stopping.stopped_epoch, early_stopping.stopped_batch)
#         early_stopping.print_log_of_best_result()
#         # series_plot(losses=loss_list, fscores=fscore_list, aucs=acc_list)
#         return early_stopping.get_log_info()
#
#
# class BenchmarkNusWideModel(BaseModel):
#
#     def __init__(self):
#         self.feature_extractor = None
#         self.logistic_regressor = None
#         self.stop_training = False
#
#     def save_model(self):
#         print("TODO: save model")
#
#     def compile(self, n_class, hidden_dim, input_dim, learning_rate=0.01, regularization_lambda=0.01):
#         self.feature_extractor = Autoencoder(1)
#         self.feature_extractor.build(input_dim=input_dim, hidden_dim_list=hidden_dim, learning_rate=learning_rate)
#
#         reprs = self.feature_extractor.get_all_hidden_reprs()
#
#         self.logistic_regressor = LogisticRegression(1)
#         self.logistic_regressor.build(input_dim=hidden_dim[-1], n_class=n_class)
#         self.logistic_regressor.set_ops(learning_rate=learning_rate, reg_lambda=regularization_lambda, tf_X_in=reprs,
#                                         tf_labels_in=None)
#
#     def _set_session(self, session):
#         self.logistic_regressor.set_session(session)
#
#     def _get_input_placeholder(self):
#         return self.feature_extractor.X_all_in
#
#     def _get_label_placeholder(self):
#         return self.logistic_regressor.target_labels
#
#     def _get_loss_op(self):
#         return self.logistic_regressor.loss
#
#     def _get_train_op(self):
#         return self.logistic_regressor.e2e_train_op
#
#     def _get_predict_op(self):
#         return self.logistic_regressor.y_hat
#
#     def _should_stopping_training(self):
#         return self.stop_training


# def test(X_train, y_train, n_class, hidden_dim, learning_rate, regularization_lambda, epochs, batch_size,
#          validation_data):
#     """
#     Build model
#     """
#     tf.reset_default_graph()
#
#     # n_class = len(target_label_list)
#     # hidden_dim = 32
#     # learning_rate = 0.01
#     # regularization_lambda = 0.001
#
#     model = BenchmarkNusWideModel()
#     model.compile(n_class=n_class,
#                   hidden_dim=hidden_dim,
#                   input_dim=X_train.shape[1],
#                   learning_rate=learning_rate,
#                   regularization_lambda=regularization_lambda)
#
#     """
#     Train model
#     """
#
#     # batch_size = 32
#     # epochs = 100
#
#     early_stopping = EarlyStoppingCheckPoint(monitor="fscore", patience=20)
#     return model.fit(X_train=X_train, y_train=y_train, epochs=epochs, batch_size=batch_size,
#                      validation_data=validation_data, early_stopping=early_stopping)


def run_experiment(X_image, X_text, Y, num_overlapping_labeled_samples, test_start_index=40000, batch_size=32):
    X_image_all, X_text_all, Y_all = shuffle(X_image, X_text, Y)
    print("X_image_all shape", X_image_all.shape)
    print("X_text_all shape", X_text_all.shape)
    print("Y_all shape", Y_all.shape)

    # num_train = int(0.86 * X_guest_all.shape[0])
    # test_start_index = 500
    # num_overlapping_labeled_samples = 19900
    # test_start_index = 40000
    print("num_train", num_overlapping_labeled_samples)
    print("test_start_index", test_start_index)

    X_text_train, y_train = X_text_all[:num_overlapping_labeled_samples], Y_all[:num_overlapping_labeled_samples]
    X_text_test, y_test = X_text_all[test_start_index:], Y_all[test_start_index:]

    X_image_train = X_image_all[:num_overlapping_labeled_samples]
    X_image_test = X_image_all[test_start_index:]

    print("X_image_train shape", X_image_train.shape)
    print("X_text_train shape", X_text_train.shape)
    print("y_train shape", y_train.shape)

    print("X_image_test shape", X_image_test.shape)
    print("X_text_test shape", X_text_test.shape)
    print("y_test shape", y_test.shape)

    n_class = len(target_label_list)
    learning_rate = 0.01
    regularization_lambda = 0.001

    epochs = 100

    """
    test guest
    """

    # hidden_dim = [32]
    # valid_data = [X_image_test, y_test]
    # image_log_info = test(X_train=X_image_train,
    #                       y_train=y_train,
    #                       n_class=n_class,
    #                       hidden_dim=hidden_dim,
    #                       learning_rate=learning_rate,
    #                       regularization_lambda=regularization_lambda,
    #                       epochs=epochs,
    #                       batch_size=batch_size,
    #                       validation_data=valid_data)
    # # print("image_log_info: {0}".format(image_log_info))
    #
    # """
    # test host
    # """
    #
    # hidden_dim = [32]
    # valid_data = [X_text_test, y_test]
    # text_log_info = test(X_train=X_text_train,
    #                      y_train=y_train,
    #                      n_class=n_class,
    #                      hidden_dim=hidden_dim,
    #                      learning_rate=learning_rate,
    #                      regularization_lambda=regularization_lambda,
    #                      epochs=epochs,
    #                      batch_size=batch_size,
    #                      validation_data=valid_data)
    # # print("text_log_info: {0}".format(text_log_info))
    #
    # """
    # test host and guest combined
    # """
    #
    # X_train = np.concatenate((X_image_train, X_text_train), axis=-1)
    # X_test = np.concatenate((X_image_test, X_text_test), axis=-1)
    #
    # print("X_train shape", X_train.shape)
    # print("X_test shape", X_test.shape)
    #
    # hidden_dim = [64]
    # valid_data = [X_test, y_test]
    # image_text_log_info = test(X_train=X_train,
    #                            y_train=y_train,
    #                            n_class=n_class,
    #                            hidden_dim=hidden_dim,
    #                            learning_rate=learning_rate,
    #                            regularization_lambda=regularization_lambda,
    #                            epochs=epochs,
    #                            batch_size=batch_size,
    #                            validation_data=valid_data)
    #
    # print("image_log_info: {0}".format(image_log_info))
    # print("text_log_info: {0}".format(text_log_info))
    # print("image_text_log_info: {0}".format(image_text_log_info))
    # image_fscore = image_log_info.get("fscore")
    # text_fscore = text_log_info.get("fscore")
    # image_text_fscore = image_text_log_info.get("fscore")
    # return image_log_info, text_log_info, image_text_log_info


if __name__ == "__main__":
    """
    Prepare data
    """

    """
     top 10 labels:
     ['sky', 'clouds', 'person', 'water', 'animal',
     'grass', 'buildings', 'window', 'plants', 'lake']
    """
    file_dir = '/Users/yankang/Documents/Data/'
    # file_dir = "../../data/"
    # target_label_list = get_top_k_labels(file_dir, top_k=10)
    # target_label_list = ['sky', 'clouds', 'person']
    target_label_list = ['sky', 'clouds', 'person', 'water', 'animal',
                         'grass', 'buildings', 'window', 'plants', 'lake']
    print("target_label_list: {0}".format(target_label_list))
    # target_label_list = ["person", "animal", "sky"]
    data_loader = TwoPartyNusWideDataLoader(file_dir)
    X_image, X_text, Y = data_loader.get_train_data(target_labels=target_label_list, binary_classification=False)

    print("X_text: ", X_text.shape)
    idx_valid_sample_list = []
    idx_invalid_sample_list = []
    for idx, X_text_i in enumerate(X_text):
        if np.all(X_text_i == 0):
            idx_invalid_sample_list.append(idx)
            # print("X_text_i", idx, np.sum(X_text_i), len(X_text_i))
        else:
            idx_valid_sample_list.append(idx)

    print("number of all-zero text sample:", len(idx_invalid_sample_list))
    print("number of not all-zero text sample:", len(idx_valid_sample_list))

    X_image = X_image[idx_valid_sample_list]
    X_text = X_text[idx_valid_sample_list]
    Y = Y[idx_valid_sample_list]

    print("X_image: ", X_image.shape)
    for idx, X_image_i in enumerate(X_image):
        if np.all(X_image_i == 0):
            print("X_image_i", idx, np.sum(X_image_i), len(X_image_i))

    print("X_text: ", X_text.shape)
    for idx, X_text_id in enumerate(X_text):
        if np.all(X_text_id == 0):
            print("X_text_id", idx, np.sum(X_text_id), len(X_text_id))

    #
    # Start training
    #

    image_fscore_list = []
    image_acc_list = []
    image_auc_list = []
    text_fscore_list = []
    text_acc_list = []
    text_auc_list = []
    image_text_fscore_list = []
    image_text_acc_list = []
    image_text_auc_list = []

    num_overlapping = 20000
    batch_size = 256
    num_experiments = 5
    for i in range(num_experiments):
        print("experiment {0}".format(i))
        run_experiment(X_image, X_text, Y,
                       num_overlapping_labeled_samples=num_overlapping,
                       test_start_index=40000,
                       batch_size=batch_size)
        # image_log, text_log, image_text_log = run_experiment(X_image, X_text, Y,
        #                                                      num_overlapping_labeled_samples=num_overlapping,
        #                                                      test_start_index=40000,
        #                                                      batch_size=batch_size)
    #     image_fscore = image_log.get("fscore")
    #     text_fscore = text_log.get("fscore")
    #     image_text_fscore = image_text_log.get("fscore")
    #
    #     image_acc = image_log.get("all_acc")
    #     text_acc = text_log.get("all_acc")
    #     image_text_acc = image_text_log.get("all_acc")
    #
    #     image_auc = image_log.get("all_auc")
    #     text_auc = text_log.get("all_auc")
    #     image_text_auc = image_text_log.get("all_auc")
    #     print("fscore {0}, {1}, {2}".format(image_fscore, text_fscore, image_text_fscore))
    #     print("acc {0}, {1}, {2}".format(image_acc, text_acc, image_text_acc))
    #     print("auc {0}, {1}, {2}".format(image_auc, text_auc, image_text_auc))
    #
    #     image_fscore_list.append(image_fscore)
    #     image_acc_list.append(image_acc)
    #     image_auc_list.append(image_auc)
    #     text_fscore_list.append(text_fscore)
    #     text_acc_list.append(text_acc)
    #     text_auc_list.append(text_auc)
    #     image_text_fscore_list.append(image_text_fscore)
    #     image_text_acc_list.append(image_text_acc)
    #     image_text_auc_list.append(image_text_auc)
    #
    # image_fscore_mean = np.mean(image_fscore_list)
    # image_fscore_std = np.std(image_fscore_list)
    # image_acc_mean = np.mean(image_acc_list)
    # image_acc_std = np.std(image_acc_list)
    # image_auc_mean = np.mean(image_auc_list)
    # image_auc_std = np.std(image_auc_list)
    #
    # text_fscore_mean = np.mean(text_fscore_list)
    # text_fscore_std = np.std(text_fscore_list)
    # text_acc_mean = np.mean(text_acc_list)
    # text_acc_std = np.std(text_acc_list)
    # text_auc_mean = np.mean(text_auc_list)
    # text_auc_std = np.std(text_auc_list)
    #
    # image_text_fscore_mean = np.mean(image_text_fscore_list)
    # image_text_fscore_std = np.std(image_text_fscore_list)
    # image_text_acc_mean = np.mean(image_text_acc_list)
    # image_text_acc_std = np.std(image_text_acc_list)
    # image_text_auc_mean = np.mean(image_text_auc_list)
    # image_text_auc_std = np.std(image_text_auc_list)
    #
    # print("num_overlapping:", num_overlapping)
    # print("[fscore] img: {0}({1}), txt: {2}({3}), img_txt: {4}({5})".format(image_fscore_mean, image_fscore_std,
    #                                                                         text_fscore_mean, image_acc_std,
    #                                                                         image_text_fscore_mean,
    #                                                                         image_text_fscore_std))
    # print("[acc] img: {0}({1}), txt: {2}({3}), img_txt: {4}({5})".format(image_acc_mean, image_acc_std,
    #                                                                      text_acc_mean, text_acc_std,
    #                                                                      image_text_acc_mean, image_text_acc_std))
    # print("[auc] img: {0}({1}), txt: {2}({3}), img_txt: {4}({5})".format(image_auc_mean, image_auc_std,
    #                                                                      text_auc_mean, text_auc_std,
    #                                                                      image_text_auc_mean, image_text_auc_std))
    # # print("acc {0}, {1}, {2}".format(image_acc, text_acc, image_text_acc))
    # # print("auc {0}, {1}, {2}".format(image_auc, text_auc, image_text_auc))
