import csv

if __name__ == "__main__":

    # I like using the codecs opening in a with
    # field_names = ['test1', 'test2', 'test3', 'test4', 'test5']
    #
    # dict_1 = dict()
    # dict_1["test1"] = 1
    # dict_1["test2"] = 2
    # dict_1["test3"] = 3
    # dict_1["test4"] = 4
    # dict_1["test5"] = 5
    #
    # dict_2 = dict()
    # dict_2["test1"] = 12
    # dict_2["test2"] = 22
    # dict_2["test3"] = 32
    # dict_2["test4"] = 42
    # dict_2["test5"] = 52
    #
    # rowdicts = [dict_1, dict_2]
    #
    # file_name = "test_csv_read"
    # with open(file_name, "a", newline='') as logfile:
    #     logger = csv.DictWriter(logfile, fieldnames=field_names)
    #     logger.writeheader()
    #     logger.writerows(rowdicts)

    # with open(file_name, "a") as logfile:
    #     logger = csv.DictWriter(logfile, fieldnames=field_names)
    #     # logger.writeheader()
    #
    #     # some more code stuff
    #     logger.writerows(rowdicts)

    log_field_names = ["fscore",
                       "all_fscore", "g_fscore", "h_fscore:",
                       "all_acc", "g_acc", "h_acc:",
                       "all_auc", "g_auc", "h_auc:",
                       "loss"]

    hyperparam_field_names = ["lambda_dis_shared_reprs",
                              "lambda_sim_shared_reprs_vs_distinct_repr",
                              "lambda_host_dis_ested_lbl_vs_true_lbl",
                              "lambda_dis_ested_repr_vs_true_repr",
                              "lambda_host_dis_two_ested_repr"]

    all_field_names = hyperparam_field_names + log_field_names

    hyperparameter_dict = dict()
    hyperparameter_dict["learning_rate"] = 1
    hyperparameter_dict["lambda_dis_shared_reprs"] = 2
    hyperparameter_dict["lambda_sim_shared_reprs_vs_distinct_repr"] = 3
    hyperparameter_dict["lambda_host_dis_ested_lbl_vs_true_lbl"] = 4
    hyperparameter_dict["lambda_dis_ested_repr_vs_true_repr"] = 5
    hyperparameter_dict["lambda_host_dis_two_ested_repr"] = 6

    log = {"fscore": 0.1,
           "all_acc": 0.1, "g_acc": 0.1, "h_acc:": 0.1,
           "all_auc": 0.1, "g_auc": 0.1, "h_auc:": 0.1,
           "all_fscore": 0.1, "g_fscore": 0.1, "h_fscore:": 0.1,
           "loss": 0.1}

    record_dict = dict()
    record_dict.update(hyperparameter_dict)
    record_dict.update(log)

    print("all_field_names: {0}".format(all_field_names))
    print("record_dict: {0}".format(record_dict))

    diff = record_dict.keys() - all_field_names
    print("diff: {0}".format(diff))



