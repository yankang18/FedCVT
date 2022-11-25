import numpy as np

from fedcvt_core.fedcvt_repr_estimator import AttentionBasedRepresentationEstimator
from fedcvt_core.fedcvt_train import VerticalFederatedTransferLearning
from fedcvt_core.param import PartyModelParam, FederatedModelParam
from utils import get_timestamp


def run_experiment(X_guest_train, X_host_train, y_train,
                   X_guest_test, X_host_test, y_test,
                   num_overlap, exp_param_dict, epoch,
                   overlap_sample_batch_size,
                   non_overlap_sample_batch_size,
                   estimation_block_size,
                   training_info_file_name,
                   sharpen_temperature,
                   label_prob_sharpen_temperature,
                   fed_label_prob_threshold,
                   host_label_prob_threshold,
                   is_hetero_reprs,
                   using_uniq,
                   using_comm,
                   hidden_dim,
                   num_class,
                   vfl_guest_constructor,
                   vfl_host_constructor,
                   other_args,
                   normalize_repr=False,
                   debug=False):
    print("# ============= Data Info =============")
    print("X_guest_train shape", X_guest_train.shape)
    print("X_host_train shape", X_host_train.shape)
    print("y_train shape", y_train.shape)

    print("X_guest_test shape", X_guest_test.shape)
    print("X_host_test shape", X_host_test.shape)
    print("y_test shape", y_test.shape)
    print("# =====================================")

    # sample_num = num_overlap / 10
    # lbl_sample_idx_dict = {}
    # for index, one_hot_lbl in enumerate(y_train):
    #     lbl_index = np.argmax(one_hot_lbl)
    #     sample_idx_list = lbl_sample_idx_dict.get(lbl_index)
    #     if sample_idx_list is None:
    #         lbl_sample_idx_dict[lbl_index] = [index]
    #     elif len(sample_idx_list) < sample_num:
    #         lbl_sample_idx_dict[lbl_index].append(index)
    # print("lbl_sample_idx_dict:\n", lbl_sample_idx_dict)
    # # compute overlap and non-overlap indices
    # overlap_indices = list()
    # for k, v in lbl_sample_idx_dict.items():
    #     overlap_indices += lbl_sample_idx_dict[k]

    overlap_indices = [i for i in range(num_overlap)]
    overlap_indices = np.array(overlap_indices)
    num_train = X_guest_train.shape[0]
    non_overlap_indices = np.setdiff1d(range(num_train), overlap_indices)
    num_non_overlap = num_train - num_overlap
    print("[DEBUG] overlap_indices:\n", overlap_indices, len(set(overlap_indices)))
    print("[DEBUG] non_overlap_indices:\n", non_overlap_indices, len(set(non_overlap_indices)))

    if using_comm and using_uniq:
        guest_hidden_dim = hidden_dim
        host_hidden_dim = hidden_dim
        guest_input_dim = guest_hidden_dim * 2
        host_input_dim = host_hidden_dim * 2
    else:
        guest_hidden_dim = hidden_dim * 2
        host_hidden_dim = hidden_dim * 2
        guest_input_dim = guest_hidden_dim
        host_input_dim = host_hidden_dim

    fed_input_dim = host_input_dim + guest_input_dim

    guest_model_param = PartyModelParam(data_folder=None,
                                        apply_dropout=False,
                                        hidden_dim_list=[guest_hidden_dim],
                                        n_class=num_class,
                                        normalize_repr=normalize_repr,
                                        data_type="tab")
    host_model_param = PartyModelParam(data_folder=None,
                                       apply_dropout=False,
                                       hidden_dim_list=[host_hidden_dim],
                                       n_class=num_class,
                                       normalize_repr=normalize_repr,
                                       data_type="tab")

    learning_rate = exp_param_dict["learning_rate"]
    lambda_dist_shared_reprs = exp_param_dict["lambda_dist_shared_reprs"]
    lambda_sim_shared_reprs_vs_unique_repr = exp_param_dict["lambda_sim_shared_reprs_vs_unique_repr"]
    lambda_host_dist_ested_lbl_vs_true_lbl = exp_param_dict["lambda_host_dist_ested_lbl_vs_true_lbl"]
    lambda_dist_ested_repr_vs_true_repr = exp_param_dict["lambda_dist_ested_repr_vs_true_repr"]
    lambda_host_dist_two_ested_lbl = exp_param_dict["lambda_host_dist_two_ested_lbl"]

    # ================================================================================================================
    # lambda for auxiliary losses, which include:
    # (1) loss for minimizing distance between shared representations between host and guest
    # (2) loss for minimizing similarity between shared representation and unique representation for guest
    # (3) loss for minimizing similarity between shared representation and unique representation for host
    # (4) loss for minimizing distance between estimated host unique overlap labels and true overlap labels
    # (5) loss for minimizing distance between estimated host common overlap labels and true overlap labels
    # (6) loss for minimizing distance between estimated guest overlap representation and true guest representation
    # (7) loss for minimizing distance between estimated host overlap representation and true host representation
    # (8) loss for minimizing distance between shared-repr-estimated host label and uniq-repr-estimated host label
    # ================================================================================================================
    loss_weight_dict = {"lambda_dist_shared_reprs": lambda_dist_shared_reprs,
                        "lambda_guest_sim_shared_reprs_vs_unique_repr": lambda_sim_shared_reprs_vs_unique_repr,
                        "lambda_host_sim_shared_reprs_vs_unique_repr": lambda_sim_shared_reprs_vs_unique_repr,
                        "lambda_host_dist_ested_uniq_lbl_vs_true_lbl": lambda_host_dist_ested_lbl_vs_true_lbl,
                        "lambda_host_dist_ested_comm_lbl_vs_true_lbl": lambda_host_dist_ested_lbl_vs_true_lbl,
                        "lambda_guest_dist_ested_repr_vs_true_repr": lambda_dist_ested_repr_vs_true_repr,
                        "lambda_host_dist_ested_repr_vs_true_repr": lambda_dist_ested_repr_vs_true_repr,
                        "lambda_host_dist_two_ested_lbl": lambda_host_dist_two_ested_lbl}

    print("* loss_weight_dict: {0}".format(loss_weight_dict))

    fed_model_param = FederatedModelParam(fed_input_dim=fed_input_dim,
                                          guest_input_dim=guest_input_dim,
                                          host_input_dim=host_input_dim,
                                          is_hetero_repr=is_hetero_reprs,
                                          using_block_idx=False,
                                          learning_rate=learning_rate,
                                          fed_reg_lambda=0.0,
                                          guest_reg_lambda=0.0,
                                          loss_weight_dict=loss_weight_dict,
                                          overlap_indices=overlap_indices,
                                          non_overlap_indices=non_overlap_indices,
                                          epoch=epoch,
                                          top_k=1,
                                          overlap_sample_batch_size=overlap_sample_batch_size,
                                          non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                          overlap_sample_batch_num=num_overlap,
                                          all_sample_block_size=estimation_block_size,
                                          label_prob_sharpen_temperature=label_prob_sharpen_temperature,
                                          sharpen_temperature=sharpen_temperature,
                                          fed_label_prob_threshold=fed_label_prob_threshold,
                                          host_label_prob_threshold=host_label_prob_threshold,
                                          training_info_file_name=training_info_file_name,
                                          valid_iteration_interval=3,
                                          using_uniq=using_uniq,
                                          using_comm=using_comm)

    # set up and train model
    guest_constructor = vfl_guest_constructor(guest_model_param, fed_model_param)
    host_constructor = vfl_host_constructor(host_model_param, fed_model_param)

    guest = guest_constructor.build(X_train=X_guest_train, Y_train=y_train,
                                    X_test=X_guest_test, Y_test=y_test,
                                    args=other_args, debug=debug)
    host = host_constructor.build(X_train=X_host_train, X_test=X_host_test,
                                  args=other_args, debug=debug)

    VFTL = VerticalFederatedTransferLearning(vftl_guest=guest, vftl_host=host,
                                             fed_model_param=fed_model_param,
                                             debug=debug)
    VFTL.set_representation_estimator(AttentionBasedRepresentationEstimator())
    VFTL.build()
    VFTL.train()


def batch_run_experiments(X_guest_train, X_host_train, Y_train, X_guest_test, X_host_test, Y_test,
                          optim_args, loss_weight_args, training_args, other_args=None):
    using_uniq = True
    using_comm = True
    file_folder = "training_log_info/"
    timestamp = get_timestamp()

    weight_decay = optim_args["weight_decay"]
    lr_list = optim_args["learning_rate_list"]

    is_hetero_reprs = training_args["is_hetero_reprs"]
    num_overlap_list = training_args["num_overlap_list"]
    overlap_sample_batch_size = training_args["overlap_sample_batch_size"]
    non_overlap_sample_batch_size = training_args["non_overlap_sample_batch_size"]
    estimation_block_size = training_args["estimation_block_size"]
    sharpen_temperature = training_args["sharpen_temperature"]
    normalize_repr = training_args["normalize_repr"]
    epoch = training_args["epoch"]

    hidden_dim = training_args["hidden_dim"]
    num_class = training_args["num_class"]

    vfl_guest_constructor = training_args["vfl_guest_constructor"]
    vfl_host_constructor = training_args["vfl_host_constructor"]

    label_prob_sharpen_temperature = training_args["label_prob_sharpen_temperature"]
    fed_label_prob_threshold = training_args["fed_label_prob_threshold"]
    host_label_prob_threshold = training_args["host_label_prob_threshold"]

    lambda_dist_shared_reprs = loss_weight_args["lambda_dist_shared_reprs"]
    lambda_sim_shared_reprs_vs_uniq_reprs = loss_weight_args["lambda_sim_shared_reprs_vs_uniq_reprs"]
    lambda_host_dist_ested_lbls_vs_true_lbls = loss_weight_args["lambda_host_dist_ested_lbls_vs_true_lbls"]
    lambda_dist_ested_reprs_vs_true_reprs = loss_weight_args["lambda_dist_ested_reprs_vs_true_reprs"]
    lambda_host_dist_two_ested_lbls = loss_weight_args["lambda_host_dist_two_ested_lbls"]

    # =============================================================================================================
    # lambda for auxiliary losses, which include:
    # (1) loss for minimizing distance between shared representations between host and guest
    # (2) loss for minimizing similarity between shared representation and unique representation for guest
    # (3) loss for minimizing similarity between shared representation and unique representation for host
    # (4) loss for minimizing distance between estimated host unique overlap labels and true overlap labels
    # (5) loss for minimizing distance between estimated host common overlap labels and true overlap labels
    # (6) loss for minimizing distance between estimated guest overlap representation and true guest representation
    # (7) loss for minimizing distance between estimated host overlap representation and true host representation
    # (8) loss for minimizing distance between shared-repr-estimated host label and uniq-repr-estimated host label
    # =============================================================================================================
    exp_param_dict = dict()
    for n_ol in num_overlap_list:
        for lbda_0 in lr_list:
            for lbda_1 in lambda_dist_shared_reprs:
                for lbda_2 in lambda_sim_shared_reprs_vs_uniq_reprs:
                    for lbda_3 in lambda_host_dist_ested_lbls_vs_true_lbls:
                        for lbda_4 in lambda_dist_ested_reprs_vs_true_reprs:
                            for lbda_5 in lambda_host_dist_two_ested_lbls:
                                file_name = file_folder + "nuswide_" + str(n_ol) + "_" + timestamp

                                exp_param_dict["learning_rate"] = lbda_0
                                exp_param_dict["weight_decay"] = weight_decay
                                exp_param_dict["lambda_dist_shared_reprs"] = lbda_1
                                exp_param_dict["lambda_sim_shared_reprs_vs_unique_repr"] = lbda_2
                                exp_param_dict["lambda_host_dist_ested_lbl_vs_true_lbl"] = lbda_3
                                exp_param_dict["lambda_dist_ested_repr_vs_true_repr"] = lbda_4
                                exp_param_dict["lambda_host_dist_two_ested_lbl"] = lbda_5
                                run_experiment(X_guest_train=X_guest_train, X_host_train=X_host_train, y_train=Y_train,
                                               X_guest_test=X_guest_test, X_host_test=X_host_test, y_test=Y_test,
                                               num_overlap=n_ol, exp_param_dict=exp_param_dict, epoch=epoch,
                                               overlap_sample_batch_size=overlap_sample_batch_size,
                                               non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                               estimation_block_size=estimation_block_size,
                                               training_info_file_name=file_name,
                                               sharpen_temperature=sharpen_temperature,
                                               label_prob_sharpen_temperature=label_prob_sharpen_temperature,
                                               fed_label_prob_threshold=fed_label_prob_threshold,
                                               host_label_prob_threshold=host_label_prob_threshold,
                                               is_hetero_reprs=is_hetero_reprs,
                                               using_uniq=using_uniq,
                                               using_comm=using_comm,
                                               hidden_dim=hidden_dim,
                                               num_class=num_class,
                                               vfl_guest_constructor=vfl_guest_constructor,
                                               vfl_host_constructor=vfl_host_constructor,
                                               other_args=other_args,
                                               normalize_repr=normalize_repr)
