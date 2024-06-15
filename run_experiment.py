import copy
import random

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.data_utils import SubsetSampler
from fedcvt_core.fedcvt_trainer import VerticalFederatedTransferLearning
from fedcvt_core.param import PartyModelParam, FederatedTrainingParam
from utils import get_timestamp


def get_loader(dataset, sampler, batch_size, shuffle=None):
    if sampler is None:
        return None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)


def prepare_data_indices(num_all_train, num_labeled_overlap_samples, num_overlap_samples, use_only_ll=False):
    train_indices = list(range(num_all_train))
    # print("new train_indices:", train_indices)
    random.shuffle(train_indices)
    overlap_indices = train_indices[:num_overlap_samples]

    # overlap_indices = [i for i in range(num_overlap_samples)]
    # print("overlap_indices:", overlap_indices, len(overlap_indices))

    non_overlap_indices = np.setdiff1d(range(num_all_train), overlap_indices)
    # non_overlap_indices = np.setdiff1d(train_indices, overlap_indices)
    num_non_overlap = num_all_train - num_overlap_samples

    labeled_overlap_indices = overlap_indices[:num_labeled_overlap_samples]

    print("labeled_overlap_indices:", labeled_overlap_indices, len(labeled_overlap_indices))

    if use_only_ll:
        unlabeled_overlap_indices = None
        guest_non_overlap_indices = None
        guest_all_indices = None
        host_non_overlap_indices = None
        host_all_indices = None
    else:
        if num_labeled_overlap_samples == len(overlap_indices):
            # all overlaping samples are labeled. In other words, there is no unlabeled overlapping samples
            unlabeled_overlap_indices = None
        else:
            unlabeled_overlap_indices = overlap_indices[num_labeled_overlap_samples:]

        half_num_non_overlap = int(num_non_overlap / 2)
        # guest_non_overlap_indices = non_overlap_indices[:half_num_non_overlap]
        guest_non_overlap_indices = non_overlap_indices
        guest_all_indices = np.concatenate([overlap_indices, guest_non_overlap_indices])

        half_num_non_overlap = int(num_non_overlap / 2)
        # host_non_overlap_indices = non_overlap_indices[half_num_non_overlap:]
        host_non_overlap_indices = non_overlap_indices
        host_all_indices = np.concatenate([overlap_indices, host_non_overlap_indices])

        # print("labeled_overlap_indices:{}, {}".format(labeled_overlap_indices, len(labeled_overlap_indices)))
        # print("unlabeled_overlap_indices:{}, {}".format(unlabeled_overlap_indices, len(unlabeled_overlap_indices)))
        # print("guest_non_overlap_indices:{}, {}".format(guest_non_overlap_indices, len(guest_non_overlap_indices)))
        # print("guest_all_indices:{}, {}".format(guest_all_indices, len(guest_all_indices)))
        # print("host_non_overlap_indices:{}, {}".format(host_non_overlap_indices, len(host_non_overlap_indices)))
        # print("host_all_indices:{}, {}".format(host_all_indices, len(host_all_indices)))
        print("[INFO] labeled_overlap_indices: {}".format(len(labeled_overlap_indices)))
        if unlabeled_overlap_indices is None:
            print("[INFO] unlabeled_overlap_indices: {}".format(None))
        else:
            print("[INFO] unlabeled_overlap_indices: {}".format(len(unlabeled_overlap_indices)))
        print("[INFO] guest_non_overlap_indices: {}".format(len(guest_non_overlap_indices)))
        print("[INFO] host_non_overlap_indices: {}".format(len(host_non_overlap_indices)))
        print("[INFO] guest_all_indices: {}".format(len(guest_all_indices)))
        print("[INFO] host_all_indices: {}".format(len(host_all_indices)))

    sample_indices_dict = dict()
    sample_indices_dict["labeled_overlap_indices"] = labeled_overlap_indices
    sample_indices_dict["unlabeled_overlap_indices"] = unlabeled_overlap_indices
    sample_indices_dict["guest_non_overlap_indices"] = guest_non_overlap_indices
    sample_indices_dict["host_non_overlap_indices"] = host_non_overlap_indices
    sample_indices_dict["guest_all_indices"] = guest_all_indices
    sample_indices_dict["host_all_indices"] = host_all_indices
    return sample_indices_dict


def generate_sampler(num_labeled_overlap_samples, num_overlap_samples, train_dataset, use_only_ll):
    num_all_train = len(train_dataset)
    sample_indices_dict = prepare_data_indices(num_all_train, num_labeled_overlap_samples, num_overlap_samples, use_only_ll)

    labeled_overlap_indices = sample_indices_dict["labeled_overlap_indices"]
    unlabeled_overlap_indices = sample_indices_dict["unlabeled_overlap_indices"]
    guest_non_overlap_indices = sample_indices_dict["guest_non_overlap_indices"]
    host_non_overlap_indices = sample_indices_dict["host_non_overlap_indices"]
    guest_all_indices = sample_indices_dict["guest_all_indices"]
    host_all_indices = sample_indices_dict["host_all_indices"]

    # labeled_overlap_indices_sampler = SubsetRandomSampler(labeled_overlap_indices)
    if labeled_overlap_indices is None or len(labeled_overlap_indices) == 0:
        labeled_overlap_indices_sampler = None
    else:
        labeled_overlap_indices_sampler = SubsetSampler(labeled_overlap_indices)
    unlabeled_overlap_indices_sampler = None if unlabeled_overlap_indices is None else SubsetRandomSampler(
        unlabeled_overlap_indices)
    guest_non_overlap_indices_sampler = None if guest_non_overlap_indices is None else SubsetRandomSampler(
        guest_non_overlap_indices)
    host_non_overlap_indices_sampler = None if host_non_overlap_indices is None else SubsetRandomSampler(
        host_non_overlap_indices)
    guest_all_indices_sampler = None if guest_all_indices is None else SubsetRandomSampler(guest_all_indices)
    host_all_indices_sampler = None if host_all_indices is None else SubsetRandomSampler(host_all_indices)

    val_indices_sampler = SubsetSampler(labeled_overlap_indices)

    sampler_dict = dict()
    sampler_dict["labeled_overlap_indices_sampler"] = labeled_overlap_indices_sampler
    sampler_dict["unlabeled_overlap_indices_sampler"] = unlabeled_overlap_indices_sampler
    sampler_dict["guest_non_overlap_indices_sampler"] = guest_non_overlap_indices_sampler
    sampler_dict["host_non_overlap_indices_sampler"] = host_non_overlap_indices_sampler
    sampler_dict["guest_all_indices_sampler"] = guest_all_indices_sampler
    sampler_dict["host_all_indices_sampler"] = host_all_indices_sampler
    sampler_dict["val_indices_sampler"] = val_indices_sampler

    return sampler_dict


def run_experiment(
        train_dataset,
        test_dataset,
        num_overlap_samples,
        num_labeled_overlap_samples,
        exp_param_dict,
        epoch,
        ul_overlap_sample_batch_size,
        ll_overlap_sample_batch_size,
        non_overlap_sample_batch_size,
        estimation_block_size,
        training_info_file_name,
        sharpen_temperature,
        label_prob_sharpen_temperature,
        fed_label_prob_threshold,
        guest_label_prob_threshold,
        host_label_prob_threshold,
        is_hetero_reprs,
        using_uniq,
        using_comm,
        hidden_dim,
        num_class,
        vfl_guest_constructor,
        vfl_host_constructor,
        other_args,
        data_type="tab",
        normalize_repr=False,
        only_use_ll=False,
        debug=False):

    sampler_dict = generate_sampler(num_labeled_overlap_samples, num_overlap_samples, train_dataset, only_use_ll)

    labeled_overlap_indices_sampler = sampler_dict["labeled_overlap_indices_sampler"]
    val_indices_sampler = sampler_dict["val_indices_sampler"]

    train_dataset_copy = copy.deepcopy(train_dataset)

    ll_dataloader = get_loader(train_dataset, labeled_overlap_indices_sampler, ll_overlap_sample_batch_size)
    val_dataloader = get_loader(train_dataset_copy, val_indices_sampler, 1024, shuffle=False)
    test_dataloader = get_loader(test_dataset, None, 2048, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=True)

    if only_use_ll:
        ul_dataloader = None
        nl_guest_dataloader = None
        nl_host_dataloader = None
        all_guest_dataloader = None
        all_host_dataloader = None

        num_ll_samples = len(ll_dataloader)
        num_iterations_per_epoch = int(num_ll_samples / ll_overlap_sample_batch_size) + 1
        print("num_ll_samples: ", num_ll_samples)
        print("num_iterations_per_epoch: ", num_iterations_per_epoch)
    else:
        unlabeled_overlap_indices_sampler = sampler_dict["unlabeled_overlap_indices_sampler"]
        guest_non_overlap_indices_sampler = sampler_dict["guest_non_overlap_indices_sampler"]
        host_non_overlap_indices_sampler = sampler_dict["host_non_overlap_indices_sampler"]
        guest_all_indices_sampler = sampler_dict["guest_all_indices_sampler"]
        host_all_indices_sampler = sampler_dict["host_all_indices_sampler"]

        ul_dataloader = get_loader(train_dataset, unlabeled_overlap_indices_sampler, ul_overlap_sample_batch_size)
        nl_guest_dataloader = get_loader(train_dataset, guest_non_overlap_indices_sampler, non_overlap_sample_batch_size)
        nl_host_dataloader = get_loader(train_dataset, host_non_overlap_indices_sampler, non_overlap_sample_batch_size)
        all_guest_dataloader = get_loader(train_dataset, guest_all_indices_sampler, estimation_block_size)
        all_host_dataloader = get_loader(train_dataset, host_all_indices_sampler, estimation_block_size)

        num_nl_samples = max(len(guest_all_indices_sampler), len(host_all_indices_sampler))
        num_iterations_per_epoch = int(num_nl_samples / non_overlap_sample_batch_size) + 1
        print("num_nl_samples: ", num_nl_samples)
        print("num_iterations_per_epoch: ", num_iterations_per_epoch)

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

    device = other_args["device"]
    guest_model_param = PartyModelParam(data_folder=None,
                                        apply_dropout=False,
                                        hidden_dim_list=[guest_hidden_dim],
                                        n_classes=num_class,
                                        normalize_repr=normalize_repr,
                                        data_type=data_type,
                                        device=device)
    host_model_param = PartyModelParam(data_folder=None,
                                       apply_dropout=False,
                                       hidden_dim_list=[host_hidden_dim],
                                       n_classes=num_class,
                                       normalize_repr=normalize_repr,
                                       data_type=data_type,
                                       device=device)

    learning_rate = exp_param_dict["learning_rate"]
    weight_decay = exp_param_dict["weight_decay"]
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

    metric = other_args["monitor_metric"]
    valid_iteration_interval = other_args["valid_iteration_interval"]
    aggregation_mode = other_args["aggregation_mode"]
    fed_input_dim = host_input_dim + guest_input_dim
    fed_training_param = FederatedTrainingParam(fed_input_dim=fed_input_dim,
                                                guest_input_dim=guest_input_dim,
                                                host_input_dim=host_input_dim,
                                                is_hetero_repr=is_hetero_reprs,
                                                using_block_idx=False,
                                                learning_rate=learning_rate,
                                                weight_decay=weight_decay,
                                                fed_reg_lambda=0.0,
                                                guest_reg_lambda=0.0,
                                                loss_weight_dict=loss_weight_dict,
                                                # overlap_indices=overlap_indices,
                                                # non_overlap_indices=non_overlap_indices,
                                                num_labeled_overlap_samples=num_labeled_overlap_samples,
                                                epoch=epoch,
                                                top_k=1,
                                                unlabeled_overlap_sample_batch_size=ul_overlap_sample_batch_size,
                                                labeled_overlap_sample_batch_size=ll_overlap_sample_batch_size,
                                                non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                                overlap_sample_batch_num=num_overlap_samples,
                                                all_sample_block_size=estimation_block_size,
                                                label_prob_sharpen_temperature=label_prob_sharpen_temperature,
                                                sharpen_temperature=sharpen_temperature,
                                                fed_label_prob_threshold=fed_label_prob_threshold,
                                                guest_label_prob_threshold=guest_label_prob_threshold,
                                                host_label_prob_threshold=host_label_prob_threshold,
                                                training_info_file_name=training_info_file_name,
                                                valid_iteration_interval=valid_iteration_interval,
                                                using_uniq=using_uniq,
                                                using_comm=using_comm,
                                                monitor_metric=metric,
                                                aggregation_mode=aggregation_mode,
                                                device=device)

    # set up and train model
    guest_constructor = vfl_guest_constructor(guest_model_param, fed_training_param)
    host_constructor = vfl_host_constructor(host_model_param, fed_training_param)

    guest = guest_constructor.build(args=other_args, debug=debug)
    host = host_constructor.build(args=other_args, debug=debug)

    VFTL = VerticalFederatedTransferLearning(vftl_guest=guest, vftl_host=host,
                                             fed_training_param=fed_training_param,
                                             debug=debug)
    VFTL.build()
    VFTL.train(ll_data_loader=ll_dataloader,
               ul_data_loader=ul_dataloader,
               nl_guest_data_loader=nl_guest_dataloader,
               nl_host_data_loader=nl_host_dataloader,
               all_guest_data_loader=all_guest_dataloader,
               all_host_data_loader=all_host_dataloader,
               val_data_loader=val_dataloader,
               test_dataloader=test_dataloader,
               num_iteration_per_epoch=num_iterations_per_epoch,
               only_use_ll=only_use_ll)


def batch_run_experiments(train_dataset, test_dataset, optim_args, loss_weight_args, training_args, other_args=None):
    using_uniq = True
    using_comm = True
    file_folder = "training_log_info_3/"
    timestamp = get_timestamp()

    weight_decay = optim_args["weight_decay"]
    lr_list = optim_args["learning_rate_list"]

    data_type = training_args["data_type"]
    is_hetero_reprs = training_args["is_hetero_reprs"]
    num_overlap_list = training_args["num_overlap_list"]
    num_labeled_overlap_list = training_args["num_labeled_overlap_list"]
    ul_overlap_sample_batch_size = training_args["ul_overlap_sample_batch_size"]
    ll_overlap_sample_batch_size = training_args["ll_overlap_sample_batch_size"]
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
    guest_label_prob_threshold = training_args["guest_label_prob_threshold"]
    host_label_prob_threshold = training_args["host_label_prob_threshold"]

    lambda_dist_shared_reprs = loss_weight_args["lambda_dist_shared_reprs"]
    lambda_sim_shared_reprs_vs_uniq_reprs = loss_weight_args["lambda_sim_shared_reprs_vs_uniq_reprs"]
    lambda_host_dist_ested_lbls_vs_true_lbls = loss_weight_args["lambda_host_dist_ested_lbls_vs_true_lbls"]
    lambda_dist_ested_reprs_vs_true_reprs = loss_weight_args["lambda_dist_ested_reprs_vs_true_reprs"]
    lambda_host_dist_two_ested_lbls = loss_weight_args["lambda_host_dist_two_ested_lbls"]

    # use only labeled aligned samples
    only_use_ll = training_args.get("only_use_ll")

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
    for n_ll in num_labeled_overlap_list:
        for n_ol in num_overlap_list:
            for lbda_0 in lr_list:
                for lbda_1 in lambda_dist_shared_reprs:
                    for lbda_2 in lambda_sim_shared_reprs_vs_uniq_reprs:
                        for lbda_3 in lambda_host_dist_ested_lbls_vs_true_lbls:
                            for lbda_4 in lambda_dist_ested_reprs_vs_true_reprs:
                                for lbda_5 in lambda_host_dist_two_ested_lbls:
                                    # TODO: refactor the creation of file name
                                    seed = other_args.get("seed")
                                    name = other_args.get("name")
                                    file_name = file_folder + name + "_" + str(n_ll) + "_" + timestamp + "_seed" + str(
                                        seed)

                                    exp_param_dict["learning_rate"] = lbda_0
                                    exp_param_dict["weight_decay"] = weight_decay
                                    exp_param_dict["lambda_dist_shared_reprs"] = lbda_1
                                    exp_param_dict["lambda_sim_shared_reprs_vs_unique_repr"] = lbda_2
                                    exp_param_dict["lambda_host_dist_ested_lbl_vs_true_lbl"] = lbda_3
                                    exp_param_dict["lambda_dist_ested_repr_vs_true_repr"] = lbda_4
                                    exp_param_dict["lambda_host_dist_two_ested_lbl"] = lbda_5

                                    run_experiment(
                                        train_dataset=train_dataset,
                                        test_dataset=test_dataset,
                                        num_overlap_samples=n_ol,
                                        num_labeled_overlap_samples=n_ll,
                                        exp_param_dict=exp_param_dict,
                                        epoch=epoch,
                                        ul_overlap_sample_batch_size=ul_overlap_sample_batch_size,
                                        ll_overlap_sample_batch_size=ll_overlap_sample_batch_size,
                                        non_overlap_sample_batch_size=non_overlap_sample_batch_size,
                                        estimation_block_size=estimation_block_size,
                                        training_info_file_name=file_name,
                                        sharpen_temperature=sharpen_temperature,
                                        label_prob_sharpen_temperature=label_prob_sharpen_temperature,
                                        fed_label_prob_threshold=fed_label_prob_threshold,
                                        guest_label_prob_threshold=guest_label_prob_threshold,
                                        host_label_prob_threshold=host_label_prob_threshold,
                                        is_hetero_reprs=is_hetero_reprs,
                                        using_uniq=using_uniq,
                                        using_comm=using_comm,
                                        hidden_dim=hidden_dim,
                                        num_class=num_class,
                                        vfl_guest_constructor=vfl_guest_constructor,
                                        vfl_host_constructor=vfl_host_constructor,
                                        data_type=data_type,
                                        other_args=other_args,
                                        normalize_repr=normalize_repr,
                                        only_use_ll=only_use_ll)
