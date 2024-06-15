from collections import OrderedDict


class PartyModelParam(object):
    def __init__(self, n_classes, dense_units=None, data_folder=None,
                 keep_probability=0.7, learning_rate=0.01,
                 input_shape=None, hidden_dim_list=None, apply_dropout=False,
                 normalize_repr=True, data_type="img", device="cpu"):
        self.nn_learning_rate = learning_rate
        self.n_classes = n_classes
        self.keep_probability = keep_probability
        self.hidden_dim_list = hidden_dim_list
        self.input_shape = input_shape
        self.apply_dropout = apply_dropout
        self.data_folder = data_folder
        self.dense_units = dense_units
        self.normalize_repr = normalize_repr
        self.data_type = data_type
        self.device = device


class FederatedTrainingParam(object):
    def __init__(self, fed_input_dim, guest_input_dim=None, host_input_dim=None,
                 fed_hidden_dim=None, guest_hidden_dim=None, host_hidden_dim=None,
                 using_block_idx=True, num_guest_nonoverlap_samples=None, num_host_nonoverlap_samples=None,
                 learning_rate=0.01, weight_decay=1e-6,
                 fed_reg_lambda=0.01, guest_reg_lambda=0.0, loss_weight_dict=None,
                 overlap_indices=None, non_overlap_indices=None, num_labeled_overlap_samples=None,
                 epoch=50, top_k=3, combine_axis=0, parallel_iterations=10,
                 unlabeled_overlap_sample_batch_size=None, labeled_overlap_sample_batch_size=None,
                 non_overlap_sample_batch_size=None, overlap_sample_batch_num=10, all_sample_block_size=500,
                 is_hetero_repr=False, sharpen_temperature=0.1, label_prob_sharpen_temperature=0.5,
                 fed_label_prob_threshold=0.7, guest_label_prob_threshold=0.7, host_label_prob_threshold=0.6,
                 training_info_file_name=None, valid_iteration_interval=5,
                 using_uniq=True, using_comm=True, aggregation_mode="cat", monitor_metric="acc", device="cpu"):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.using_block_idx = using_block_idx
        self.fed_input_dim = fed_input_dim
        self.fed_hidden_dim = fed_hidden_dim
        self.guest_input_dim = guest_input_dim
        self.host_input_dim = host_input_dim
        self.guest_hidden_dim = guest_hidden_dim
        self.host_hidden_dim = host_hidden_dim
        self.fed_reg_lambda = fed_reg_lambda
        self.guest_reg_lamba = guest_reg_lambda
        self.loss_weight_dict = loss_weight_dict
        self.overlap_indices = overlap_indices
        self.non_overlap_indices = non_overlap_indices
        self.num_labeled_overlap_samples = num_labeled_overlap_samples
        self.epoch = epoch
        self.top_k = top_k
        self.combine_axis = combine_axis
        self.parallel_iterations = parallel_iterations
        self.num_guest_nonoverlap_samples = num_guest_nonoverlap_samples
        self.num_host_nonoverlap_samples = num_host_nonoverlap_samples
        self.unlabeled_overlap_sample_batch_size = unlabeled_overlap_sample_batch_size
        self.labeled_overlap_sample_batch_size = labeled_overlap_sample_batch_size
        self.non_overlap_sample_batch_size = non_overlap_sample_batch_size
        self.overlap_sample_batch_num = overlap_sample_batch_num
        self.all_sample_block_size = all_sample_block_size
        self.is_hetero_repr = is_hetero_repr
        self.label_prob_sharpen_temperature = label_prob_sharpen_temperature
        self.sharpen_temperature = sharpen_temperature
        self.fed_label_prob_threshold = fed_label_prob_threshold
        self.guest_label_prob_threshold = guest_label_prob_threshold
        self.host_label_prob_threshold = host_label_prob_threshold
        self.training_info_file_name = training_info_file_name
        self.valid_iteration_interval = valid_iteration_interval
        self.using_uniq = using_uniq
        self.using_comm = using_comm
        self.monitor_metric = monitor_metric
        self.aggregation_mode = aggregation_mode
        self.device = device

    def get_parameters(self):
        param_dict = OrderedDict()
        param_dict["reg_lambda"] = self.fed_reg_lambda
        # for idx in range(len(self.loss_weight_list)):
        #     param_dict["loss_weight_" + str(idx)] = self.loss_weight_list[idx]
        param_dict["loss_lambda_dict"] = self.loss_weight_dict
        param_dict["learning_rate"] = self.learning_rate
        param_dict["input_dim"] = self.fed_input_dim
        param_dict["epoch"] = self.epoch
        param_dict["metric"] = self.monitor_metric
        param_dict["overlap_sample_batch_sum"] = self.overlap_sample_batch_num
        param_dict["all_sample_block_size"] = self.all_sample_block_size
        param_dict["is_hetero_repr"] = self.is_hetero_repr
        param_dict["sharpen_temperature"] = self.sharpen_temperature
        param_dict["label_prob_threshold"] = self.fed_label_prob_threshold
        return param_dict
