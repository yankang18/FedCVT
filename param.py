from collections import OrderedDict


class PartyModelParam(object):
    def __init__(self, n_class, dense_units=None, data_folder=None, keep_probability=0.7, learning_rate=0.01,
                 input_shape=None, hidden_dim_list=None, apply_dropout=False):
        self.nn_learning_rate = learning_rate
        self.n_class = n_class
        self.keep_probability = keep_probability
        self.hidden_dim_list = hidden_dim_list
        self.input_shape = input_shape
        self.apply_dropout = apply_dropout
        self.data_folder = data_folder
        self.dense_units = dense_units


class FederatedModelParam(object):
    def __init__(self, fed_input_dim, guest_input_dim=None, fed_hidden_dim=None, guest_hidden_dim=None, using_block_idx=True,
                 num_guest_nonoverlap_samples=None, num_host_nonoverlap_samples=None,
                 learning_rate=0.01, fed_reg_lambda=0.01, guest_reg_lambda=0.0, loss_weight_dict=None, overlap_indices=None,
                 non_overlap_indices=None, epoch=50, top_k=3, combine_axis=0, parallel_iterations=10,
                 overlap_sample_batch_size=None, non_overlap_sample_batch_size=None,
                 overlap_sample_batch_num=10, all_sample_block_size=500,
                 is_hetero_repr=False, sharpen_temperature=0.1, label_prob_sharpen_temperature=0.5, 
                 fed_label_prob_threshold=0.7, host_label_prob_threshold=0.6,
                 training_info_file_name=None, valid_iteration_interval=5):
        self.learning_rate = learning_rate
        self.using_block_idx = using_block_idx
        self.fed_input_dim = fed_input_dim
        self.fed_hidden_dim = fed_hidden_dim
        self.guest_input_dim = guest_input_dim
        self.guest_hidden_dim = guest_hidden_dim
        self.fed_reg_lambda = fed_reg_lambda
        self.guest_reg_lamba = guest_reg_lambda
        self.loss_weight_dict = loss_weight_dict
        self.overlap_indices = overlap_indices
        self.non_overlap_indices = non_overlap_indices
        self.epoch = epoch
        self.top_k = top_k
        self.combine_axis = combine_axis
        self.parallel_iterations = parallel_iterations
        self.num_guest_nonoverlap_samples = num_guest_nonoverlap_samples
        self.num_host_nonoverlap_samples = num_host_nonoverlap_samples
        self.overlap_sample_batch_size = overlap_sample_batch_size
        self.non_overlap_sample_batch_size = non_overlap_sample_batch_size
        self.overlap_sample_batch_num = overlap_sample_batch_num
        self.all_sample_block_size = all_sample_block_size
        self.is_hetero_repr = is_hetero_repr
        self.label_prob_sharpen_temperature = label_prob_sharpen_temperature
        self.sharpen_temperature = sharpen_temperature
        self.host_label_prob_threshold = host_label_prob_threshold
        self.fed_label_prob_threshold = fed_label_prob_threshold
        self.training_info_file_name = training_info_file_name
        self.valid_iteration_interval = valid_iteration_interval

    def get_parameters(self):
        param_dict = OrderedDict()
        param_dict["reg_lambda"] = self.fed_reg_lambda
        # for idx in range(len(self.loss_weight_list)):
        #     param_dict["loss_weight_" + str(idx)] = self.loss_weight_list[idx]
        param_dict["loss_lambda_dict"] = self.loss_weight_dict
        param_dict["learning_rate"] = self.learning_rate
        param_dict["input_dim"] = self.fed_input_dim
        param_dict["epoch"] = self.epoch
        # param_dict["non_overlap_sample_batch_num"] = self.non_overlap_sample_batch_num
        param_dict["overlap_sample_batch_sum"] = self.overlap_sample_batch_num
        param_dict["all_sample_block_size"] = self.all_sample_block_size
        param_dict["is_hetero_repr"] = self.is_hetero_repr
        param_dict["sharpen_temperature"] = self.sharpen_temperature
        param_dict["label_prob_threshold"] = self.fed_label_prob_threshold
        return param_dict