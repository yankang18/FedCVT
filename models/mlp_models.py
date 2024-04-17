import torch.nn as nn


class FeatureExtractor(object):

    def get_all_samples(self):
        pass

    def get_overlap_samples(self):
        pass

    def get_non_overlap_samples(self):
        pass

    def get_all_hidden_reprs(self):
        pass

    def get_overlap_hidden_reprs(self):
        pass

    def get_non_overlap_hidden_reprs(self):
        pass

    def get_encode_dim(self):
        pass

    def get_model_parameters(self):
        pass

    def set_session(self, sess):
        pass

    def get_session(self):
        pass

    def get_is_train(self):
        pass

    def get_keep_probability(self):
        pass


class SoftmaxRegression(nn.Module):

    def __init__(self, an_id):
        super(SoftmaxRegression, self).__init__()
        self.id = str(an_id)
        self.input_dim = None
        self.hidden_dim = None
        self.output_dim = None
        self.sess = None
        self.loss_factor_dict = None
        self.loss_factor_weight_dict = None
        self.y_hat_two_side = None
        self.y_hat_guest_side = None
        self.y_hat_host_side = None
        self.tf_X_in = None
        self.tf_labels_in = None
        self.lr = 0.01
        self.reg_lambda = 0.01
        self.stddev = None
        self.stop_training = False
        self.reprs = None
        self.lr_trainable_variables = None
        self.comb_loss_lambda = 1.0

    def build(self, input_dim, output_dim, hidden_dim=None, stddev=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.stddev = stddev
        # self.lr = learning_rate
        self._build_model()

    def _build_model(self):
        if self.hidden_dim is None:
            print(
                "[INFO] Softmax Regression applies one layer with input_dim:{0}, num_class:{1}".format(self.input_dim,
                                                                                                       self.output_dim))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=self.output_dim),
                nn.Sigmoid()
                # nn.Tanh()
                # nn.LeakyReLU(inplace=True, negative_slope=0.1)
            )
        else:
            print("[INFO] Softmax Regression applies two layers with hidden_dim:{0}".format(self.hidden_dim))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                # nn.Sigmoid(),
                nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim),
                # nn.Sigmoid()
            )

    def forward(self, x):
        return self.classifier(x)
