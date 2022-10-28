import torch
import torch.nn as nn


class ClientVGG8(nn.Module):

    def __init__(self, an_id):
        super(ClientVGG8, self).__init__()
        self.id = str(an_id)
        print("[INFO] {0} is using ClientVGG8".format(an_id))
        act = nn.LeakyReLU
        ks = 3
        self.feature_extractor = nn.Sequential(
            # first conv block
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            # second conv block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            # third conv block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=48),
            act(inplace=True)
        )

    def forward(self, x):
        # print(f"x:{x.shape} {x.dtype}")
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


# class SoftmaxRegression(nn.Module):
#
#     def __init__(self, an_id):
#         super(SoftmaxRegression, self).__init__()
#         self.id = str(an_id)
#         self.input_dim = None
#         self.hidden_dim = None
#         self.n_class = None
#         self.sess = None
#         self.loss_factor_dict = None
#         self.loss_factor_weight_dict = None
#         self.y_hat_two_side = None
#         self.y_hat_guest_side = None
#         self.y_hat_host_side = None
#         self.tf_X_in = None
#         self.tf_labels_in = None
#         self.lr = 0.01
#         self.reg_lambda = 0.01
#         self.stddev = None
#         self.stop_training = False
#         self.reprs = None
#         self.lr_trainable_variables = None
#         self.comb_loss_lambda = 1.0
#
#     def build(self, input_dim, n_class, hidden_dim=None, stddev=0.1):
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.n_class = n_class
#         self.stddev = stddev
#         # self.lr = learning_rate
#         self._build_model()
#
#     def _build_model(self):
#         if self.hidden_dim is None:
#             print("[INFO] Softmax Regressor applies one layer with input_dim: {0}, num_class: {1}".format(self.input_dim,
#                                                                                                          self.n_class))
#             self.classifier = nn.Sequential(
#                 nn.Linear(in_features=self.input_dim, out_features=self.n_class),
#             )
#         else:
#             print("[INFO] Softmax Regressor applies two layers with hidden_dim: {0}".format(self.hidden_dim))
#             self.classifier = nn.Sequential(
#                 nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Linear(in_features=self.hidden_dim, out_features=self.n_class)
#             )
#
#     def forward(self, x):
#         return self.classifier(x)
