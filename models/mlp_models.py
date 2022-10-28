import torch.nn as nn


# import matplotlib.pyplot as plt


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
        self.n_class = None
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

    def build(self, input_dim, n_class, hidden_dim=None, stddev=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.stddev = stddev
        # self.lr = learning_rate
        self._build_model()

    def _build_model(self):
        if self.hidden_dim is None:
            print(
                "[INFO] Softmax Regressor applies one layer with input_dim: {0}, num_class: {1}".format(self.input_dim,
                                                                                                        self.n_class))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=self.n_class),
            )
        else:
            print("[INFO] Softmax Regressor applies two layers with hidden_dim: {0}".format(self.hidden_dim))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(in_features=self.hidden_dim, out_features=self.n_class)
            )

    def forward(self, x):
        return self.classifier(x)

# class Autoencoder(FeatureExtractor):
#
#     def __init__(self, an_id):
#         super(Autoencoder, self).__init__()
#         self.id = str(an_id)
#         self.sess = None
#         self.enc_layer_vars_initializer_map = {}
#         self.dec_layer_vars_initializer_map = {}
#         self.enc_layer_vars_map = {}
#         self.dec_layer_vars_map = {}
#
#         self.lr = None
#         self.input_dim = None
#         self.hidden_dim_list = None
#
#     def set_session(self, sess):
#         self.sess = sess
#
#     def get_session(self):
#         return self.sess
#
#     def build(self, input_dim, hidden_dim_list, learning_rate=1e-2):
#         self.lr = learning_rate
#         self.input_dim = input_dim
#         self.hidden_dim_list = hidden_dim_list
#         self._set_variable_initializer()
#         self._build_model()
#
#     def _glorot_normal(self, fan_in, fan_out):
#         stddev = np.sqrt(2.0 / (fan_in + fan_out))
#         return tf.random.normal(shape=(fan_in, fan_out), mean=0.0, stddev=stddev, dtype=tf.dtypes.float32)
#
#     def _initialize(self, in_dim, out_dim):
#         return tf.random.normal((in_dim, out_dim), dtype=tf.float32)
#         # return self._glorot_normal(in_dim, out_dim)
#
#     def _set_variable_initializer(self):
#
#         in_dim = self.input_dim
#
#         print("in_dim", in_dim)
#         layer = 0
#         for out_dim in self.hidden_dim_list:
#             print("out_dim", out_dim)
#             self.enc_layer_vars_initializer_map["We_" + str(layer)] = self._initialize(in_dim, out_dim)
#             self.enc_layer_vars_initializer_map["be_" + str(layer)] = np.zeros(out_dim).astype(np.float32)
#             layer += 1
#             in_dim = out_dim
#
#         layer = 0
#         for out_dim in reversed(self.hidden_dim_list[:-1]):
#             self.dec_layer_vars_initializer_map["Wd_" + str(layer)] = self._initialize(in_dim, out_dim)
#             self.dec_layer_vars_initializer_map["bd_" + str(layer)] = np.zeros(out_dim).astype(np.float32)
#             layer += 1
#             in_dim = out_dim
#
#         self.dec_layer_vars_initializer_map["Wd_" + str(layer)] = self._initialize(in_dim, self.input_dim)
#         self.dec_layer_vars_initializer_map["bd_" + str(layer)] = np.zeros(self.input_dim).astype(np.float32)
#
#     def _build_model(self):
#         # self._add_input_placeholder()
#         self._add_encoder_decoder_ops()
#         self._add_forward_ops()
#         # self._add_representation_training_ops()
#         self._add_e2e_training_ops()
#         # self._add_encrypt_grad_update_ops()
#
#     # def _add_input_placeholder(self):
#     #     self.X_all_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.input_dim), name="X_input_all")
#     #     self.X_overlap_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.input_dim), name="X_input_overlap")
#     #     self.X_non_overlap_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.input_dim), name="X_input_non_overlap")
#     #     # self.X_in = self._gaussian_additive_noise(self.X_in, 0.01)
#     #     self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
#     #     self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
#     #
#     # def get_all_samples(self):
#     #     return self.X_all_in
#     #
#     # def get_overlap_samples(self):
#     #     return self.X_overlap_in
#     #
#     # def get_non_overlap_samples(self):
#     #     return self.X_non_overlap_in
#
#     @staticmethod
#     def _gaussian_additive_noise(X_in, std):
#         return X_in + tf.random.normal(shape=tf.shape(input=X_in), dtype=tf.float32, mean=0.0, stddev=std)
#
#     def _add_encoder_decoder_ops(self):
#         self.encoder_vars_scope = self.id + "_encoder_vars"
#         with tf.compat.v1.variable_scope(self.encoder_vars_scope):
#             for i in range(len(self.hidden_dim_list)):
#                 self.enc_layer_vars_map["We_" + str(i)] = tf.compat.v1.get_variable(name="We_" + str(i),
#                                                                           initializer=self.enc_layer_vars_initializer_map["We_" + str(i)],
#                                                                           dtype=tf.float32)
#                 self.enc_layer_vars_map["be_" + str(i)] = tf.compat.v1.get_variable(name="be_" + str(i),
#                                                                           initializer=self.enc_layer_vars_initializer_map["be_" + str(i)],
#                                                                           dtype=tf.float32)
#
#         self.decoder_vars_scope = self.id + "_decoder_vars"
#         with tf.compat.v1.variable_scope(self.decoder_vars_scope):
#             for i in range(len(self.hidden_dim_list)):
#                 self.dec_layer_vars_map["Wd_" + str(i)] = tf.compat.v1.get_variable(name="Wd_" + str(i),
#                                                                           initializer=self.dec_layer_vars_initializer_map["Wd_" + str(i)],
#                                                                           dtype=tf.float32)
#                 self.dec_layer_vars_map["bd_" + str(i)] = tf.compat.v1.get_variable(name="bd_" + str(i),
#                                                                           initializer=self.dec_layer_vars_initializer_map["bd_" + str(i)],
#                                                                           dtype=tf.float32)
#
#     # def _add_forward_ops(self):
#     #     self.all_hidden_reprs = self._forward_hidden(self.X_all_in)
#     #     self.overlap_hidden_reprs = self._forward_hidden(self.X_overlap_in)
#     #     self.non_overlap_hidden_reprs = self._forward_hidden(self.X_non_overlap_in)
#     #     self.all_logits = self._forward_logits(self.X_all_in)
#     #     self.all_y_hat = self._forward_output(self.X_all_in)
#
#     # def _add_representation_training_ops(self):
#         # vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_vars_scope)
#         # self.init_grad = tf.placeholder(tf.float32, shape=(None, self.hidden_dim))
#         # self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.Z, var_list=vars_to_train, grad_loss=self.init_grad)
#
#     def _add_e2e_training_ops(self):
#         encoder_vars_to_train = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_vars_scope)
#         decoder_vars_to_train = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.decoder_vars_scope)
#         vars_to_train = encoder_vars_to_train + decoder_vars_to_train
#         # print("vars_to_train:\n", vars_to_train)
#         l2_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in vars_to_train])
#         self.loss_ = tf.cast(tf.compat.v1.losses.mean_squared_error(predictions=self.all_logits, labels=self.X_all_in, reduction="none"), tf.float32)
#         self.loss = tf.reduce_mean(input_tensor=self.loss_) + 0.01 * l2_loss
#         self.e2e_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
#
#     def _forward_hidden(self, X):
#         layer_in = X
#         for i in range(len(self.hidden_dim_list)):
#             layer_in = self._do_forward_encode(layer_in, i)
#         return layer_in
#
#     def _do_forward_encode(self, X, layer_index):
#         We = self.enc_layer_vars_map["We_" + str(layer_index)]
#         be = self.enc_layer_vars_map["be_" + str(layer_index)]
#         # return tf.matmul(X, We)
#         # return tf.nn.tanh(tf.matmul(X, We))
#         return tf.matmul(X, We) + be
#         # return tf.nn.tanh(tf.matmul(X, We) + be)
#         # return tf.nn.leaky_relu(tf.matmul(X, We) + be)
#         # return tf.nn.sigmoid(tf.matmul(X, We))
#
#     def _forward_logits(self, X):
#         Z = self._forward_hidden(X)
#         layer_in = Z
#         for i in range(len(self.hidden_dim_list) - 1):
#             layer_in = self._do_forward_decode(layer_in, i)
#         last_layer_index = len(self.hidden_dim_list) - 1
#         Wd = self.dec_layer_vars_map["Wd_" + str(last_layer_index)]
#         bd = self.dec_layer_vars_map["bd_" + str(last_layer_index)]
#         return tf.matmul(layer_in, Wd) + bd
#
#     def _do_forward_decode(self, X, layer_index):
#         Wd = self.dec_layer_vars_initializer_map["Wd_" + str(layer_index)]
#         bd = self.dec_layer_vars_initializer_map["bd_" + str(layer_index)]
#         # return tf.matmul(X, Wd) + bd
#         return tf.nn.tanh(tf.matmul(X, Wd) + bd)
#         # return tf.nn.leaky_relu(tf.matmul(X, Wd) + bd)
#
#     def _forward_output(self, X):
#         return tf.sigmoid(self._forward_logits(X))
#
#     def compute_loss(self, X):
#         return self.sess.run(self.loss_, feed_dict={self.X_all_in: X})
#         # return self.sess.run(self.logits, feed_dict={self.X_in: X})
#
#     def transform(self, X):
#         return self.sess.run(self.all_hidden_reprs, feed_dict={self.X_all_in: X})
#
#     def get_all_hidden_reprs(self):
#         return self.all_hidden_reprs
#
#     def get_overlap_hidden_reprs(self):
#         return self.overlap_hidden_reprs
#
#     def get_non_overlap_hidden_reprs(self):
#         return self.non_overlap_hidden_reprs
#
#     def get_is_train(self):
#         return self.is_train
#
#     def get_keep_probability(self):
#         return self.keep_prob
#
#     def predict(self, X):
#         return self.sess.run(self.all_y_hat, feed_dict={self.X_all_in: X})
#
#     def get_encode_dim(self):
#         return self.hidden_dim_list[-1]
#
#     def get_model_parameters(self):
#         # _Wh = self.sess.run(self.Wh)
#         # _Wo = self.sess.run(self.Wo)
#         # _bh = self.sess.run(self.bh)
#         # _bo = self.sess.run(self.bo)
#         _Wh = self.sess.run(self.enc_layer_vars_map["We_" + str(0)])
#         _bo = self.sess.run(self.enc_layer_vars_map["be_" + str(0)])
#
#         # model_meta = {"learning_rate": self.lr,
#         #               "input_dim": self.input_dim,
#         #               "hidden_dim": self.hidden_dim}
#         # return {"Wh": _Wh, "bh": _bh, "Wo": _Wo, "bo": _bo, "model_meta": model_meta}
#         return {"Wh": _Wh, "bo": _bo}
#
#     # def restore_model(self, model_parameters):
#     #     self.Wh_initializer = model_parameters["Wh"]
#     #     self.bh_initializer = model_parameters["bh"]
#     #     self.Wo_initializer = model_parameters["Wo"]
#     #     self.bo_initializer = model_parameters["bo"]
#     #     model_meta = model_parameters["model_meta"]
#     #
#     #     self.lr = model_meta["learning_rate"]
#     #     self.input_dim = model_meta["input_dim"]
#     #     self.hidden_dim = model_meta["hidden_dim"]
#     #     self._build_model()
#
#     def fit(self, X, batch_size=32, epoch=1, show_fig=False):
#         N, D = X.shape
#         n_batches = N // batch_size
#         costs = []
#         for ep in range(epoch):
#             for i in range(n_batches + 1):
#                 batch = X[i * batch_size: i * batch_size + batch_size]
#                 _, c = self.sess.run([self.e2e_train_op, self.loss], feed_dict={self.X_all_in: batch})
#                 if i % 5 == 0:
#                     print(ep, i, "/", n_batches, "cost:", c)
#                 costs.append(c)
#
#             if len(costs) > 1 and costs[-1] < 0.02 and abs(costs[-1] - costs[len(costs) - 2]) < 0.01:
#                 # print(costs)
#                 # print("converged")
#                 break
#
#         if show_fig:
#             plt.plot(costs)
#             plt.show()
