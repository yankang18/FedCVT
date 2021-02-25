
import numpy as np
import tensorflow as tf


class LogisticRegression(object):

    def __init__(self, an_id):
        super(LogisticRegression, self).__init__()
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

    def save_model(self):
        print("TODO: save model")

    def set_session(self, sess):
        self.sess = sess

    def get_session(self):
        return self.sess

    def build(self, input_dim, n_class, hidden_dim=None, stddev=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.stddev = stddev
        self._set_variable_initializer()
        self._build_model()

    def _set_variable_initializer(self):
        if self.hidden_dim is None:
            print("Applied one layer with input_dim: {0}, hidden_dim: {1}, num_class: {2}".format(self.input_dim,
                                                                                                  self.hidden_dim,
                                                                                                  self.n_class))
            self.W_1_initializer = tf.random.normal((self.input_dim, self.n_class), stddev=self.stddev, dtype=tf.float32)
            self.b_1_initializer = np.zeros(self.n_class).astype(np.float32)
        else:
            print("Applied two layers with hidden_dim: {0}".format(self.hidden_dim))
            self.W_1_initializer = tf.random.normal((self.input_dim, self.hidden_dim), stddev=self.stddev, dtype=tf.float32)
            self.b_1_initializer = np.zeros(self.hidden_dim).astype(np.float32)
            self.W_2_initializer = tf.random.normal((self.hidden_dim, self.n_class), stddev=self.stddev, dtype=tf.float32)
            self.b_2_initializer = np.zeros(self.n_class).astype(np.float32)

    def set_ops(self, learning_rate=1e-2, reg_lambda=0.01, tf_X_in=None, tf_labels_in=None):
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        self.tf_X_in = tf_X_in
        self.tf_labels_in = tf_labels_in
        self._add_forward_ops()
        self._add_e2e_training_ops()
        self._add_evaluate_ops()

    def _build_model(self):
        self._add_input_placeholder()
        self._add_model_variables()
        # self._add_forward_ops()
        # self._add_e2e_training_ops()
        # self._add_evaluate_ops()

    def _add_model_variables(self):
        self.model_vars_scope = "lr_" + self.id + "_model_vars"
        self.lr_trainable_variables = []
        with tf.compat.v1.variable_scope(self.model_vars_scope):
            self.W_1 = tf.compat.v1.get_variable("weights_1", initializer=self.W_1_initializer, dtype=tf.float32)
            self.b_1 = tf.compat.v1.get_variable("bias_1", initializer=self.b_1_initializer, dtype=tf.float32)
            self.lr_trainable_variables.append(self.W_1)
            self.lr_trainable_variables.append(self.b_1)
            if self.hidden_dim is not None:
                self.W_2 = tf.compat.v1.get_variable("weights_2", initializer=self.W_2_initializer, dtype=tf.float32)
                self.b_2 = tf.compat.v1.get_variable("bias_2", initializer=self.b_2_initializer, dtype=tf.float32)
                self.lr_trainable_variables.append(self.W_2)
                self.lr_trainable_variables.append(self.b_2)

    def _add_input_placeholder(self):
        self.X_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.input_dim))
        self.labels = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_class), name="labels_input")

    def _add_forward_ops(self):
        if self.tf_X_in is None:
            self.logits = self._forward_logits(self.X_in)
            self.y_hat = self._forward_output(self.X_in)
        else:
            self.logits = self._forward_logits(self.tf_X_in)
            self.y_hat = self._forward_output(self.tf_X_in)

    def set_two_sides_predict_ops(self, reprs):
        self.y_hat_two_side = self._forward_output(reprs)

    def set_guest_side_predict_ops(self, reprs):
        self.y_hat_guest_side = self._forward_output(reprs)

    def predict_lbls_for_reprs(self, reprs):
        self.y_hat_host_side = self._forward_output(reprs)
        self.reprs = reprs
        return self.y_hat_host_side

    def _forward_logits(self, X):
        layer_1 = tf.matmul(X, self.W_1) + self.b_1
        if self.hidden_dim is None:
            return layer_1
        else:
            layer_1 = tf.nn.leaky_relu(layer_1)
            return tf.matmul(layer_1, self.W_2) + self.b_2

    def _forward_output(self, X):
        return tf.nn.softmax(self._forward_logits(X))
        # return tf.sigmoid(self._forward_logits(X))

    def _add_e2e_training_ops(self):
        # regularization loss
        self.reg_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.compat.v1.trainable_variables()])
        if self.tf_labels_in is None:
            self.pred_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        else:
            print("using feature extraction loss")
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.stop_gradient(self.tf_labels_in))
            # self.pred_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.tf_labels_in)
        self.mean_pred_loss = tf.reduce_mean(input_tensor=self.pred_loss)
        self.loss = self.mean_pred_loss + self.reg_lambda * self.reg_loss
        self.loss = self.append_loss_factors(self.loss)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr)
        # optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        # self.e2e_train_op = optimizer.minimize(self.loss)
        self.computed_gradients = optimizer.compute_gradients(self.loss)
        self.e2e_train_op = optimizer.apply_gradients(self.computed_gradients)

    def set_loss_factors(self, loss_factor_dict, loss_factor_weight_dict):
        self.loss_factor_dict = loss_factor_dict
        self.loss_factor_weight_dict = loss_factor_weight_dict

    def append_loss_factors(self, loss):
        if self.loss_factor_dict is None or self.loss_factor_weight_dict is None:
            return loss

        print("[DEBUG] append loss factors:")
        for key, loss_fac in self.loss_factor_dict.items():
            loss_fac_weight = self.loss_factor_weight_dict[key]
            print(f"[DEBUG] append loss factor: {key}, [{loss_fac_weight}], {loss_fac}")
            loss = loss + loss_fac_weight * loss_fac
        return loss

    def predict(self, X):
        return self.sess.run(self.y_hat, feed_dict={self.X_in: X})

    def _add_evaluate_ops(self):
        # The default threshold is 0.5, rounded off directly
        prediction = tf.round(self.y_hat)
        correct = tf.cast(tf.equal(prediction, self.labels), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(input_tensor=correct)

    def get_model_parameters(self):
        _W = self.sess.run(self.W_1)
        _b = self.sess.run(self.b_1)
        return {"W": _W, "b": _b}
