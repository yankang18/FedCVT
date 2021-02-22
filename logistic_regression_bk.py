
import numpy as np
import tensorflow as tf


class LogisticRegression(object):

    def __init__(self, an_id):
        super(LogisticRegression, self).__init__()
        self.id = str(an_id)
        self.sess = None
        self.loss_factor_list = None
        self.loss_factor_weight_list = None
        self.y_hat_two_side = None
        self.y_hat_guest_side = None
        self.y_hat_host_side = None

    def set_session(self, sess):
        self.sess = sess

    def get_session(self):
        return self.sess

    def build(self, input_dim, learning_rate=1e-2, reg_lambda=0.01, tf_X_in=None, tf_labels_in=None):
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        self.input_dim = input_dim
        self.tf_X_in = tf_X_in
        self.tf_labels_in = tf_labels_in
        self._set_variable_initializer()
        self._build_model()

    # def set_representation(self, representation):
    #     self.representation = representation

    def _set_variable_initializer(self):
        # self.W_initializer = tf.random_normal((self.input_dim, 1), stddev=np.sqrt(2./(self.input_dim + 1)), dtype=tf.float64)
        self.W_initializer = tf.random.normal((self.input_dim, 1), dtype=tf.float64)
        self.b_initializer = np.zeros(1).astype(np.float64)

    def _build_model(self):
        self._add_input_placeholder()
        self._add_model_variables()
        self._add_forward_ops()
        self._add_e2e_training_ops()
        self._add_evaluate_ops()

    def _add_model_variables(self):
        self.model_vars_scope = "lr_" + self.id + "_model_vars"
        with tf.compat.v1.variable_scope(self.model_vars_scope):
            self.W = tf.compat.v1.get_variable("weights", initializer=self.W_initializer, dtype=tf.float64)
            self.b = tf.compat.v1.get_variable("bias", initializer=self.b_initializer, dtype=tf.float64)

    def _add_input_placeholder(self):
        self.X_in = tf.compat.v1.placeholder(tf.float64, shape=(None, self.input_dim))
        # self.weights = tf.placeholder(tf.float64, shape=(None, 1))
        self.labels = tf.compat.v1.placeholder(tf.float64, shape=(None, 1), name="labels_input")

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

    def set_guest_side_train_ops(self, reprs, guest_train_labels):
        logits = self._forward_logits(reprs)
        guest_train_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=guest_train_labels)
        self.guest_train_loss = tf.reduce_sum(input_tensor=guest_train_loss)
        # self.guest_train_loss = tf.nn.l2_loss(logits - guest_train_labels)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        self.guest_train_op = optimizer.minimize(self.guest_train_loss)

    def set_host_side_predict_ops(self, reprs):
        self.y_hat_host_side = self._forward_output(reprs)

    def _forward_logits(self, X):
        return tf.matmul(X, self.W) + self.b

    def _forward_output(self, X):
        return tf.sigmoid(self._forward_logits(X))

    def _add_e2e_training_ops(self):
        self.reg_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float64)) for v in tf.compat.v1.trainable_variables()])
        # pred_loss = tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels), self.weights)
        if self.tf_labels_in is None:
            self.pred_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        else:
            print("using feature extraction loss")
            self.pred_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.tf_labels_in)
        self.pred_loss = tf.reduce_sum(input_tensor=self.pred_loss)
        self.loss = self.pred_loss + self.reg_lambda * self.reg_loss
        self.loss = self.append_loss_factors(self.loss)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        # self.computed_gradients = optimizer.compute_gradients(self.loss)
        self.e2e_train_op = optimizer.minimize(self.loss)

    def set_loss_factors(self, loss_factor_list, loss_factor_weight_list):
        self.loss_factor_list = loss_factor_list
        self.loss_factor_weight_list = loss_factor_weight_list

    def append_loss_factors(self, loss):
        for loss_fac, loss_fac_weight in zip(self.loss_factor_list, self.loss_factor_weight_list):
            print("append loss factor:", loss_fac_weight, loss_fac)
            loss = loss + loss_fac_weight * loss_fac
            print("appended loss")
        return loss

    def predict(self, X):
        return self.sess.run(self.y_hat, feed_dict={self.X_in: X})

    def _add_evaluate_ops(self):
        # The default threshold is 0.5, rounded off directly
        prediction = tf.round(self.y_hat)
        correct = tf.cast(tf.equal(prediction, self.labels), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(input_tensor=correct)

    def get_model_parameters(self):
        _W = self.sess.run(self.W)
        _b = self.sess.run(self.b)
        return {"W": _W, "b": _b}
