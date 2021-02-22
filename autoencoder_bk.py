
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(object):

    def __init__(self, an_id):
        super(Autoencoder, self).__init__()
        self.id = str(an_id)
        self.sess = None

    def set_session(self, sess):
        self.sess = sess

    def get_session(self):
        return self.sess

    def build(self, input_dim, hidden_dim, learning_rate=1e-2):
        self.lr = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._set_variable_initializer()
        self._build_model()

    def _set_variable_initializer(self):
        self.Wh_initializer = tf.random.normal((self.input_dim, self.hidden_dim), dtype=tf.float64)
        self.bh_initializer = np.zeros(self.hidden_dim).astype(np.float64)
        self.Wo_initializer = tf.random.normal((self.hidden_dim, self.input_dim), dtype=tf.float64)
        self.bo_initializer = np.zeros(self.input_dim).astype(np.float64)

    def _build_model(self):
        self._add_input_placeholder()
        self._add_encoder_decoder_ops()
        self._add_forward_ops()
        self._add_representation_training_ops()
        self._add_e2e_training_ops()
        # self._add_encrypt_grad_update_ops()

    def _add_input_placeholder(self):
        self.X_in = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, self.input_dim))
        # self.X_in = self._gaussian_additive_noise(self.X_in_1, 0.01)

        # self.X_in_2 = tf.placeholder(dtype=tf.float64, shape=(None, self.input_dim))
        # self.X_in_2 = self._gaussian_additive_noise(self.X_in_2, 0.01)

    # def _gaussian_additive_noise(self, X_in, std):
    #     return X_in + tf.random_normal(shape=tf.shape(X_in), dtype=tf.float64, mean=0.0, stddev=std)

    def _add_encoder_decoder_ops(self):
        self.encoder_vars_scope = self.id + "_encoder_vars"
        with tf.compat.v1.variable_scope(self.encoder_vars_scope):
            self.Wh = tf.compat.v1.get_variable("weights", initializer=self.Wh_initializer, dtype=tf.float64)
            self.bh = tf.compat.v1.get_variable("bias", initializer=self.bh_initializer, dtype=tf.float64)

        self.decoder_vars_scope = self.id + "_decoder_vars"
        with tf.compat.v1.variable_scope(self.decoder_vars_scope):
            self.Wo = tf.compat.v1.get_variable("weights", initializer=self.Wo_initializer, dtype=tf.float64)
            self.bo = tf.compat.v1.get_variable("bias", initializer=self.bo_initializer, dtype=tf.float64)

    def _add_forward_ops(self):
        self.Z = self._forward_hidden(self.X_in)
        self.logits = self._forward_logits(self.X_in)
        self.X_hat = self._forward_output(self.X_in)

    def _add_representation_training_ops(self):
        vars_to_train = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_vars_scope)
        self.init_grad = tf.compat.v1.placeholder(tf.float64, shape=(None, self.hidden_dim))
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.Z, var_list=vars_to_train, grad_loss=self.init_grad)

    def _add_e2e_training_ops(self):
        encoder_vars_to_train = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_vars_scope)
        decoder_vars_to_train = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.decoder_vars_scope)
        vars_to_train = encoder_vars_to_train + decoder_vars_to_train
        l2_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float64)) for v in vars_to_train])
        self.loss_ = tf.cast(tf.compat.v1.losses.mean_squared_error(predictions=self.logits, labels=self.X_in, reduction="none"), tf.float64)
        self.loss = tf.reduce_mean(input_tensor=self.loss_) + 0.01 * l2_loss
        self.e2e_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _forward_hidden(self, X):
        # return tf.sigmoid(tf.matmul(X, self.Wh) + self.bh)
        return tf.nn.tanh(tf.matmul(X, self.Wh) + self.bh)

    def _forward_logits(self, X):
        Z = self._forward_hidden(X)
        return tf.matmul(Z, self.Wo) + self.bo

    def _forward_output(self, X):
        return tf.sigmoid(self._forward_logits(X))

    def compute_loss(self, X):
        return self.sess.run(self.loss_, feed_dict={self.X_in: X})

    def transform(self, X):
        return self.sess.run(self.Z, feed_dict={self.X_in: X})

    def get_representation(self):
        return self.Z

    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X_in: X})

    def get_encode_dim(self):
        return self.hidden_dim

    def get_model_parameters(self):
        _Wh = self.sess.run(self.Wh)
        _Wo = self.sess.run(self.Wo)
        _bh = self.sess.run(self.bh)
        _bo = self.sess.run(self.bo)

        model_meta = {"learning_rate": self.lr,
                      "input_dim": self.input_dim,
                      "hidden_dim": self.hidden_dim}
        return {"Wh": _Wh, "bh": _bh, "Wo": _Wo, "bo": _bo, "model_meta": model_meta}

    def restore_model(self, model_parameters):
        self.Wh_initializer = model_parameters["Wh"]
        self.bh_initializer = model_parameters["bh"]
        self.Wo_initializer = model_parameters["Wo"]
        self.bo_initializer = model_parameters["bo"]
        model_meta = model_parameters["model_meta"]

        self.lr = model_meta["learning_rate"]
        self.input_dim = model_meta["input_dim"]
        self.hidden_dim = model_meta["hidden_dim"]
        self._build_model()

    def fit(self, X, batch_size=32, epoch=1, show_fig=False):
        N, D = X.shape
        n_batches = N // batch_size
        costs = []
        for ep in range(epoch):
            for i in range(n_batches + 1):
                batch = X[i * batch_size: i * batch_size + batch_size]
                _, c = self.sess.run([self.e2e_train_op, self.loss], feed_dict={self.X_in: batch})
                # if i % 5 == 0:
                #     print(ep, i, "/", n_batches, "cost:", c)
                costs.append(c)

            if len(costs) > 1 and costs[-1] < 0.02 and abs(costs[-1] - costs[len(costs) - 2]) < 0.01:
                # print(costs)
                # print("converged")
                break

        if show_fig:
            plt.plot(costs)
            plt.show()

