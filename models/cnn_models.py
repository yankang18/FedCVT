import tensorflow as tf

from models.autoencoder import FeatureExtractor


class CNNFeatureExtractor(FeatureExtractor):

    def __init__(self, an_id):
        super(CNNFeatureExtractor, self).__init__()
        self.id = str(an_id)
        self._sess = None
        self._input_shape = None
        self.learning_rate = None
        self._num_classes = None

    def set_session(self, sess):
        self._sess = sess

    def get_session(self):
        return self._sess

    def build(self, input_shape, num_classes=10, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._add_input_placeholder()
        self.set_filters()
        self._add_forward_ops()
        self._add_loss_op()

    def _add_input_placeholder(self):
        input_dim = len(self._input_shape)
        print("input dim : {0}".format(input_dim))
        if input_dim == 3:
            self.X_all_in = tf.compat.v1.placeholder(dtype=tf.float32,
                                                     shape=(None,
                                                            self._input_shape[0],
                                                            self._input_shape[1],
                                                            self._input_shape[2]),
                                                     name="X_input_all")
            self.X_overlap_in = tf.compat.v1.placeholder(dtype=tf.float32,
                                                         shape=(None,
                                                                self._input_shape[0],
                                                                self._input_shape[1],
                                                                self._input_shape[2]),
                                                         name="X_input_overlap")
            self.X_non_overlap_in = tf.compat.v1.placeholder(dtype=tf.float32,
                                                             shape=(None,
                                                                    self._input_shape[0],
                                                                    self._input_shape[1],
                                                                    self._input_shape[2]),
                                                             name="X_input_non_overlap")
        else:
            raise Exception("input dim mush be 3, but is {0}".format(input_dim))

        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
        self.y = tf.compat.v1.placeholder(tf.float32, shape=(None, self._num_classes), name='output_y')

    def get_all_samples(self):
        return self.X_all_in

    def get_overlap_samples(self):
        return self.X_overlap_in

    def get_non_overlap_samples(self):
        return self.X_non_overlap_in

    def get_is_train(self):
        return self.is_train

    def get_keep_probability(self):
        return self.keep_prob

    def _add_forward_ops(self):
        self.all_hidden_reprs = self.forward_hidden(self.X_all_in)
        self.overlap_hidden_reprs = self.forward_hidden(self.X_overlap_in)
        self.non_overlap_hidden_reprs = self.forward_hidden(self.X_non_overlap_in)

    def _add_loss_op(self):
        representation = self.forward_hidden(self.X_all_in)
        logits = tf.compat.v1.layers.dense(inputs=representation, units=self._num_classes, activation=None)
        # Name logits Tensor, so that can be loaded from disk after training
        # model = tf.identity(logits, name='logits')

        # Loss and Optimizer
        self.cost = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(self.y)))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=self.y, axis=1))
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32), name='accuracy')

    def set_filters(self):
        pass

    def forward_hidden(self, x):
        pass

    def get_all_hidden_reprs(self):
        return self.all_hidden_reprs

    def get_overlap_hidden_reprs(self):
        return self.overlap_hidden_reprs

    def get_non_overlap_hidden_reprs(self):
        return self.non_overlap_hidden_reprs


class ClientCNNFeatureExtractor(CNNFeatureExtractor):

    def __init__(self, an_id):
        super(ClientCNNFeatureExtractor, self).__init__(an_id)
        self.conv1_filter = None
        self.conv2_filter = None
        self.conv3_filter = None
        self.conv_1x1_filter_1 = None
        self.conv_1x1_filter_2 = None

    def set_filters(self):
        self.conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 1, 16], mean=0, stddev=0.1))
        self.conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 16], mean=0, stddev=0.1))
        self.conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 32], mean=0, stddev=0.1))
        self.conv_1x1_filter_1 = tf.Variable(tf.random.truncated_normal(shape=[1, 1, 16, 16], mean=0, stddev=0.1))
        self.conv_1x1_filter_2 = tf.Variable(tf.random.truncated_normal(shape=[1, 1, 32, 16], mean=0, stddev=0.1))

    def forward_hidden(self, x):
        # define CNN architecture
        conv1 = tf.nn.conv2d(input=x, filters=self.conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.leaky_relu(conv1)
        conv1 = tf.nn.max_pool2d(input=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = tf.nn.conv2d(input=conv1, filters=self.conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.compat.v1.layers.batch_normalization(conv2)
        conv2 = tf.nn.leaky_relu(conv2)
        conv2 = tf.nn.max_pool2d(input=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        side_conv_1 = tf.nn.conv2d(input=conv2, filters=self.conv_1x1_filter_1, strides=[1, 1, 1, 1], padding='SAME')
        side_flat_1 = tf.compat.v1.layers.flatten(side_conv_1)

        conv3 = tf.nn.conv2d(input=conv2, filters=self.conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.compat.v1.layers.batch_normalization(conv3)
        conv3 = tf.nn.leaky_relu(conv3)

        side_conv_2 = tf.nn.conv2d(input=conv3, filters=self.conv_1x1_filter_2, strides=[1, 1, 1, 1], padding='SAME')
        side_flat_2 = tf.compat.v1.layers.flatten(side_conv_2)

        print("side_flat_1 shape:", side_flat_1.shape)
        print("side_flat_2 shape:", side_flat_2.shape)

        flat = tf.concat(values=[side_flat_1, side_flat_2], axis=1)

        print("flat shape:", flat.shape)

        repr = tf.compat.v1.layers.dense(inputs=flat, num_outputs=64, activation_fn=tf.nn.leaky_relu)
        return repr


class ClientDeeperCNNFeatureExtractor(CNNFeatureExtractor):

    def __init__(self, an_id):
        super(ClientDeeperCNNFeatureExtractor, self).__init__(an_id)

    def forward_hidden(self, x):
        # define filters
        conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 1, 16], mean=0, stddev=0.1))
        conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 16], mean=0, stddev=0.1))
        conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 32], mean=0, stddev=0.1))
        conv4_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1))
        conv5_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.1))
        conv6_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1))

        conv_1x1_filter_1 = tf.Variable(tf.random.truncated_normal(shape=[1, 1, 32, 16], mean=0, stddev=0.1))
        conv_1x1_filter_2 = tf.Variable(tf.random.truncated_normal(shape=[1, 1, 64, 16], mean=0, stddev=0.1))
        conv_1x1_filter_3 = tf.Variable(tf.random.truncated_normal(shape=[1, 1, 64, 16], mean=0, stddev=0.1))

        # define CNN architecture
        conv1 = tf.nn.conv2d(input=x, filters=conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.compat.v1.layers.batch_normalization(conv1)
        conv1 = tf.nn.leaky_relu(conv1)

        conv2 = tf.nn.conv2d(input=conv1, filters=conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.compat.v1.layers.batch_normalization(conv2)
        conv2 = tf.nn.leaky_relu(conv2)

        conv3 = tf.nn.conv2d(input=conv2, filters=conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.compat.v1.layers.batch_normalization(conv3)
        conv3 = tf.nn.leaky_relu(conv3)

        conv4 = tf.nn.conv2d(input=conv3, filters=conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.compat.v1.layers.batch_normalization(conv4)
        conv4 = tf.nn.leaky_relu(conv4)
        conv4_pool = tf.nn.max_pool2d(input=conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        side_conv_1 = tf.nn.conv2d(input=conv4_pool, filters=conv_1x1_filter_1, strides=[1, 1, 1, 1], padding='SAME')
        side_flat_1 = tf.compat.v1.layers.flatten(side_conv_1)

        conv5 = tf.nn.conv2d(input=conv4_pool, filters=conv5_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = tf.compat.v1.layers.batch_normalization(conv5)
        conv5 = tf.nn.leaky_relu(conv5)

        side_conv_2 = tf.nn.conv2d(input=conv5, filters=conv_1x1_filter_2, strides=[1, 1, 1, 1], padding='SAME')
        side_flat_2 = tf.compat.v1.layers.flatten(side_conv_2)

        conv6 = tf.nn.conv2d(input=conv5, filters=conv6_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv6 = tf.compat.v1.layers.batch_normalization(conv6)
        conv6 = tf.nn.leaky_relu(conv6)
        conv6_pool = tf.nn.max_pool2d(input=conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        side_conv_3 = tf.nn.conv2d(input=conv6_pool, filters=conv_1x1_filter_3, strides=[1, 1, 1, 1], padding='SAME')
        side_flat_3 = tf.compat.v1.layers.flatten(side_conv_3)

        print("side_flat_1 shape:", side_flat_1.shape)
        print("side_flat_2 shape:", side_flat_2.shape)
        print("side_flat_3 shape:", side_flat_3.shape)

        flat = tf.concat(values=[side_flat_1, side_flat_2, side_flat_3], axis=1)

        print("flat shape : {0}".format(flat.shape))
        repr = tf.compat.v1.layers.dense(inputs=flat, num_outputs=64, activation_fn=tf.nn.leaky_relu)
        return repr


class ClientMiniVGG(CNNFeatureExtractor):

    def __init__(self, an_id):
        super(ClientMiniVGG, self).__init__(an_id)
        print("[INFO] Using ClientMiniVGG")
        # self.conv1_filter = None
        # self.conv2_filter = None
        # self.conv3_filter = None
        # self.conv4_filter = None
        # self.activation = tf.nn.relu
        self.activation = tf.nn.leaky_relu

    def set_filters(self):
        # self.conv1_filter = tf.truncated_normal(shape=[3, 3, 1, 32], mean=0, stddev=0.1)
        # self.conv2_filter = tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1)
        # self.conv3_filter = tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.1)
        # self.conv4_filter = tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1)

        self.conv1 = tf.compat.v1.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                                kernel_initializer="glorot_normal",
                                                strides=(1, 1), padding="same")
        self.bn1 = tf.compat.v1.layers.BatchNormalization()

        self.conv2 = tf.compat.v1.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                                kernel_initializer="glorot_normal",
                                                strides=(1, 1), padding="same")
        self.bn2 = tf.compat.v1.layers.BatchNormalization()
        self.mp2 = tf.compat.v1.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        self.conv3 = tf.compat.v1.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                                kernel_initializer="glorot_normal",
                                                strides=(1, 1), padding="same")
        self.bn3 = tf.compat.v1.layers.BatchNormalization()
        self.conv4 = tf.compat.v1.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                                kernel_initializer="glorot_normal",
                                                strides=(1, 1), padding="same")
        self.bn4 = tf.compat.v1.layers.BatchNormalization()
        self.mp4 = tf.compat.v1.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        self.flat = tf.compat.v1.layers.Flatten()
        self.dense1 = tf.compat.v1.layers.Dense(units=256, activation=self.activation)
        self.repr = tf.compat.v1.layers.Dense(units=512, activation=self.activation)
        self.bn5 = tf.compat.v1.layers.BatchNormalization()

    def forward_hidden(self, x):
        # define CNN architecture
        x = self.conv1(x)
        x = self.activation(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.mp2(x)
        x = tf.compat.v1.layers.dropout(x, rate=0.25)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.bn4(x)
        x = self.mp4(x)
        x = tf.compat.v1.layers.dropout(x, rate=0.25)

        x = self.flat(x)
        # x = self.dense1(x)
        x = self.repr(x)
        x = self.bn5(x)
        # x = tf.layers.dropout(x, rate=0.25)

        return x


class ClientVGG8(CNNFeatureExtractor):

    def __init__(self, an_id, dense_units=48):
        super(ClientVGG8, self).__init__(an_id)
        print("[INFO] {0} is using ClientVGG8".format(an_id))
        self.activation = tf.nn.leaky_relu
        self.dense_units = dense_units

    def set_filters(self):
        initializer = "glorot_normal"
        # initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.08)
        # first CONV block
        self.conv1 = tf.compat.v1.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                                kernel_initializer=initializer,
                                                strides=(1, 1), padding="same")
        self.bn1 = tf.compat.v1.layers.BatchNormalization()

        self.conv2 = tf.compat.v1.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                                kernel_initializer=initializer,
                                                strides=(1, 1), padding="same")
        self.bn2 = tf.compat.v1.layers.BatchNormalization()
        self.mp2 = tf.compat.v1.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # second CONV block
        self.conv3 = tf.compat.v1.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                                kernel_initializer=initializer,
                                                strides=(1, 1), padding="same")
        self.bn3 = tf.compat.v1.layers.BatchNormalization()
        self.conv4 = tf.compat.v1.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                                kernel_initializer=initializer,
                                                strides=(1, 1), padding="same")
        self.bn4 = tf.compat.v1.layers.BatchNormalization()
        self.mp4 = tf.compat.v1.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # third CONV block
        self.conv5 = tf.compat.v1.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                                kernel_initializer=initializer,
                                                strides=(1, 1), padding="same")
        self.bn5 = tf.compat.v1.layers.BatchNormalization()
        self.conv6 = tf.compat.v1.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                                kernel_initializer=initializer,
                                                strides=(1, 1), padding="same")

        # self.conv6_5_5 = tf.compat.v1.layers.Conv2D(filters=128, kernel_size=(5, 5),
        #                                             kernel_initializer=initializer,
        #                                             strides=(1, 1), padding="same")
        #
        # self.conv6_1_1 = tf.compat.v1.layers.Conv2D(filters=128, kernel_size=(1, 1),
        #                                             kernel_initializer=initializer,
        #                                             strides=(1, 1), padding="same")

        self.bn6 = tf.compat.v1.layers.BatchNormalization()
        # self.bn6_1_1 = tf.compat.v1.layers.BatchNormalization()
        # self.bn6_5_5 = tf.compat.v1.layers.BatchNormalization()

        self.mp6 = tf.compat.v1.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
        self.ave_pool = tf.compat.v1.layers.AveragePooling2D(pool_size=(5, 5), strides=(1, 1), padding="valid")

        self.conv7 = tf.compat.v1.layers.Conv2D(filters=128, kernel_size=(1, 1),
                                                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                    stddev=0.08),
                                                strides=(1, 1), padding="same")

        # Dense layer
        self.flat = tf.compat.v1.layers.Flatten()
        # self.dense1 = tf.layers.Dense(units=256, activation=self.activation)
        self.repr = tf.compat.v1.layers.Dense(units=self.dense_units, activation=self.activation)
        self.bn7 = tf.compat.v1.layers.BatchNormalization()

    def forward_hidden(self, x):
        # define CNN architecture

        is_training = self.get_is_train()

        x = self.conv1(x)
        x = self.activation(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.mp2(x)
        # x = tf.compat.v1.layers.dropout(x, rate=0.25, training=is_training)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.bn4(x)
        x = self.mp4(x)
        # x = tf.compat.v1.layers.dropout(x, rate=0.25, training=is_training)

        x = self.conv5(x)
        x = self.activation(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.activation(x)
        x = self.bn6(x)
        x = self.mp6(x)
        print("x max/ave pooling shape {0}".format(x.shape))

        x = self.flat(x)
        print("x flatten shape {0}".format(x.shape))

        # x = self.dense1(x)
        x = self.repr(x)
        print("x repr shape {0}".format(x.shape))

        # x = self.bn7(x)
        # x = tf.layers.dropout(x, rate=0.25)

        return x


class ClientVGG8B(CNNFeatureExtractor):

    def __init__(self, an_id):
        super(ClientVGG8B, self).__init__(an_id)
        print("[INFO] {0} is using ClientVGG8B".format(an_id))
        self.activation = tf.nn.leaky_relu

    def set_filters(self):
        # first CONV block
        self.conv1 = tf.compat.v1.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                    stddev=0.08),
                                                strides=(1, 1), padding="same")
        self.bn1 = tf.compat.v1.layers.BatchNormalization()

        self.conv2 = tf.compat.v1.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                    stddev=0.08),
                                                strides=(1, 1), padding="same")
        self.bn2 = tf.compat.v1.layers.BatchNormalization()
        self.mp2 = tf.compat.v1.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # second CONV block
        self.conv3 = tf.compat.v1.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                    stddev=0.08),
                                                strides=(1, 1), padding="same")
        self.bn3 = tf.compat.v1.layers.BatchNormalization()
        self.conv4 = tf.compat.v1.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                    stddev=0.08),
                                                strides=(1, 1), padding="same")
        self.bn4 = tf.compat.v1.layers.BatchNormalization()
        self.mp4 = tf.compat.v1.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # third CONV block
        self.conv5 = tf.compat.v1.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                    stddev=0.08),
                                                strides=(1, 1), padding="same")
        self.bn5 = tf.compat.v1.layers.BatchNormalization()
        self.conv6 = tf.compat.v1.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                    stddev=0.08),
                                                strides=(1, 1), padding="same")

        self.bn6 = tf.compat.v1.layers.BatchNormalization()
        self.bn6_1_1 = tf.compat.v1.layers.BatchNormalization()
        self.bn6_5_5 = tf.compat.v1.layers.BatchNormalization()

        self.mp6 = tf.compat.v1.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # Dense layer
        self.flat = tf.compat.v1.layers.Flatten()
        # self.dense1 = tf.layers.Dense(units=256, activation=self.activation)
        self.repr = tf.compat.v1.layers.Dense(units=1024, activation=self.activation)
        self.bn7 = tf.compat.v1.layers.BatchNormalization()

    def forward_hidden(self, x):
        # define CNN architecture

        is_training = self.get_is_train()

        # first CONV block
        x = self.conv1(x)
        x = self.activation(x)
        # x = self.bn1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.mp2(x)
        x = tf.compat.v1.layers.dropout(x, rate=0.25, training=is_training)

        # second CONV block
        x = self.conv3(x)
        x = self.activation(x)
        # x = self.bn3(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.bn4(x)
        x = self.mp4(x)
        x = tf.compat.v1.layers.dropout(x, rate=0.25, training=is_training)

        # third CONV block
        x = self.conv5(x)
        x = self.activation(x)
        # x = self.bn5(x)

        x = self.conv6(x)
        x = self.activation(x)
        x = self.bn6(x)
        x = self.mp6(x)
        print("x max/ave pooling shape {0}".format(x.shape))

        x = self.flat(x)
        print("x flatten shape {0}".format(x.shape))

        # x = self.dense1(x)
        x = self.repr(x)
        print("x repr shape {0}".format(x.shape))

        # x = self.bn7(x)
        # x = tf.layers.dropout(x, rate=0.25)

        return x


class MiniGoogLeNetConvolutionModule(object):

    def __init__(self, parent_id, an_id, out_filters, filter_h, filter_w, stride, padding="same",
                 activation_func=tf.nn.leaky_relu):
        super(MiniGoogLeNetConvolutionModule, self).__init__()
        self.id = str(parent_id) + "_MiniGoogLeNetConvolutionModule_" + str(an_id)
        print("[INFO] Using MiniGoogLeNetConvolutionModule: {0}".format(self.id))
        self.activation = activation_func
        self.conv = tf.compat.v1.layers.Conv2D(filters=out_filters, kernel_size=(filter_h, filter_w),
                                               kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                   stddev=0.08),
                                               strides=stride, padding=padding)
        self.bn = tf.compat.v1.layers.BatchNormalization()

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class MiniGoogLeNetInceptionModule(object):
    def __init__(self, parent_id, an_id, ch1, ch3):
        super(MiniGoogLeNetInceptionModule, self).__init__()
        self.id = str(parent_id) + "_InceptionModule_" + str(an_id)
        print("[INFO] Using InceptionModule: {0}".format(self.id))
        self.conv_1_1 = MiniGoogLeNetConvolutionModule(self.id, 1, ch1, 1, 1, (1, 1))
        self.conv_3_3 = MiniGoogLeNetConvolutionModule(self.id, 2, ch3, 3, 3, (1, 1))

    def __call__(self, x):
        conv_11 = self.conv_1_1(x)
        conv_33 = self.conv_3_3(x)
        concatenated = tf.concat(values=[conv_11, conv_33], axis=-1)
        return concatenated


class MiniGoogLeNetDownsampleModule(object):
    def __init__(self, parent_id, an_id, ch3):
        super(MiniGoogLeNetDownsampleModule, self).__init__()
        self.id = str(parent_id) + "_MiniGoogLeNetDownsampleModule_" + str(an_id)
        print("[INFO] Using MiniGoogLeNetDownsampleModule: {0}".format(self.id))
        self.conv_3_3 = MiniGoogLeNetConvolutionModule(self.id, 1, ch3, 3, 3, (2, 2), "valid")
        self.max_pool = tf.compat.v1.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")

    def __call__(self, x):
        conv_33 = self.conv_3_3(x)
        mp = self.max_pool(x)
        concatenated = tf.concat(values=[conv_33, mp], axis=-1)
        return concatenated


class ClientMiniGoogLeNet(CNNFeatureExtractor):

    def __init__(self, an_id):
        super(ClientMiniGoogLeNet, self).__init__(an_id)
        self.id = "ClientMiniGoogLeNet_" + str(an_id)
        print("[INFO] Using ClientMiniGoogLeNet: {0}".format(self.id))
        self.activation = tf.nn.leaky_relu

    def set_filters(self):
        self.conv_module_11 = MiniGoogLeNetConvolutionModule(parent_id=self.id, an_id=1,
                                                             out_filters=96, filter_h=3, filter_w=3, stride=(1, 1))

        self.inpt_module_21 = MiniGoogLeNetInceptionModule(parent_id=self.id, an_id=1, ch1=32, ch3=32)
        self.inpt_module_22 = MiniGoogLeNetInceptionModule(parent_id=self.id, an_id=2, ch1=32, ch3=48)
        self.down_module_21 = MiniGoogLeNetDownsampleModule(parent_id=self.id, an_id=1, ch3=80)

        self.inpt_module_31 = MiniGoogLeNetInceptionModule(parent_id=self.id, an_id=3, ch1=112, ch3=48)
        self.inpt_module_32 = MiniGoogLeNetInceptionModule(parent_id=self.id, an_id=4, ch1=96, ch3=64)
        self.inpt_module_33 = MiniGoogLeNetInceptionModule(parent_id=self.id, an_id=5, ch1=80, ch3=80)
        self.inpt_module_34 = MiniGoogLeNetInceptionModule(parent_id=self.id, an_id=6, ch1=48, ch3=96)
        self.down_module_35 = MiniGoogLeNetDownsampleModule(parent_id=self.id, an_id=2, ch3=96)

        self.inpt_module_41 = MiniGoogLeNetInceptionModule(parent_id=self.id, an_id=7, ch1=176, ch3=160)
        self.inpt_module_42 = MiniGoogLeNetInceptionModule(parent_id=self.id, an_id=8, ch1=176, ch3=160)

        self.ave_pool = tf.compat.v1.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding="valid")
        self.flat = tf.compat.v1.layers.Flatten()

    def forward_hidden(self, x):
        x = self.conv_module_11(x)

        x = self.inpt_module_21(x)
        x = self.inpt_module_22(x)
        x = self.down_module_21(x)

        x = self.inpt_module_31(x)
        x = self.inpt_module_32(x)
        x = self.inpt_module_33(x)
        x = self.inpt_module_34(x)
        x = self.down_module_35(x)

        x = self.inpt_module_41(x)
        x = self.inpt_module_42(x)

        print("===> output shape of last inception layer {0}".format(x.shape))

        x = self.ave_pool(x)

        print("===> output shape of average pooling layer {0}".format(x.shape))

        x = self.flat(x)

        print("===> output shape of flatten layer {0}".format(x.shape))

        return x


class BenchmarkDeeperHalfImageCNNFeatureExtractor(CNNFeatureExtractor):

    def __init__(self, an_id):
        super(BenchmarkDeeperHalfImageCNNFeatureExtractor, self).__init__(an_id)

    def forward_hidden(self, x):
        # define filters
        conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 1, 16], mean=0, stddev=0.1))
        conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 16], mean=0, stddev=0.1))
        conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 32], mean=0, stddev=0.1))
        conv4_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1))
        conv5_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.1))
        conv6_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1))

        # define CNN architecture
        conv1 = tf.nn.conv2d(input=x, filters=conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.compat.v1.layers.batch_normalization(conv1)
        conv1 = tf.nn.leaky_relu(conv1)

        conv2 = tf.nn.conv2d(input=conv1, filters=conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.compat.v1.layers.batch_normalization(conv2)
        conv2 = tf.nn.leaky_relu(conv2)

        conv3 = tf.nn.conv2d(input=conv2, filters=conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.compat.v1.layers.batch_normalization(conv3)
        conv3 = tf.nn.leaky_relu(conv3)

        conv4 = tf.nn.conv2d(input=conv3, filters=conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.compat.v1.layers.batch_normalization(conv4)
        conv4 = tf.nn.leaky_relu(conv4)
        conv4 = tf.nn.max_pool2d(input=conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv5 = tf.nn.conv2d(input=conv4, filters=conv5_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = tf.compat.v1.layers.batch_normalization(conv5)
        conv5 = tf.nn.leaky_relu(conv5)

        conv6 = tf.nn.conv2d(input=conv5, filters=conv6_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv6 = tf.compat.v1.layers.batch_normalization(conv6)
        conv6 = tf.nn.leaky_relu(conv6)
        conv6 = tf.nn.max_pool2d(input=conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        print("conv6 shape : {0}".format(conv6.shape))
        flat = tf.compat.v1.layers.flatten(conv6)
        print("flat shape : {0}".format(flat.shape))
        repr = tf.compat.v1.layers.dense(inputs=flat, num_outputs=64, activation_fn=tf.nn.leaky_relu)
        return repr


class BenchmarkHalfImageCNNFeatureExtractor(CNNFeatureExtractor):

    def __init__(self, an_id):
        super(BenchmarkHalfImageCNNFeatureExtractor, self).__init__(an_id)

    def forward_hidden(self, x):
        # define filters
        conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 1, 16], mean=0, stddev=0.1))
        conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 16], mean=0, stddev=0.1))
        conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 16, 32], mean=0, stddev=0.1))

        # define CNN architecture
        conv1 = tf.nn.conv2d(input=x, filters=conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.leaky_relu(conv1)
        conv1 = tf.nn.max_pool2d(input=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = tf.nn.conv2d(input=conv1, filters=conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.compat.v1.layers.batch_normalization(conv2)
        conv2 = tf.nn.leaky_relu(conv2)
        conv2 = tf.nn.max_pool2d(input=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = tf.nn.conv2d(input=conv2, filters=conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.compat.v1.layers.batch_normalization(conv3)
        conv3 = tf.nn.leaky_relu(conv3)
        flat = tf.compat.v1.layers.flatten(conv3)
        repr = tf.compat.v1.layers.dense(inputs=flat, num_outputs=64, activation_fn=tf.nn.leaky_relu)
        return repr


class BenchmarkFullImageCNNFeatureExtractor(CNNFeatureExtractor):

    def __init__(self, an_id):
        super(BenchmarkFullImageCNNFeatureExtractor, self).__init__(an_id)

    def forward_hidden(self, x):
        # define filters
        # conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16], mean=0, stddev=0.1))
        # conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 16], mean=0, stddev=0.1))
        # conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], mean=0, stddev=0.1))
        # conv4_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1))
        # conv5_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.1))
        # conv6_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1))

        conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.08))
        conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.08))
        conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.08))
        conv4_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.08))
        conv5_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
        conv6_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 128, 128], mean=0, stddev=0.08))

        # conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.08))
        # conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.08))
        # conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], mean=0, stddev=0.08))
        # conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], mean=0, stddev=0.08))
        # conv5_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 128], mean=0, stddev=0.08))
        # conv6_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 128], mean=0, stddev=0.08))

        # define CNN architecture
        conv1 = tf.nn.conv2d(input=x, filters=conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.compat.v1.layers.batch_normalization(conv1)
        conv1 = tf.nn.leaky_relu(conv1)

        conv2 = tf.nn.conv2d(input=conv1, filters=conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.compat.v1.layers.batch_normalization(conv2)
        conv2 = tf.nn.leaky_relu(conv2)

        conv3 = tf.nn.conv2d(input=conv2, filters=conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.compat.v1.layers.batch_normalization(conv3)
        conv3 = tf.nn.leaky_relu(conv3)

        conv4 = tf.nn.conv2d(input=conv3, filters=conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.compat.v1.layers.batch_normalization(conv4)
        conv4 = tf.nn.leaky_relu(conv4)
        conv4 = tf.nn.max_pool2d(input=conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv5 = tf.nn.conv2d(input=conv4, filters=conv5_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = tf.compat.v1.layers.batch_normalization(conv5)
        conv5 = tf.nn.leaky_relu(conv5)

        conv6 = tf.nn.conv2d(input=conv5, filters=conv6_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv6 = tf.compat.v1.layers.batch_normalization(conv6)
        conv6 = tf.nn.leaky_relu(conv6)
        conv6 = tf.nn.max_pool2d(input=conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        print("conv6 shape : {0}".format(conv6.shape))
        flat = tf.compat.v1.layers.flatten(conv6)
        print("flat shape : {0}".format(flat.shape))
        repr = tf.compat.v1.layers.dense(inputs=flat, num_outputs=128, activation_fn=tf.nn.leaky_relu)
        return repr
