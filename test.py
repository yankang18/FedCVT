import numpy as np
# import tensorflow as tf
# from instance_weighting_based_transfer import train_autoencoder
from scipy.stats import multivariate_normal
#
# from autoencoder import Autoencoder
#
#
# def train_autoencoder(partyB_overlap_X, partA_overlap_X, hidden_dim=8):
#     print("partyB_overlap_X shape", partyB_overlap_X.shape)
#     print("partA_overlap_X shape", partA_overlap_X.shape)
#
#     tf.reset_default_graph()
#     autoencoder = Autoencoder(0)
#     autoencoder.build(input_dim=partyB_overlap_X.shape[1], hidden_dim_list=[40, 20], learning_rate=0.01)
#     init_op = tf.global_variables_initializer()
#     with tf.Session() as session:
#         autoencoder.set_session(session)
#         session.run(init_op)
#         autoencoder.fit(partyB_overlap_X, batch_size=512, epoch=200, show_fig=True)
#
#         # model_parameters = autoencoder.get_model_parameters()
#         # # print("model_parameters", model_parameters)
#         # with open('./autoencoder_model_parameters', 'wb') as outfile:
#         #     # json.dump(model_parameters, outfile)
#         #     pickle.dump(model_parameters, outfile)
#
#         output = autoencoder.compute_loss(partA_overlap_X)
#         print("output", output, output.shape)
#         weights = np.mean(output, axis=1)
#
#     # print("weights", weights, weights.shape)
#     return 1 / weights
#
#
# def calculate_multivariate_normal(samples):
#     cov_mattrix = np.cov(samples.T)
#     samples_mean = np.mean(samples, axis=0)
#     return multivariate_normal(mean=samples_mean, cov=cov_mattrix)
#
#
# def calculate_weights(target_samples, source_samples):
#     targe_multivariate_normal = calculate_multivariate_normal(target_samples)
#     source_multivariate_normal = calculate_multivariate_normal(source_samples)
#
#     weights = []
#     for sample in source_samples:
#         ratio = targe_multivariate_normal.logpdf(sample) / source_multivariate_normal.logpdf(sample)
#         weights.append(np.sqrt(ratio))
#     weights = np.array(weights)
#     return weights / np.sum(weights)


# def cosine_sim(x1, x2, name='Cosine_loss'):
#     with tf.name_scope(name):
#         # x1_val = tf.sqrt(tf.reduce_sum(tf.matmul(x1, tf.transpose(x1)), axis=1))
#         # x2_val = tf.sqrt(tf.reduce_sum(tf.matmul(x2, tf.transpose(x2)), axis=1))
#         x1_val = tf.sqrt(tf.reduce_sum(tf.multiply(x1, x1), axis=1))
#         x2_val = tf.sqrt(tf.reduce_sum(tf.multiply(x2, x2), axis=1))
#         denom = tf.multiply(x1_val, x2_val)
#         # print(denom.shape)
#         num = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
#         # print(num.shape)
#         return tf.div(num, denom)


if __name__ == "__main__":

    # v = np.log(-9.87616100e-11)
    # print(v)
    # v = np.log(0.0000000e+00)
    # print(v)
    # partyA = np.loadtxt("./datasets/20190224170051_partyA_3084_800.datasets", delimiter=",")
    # partyB = np.loadtxt("./datasets/20190224170051_partyB_1517_1013.datasets", delimiter=",")
    #
    # np.random.shuffle(partyA)
    # np.random.shuffle(partyB)
    #
    # weights, _ = train_autoencoder(partyA, partyB)
    #
    # weights = np.sort(weights)
    # index = int(weights.shape[0] * 0.1)
    # print(weights)
    # print(weights[index])

    # print("partyA.shape", partyA.shape)
    # print("partyB.shape", partyB.shape)
    # partyB_multivariate_normal = calculate_multivariate_normal(partyB)
    # partyA_multivariate_normal = calculate_multivariate_normal(partyA)
    #
    # count0 = 0
    # count1 = 0
    # count5 = 0
    # count10 = 0
    # start = time.time()
    # for sample in partyA:
    #     # sample = partyA[10]
    #     # print("pdf", partyB_multivariate_normal.pdf(sample))
    #     # print("cdf", partyB_multivariate_normal.cdf(sample))
    #     ratio = partyB_multivariate_normal.logpdf(sample) / partyA_multivariate_normal.logpdf(sample)
    #     # ratio_2 = partyB_multivariate_normal.logcdf(sample) / partyA_multivariate_normal.logcdf(sample)
    #     if ratio < 1:
    #         count0 += 1
    #     elif ratio > 10:
    #         count10 += 1
    #     elif ratio > 5:
    #         count5 += 1
    #     elif ratio >= 1:
    #         count1 += 1
    # end = time.time()
    #
    # print(end - start, count0, count1, count5, count10)

    # tf.reset_default_graph()
    # a = tf.convert_to_tensor([[10, 20, 30, 40],
    #                           [10, 20, 30, 40]], dtype=tf.float64)
    # a_1 = tf.convert_to_tensor([10, 20, 30, 40], dtype=tf.float64)
    # b = tf.convert_to_tensor([[10, 20, 30, 40],
    #                           [20, 20, 30, 40]], dtype=tf.float64)
    # #
    # # a = tf.convert_to_tensor([[10, 20, 30, 40]], dtype=tf.float64)
    # # b = tf.convert_to_tensor([[10, 20, 30, 40]], dtype=tf.float64)
    #
    # # c = tf.convert_to_tensor([10, 20, 30, 40])
    # # d = tf.convert_to_tensor([10, 20, 30, 40])
    # sim = tf_dot_sim(tf.expand_dims(a_1, axis=0), b)
    # res = tf.tile(tf.expand_dims(a_1, axis=0), (a.shape[0], 1))
    # # a_b_1, _ = tf.metrics.mean_cosine_distance(a, b, dim=0)
    # # a_b_2 = tf.losses.cosine_distance(a, b, axis=0, reduction=tf.losses.Reduction.MEAN)
    #
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     a_b_1_val = sess.run(sim)
    #     out = sess.run(res)
    #     # a_b_1_va2 = sess.run(a_b_2)
    #
    #     print("out", out)
    #     print("a_b_1_val", a_b_1_val)
    #     # print("a_b_1_va2", a_b_1_va2)
    #
    #     # d = tf.nn.top_k(a, 2)
    #     # values, indices = sess.run(b)
    #     # print(values)
    #     # print(indices)

    auc = np.array([0.811, 0.820, 0.819, 0.813, 0.817])
    auc_mean = np.mean(auc)
    auc_std = np.std(auc)

    ks = np.array([0.472, 0.481, 0.486, 0.481, 0.485])
    ks_mean = np.mean(ks)
    ks_std = np.std(ks)

    print(auc_mean)
    print(auc_std)
    print(ks_mean)
    print(ks_std)

    auc = np.array([0.911, 0.902, 0.903, 0.906, 0.900])
    auc_mean = np.mean(auc)
    auc_std = np.std(auc)

    ks = np.array([0.698, 0.683, 0.692, 0.689, 0.699])
    ks_mean = np.mean(ks)
    ks_std = np.std(ks)

    print(auc_mean)
    print(auc_std)
    print(ks_mean)
    print(ks_std)

