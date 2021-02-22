import pickle

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

from autoencoder_bk import Autoencoder


def calculate_multivariate_normal(samples):
    cov_mattrix = np.cov(samples.T)
    samples_mean = np.mean(samples, axis=0)
    # print(cov_mattrix.shape, samples_mean.shape)
    return multivariate_normal(mean=samples_mean, cov=cov_mattrix, allow_singular=True)


def train_autoencoder(partyB_overlap_X, partA_overlap_X, hidden_dim=8):

    tf.compat.v1.reset_default_graph()
    autoencoder = Autoencoder(0)
    autoencoder.build(input_dim=partyB_overlap_X.shape[1], hidden_dim=hidden_dim, learning_rate=0.01)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        autoencoder.set_session(session)
        session.run(init_op)
        autoencoder.fit(partyB_overlap_X, batch_size=512, epoch=1000, show_fig=False)

        model_parameters = autoencoder.get_model_parameters()
        # print("model_parameters", model_parameters)

        with open('./autoencoder_model_parameters', 'wb') as outfile:
            # json.dump(model_parameters, outfile)
            pickle.dump(model_parameters, outfile)

        output = autoencoder.compute_loss(partA_overlap_X)
        # print("output", output.shape)
        weights = np.mean(output, axis=1)

    # print("weights", weights, weights.shape)
    return 1 / weights, model_parameters


def calculate_weights_based_on_autoencoder_loss(target_overlap_X, source_overlap_X, source_samples_X, source_samples_Y, **kwargs):
    hidden_dim = kwargs["hidden_dim"]
    exclude_ratio = kwargs["exclude_ratio"]
    # print("hidden_dim", hidden_dim)
    # print("exclude_ratio", exclude_ratio)

    weights, _ = train_autoencoder(target_overlap_X, source_overlap_X, hidden_dim)

    sorted_weights = np.sort(weights)
    index = int(sorted_weights.shape[0] * exclude_ratio)
    threshold = sorted_weights[index]
    # print("weights", weights)
    # print("sorted_weights", sorted_weights)
    # print("index", index)
    # print("threshold", threshold)
    #
    # print(len(weights), type(weights))
    # print(len(source_samples_X), type(source_samples_X))
    # print(len(source_samples_Y), type(source_samples_Y))

    filtered_source_samples_x = []
    filtered_source_samples_y = []
    r_weights = []
    for weight, sample_x, sample_y in zip(weights, source_samples_X, source_samples_Y):
        if weight > threshold:
            filtered_source_samples_x.append(sample_x)
            filtered_source_samples_y.append(sample_y)
            r_weights.append(weight)
    r_weights = np.array(r_weights)
    return r_weights / np.sum(r_weights), np.array(filtered_source_samples_x), np.array(filtered_source_samples_y)


def calculate_weights_based_on_pdf_ratio(target_overlap_X, source_overlap_X, source_samples_X, source_samples_Y):

    target_multivariate_normal = calculate_multivariate_normal(target_overlap_X)
    source_multivariate_normal = calculate_multivariate_normal(source_overlap_X)

    filtered_source_samples_x = []
    filtered_source_samples_y = []
    weights = []
    for sample, sample_x, sample_y in zip(source_overlap_X, source_samples_X, source_samples_Y):
        ratio = target_multivariate_normal.logpdf(sample) / source_multivariate_normal.logpdf(sample)
        if ratio > 1:
            filtered_source_samples_x.append(sample_x)
            filtered_source_samples_y.append(sample_y)
            weights.append(np.sqrt(ratio))
    weights = np.array(weights)
    return weights / np.sum(weights), np.array(filtered_source_samples_x), np.array(filtered_source_samples_y)
