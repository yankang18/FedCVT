
import numpy

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from autoencoder import Autoencoder
import tensorflow as tf
import numpy as np
import json
import pickle


def local_train(x1, y1, x2, y2):
	# print(x1.shape, x2.shape)
	clf = LogisticRegression(solver="lbfgs").fit(x1, y1)
	return clf.score(x2, y2)

# def local_train_with_clustering(x1, y1, x2, y2):
# 	print(x1.shape, x2.shape)
# 	clf = LogisticRegression().fit(x1, y1)
# 	y2_hat = clf.predict(x2)
# 	print("y2_hat", y2_hat)
# 	kmeans = KMeans(n_clusters=2).fit(x2)
# 	print("lbl:", kmeans.labels_, np.sum(kmeans.labels_))
#
# 	return clf.score(x2, y2)
	
def transfer_feature_A_2_B(partA_X, partB_X, A_start, A_end, B_start, B_end, A_sample_weight=None):
	# poly_reg =PolynomialFeatures(degree=1)
	# x_ploy =poly_reg.fit_transform()
	# print("test_x", x.shape, test_x.shape)
	# print("B_start", B_start)
	# print("A_end", A_end)
	# print("A_end - B_start", A_end - B_start)
	# print("B_start - A_start", B_start-A_start)
	# print("A_end - A_start", A_end-A_start)
	# reg = Ridge().fit(x[:, B_start - A_start:A_end - A_start], x[:, A_start - A_start:B_start - A_start], sample_weight=A_sample_weight)
	reg = DecisionTreeRegressor(max_depth=4).fit(partA_X[:, B_start - A_start:A_end - A_start], partA_X[:, A_start - A_start:B_start - A_start], sample_weight=A_sample_weight)
	output = reg.predict(partB_X[:, 0:(A_end - B_start)])
	return output


def transfer_feature_B_2_A(x, test_x, A_start, A_end, B_start, B_end):
	# poly_reg =PolynomialFeatures(degree=1)
	# x_ploy =poly_reg.fit_transform(x[:, 0:28])
	reg = Ridge(solver="saga").fit(x[:, 0:(A_end-B_start)], x[:, A_end-B_start:(B_end-B_start)])
	print("B2A", test_x.shape)
	# reg = DecisionTreeRegressor(max_depth=4).fit(x[:, 0:(A_end - B_start)], x[:, (A_end - B_start):(B_end - B_start)])
	output = reg.predict(test_x[:, B_start-A_start:])
	return output	


def train_autoencoder(Xtrain, partA_X, hidden_dim=8):
	tf.compat.v1.reset_default_graph()
	autoencoder = Autoencoder(0)
	autoencoder.build(input_dim=Xtrain.shape[1], hidden_dim=hidden_dim, learning_rate=0.01)
	init_op = tf.compat.v1.global_variables_initializer()
	with tf.compat.v1.Session() as session:
		autoencoder.set_session(session)
		session.run(init_op)
		autoencoder.fit(Xtrain, batch_size=512, epoch=1000, show_fig=False)

		model_parameters = autoencoder.get_model_parameters()
		# print("model_parameters", model_parameters)

		with open('./autoencoder_model_parameters', 'wb') as outfile:
			# json.dump(model_parameters, outfile)
			pickle.dump(model_parameters, outfile)

		output = autoencoder.compute_loss(partA_X)
		# print("output", output.shape)
		weights = np.mean(output, axis=1)
		# print("raw weights", weights, weights.shape)
	return weights, model_parameters


def get_weights(partA_X):
	# test whether autoencoder can be restored from stored model parameters

	with open('./autoencoder_model_parameters', 'rb') as file:
		model_parameters = pickle.load(file)

	tf.compat.v1.reset_default_graph()

	autoencoder = Autoencoder(0)
	autoencoder.restore_model(model_parameters)
	init_op = tf.compat.v1.global_variables_initializer()
	with tf.compat.v1.Session() as session:
		autoencoder.set_session(session)
		session.run(init_op)

		output = autoencoder.compute_loss(partA_X)
		weights = np.mean(output, axis=1)
	return weights


def normalize_weights(weights):
	weights = 1 / weights
	weights_1 = weights / np.sum(weights)
	weights_2 = softmax(weights)
	# print("weights_1", weights_1, weights_1.shape)
	# print("weights_2", weights_2, weights_2.shape)
	return weights_1, weights_2


def softmax(x, axis=-1):
	y = np.exp(x - np.max(x, axis, keepdims=True))
	return y / np.sum(y, axis, keepdims=True)


def train(data_X, data_Y, idx, A_start, A_end, B_start, B_end):
	print("A_start, A_end, B_start, B_end:", A_start, A_end, B_start, B_end)

	num_train_data = 4000
	num_part_A_data = 2000
	num_party_A_train_data = 1600
	num_party_B_train_data = 1600

	train_X = data_X[idx[0:num_train_data]]
	train_Y = data_Y[idx[0:num_train_data]]
	test_X = data_X[idx[num_train_data:]]
	test_Y = data_Y[idx[num_train_data:]]

	partA_X = data_X[idx[0:num_part_A_data], A_start:A_end]
	partA_Y = data_Y[idx[0:num_part_A_data]]

	# 6-24
	partB_X = data_X[idx[num_part_A_data:], B_start:B_end]
	partB_Y = data_Y[idx[num_part_A_data:]]

	partA_train_X = partA_X[:num_party_A_train_data]
	partA_train_Y = partA_Y[:num_party_A_train_data]
	partA_test_X = partA_X[num_party_A_train_data:]
	partA_test_Y = partA_Y[num_party_A_train_data:]

	partB_train_X = partB_X[:num_party_B_train_data]
	partB_train_Y = partB_Y[:num_party_B_train_data]
	partB_test_X = partB_X[num_party_B_train_data:]
	partB_test_Y = partB_Y[num_party_B_train_data:]

	# score = local_train(train_X, train_Y, test_X, test_Y)
	# print("all model:", score)

	# score = local_train(partA_train_X, partA_train_Y, partA_test_X, partA_test_Y)
	# print("part A model score:", score)

	# b_model_alone_score = local_train(partB_train_X, partB_train_Y, partB_test_X, partB_test_Y)
	# print("part B model score:", b_model_alone_score, partB_train_X.shape)
	b_model_alone_score = local_train(partA_train_X, partA_train_Y, partB_test_X, partB_test_Y)

	output = transfer_feature_A_2_B(partA_X, partB_X, A_start, A_end, B_start, B_end)
	# print(output.shape)
	new_partB_X = numpy.concatenate((output, partB_X), axis=1)
	# print("new_partB_X shape", new_partB_X.shape)
	new_partB_train_X = new_partB_X[:num_party_B_train_data]
	new_partB_test_X = new_partB_X[num_party_B_train_data:]
	TrPartB_score = local_train(new_partB_train_X, partB_train_Y, new_partB_test_X, partB_test_Y)
	# print("TrPartB model score", TrPartB_score)

	mean_w_score_list = []
	soft_w_score_list = []
	iterations = 20
	for i in range(iterations):

		A_sample_weight, _ = train_autoencoder(partB_X[:, 0:(A_end - B_start)], partA_X[:, B_start - A_start:])
		# A_sample_weight = get_weights(partA_X[:, B_start - A_start:])
		A_sample_weight_mean, A_sample_weight_softmax = normalize_weights(A_sample_weight)

		output_with_sample_weight = transfer_feature_A_2_B(partA_X, partB_X, A_start, A_end, B_start, B_end, A_sample_weight_mean)
		output_with_sample_weight_soft = transfer_feature_A_2_B(partA_X, partB_X, A_start, A_end, B_start, B_end, A_sample_weight_softmax)

		# print(output_with_sample_weight.shape)
		# print(output_with_sample_weight_soft.shape)

		new_partB_X_with_sample_w = numpy.concatenate((output_with_sample_weight, partB_X), axis=1)
		new_partB_X_with_sample_w_soft = numpy.concatenate((output_with_sample_weight_soft, partB_X), axis=1)

		# print("new_partB_X_with_sample_w shape", new_partB_X_with_sample_w.shape)
		# print("new_partB_X_with_sample_w_soft shape", new_partB_X_with_sample_w_soft.shape)

		new_partB_train_X_with_w = new_partB_X_with_sample_w[:num_party_B_train_data]
		new_partB_test_X_with_w = new_partB_X_with_sample_w[num_party_B_train_data:]
		mean_w_score = local_train(new_partB_train_X_with_w, partB_train_Y, new_partB_test_X_with_w, partB_test_Y)
		mean_w_score_list.append(mean_w_score)
		# print("TrPartB model score /w w", mean_w_score)

		new_partB_train_X_with_w_soft = new_partB_X_with_sample_w_soft[:num_party_B_train_data]
		new_partB_test_X_with_w_soft = new_partB_X_with_sample_w_soft[num_party_B_train_data:]
		soft_w_score = local_train(new_partB_train_X_with_w_soft, partB_train_Y, new_partB_test_X_with_w_soft, partB_test_Y)
		soft_w_score_list.append(soft_w_score)
		# print("TrPartB model score /w w_soft", soft_w_score)

	print("-"*20)
	print("part B model score:", b_model_alone_score)
	print("TrPartB model score", TrPartB_score)
	print("TrPartB mean_w_score mean", np.mean(mean_w_score_list))
	print("TrPartB soft_w_score mean", np.mean(soft_w_score_list))
	print("-" * 20)

	# output = transfer_feature_B_2_A(partB_X, partA_X, A_start, A_end, B_start, B_end)
	# new_partA_X = numpy.concatenate((partA_X, output), axis=1)
	#
	# new_partA_train_X = new_partA_X[:num_party_A_train_data]
	# new_partA_test_X = new_partA_X[num_party_A_train_data:]
	# score = local_train(new_partA_train_X, partA_train_Y, new_partA_test_X, partA_test_Y)
	# print("TrPartA model score", score)


if __name__ == "__main__":

	data_X, data_Y = [], []
	with open("./spambase/spambase.datasets") as fin:
		# with open("./credit_g/credit_g_num_data") as fin:
		for line in fin:
			data = line.split(",")
			data_X.append([float(e) for e in data[:-1]])
			data_Y.append(int(data[-1]))
	# print(data_X[0])

	data_X = numpy.array(data_X)
	data_Y = numpy.array(data_Y)

	# print(data_X.shape, data_Y.shape)
	# print("datasets Y", data_Y)

	scaler = StandardScaler()
	data_X = scaler.fit_transform(data_X)
	# print(data_X[0])

	idx = numpy.arange(data_X.shape[0])
	numpy.random.shuffle(idx)

	# A_start = 0
	# A_end = 20
	# B_start = 10
	# B_end = 30

	over_lap_list = [10, 15]
	fore_num_list = [10, 15]
	back_num_list = [10, 15]
	start_index_list = [0, 5]

	# over_lap_list = [10, 15, 20, 25, 30]
	# fore_num_list = [10, 15, 20, 25, 30]
	# back_num_list = [10, 15, 20, 25, 30]
	# start_index_list = [0, 5, 10, 20]

	for over_lap in over_lap_list:
		for fore_num in fore_num_list:
			for back_num in back_num_list:
				for start_index in start_index_list:
					A_start = start_index
					A_end = start_index + fore_num + over_lap
					B_start = start_index + fore_num
					B_end = start_index + fore_num + over_lap + back_num
					if A_end >= 57 or B_end >= 57:
						continue
					train(data_X, data_Y, idx, A_start, A_end, B_start, B_end)








	
