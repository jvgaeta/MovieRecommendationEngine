import numpy as np
import math
from operator import itemgetter
import scipy.spatial.distance as sp

# want to predict the rating they will give to movies they have not seen
# so we need to only predict on movies that they have not seen from the test
# set vs previously we were dropping the rating that they had given and trying
# to figure out what it was

def get_naive_rating(X_train):
	return round((sum(X_train[:, -1]) / float(len(X_train))))

# return the k nearest neighbors
def get_k_nearest_neighbors(X_train, x_test, k, metric='euclidean'):
	distances = []
	# compute all of the distances
	y_train = X_train[:,-1]
	X_train = X_train[:, 0:-1]
	if metric == 'euclidean':
		difference_matrix = np.tile(x_test, (X_train.shape[0], 1)) - X_train
		sq_difference_matrix = difference_matrix**2
		sq_distances = sq_difference_matrix.sum(axis=1)
		distances = sq_distances**0.5
	elif metric == 'manhattan':
		difference_matrix = np.absolute(np.tile(x_test, (Xtrain.shape[0], 1)) - X_train)
		distances = difference_matrix.sum(axis=1)
	elif metric == 'lmax':
		distances = np.absolute(np.tile(x_test, (Xtrain.shape[0], 1)) - X_train)
	else:
		print('Invalid metric.')
		return
	# now get the k nearest neighbors
	return distances.argsort()

# here we actually do the classifying
def make_classification(knn, labels, k):
	votes = {}
	if k > len(knn):
		k = len(knn)
	for i in range(0, k):
		voting_label = labels[knn[i]]
		votes[voting_label] = votes.get(voting_label, 0) + 1
	results = sorted(votes.iteritems(), key=itemgetter(1), reverse=True)
	return results[0][0]

# make predictions for all users given a dataset
# takes the dataset, the userlist, and movie info, and # of desired neighbors
def predict_all_users(dataset, users, movie_info, k=25, distance='euclidean'):
	predictions = []
	naive_rating = get_naive_rating(dataset.get_values())
	features = list(dataset.columns)
	features.remove('user_id')
	features.remove('item_id')
	for i in range(0, len(users)):
		print('Now examining the ' + str(i + 1) +'-th user.')
		user = user_data.iloc[i][features_user].get_values()
		for j in range(0, len(movie_info)):
			watched_movie_list = dataset[dataset['item_id'] == movie_info[['item_id']].iloc[j].get_values()[0]]
			item = movie_info[['item_id']].iloc[j].get_values()[0]
			if user not in watched_movie_list[['user_id']].get_values():
				knn = get_k_nearest_neighbors(dataset[dataset['item_id'] == item][features].get_values(), user, k, metric=distance)
				classification = make_classification(knn, dataset[dataset['item_id'] == item][features].get_values()[:, -1], k)
				predictions.append([user_data.iloc[i].get_values()[0], item, classification])
	return predictions, naive_rating

# given a test set, make predictions, we use this to test accuracy since the predict_all_users
# is not viable to test for accuracy
def predict_from_set(X_train, X_test, movie_info, k=25, distance='euclidean'):
	predictions = []
	naive_rating = get_naive_rating(X_train.get_values())
	features = list(X_train.columns)
	features.remove('user_id')
	features_test = list(X_test.columns)
	features_test.remove('user_id')
	x_train = X_train[features].get_values()
	x_test = X_test[features_test].get_values()
	for i in range(len(x_test)):
		knn = get_k_nearest_neighbors(x_train, x_test[i], k)
		classification = make_classification(knn, x_train[:, -1], k)
		predictions.append(classification)
	return predictions, naive_rating