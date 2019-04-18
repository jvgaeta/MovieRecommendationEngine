import numpy as np
import math
from operator import itemgetter

# want to predict the rating they will give to movies they have not seen
# so we need to only predict on movies that they have not seen from the test
# set vs previously we were dropping the rating that they had given and trying
# to figure out what it was

# compute the distance
def euclidean_distance(x1, x2):
	return np.linalg.norm(x1 - x2)

# return the k nearest neighbors
def get_k_nearest_neighbors(X_train, y_train, x_test, k):
	euclidean_distances = []
	# compute all of the distances
	for i in range(len(X_train)):
		euclidean_distances.append((X_train[i], y_train[i], euclidean_distance(x_test, X_train[i])))
	euclidean_distances.sort(key=itemgetter(2))
	knn = []
	# now get the k nearest neighbors
	for i in range(k):
		knn.append(euclidean_distances[i][0:2])
	return knn

# here we actually do the classifying
def make_classification(knn):
	votes = {}
	for i in range(len(knn)):
		result = knn[i][1]
		if result in votes:
			votes[result] += 1
		else:
			votes[result] = 1
	results = sorted(votes.iteritems(), key=itemgetter(1), reverse=True)
	return results[0][0]

# compute the accuracy of our predictions
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0