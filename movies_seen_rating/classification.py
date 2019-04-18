import numpy as np
import math
import pandas as pd
from sklearn.cross_validation import train_test_split
import algorithm as alg
import time
from sklearn import neighbors
from random import randint

# read in the data, this contains the users and their rating
file_path = 'ml-100k/u.data'
dataset = pd.read_csv(file_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
dataset = dataset.drop(['timestamp'], axis=1)

# read in the user info
file_path_user_info = 'ml-100k/u.user'
user_info = pd.read_csv(file_path_user_info, sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip'])

file_path_movie = 'ml-100k/u.item'
headers = ['item_id', 'title', 'release_date', 'video_rd', 'url', 'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 
		   'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
		   'Thriller', 'War', 'Western']
movie_info = pd.read_csv(file_path_movie, sep='|', header=None, names=headers)
movie_info = movie_info.drop(['title', 'release_date', 'video_rd', 'url'], axis=1)

# now let's start building our data
user_job = pd.get_dummies(user_info.occupation)
# normalize the age so that it isn't given too much weight in computing the distance
max_age = max(user_info['age'])
user_info['age'] = user_info.age.map(lambda x : x / float(max_age))
# map males to 1 and females to zero
user_gender = user_info.gender.map(lambda x : 1 if x == 'M' else 0)
# drop the zip for now
user_data = pd.concat([user_info['user_id'], user_info['age'], user_gender, user_job], axis=1)

# let's join on the user id here and we have our dataset
full_dataset = pd.merge(user_data, dataset, on='user_id')
full_dataset = pd.merge(full_dataset, movie_info, on='item_id')

# now let's split into train and test
train_set, test_set = train_test_split(full_dataset, train_size=0.03, test_size=0.01)

# now drop the rating from the test set
y_test = test_set.rating
X_test = test_set.drop(['rating'], axis=1)

y_train = train_set.rating
X_train = train_set.drop(['rating'], axis=1)
predictions = []
user_ids = []

# build the feature list, don't want to use user id in computing euclidean dist
features = list(X_train.columns)
features.remove('user_id')

item_index = X_test.columns.get_loc('item_id')

k = 25
t0 = time.time()
for i in range(len(X_test)):
	knn = alg.get_k_nearest_neighbors(X_train[features].get_values(), y_train.get_values(), X_test[features].iloc[i].get_values(), k)
	classification = alg.make_classification(knn)
	predictions.append(classification)
	# this will give us all of our predictions if it is uncommented
	# user_ids.append(("user_id: " + str(X_test.get_values()[i, 0]), "item_id: " + str(X_test.get_values()[i, item_index]), "rating: " + str(classification)))
accuracy = alg.getaccuracy(y_test.get_values(), predictions)
t1 = time.time()
# uncomment this as well to print actual predictions
# print(user_ids)
print('Time for KNN : ' + str(t1 - t0) + ' seconds')
print("Accuracy: " + str(accuracy))

t2 = time.time()
clf = neighbors.KNeighborsClassifier(25, algorithm='brute')
clf.fit(X_train[features], y_train)
sk_predictions = clf.predict(X_test[features])
sk_accuracy = alg.getaccuracy(y_test.get_values(), sk_predictions)
t3 = time.time()
print('Time for KNN-sklearn : ' + str(t3 - t2) + ' seconds')
print("Accuracy: " + str(sk_accuracy))
random_predictions = [randint(1 , 5) for i in range(len(y_test))]
random_accuracy = alg.getaccuracy(random_predictions, predictions)
print("Random Accuracy: " + str(random_accuracy))










