import numpy as np
import math
import pandas as pd
import knn as alg
import utilities as utils
import time

u1_base_path = 'ml-100k/u1.base'
u1_test_path = 'ml-100k/u1.test'
u2_base_path = 'ml-100k/u2.base'
u2_test_path = 'ml-100k/u2.test'
u3_base_path = 'ml-100k/u3.base'
u3_test_path = 'ml-100k/u3.test'
u4_base_path = 'ml-100k/u4.base'
u4_test_path = 'ml-100k/u4.test'
u5_base_path = 'ml-100k/u5.base'
u5_test_path = 'ml-100k/u5.test'

# training sets
u1_base = pd.read_csv(u1_base_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u1_base = u1_base.drop(['timestamp'], axis=1)
u2_base = pd.read_csv(u2_base_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u2_base = u2_base.drop(['timestamp'], axis=1)
u3_base = pd.read_csv(u3_base_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u3_base = u3_base.drop(['timestamp'], axis=1)
u4_base = pd.read_csv(u4_base_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u4_base = u4_base.drop(['timestamp'], axis=1)
u5_base = pd.read_csv(u5_base_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u5_base = u5_base.drop(['timestamp'], axis=1)

# test sets
u1_test = pd.read_csv(u1_test_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u1_test = u1_test.drop(['timestamp'], axis=1)
u2_test = pd.read_csv(u2_test_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u2_test = u2_test.drop(['timestamp'], axis=1)
u3_test = pd.read_csv(u3_test_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u3_test = u3_test.drop(['timestamp'], axis=1)
u4_test = pd.read_csv(u4_test_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u4_test = u4_test.drop(['timestamp'], axis=1)
u5_test = pd.read_csv(u5_test_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u5_test = u5_test.drop(['timestamp'], axis=1)

# user data set
file_path_user_info = 'ml-100k/u.user'
user_info = pd.read_csv(file_path_user_info, sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip'])

# movie data set
file_path_movie = 'ml-100k/u.item'
headers = ['item_id', 'title', 'release_date', 'video_rd', 'url', 'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 
		   'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
		   'Thriller', 'War', 'Western']
movie_info = pd.read_csv(file_path_movie, sep='|', header=None, names=headers)
movie_info = movie_info.drop(['title', 'release_date', 'video_rd', 'url'], axis=1)

# construct the data for each user
user_job = pd.get_dummies(user_info.occupation)
max_age = max(user_info['age'])
user_info['age'] = user_info.age.map(lambda x : x / float(max_age))
user_gender = user_info.gender.map(lambda x : 1 if x == 'M' else 0)
user_data = pd.concat([user_info['user_id'], user_info['age'], user_gender, user_job], axis=1)

dataset1 = pd.merge(user_data, u1_base, on='user_id')
dataset2 = pd.merge(user_data, u2_base, on='user_id')
dataset3 = pd.merge(user_data, u3_base, on='user_id')
dataset4 = pd.merge(user_data, u4_base, on='user_id')
dataset5 = pd.merge(user_data, u5_base, on='user_id')
datasets = [dataset1, dataset2, dataset3, dataset4, dataset5]

u1_test = pd.merge(user_data, u1_test, on='user_id')
u2_test = pd.merge(user_data, u2_test, on='user_id')
u3_test = pd.merge(user_data, u3_test, on='user_id')
u4_test = pd.merge(user_data, u4_test, on='user_id')
u5_test = pd.merge(user_data, u5_test, on='user_id')

# now drop the rating from the test set
u1_test_ans = u1_test.rating
u2_test_ans = u2_test.rating
u3_test_ans = u3_test.rating
u4_test_ans = u4_test.rating
u5_test_ans = u5_test.rating
u1_test = u1_test.drop(['rating'], axis=1)
u2_test = u2_test.drop(['rating'], axis=1)
u3_test = u3_test.drop(['rating'], axis=1)
u4_test = u4_test.drop(['rating'], axis=1)
u5_test = u5_test.drop(['rating'], axis=1)

validation_sets = [u1_test, u2_test, u3_test, u4_test, u5_test]
validation_answers = [u1_test_ans, u2_test_ans, u3_test_ans, u4_test_ans, u5_test_ans]

print('Data has been loaded, now running algorithm.')

k = 161
n_folds = 5
accuracy_total = 0
naive_accuracy_total = 0
t0 = time.time()
for i in range(0, n_folds):
	print('Making predictions for the ' + str(i+1) + '-th fold')
	predictions, naive_rating = alg.predict_from_set(X_train=datasets[i], X_test=validation_sets[i], movie_info=movie_info, k=k, distance='euclidean')
	print('Computing Accuracy for predictions')
	accuracy = utils.getaccuracy(validation_answers[i], predictions)
	naive_accuracy = utils.get_mean_accuracy(validation_answers[i], naive_rating)
	print('Accuracy for ' + str(i + 1) + '-th fold: ' + str(accuracy))
	print('Naive Accuracy for ' + str(i + 1) + '-th fold: ' + str(naive_accuracy))
	accuracy_total += accuracy
	naive_accuracy_total += naive_accuracy
t1 = time.time()

# this will make a prediction for every user for every movie, dataset can be any specified set
# that is of the form of the datasets in the 'datasets' array above
# need this to run on all users
# users = user_data[['user_id']].get_values()
# predictions, naive_rating = alg.predict_all_users(dataset, users, movie_info, k, distance='euclidean')

# write our predictions to a csv file
# print('writing results to csv...')
# df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'rating'])
# df.to_csv('predictions.csv', index=False)
# now we need to evaluate our metric
print("Total accuracy of our predictions: " + str(accuracy_total/n_folds))
print("Accuracy of the mean rating: " + str(naive_accuracy_total/n_folds))
print('Time for KNN and Accuracy Measure: ' + str(t1 - t0) + ' seconds')











