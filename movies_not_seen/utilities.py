from __future__ import division  # floating point division
import math
import numpy as np
from scipy.optimize import fmin,fmin_bfgs
import random

# Some functions you might find useful

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# get the accuracy of those available
def get_acc(validation_sets, predictions):
    correct = 0
    available = 0
    n = 1
    for validation_set in validation_sets:
        print('Validation set number: ' + str(n))
        validation_set_arr = validation_set.get_values()
        current_fold_available = 0
        current_fold_correct = 0
        for i in range(len(predictions)):
            for j in range(len(validation_set_arr)):
                # in the case that a rating does exist
                if (validation_set_arr[j][0] == predictions[i][0]) and (validation_set_arr[j][1] == predictions[i][1]):
                    available += 1
                    current_fold_available += 1
                    if validation_set_arr[j][2] == predictions[i][2]:
                        correct += 1
                        current_fold_correct += 1
        print('Accuracy for the ' + str(n) +'-th fold: ' + str((current_fold_correct/float(current_fold_available)) * 100.0))
        n += 1

    return (correct / float(available)) * 100.0

# accuracy for the mean set over validation sets
def get_mean_accuracy(validation_set, naive_rating):
    correct = 0
    v_set = validation_set.get_values()
    for i in range(len(v_set)):
        if v_set[i] == naive_rating:
            correct += 1
    return (correct / float(len(v_set))) * 100.0


# compute the accuracy of our predictions
def getaccuracy(ytest, predictions):
    correct = 0
    ytest_arr = ytest.get_values()
    for i in range(len(ytest_arr)):
        if ytest_arr[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest_arr))) * 100.0