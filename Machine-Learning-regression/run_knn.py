from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


## Calculates test accuracy ##
def knn2(k, train_data, train_labels, test_data):
    dist = l2_distance(test_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    test_labels = train_labels[nearest]

    # Note this only works for binary labels:
    test_labels = (np.mean(test_labels, axis=1) >= 0.5).astype(int)
    test_labels = test_labels.reshape(-1, 1)

    return test_labels

def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    valid1 = knn(1, train_inputs, train_targets, valid_inputs)
    valid3 = knn(3, train_inputs, train_targets, valid_inputs)
    valid5 = knn(5, train_inputs, train_targets, valid_inputs)
    valid7 = knn(7, train_inputs, train_targets, valid_inputs)
    valid9 = knn(9, train_inputs, train_targets, valid_inputs)
    train3 = knn(3, train_inputs, train_targets, test_inputs)
    train5 = knn(5, train_inputs, train_targets, test_inputs)
    train7 = knn(7, train_inputs, train_targets, test_inputs)
    n = len(valid_inputs)
    ntest = len(test_inputs)
    collect3 = 0
    collect5 = 0
    collect7 = 0
    for i in range(ntest):
        if train3[i] == test_targets[i]:
            collect3 += 1
        if train5[i] == test_targets[i]:
            collect5 += 1
        if train7[i] == test_targets[i]:
            collect7 += 1
    test_acc = [collect3/ntest, collect5/ntest, collect7/ntest]
    count1 = 0
    count3 = 0
    count5 = 0
    count7 = 0
    count9 = 0
    for i in range(n):
        if valid1[i] == valid_targets[i]:
            count1 += 1
        if valid3[i] == valid_targets[i]:
            count3 += 1
        if valid5[i] == valid_targets[i]:
            count5 += 1
        if valid7[i] == valid_targets[i]:
            count7 += 1
        if valid9[i] == valid_targets[i]:
            count9 += 1
    accuracy = [count1/n, count3/n, count5/n, count7/n, count9/n]
    plt.scatter(x=[1,3,5,7,9], y=accuracy)
    plt.title("Scatterplot between k and validation prediction accuracy")
    plt.xlabel("k")
    plt.ylabel("validation prediction accuracy")
    plt.show()
    return accuracy, test_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    print(run_knn())
