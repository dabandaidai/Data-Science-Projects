from utils import sigmoid
import numpy as np
import math


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    w = weights[:-1]
    b = weights[-1]
    n = data.shape[0]
    y = np.empty(0)
    for i in range(n):
        wt = np.transpose(w)
        xt = np.transpose(data[i])
        xt = xt.reshape(len(w),1)
        wtx = np.matmul(wt, xt) + b
        prediction = sigmoid(wtx)
        y = np.append(y, prediction)
    y = y.reshape(n,1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """

    count = 0
    n = len(y)
    total = 0

    for i in range(n):
        t = targets[i]
        y_i = y[i].item()
        total += (-1) * t * math.log(y_i) - (1-t)*math.log(1-y_i)
        if round(y_i) == t:
            count += 1

    ce = total / n
    frac_correct = count / n
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    f, frac_correct = evaluate(targets, y)
    D = len(weights) - 1
    n = len(targets)
    df = np.empty(0)
    ytt = np.transpose(y - targets)
    for j in range(D):
        xj = data[:, [j]]
        new = np.matmul(ytt, xj)
        df = np.append(df, new)
    dzdb = np.ones((n,1))
    djdb = np.matmul(ytt, dzdb)
    item = djdb.item()
    df = np.append(df, item)
    df = df.reshape(D+1, 1)
    df = df / n
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
