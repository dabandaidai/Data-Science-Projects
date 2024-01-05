from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def best_hyper(train_inputs, train_targets, valid_inputs, valid_targets):
    N, M = train_inputs.shape
    weights = np.zeros((M + 1, 1))
    alphas = [0.05, 0.03, 0.01]
    regs = [0.1, 0.01]
    iters = [50, 100, 1000]
    stat = {'train_ce': [], 'train_acc': [], 'valid_ce': [], 'valid_acc': [], 'weights': []}
    for alpha in alphas:
        for reg in regs:
            for iter in iters:
                hyperparameters = {
                    "learning_rate": alpha,
                    "weight_regularization": reg,
                    "num_iterations": iter
                }
                for t in range(hyperparameters["num_iterations"]):
                    f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
                    alpha = hyperparameters["learning_rate"]
                    reg = hyperparameters["weight_regularization"]
                    weights = weights - alpha * df - alpha * reg * weights
                ## Final training cross entropy and accuracy
                f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
                stat['train_ce'].append(f.item())
                ce, train_acc = evaluate(train_targets, y)
                stat['train_acc'].append(train_acc)
                fv, dyv, yv = logistic(weights, valid_inputs, valid_targets, hyperparameters)
                stat['valid_ce'].append(fv.item())
                ce, valid_acc = evaluate(valid_targets, yv)
                stat['valid_acc'].append(valid_acc)
                stat['weights'].append(weights)
    maximum = stat['valid_acc'].index(max(stat['valid_acc']))
    return maximum, stat


def cross_entropy(hyperparameters, train_inputs, train_targets, valid_inputs, valid_targets):
    N, M = train_inputs.shape
    weights = np.zeros((M + 1, 1))
    ce1 = []
    ce2 = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        f2, df2, y2 = logistic(weights, valid_inputs, valid_targets, hyperparameters)
        ce1.append(f.item())
        ce2.append(f2.item())
        alpha = hyperparameters["learning_rate"]
        reg = hyperparameters["weight_regularization"]
        weights = weights - alpha * df - alpha * reg * weights
    return ce1, ce2


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    train_inputs2, train_targets2 = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    maximum, stat = best_hyper(train_inputs, train_targets, valid_inputs, valid_targets)
    maximum2, stat2 = best_hyper(train_inputs2, train_targets2, valid_inputs, valid_targets)
    print(maximum, maximum2)
    weights1 = stat['weights'][maximum]
    weights2 = stat2['weights'][maximum2]
    hyperparameters1 = {
        "learning_rate": 0.05,
        "weight_regularization": 0.1,
        "num_iterations": 100
    }

    hyperparameters2 = {
        "learning_rate": 0.01,
        "weight_regularization": 0.01,
        "num_iterations": 100
    }
    f1, df1, y1 = logistic(weights1, test_inputs, test_targets, hyperparameters1)
    ce1, acc1 = evaluate(test_targets, y1)
    f2, df2, y2 = logistic(weights2, test_inputs, test_targets, hyperparameters2)
    ce2, acc2 = evaluate(test_targets, y2)

    print('The optimal hyperparameter has learning rate 0.05, weight_regularization 0.1, iterations 100, '
          'training loss {trainl}, training accuracy {traina}, validation loss {validl}, validation accuracy {valida}, test loss {testl}, test accuracy {testa}.'.format(
        trainl=stat['train_ce'][maximum], traina=stat['train_acc'][maximum], validl=stat['valid_ce'][maximum],
        valida=stat['valid_acc'][maximum], testl = f1.item(), testa = acc1))

    print('The optimal hyperparameter has learning rate 0.05, weight_regularization 0.01, iterations 100, '
          'training loss {trainl}, training accuracy {traina}, validation loss {validl}, validation accuracy {valida}, test loss {testl}, test accuracy {testa}.'.format(
        trainl=stat2['train_ce'][maximum2], traina=stat2['train_acc'][maximum2], validl=stat2['valid_ce'][maximum2],
        valida=stat2['valid_acc'][maximum2], testl = f2.item(), testa = acc2))
    alltraince1,allvalidce1 = cross_entropy(hyperparameters1, train_inputs, train_targets,valid_inputs, valid_targets)
    alltraince2,allvalidce2 = cross_entropy(hyperparameters2, train_inputs2, train_targets2,valid_inputs, valid_targets)
    plt.plot(alltraince1)
    plt.plot(allvalidce1)
    plt.ylabel('cross entropy loss')
    plt.xlabel('number of iterations')
    plt.title('number of iterations vs cross entropy loss plot for train data')
    plt.show()
    plt.plot(alltraince2)
    plt.plot(allvalidce2)
    plt.ylabel('cross entropy loss')
    plt.xlabel('number of iterations')
    plt.title('number of iterations vs cross entropy loss plot for small train data')
    plt.show()

def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()