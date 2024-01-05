from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np

def best_hyper(train_inputs, train_targets, valid_inputs, valid_targets):
    N, M = train_inputs.shape
    weights = np.zeros((M + 1, 1))
    alphas = [0.05, 0.01]
    regs = [0.1, 0.01]
    iters = [50, 100]
    stat = {'train_ce': [], 'train_acc': [], 'valid_ce': [], 'valid_acc': [], 'weights':[]}
    maximum = 0
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
                if valid_acc > maximum:
                    maximum = len(stat) - 1
    return maximum, stat

def run_logistic_regression():
    train_inputs, train_targets = load_train()
    train_inputs2, train_targets2 = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape
    N2, M2 = train_inputs2.shape
    weights = np.zeros((M + 1, 1))
    weights2 = np.zeros((M2 + 1, 1))
    alphas = [0.05, 0.01]
    regs = [0.1, 0.01]
    iters = [50, 100]
    stat = {'train_ce': [], 'train_acc': [], 'valid_ce': [], 'valid_acc': [], 'train_ce2': [],
            'train_acc2': [], 'valid_ce2': [], 'valid_acc2': []}
    # run_check_grad(hyperparameters)
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
                    f2, df2, y2 = logistic(weights2, train_inputs2, train_targets2, hyperparameters)
                    alpha = hyperparameters["learning_rate"]
                    reg = hyperparameters["weight_regularization"]
                    weights = weights - alpha * df - alpha * reg * weights
                    weights2 = weights2 - alpha * df2 - alpha * reg * weights2
                ## Final training cross entropy and accuracy
                f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
                stat['train_ce'].append(f.item())
                ce, train_acc = evaluate(train_targets, y)
                stat['train_acc'].append(train_acc)
                fv, dyv, yv = logistic(weights, valid_inputs, valid_targets, hyperparameters)
                stat['valid_ce'].append(fv.item())
                ce, valid_acc = evaluate(valid_targets, yv)
                stat['valid_acc'].append(valid_acc)

                f2, df2, y2 = logistic(weights2, train_inputs2, train_targets2, hyperparameters)
                stat['train_ce2'].append(f2.item())
                ce, train_acc = evaluate(train_targets2, y2)
                stat['train_acc2'].append(train_acc)
                fv2, dyv2, yv2 = logistic(weights2, valid_inputs, valid_targets, hyperparameters)
                stat['valid_ce2'].append(fv2.item())
                ce, valid_acc = evaluate(valid_targets, yv2)
                stat['valid_acc2'].append(valid_acc)

    for key, value in stat.items():
        print(key, ' : ', value)

    ## Now Rerun with best hyperparameters to get test accuracies
    weights = np.zeros((M + 1, 1))
    weights2 = np.zeros((M2 + 1, 1))

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

    result = {'train_loss': 0, 'train_accuracy': 0, 'valid_loss': 0,
              'valid_accuracy': 0, 'test_loss': 0, 'test_accuracy': 0}
    result2 = {'train_loss': 0, 'train_accuracy': 0, 'valid_loss': 0,
               'valid_accuracy': 0, 'test_loss': 0, 'test_accuracy': 0}

    ## For train set ##
    for t in range(hyperparameters1["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters1)
        alpha = hyperparameters1["learning_rate"]
        reg = hyperparameters1["weight_regularization"]
        weights = weights - alpha * df - alpha * reg * weights
    f1, df1, y1 = logistic(weights, train_inputs, train_targets, hyperparameters1)
    f2, df2, y2 = logistic(weights, valid_inputs, valid_targets, hyperparameters1)
    f3, df3, y3 = logistic(weights, test_inputs, test_targets, hyperparameters1)
    ce1, acc1 = evaluate(train_targets, y1)
    ce2, acc2 = evaluate(valid_targets, y2)
    ce3, acc3 = evaluate(test_targets, y3)
    result['train_loss'] += f1
    result['valid_loss'] += f2
    result['test_loss'] += f3
    result['train_accuracy'] += acc1
    result['valid_accuracy'] += acc2
    result['test_accuracy'] += acc3

    ## For small train set ##
    for t in range(hyperparameters2["num_iterations"]):
        f, df, y = logistic(weights, train_inputs2, train_targets2, hyperparameters2)
        alpha = hyperparameters2["learning_rate"]
        reg = hyperparameters2["weight_regularization"]
        weights2 = weights2 - alpha * df - alpha * reg * weights2
    f1, df1, y1 = logistic(weights2, train_inputs2, train_targets2, hyperparameters2)
    f2, df2, y2 = logistic(weights2, valid_inputs, valid_targets, hyperparameters2)
    f3, df3, y3 = logistic(weights2, test_inputs, test_targets, hyperparameters2)
    ce1, acc1 = evaluate(train_targets2, y1)
    ce2, acc2 = evaluate(valid_targets, y2)
    ce3, acc3 = evaluate(test_targets, y3)
    result2['train_loss'] += f1
    result2['valid_loss'] += f2
    result2['test_loss'] += f3
    result2['train_accuracy'] += acc1
    result2['valid_accuracy'] += acc2
    result2['test_accuracy'] += acc3

    for key, value in result.items():
        print(key, ' : ', value)

    for key, value in result2.items():
        print(key, ' : ', value)

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
