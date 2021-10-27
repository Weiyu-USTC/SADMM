import numpy as np

true_x = 0.5
num_iter = 100


def cal_grad_f0(x):
    grad_f = x
    return grad_f


def cal_grad_f1(x):
    grad_f = x-1
    return grad_f


def update_x0(x0, x1, l1_lambda, alpha):
    grad_f = cal_grad_f0(x0)

    new_x1 = x0 - alpha * (grad_f + l1_lambda * (np.sign(x0-x1) + 1))
    new_x2 = x0 - alpha * (grad_f + l1_lambda * (np.sign(x0-x1) - 1))
    if abs(new_x1 - true_x) > abs(new_x2-true_x):
        return new_x1
    return new_x2


def update_x1(x0, x1, l1_lambda, alpha):
    grad_f = cal_grad_f1(x1)
    new_x1 = x1 - alpha * (grad_f + l1_lambda * (np.sign(x1-x0)))
    return new_x1


def main():
    x0_list = []
    x1_list = []
    x0 = 0
    x1 = 0
    alpha = 0.001
    l1_lambda = 0.01
    for i in range(num_iter):
        alpha = 0.001 / np.sqrt(i+1)
        new_x0 = update_x0(x0, x1, l1_lambda, alpha)
        new_x1 = update_x1(x0, x1, l1_lambda, alpha)
        x0_list.append(new_x0)
        x1_list.append(new_x1)
        x0 = new_x0
        x1 = new_x1
        print(i, new_x0)


main()












