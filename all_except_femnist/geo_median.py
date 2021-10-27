import numpy as np


def cal_mean(grad_li):
    m = len(grad_li)
    # print "grad_len:", m
    grad = np.zeros_like(grad_li[0])
    for i, item in enumerate(grad_li):
        grad += item
    #grad = sum(grad_li)
    grad = grad / m
    #print(grad)
    return grad


def geometric_median(mean_li):
    max_iter = 10000
    tol = 1e-7
    guess = cal_mean(mean_li)
    iter = 0
    while iter < max_iter:
        dist_li = [np.linalg.norm(item - guess) for _, item in enumerate(mean_li)]
        temp1 = np.zeros_like(mean_li[0])
        temp2 = 0.0
        for elem1, elem2 in zip(mean_li, dist_li):
            if elem2 == 0:
                elem2 = 1.0
            temp1 += elem1 / elem2
            temp2 += 1.0 / elem2
        guess_next = temp1 / temp2
        guess_movement = np.linalg.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= tol:
            break
        iter += 1
    # print "geo_iter:", iter
    # print "diff:", guess_movement
    return guess


def main():
    grad_li = []
    grad_li.append([2.0, 1.0])
    grad_li.append([1.0, 2.0])
    grad_li.append([1.0, 3.0])
    grad_li.append([2.0, 2.0])
    grad_li.append([2.0, 3.0])
    grad_li.append([3.0, 2.0])
    grad_li.append([2.0, 4.0])
    grad_li.append([3.0, 3.0])
    grad_li.append([20.0, 20.0])
    t = geometric_median(grad_li)
    print(t)

main()
