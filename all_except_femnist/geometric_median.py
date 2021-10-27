import numpy as np


def cal_median(grad_li):
    grad = [0, 0]
    for i in range(len(grad_li)):
        grad[0] += grad_li[i][0]
        grad[1] += grad_li[i][1]

    grad[0] = grad[0]/8
    grad[1] = grad[1]/8

    return grad


def geometric_median(mean_li):
    max_iter = 1000
    tol = 1e-7
    guess = cal_median(mean_li)
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
    grad_li.append([1, 1])
    grad_li.append([2, 1])
    grad_li.append([3, 1])
    grad_li.append([2, 2])
    grad_li.append([2, 3])
    grad_li.append([10, 10])
    grad_li.append([10, 12])
    print(geometric_median(grad_li))

main()
