import numpy as np
import random

# import matplotlib.pyplot as plt
# import time
# import pandas as pd
np.set_printoptions(threshold=10000000)
#
# num_class = 7
# num_feature = 54
# num_train = 465264
# num_test = 115748
# num_machines = 20
# batch_size = 32

num_class = 10
num_feature = 784
num_train = 60000
num_test = 10000
num_machines = 20
batch_size = 32

num_iter = 1100
exit_byzantine = True
num_byz = 8

f = './result/GD/wei0.01_alpha0.00001/theta_li.npy'
# theta_star = np.load(f)[-1]
theta_star = np.zeros((10, 785))


def cal_total_grad(X, Y, theta, weight_lambda):
    """
    :param X: shape(num_samples, features + 1)
    :param Y: labels' one_hot array, shape(num_samples, num_classes)
    :param theta: shape (num_classes, feature+1)
    :param weight_lambda: scalar
    :return: grad, shape(num_classes, feature+1)
    """
    m = X.shape[0]
    # loss = 0.0
    t = np.dot(theta, X.T)  # (num_classes, num_samples)
    # t = t - np.min(t, axis=0)
    t = t - np.max(t, axis=0)
    t[np.isnan(t)] = 0
    pro = np.exp(t) / np.sum(np.exp(t), axis=0)
    total_grad = -np.dot((Y.T - pro), X) / m  # + weight_lambda * theta
    # loss = -np.sum(Y.T * np.log(pro)) / m + weight_lambda / 2 * np.sum(theta ** 2)
    return total_grad


def cal_loss(X, Y, theta, weight_lambda):
    m = X.shape[0]
    t1 = np.dot(theta, X.T)
    t1 = float(t1) - float(np.max(t1, axis=0))
    t = np.exp(t1)
    tmp = t / np.sum(t, axis=0)
    loss = -np.sum(Y.T * np.log(tmp)) / m + weight_lambda * np.sum(theta ** 2) / 2
    return loss


def cal_acc(test_x, test_y, theta):
    # pred = []
    num = 0
    m = test_x.shape[0]
    for i in range(m):
        # pro = np.exp(np.dot(theta, test_x[i]))
        t1 = np.dot(theta, test_x[i])
        t1 = t1 - np.max(t1, axis=0)
        # pro = np.exp(np.dot(theta, test_x[i]))
        pro = np.exp(t1)
        index = np.argmax(pro)
        if index == test_y[i]:
            num += 1
    acc = float(num) / m
    return acc


def cal_max_norm_grad(theta):
    if np.all(theta == 0):
        return theta
    tmp = np.abs(theta)
    re = np.where(tmp == np.max(tmp))
    row = re[0][0]
    col = re[1][0]
    max_val = tmp[row, col]
    # max_val = theta[row, col]
    tmp_theta = np.zeros_like(theta)
    # if max_val > 0:
    #     tmp_theta[row, col] = 1.0
    # elif max_val < 0:
    #     tmp_theta[row, col] = -1.0
    n = len(re[0])
    theta[tmp != np.max(tmp)] = 0
    theta[theta == -max_val] = -1.0 / n
    theta[theta == max_val] = 1.0 / n
    return theta
    # return tmp_theta


def cal_var(theta):
    mean_theta = np.mean(theta, axis=0)
    mean_arr = np.tile(mean_theta, (theta.shape[0], 1))
    tmp = theta - mean_arr
    var = np.trace(np.dot(tmp, tmp.T))
    return var


def huber_loss_grad(e, d):
    t = (np.abs(e) <= d) * e
    e[np.abs(e) <= d] = 0
    grad = t + d * np.sign(e)
    return grad


class Machine:
    def __init__(self, data_x, data_y, machine_id):
        """Initializes the machine with the data
        Accepts data, a numpy array of shape :(num_samples/num_machines, dimension)
        data_x : a numpy array has shape :num_samples/num_machines, dimension)
        data_y: a list of length 'num_samples/num_machine', the label of the data_x"""

        self.data_x = data_x
        self.data_y = data_y
        self.machine_id = machine_id
        self.beta1 = np.zeros((num_class, num_feature + 1))
        self.beta0 = np.zeros((num_class, num_feature + 1))

    def update(self, theta0, theta, alpha, l1_lambda, weight_lambda, delta):
        """Calculates gradient with a randomly selected sample, given the current theta
         Accepts theta, a np array with shape of (dimension,)
         Returns the calculated gradient"""
        m = self.data_x.shape[0]
        # print "machine%d:"%(self.machine_id), m
        id = random.randint(0, m - batch_size)
        grad_f = cal_total_grad(self.data_x[id:(id + batch_size)], self.data_y[id:(id + batch_size)], theta,
                                weight_lambda)  # /num_machines

        # L1 norm
        # grad = grad_f / num_machines + weight_lambda * theta + l1_lambda * np.sign(theta - theta0) # ||x_i||2

        self.beta1 = self.beta0 + (alpha / 2) * (theta - theta0)
        self.beta1[self.beta1 >= l1_lambda] = l1_lambda
        self.beta1[self.beta1 <= -l1_lambda] = -l1_lambda
        # tmp = self.beta1 - 0.5 * self.beta0
        # new_theta = theta + (alpha*delta/(1+alpha*delta))*tmp - (delta*(1+alpha*delta))*grad_f
        new_theta = theta - (1 / (delta + alpha)) * (2 * self.beta1 - self.beta0 + grad_f)

        self.beta0 = self.beta1
        # if(exit_byzantine == True and self.machine_id >= num_machines - num_byz):
        # if (exit_byzantine == True and self.machine_id%5 == 0):
        # new_theta = -3*new_theta
        # new_theta = np.random.standard_normal((10, 785)) * 10000
        # new_theta = np.ones_like(theta0) *100

        return new_theta


class Parameter_server:
    def __init__(self):
        """Initializes all machines"""
        self.theta0_li = []
        self.theta_li = []  # list that stores each theta, grows by one iteration
        self.acc_li = []
        self.grad_li = []
        self.grad_norm = []
        self.theta0_star_norm = []
        self.acc_li = []
        self.loss_li = []
        self.theta_li_diff = []
        self.theta0_li_diff = []
        self.time_li = []
        self.var_li = []

        # train_img = np.load('./data/train_img1.npy')  # shape(60000, 784)
        # train_lbl = np.load('./data/train_lbl1.npy')  # shape(60000,)
        # one_train_lbl = np.load('./data/one_train_lbl1.npy')  # shape(60000, 10)
        # test_img = np.load('./data/test_img1.npy')  # shape(10000, 784)
        # test_lbl = np.load('./data/test_lbl1.npy')  # shape(10000,)

        train_img = np.load('./data/mnist/train_img.npy')  # shape(60000, 784)
        train_lbl = np.load('./data/mnist/train_lbl.npy')  # shape(60000,)
        one_train_lbl = np.load('./data/mnist/one_train_lbl.npy')  # shape(60000, 10)
        test_img = np.load('./data/mnist/test_img.npy')  # shape(10000, 784)
        test_lbl = np.load('./data/mnist/test_lbl.npy')  # shape(10000,)

        bias_train = np.ones(num_train)
        train_img_bias = np.column_stack((train_img, bias_train))

        bias_test = np.ones(num_test)
        test_img_bias = np.column_stack((test_img, bias_test))

        self.test_img_bias = test_img_bias
        self.test_lbl = test_lbl
        self.train_img_bias = train_img_bias
        self.one_train_lbl = one_train_lbl
        self.train_lbl = train_lbl

        samples_per_machine = int(num_train / num_machines)
        self.machines = []
        self.beta1 = np.zeros((num_class, num_feature + 1))
        self.beta0 = np.zeros((num_class, num_feature + 1))
        self.beta = np.zeros([num_machines, num_class, num_feature + 1])

        #######  i.i.d case
        for i in range(num_machines):
            new_machine = Machine(train_img_bias[i * samples_per_machine:(i + 1) * samples_per_machine, :],
                                  one_train_lbl[i * samples_per_machine:(i + 1) * samples_per_machine, :], i)
            self.machines.append(new_machine)

        ###############   every 2 machine share the same digit image (non i.i.d. case)
        # for i in range(num_class):
        #     s1 = './data/mnist/2/train_img' + str(i) + '.npy'
        #     s2 = './data/mnist/2/one_train_lbl' + str(i) + '.npy'
        #     # s1 = './data/3/train_img' + str(i) + '.npy'
        #     # s2 = './data/3/one_train_lbl' + str(i) + '.npy'
        #     train = np.load(s1)
        #     label = np.load(s2)
        #     size = train.shape[0]
        #     num1 = int(size / 2)
        #     tmp_bias = np.ones(size)
        #     train_bias = np.column_stack((train, tmp_bias))
        #     new_machine1 = Machine(train_bias[0:num1, :], label[0:num1, :], i*2)
        #     new_machine2 = Machine(train_bias[num1:, :], label[num1:, :], i * 2 + 1)
        #     self.machines.append(new_machine1)
        #     self.machines.append(new_machine2)
        # if (i < 1):
        #     new_machine1 = Machine(train_bias[0:num1, :]*(-100), label[0:num1, :], i * 2)
        #     new_machine2 = Machine(train_bias[num1:, :]*(-100), label[num1:, :], i * 2 + 1)
        #     self.machines.append(new_machine1)
        #     self.machines.append(new_machine2)
        # else:
        #     new_machine1 = Machine(train_bias[0:num1, :], label[0:num1, :], i * 2)
        #     new_machine2 = Machine(train_bias[num1:, :], label[num1:, :], i * 2 + 1)
        #     self.machines.append(new_machine1)
        #     self.machines.append(new_machine2)

    def broadcast(self, theta0, theta_li, alpha, l1_lambda, weight_lambda, delta):
        """Broadcast theta
        Accepts theta, a numpy array of shape:(dimension,)
        Return a list of length 'num_machines' containing the updated theta of each machine"""

        new_theta_li = []
        for i, mac in enumerate(self.machines):
            new_theta_li.append(mac.update(theta0, theta_li[i], alpha, l1_lambda, weight_lambda, delta))
        # tmp = np.zeros_like(theta0)
        # same_attack = np.ones_like(theta0) * 100
        same_attack = np.random.standard_normal((num_class, num_feature + 1)) * 10000
        temp = np.zeros_like(new_theta_li[0])
        self.beta1 = np.zeros((num_class, num_feature + 1))
        for j in range(0, num_machines - num_byz):
            temp += self.machines[i].beta1
        for i in range(num_machines):
            if (exit_byzantine == True and i >= num_machines - num_byz):
                # self.beta[i] = self.beta[i] + (alpha / 2) * (-3 * theta_li[i] - theta0)
                # self.beta[i] = self.beta[i] + (alpha / 2) * (same_attack - theta0)
                self.beta[i] = - (temp / num_byz)
                self.beta[i][self.beta[i] <= -l1_lambda] = -l1_lambda
                self.beta[i][self.beta[i] >= l1_lambda] = l1_lambda
            else:
                self.beta[i] = self.machines[i].beta1
            self.beta1 = self.beta1 + self.beta[i]
        # for i in range(len(new_theta_li)):
        #     if(exit_byzantine == True and i >= num_machines - num_byz):
        #         self.beta[i] = self.beta[i] + (alpha/2)*(same_attack - theta0)
        #         #self.beta[i] = self.beta[i] + (alpha / 2) * (-3 * new_theta_li[i] - theta0)
        #         #self.beta[i] = self.beta[i] + (alpha / 2) * (new_theta_li[0] - theta0)
        #     else:
        #         self.beta[i] = self.beta[i] + (alpha/2)*(new_theta_li[i] - theta0)
        #
        #     self.beta[i][self.beta[i] <= -l1_lambda ] = -l1_lambda
        #     self.beta[i][self.beta[i] >= l1_lambda] = l1_lambda
        #
        #     self.beta1 = self.beta1 + self.beta[i]

        # tmp = self.beta1 - 0.5 * self.beta0
        # new_theta0 = theta0 + (alpha*delta/(1+alpha*delta))*tmp - weight_lambda*(delta*(1+alpha*delta)) * theta0

        # tmp = 2*self.beta1 - self.beta0
        new_theta0 = theta0 - (1 / (delta + num_machines * alpha)) * (
                    weight_lambda * theta0 - (2 * self.beta1 - self.beta0))
        # new_theta0 = theta0 + (1/(delta + alpha))*(tmp- weight_lambda*theta0)
        self.beta0 = self.beta1

        # new_theta0 = theta0 - alpha * (l1_lambda * tmp + weight_lambda * theta0)
        # new_theta0 = theta0 - alpha * (l1_lambda * tmp + weight_lambda * theta0)
        return new_theta0, new_theta_li

    def train(self, init_theta0, init_theta, alpha, l1_lambda, weight_lambda, d):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""

        # tmp = np.load('./result/GD/wei0.01_alpha0.00001/theta_li.npy')
        # tmp = list(tmp)
        # theta_star = tmp[-1]

        self.theta0_li.append(init_theta0)
        self.theta_li.append(init_theta)
        k = 0
        delta = d

        # start = time.clock()
        for i in range(num_iter):
            # if (i + 1) == 1000:
            #     alpha = alpha / 10
            # alpha = d / np.sqrt(i + 1)
            if (i > 250):
                delta = d * np.sqrt(1 * (i + 1))
            else:
                delta = d * np.sqrt(1 * (i + 1))

            # alpha = d / (i + 1)
            rec_theta0, rec_theta = self.broadcast(self.theta0_li[-1], self.theta_li[-1], alpha, l1_lambda,
                                                   weight_lambda, delta)
            # if (i + 1) % 1000 == 0:
            # #     s1 = 'lam1.0_wei0.01_alpha0.0005_sqrt_theta'
            # #     np.save('./result/RSGD/L2/fault/same_attack/q10/' + s1 + '/theta_li' + str(i) + '.npy', self.theta_li)
            # #     np.save('./result/RSGD/L2/fault/same_attack/q10/' + s1 + '/theta0_li' + str(i) + '.npy', self.theta0_li)
            #     self.theta_li = []
            #     self.theta0_li = []
            self.theta0_li.append(rec_theta0)
            self.theta_li.append(rec_theta)
            # print "rec_theta shape:"

            # for j in range(20):
            #     tmp.append(np.linalg.norm(self.theta_li[-1][j] - self.theta_li[-2][j]))
            # self.theta_li_diff.append(tmp)
            # tmp0 = self.theta0_li[-1] - self.theta0_li[-2]
            # self.theta0_li_diff.append(np.linalg.norm(tmp0))
            # self.theta0_star_norm.append(np.linalg.norm(rec_theta0 - theta_star))
            # total_grad = CalTotalGrad(self.train_x[:1280], self.train_y[:1280], self.theta0_li[-1])
            # self.grad_li.append(total_grad)
            # self.grad_norm.append(np.linalg.norm(total_grad))
            # print "step: ", i, "||theta0 - thata*||:", self.theta0_star_norm[-1]
            # loss = cal_loss(self.train_img_bias, self.one_train_lbl, rec_theta0, weight_lambda)
            # total_grad = cal_total_grad(self.train_img_bias, self.one_train_lbl, rec_theta0, weight_lambda) + weight_lambda * rec_theta0
            # # total_grad = cal_total_grad(self.train_img_bias, self.one_train_lbl, rec_theta[0][0],weight_lambda) + weight_lambda * rec_theta[0][0]
            # self.grad_norm.append(np.linalg.norm(total_grad))
            # self.loss_li.append(loss)
            if (i) % 10 == 0:
                # iter_time = time.clock()
                # self.time_li.append(iter_time - start)
                acc = cal_acc(self.test_img_bias, self.test_lbl, rec_theta0)
                # acc1 = cal_acc(self.test_img_bias, self.test_lbl, rec_theta[0])
                # # acc, _ = cal_acc(self.test_img_bias, self.test_lbl, rec_theta[0][0])
                # # tr_acc, _ = cal_acc(self.train_img_bias, self.train_lbl, rec_theta0)
                # self.acc_li.append(acc)
                # print(rec_theta0[:,784])
                print(i, acc, alpha, d, l1_lambda)

                # print "step:", i, "train acc:", tr_acc
            # if (i > 550):
            #   print(rec_theta0)
            # theta_tmp = []
            # for k in range(num_machines - num_byz):
            #     theta_tmp.append(rec_theta[k])
            # # print len(theta_tmp)
            #
            # theta_tmp = np.array(theta_tmp)
            # theta_tmp = theta_tmp.reshape(num_machines - num_byz, 10*785)
            # rec_theta0 = rec_theta0.reshape(1, 10*785)
            # # var_theta = theta_tmp[:num_machines-num_byz, :]
            # var_theta = np.row_stack((theta_tmp, rec_theta0))
            # # print "var shape:", var_theta.shape
            # self.var_li.append(cal_var(var_theta))

            # print "step: ", i, " grad_norm: ", self.grad_norm[-1]
        print("train end!")

        # print "var:", self.var_li
        # print("time:", self.time_li)


def init():
    server = Parameter_server()
    return server


def main():
    server = init()
    # init_theta0 = np.random.randn(num_class, num_feature + 1)
    init_theta0 = np.zeros((num_class, num_feature + 1))
    init_theta = []
    for i in range(num_machines):
        # tmp = []
        # tmp.append(np.zeros((num_class, num_feature + 1)))
        # tmp.append(np.zeros((num_class, num_feature + 1)))
        # init_theta.append(tmp)
        # init_theta.append(np.random.randn(num_class, num_feature + 1))
        init_theta.append(np.zeros((num_class, num_feature + 1)))
    alpha = 10000
    d = 20000
    l1_lambda = 0.8
    weight_lambda = 0.01
    server.train(init_theta0, init_theta, alpha, l1_lambda, weight_lambda, d)

    # server.plot_curve()

    # file = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt(time)/time_li.npy'
    # time_li =
    # np.load(file)
    # print time_li
    # print "len:", len(time_li)


main()

# p = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.001_sqrt/theta0_li.npy'
# t = np.load(p)
# print t.shape
# print t
