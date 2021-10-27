import numpy as np
# import matplotlib.pyplot as plt
import random

# import time
# np.set_printoptions(threshold='nan')


num_class = 10
num_feature = 784
num_train = 60000
num_test = 10000
num_machines = 20
batch_size = 32

# num_class = 7
# num_feature = 54
# num_train = 465264
# num_test = 115748
# batch_size = 32
# num_machines = 14

num_iter = 1500

exit_byzantine = True
num_byz = 8


def cal_grad(x, y, theta, weight_lambda):
    x = list(x)
    x.append(1.0)
    x = np.array(x)
    tmp = [int(y == i) for i in range(num_class)]
    indice_y = np.array(tmp)
    indice_y = indice_y.reshape((num_class, 1))
    t = np.dot(theta, x)
    t = t - np.max(t, axis=0)
    pro = np.exp(t) / sum(np.exp(t))
    pro = pro.reshape((num_class, 1))
    x = x.reshape((1, num_feature + 1))
    grad = -((indice_y - pro) * x + weight_lambda * theta)
    return grad


def cal_total_grad(X, Y, theta, weight_lambda):
    """
    :param X: shape(num_samples, features + 1)
    :param Y: labels' one_hot array, shape(num_samples, features + 1)
    :param theta: shape (num_classes, feature+1)
    :param weight_lambda: scalar
    :return: grad, shape(num_classes, feature+1)
    """

    m = X.shape[0]
    t = np.dot(theta, X.T)
    t = t - np.max(t, axis=0)
    pro = np.exp(t) / np.sum(np.exp(t), axis=0)
    total_grad = -np.dot((Y.T - pro), X) / m + weight_lambda * theta
    # loss = -np.sum(Y.T * np.log(pro)) / m + weight_lambda / 2 * np.sum(theta ** 2)
    return total_grad


def cal_loss(X, Y, theta, weight_lambda):
    loss = 0.0
    m = X.shape[0]
    t1 = np.dot(theta, X.T)
    t1 = t1 - np.max(t1, axis=0)
    t = np.exp(t1)
    tmp = t / np.sum(t, axis=0)
    loss = -np.sum(Y.T * np.log(tmp)) / m + weight_lambda * np.sum(theta ** 2) / 2
    return loss


def cal_acc(test_x, test_y, theta):
    pred = []
    num = 0
    m = test_x.shape[0]
    for i in range(m):
        t1 = np.dot(theta, test_x[i])
        t1 = t1 - np.max(t1, axis=0)
        # pro = np.exp(np.dot(theta, test_x[i]))
        pro = np.exp(t1)
        # pro = list(pro)
        # index = pro.index(max(pro))
        index = np.argmax(pro)
        # index = int(index)
        # pred.append(index)
        if index == test_y[i]:
            num += 1
    acc = float(num) / m
    return acc, pred


def cal_mean(grad_li):
    m = len(grad_li)
    # print "grad_len:", m
    grad = np.zeros_like(grad_li[0])
    for i, item in enumerate(grad_li):
        grad += item
    # grad = sum(grad_li)
    grad = grad / m
    # print(grad)
    return grad


def cal_median(grad_li):
    grad = np.zeros_like(grad_li[0])

    for i in range(num_class):
        for j in range(num_feature + 1):
            tmp = []
            for t in range(num_machines):
                tmp.append(grad_li[t][i][j])
            grad[i][j] = np.median(tmp)
    return grad


def geometric_median(mean_li):
    max_iter = 1000
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


class Machine:
    def __init__(self, data_x, data_y, machine_id):
        """Initializes the machine with the data
        Accepts data, a numpy array of shape :(num_samples/num_machines, dimension)
        data_x : a numpy array has shape :num_samples/num_machines, dimension)
        data_y: a list of length 'num_samples/num_machine', the label of the data_x"""

        self.data_x = data_x
        self.data_y = data_y
        self.machine_id = machine_id

    def update(self, theta, weight_lambda):
        """Calculates gradient with a randomly selected sample, given the current theta
         Accepts theta, a np array with shape of (dimension,)
         Returns the calculated gradient"""
        m = self.data_x.shape[0]
        # print "machine %d:"%(self.machine_id), m
        id = random.randint(0, m - batch_size)
        grad_d = cal_total_grad(self.data_x[id:(id + batch_size)], self.data_y[id:(id + batch_size)], theta,
                                weight_lambda)
        if exit_byzantine == True and self.machine_id >= num_machines - num_byz:
            # grad_d= np.random.standard_cauchy((10, 785))
            # grad_d = np.random.standard_normal((10,785)) * 10000
            # grad_d = np.ones_like(theta) * 100
            grad_d = -3 * grad_d
        # if (exit_byzantine == True and self.machine_id == num_machines - 1):
        #     grad_d = -4*grad_d
        # elif(exit_byzantine == True and self.machine_id == num_machines - 2):
        #     grad_d = -4*grad_d
        # elif (exit_byzantine == True and self.machine_id == num_machines - 3):
        #     grad_d = -4*grad_d
        # elif (exit_byzantine == True and self.machine_id == num_machines - 4):
        #     grad_d = -4*grad_d
        # elif (exit_byzantine == True and self.machine_id == num_machines - 5):
        #     grad_d = -4*grad_d
        # elif (exit_byzantine == True and self.machine_id == num_machines - 6):
        #     grad_d = -4*grad_d
        # elif (exit_byzantine == True and self.machine_id == num_machines - 7):
        #     grad_d = -4*grad_d
        # elif (exit_byzantine == True and self.machine_id == num_machines - 8):
        #     grad_d = -4*grad_d
        # elif (exit_byzantine == True and self.machine_id == num_machines - 9):
        #     grad_d = -4*grad_d
        # elif (exit_byzantine == True and self.machine_id == num_machines - 10):
        #     grad_d = -4*grad_d
        return grad_d


class Parameter_server:
    def __init__(self):
        """Initializes all machines"""
        self.theta_li = []  # list that stores each theta, grows by one iteration
        self.acc_li = []
        self.grad_norm = []
        self.acc_li = []
        self.time_li = []

        train_img = np.load('./data/mnist/train_img.npy')  # shape(60000, 784)
        train_lbl = np.load('./data/mnist/train_lbl.npy')  # shape(60000,)
        one_train_lbl = np.load('./data/mnist/one_train_lbl.npy')  # shape(60000, 10)
        test_img = np.load('./data/mnist/test_img.npy')  # shape(10000, 784)
        test_lbl = np.load('./data/mnist/test_lbl.npy')  # shape(10000,)

        # train_img = np.load('./data/train_img1.npy')  # shape(60000, 784)
        # train_lbl = np.load('./data/train_lbl1.npy')  # shape(60000,)
        # one_train_lbl = np.load('./data/one_train_lbl1.npy')  # shape(60000, 10)
        # test_img = np.load('./data/test_img1.npy')  # shape(10000, 784)
        # test_lbl = np.load('./data/test_lbl1.npy')  # shape(10000,)

        bias_train = np.ones(num_train)
        train_img_bias = np.column_stack((train_img, bias_train))

        bias_test = np.ones(num_test)
        test_img_bias = np.column_stack((test_img, bias_test))

        self.test_img_bias = test_img_bias
        self.test_lbl = test_lbl
        self.train_img_bias = train_img_bias
        self.one_train_lbl = one_train_lbl

        samples_per_machine = int(num_train / num_machines)
        self.machines = []

        #########  i.i.d case
        for i in range(num_machines):
            new_machine = Machine(train_img_bias[i * samples_per_machine:(i + 1) * samples_per_machine, :],
                                  one_train_lbl[i * samples_per_machine:(i + 1) * samples_per_machine, :], i)
            self.machines.append(new_machine)

        ###############   every 2 machine share the same digit image (non i.i.d. case)
        # for i in range(num_class):
        #     s1 = './data/mnist/2/train_img' + str(i) + '.npy'
        #     s2 = './data/mnist/2/one_train_lbl' + str(i) + '.npy'
        #     # s1 = './data/2/train_img' + str(i) + '.npy'
        #     # s2 = './data/2/one_train_lbl' + str(i) + '.npy'
        #     train = np.load(s1)
        #     label = np.load(s2)
        #     size = train.shape[0]
        #     num1 = int(size / 2)
        #     tmp_bias = np.ones(size)
        #     train_bias = np.column_stack((train, tmp_bias))
        #     new_machine1 = Machine(train_bias[0:num1, :], label[0:num1, :], i * 2)
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

    def broadcast(self, theta, weight_lambda):
        """Broadcast theta
        Accepts theta, a numpy array of shape:(dimension,)
        Return a list of length 'num_machines' containing the updated theta of each machine"""

        new_grad_li = []
        for i, mac in enumerate(self.machines):
            new_grad_li.append(mac.update(theta, weight_lambda))

        #  for i in range(len(new_grad_li)):
        #     if(exit_byzantine == True and i >= num_machines - num_byz):
        #         new_grad_li[i] = new_grad_li[0]
        return new_grad_li

    def train(self, init_theta, alpha, weight_lambda):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""

        self.theta_li.append(init_theta)
        k = 0
        #d = 0.00003
        d = 0.00005
        #d = 0.0005
        # start = time.clock()
        for i in range(num_iter):
            alpha = d / np.sqrt(i + 1)
            # alpha = d / (i + 1)
            cur_grad = self.broadcast(self.theta_li[-1], weight_lambda)
            # time_m1 = time.clock()
            #mean_grad = cal_mean(cur_grad)
            mean_grad = cal_median(cur_grad)
            #mean_grad = geometric_median(cur_grad)
            # time_m2 = time.clock()
            # print "diff time:", time_m2 - time_m1
            theta = self.theta_li[-1] - alpha * (weight_lambda * self.theta_li[-1] + mean_grad)
            self.theta_li.append(theta)
            # total_grad = cal_total_grad(self.train_img_bias, self.one_train_lbl, theta, weight_lambda)
            # self.grad_norm.append(np.linalg.norm(total_grad))
            if i % 10 == 0:
                # iter_time = time.clock()
                # self.time_li.append(iter_time - start)
                acc, _ = cal_acc(self.test_img_bias, self.test_lbl, theta)
                # print(theta)
                # self.acc_li.append(acc)
                print(i, acc)
            # print "step: ", i, " grad_norm: ", self.grad_norm[-1]
        print("train end!")
        # print "time_li:", self.time_li

    def plot_curve(self):
        """plot the loss curve and the acc curve
        save the learned theta to a numpy array and a txt file"""
        # sign_attack/

        s1 = 'wei0.01_alpha0.0001_sqrt(time)(test4)'
        # np.save('./result/SGD_mean/' + s1 + '/acc.npy', self.acc_li)
        np.save('./result/SGD_mean/' + s1 + '/theta_li.npy', self.theta_li[-1])
        # np.save('./result/SGD_mean/no_fault/same_digit/' + s1 + '/grad_norm.npy', self.grad_norm)
        np.save('./result/SGD_mean/' + s1 + '/time_li.npy', self.time_li)
        # np.save('./result/RSGD/fault/sign_attack/q10/' + s1 + '/theta0_li.npy', self.theta0_li[-1])
        # np.save('./result/RSGD/fault/sign_attack/q1/' + s1 + '/theta0_li_diff.npy', self.theta0_li_diff)
        # np.save('./result/RSGD/fault/sign_attack/q1/' + s1 + '/theta_li_diff.npy', self.theta_li_diff)

        # plt.plot(np.arange(len(self.acc_li)) * 10, self.acc_li)
        # plt.xlabel('iter')
        # plt.ylabel('accuracy')
        # # plt.title(s1)
        # plt.savefig('./result/SGD_mean/' + s1 + '/acc.png')
        # plt.show()

        # plt.semilogy(np.arange(num_iter), self.grad_norm)
        # plt.xlabel('iter')
        # plt.ylabel('log||grad||')
        # # plt.title(s1)
        # plt.savefig('./result/SGD_mean/no_fault/same_digit/' + s1 + '/grad_norm.png')
        # plt.show()


def init():
    server = Parameter_server()
    return server


def main():
    server = init()
    init_theta = np.zeros((num_class, num_feature + 1))
    # init_theta = np.ones((num_class, num_feature + 1))
    alpha = 0.0001
    weight_lambda = 0.01
    server.train(init_theta, alpha, weight_lambda)
    # server.plot_curve()


import cProfile
import pstats

# cProfile.run("main()", filename="./result/SGD_mean/" + s1 + "/result_profile.out", sort="tottime")
# p = pstats.Stats("./result/SGD_mean/" + s1 + "/result_profile.out")
# p.strip_dirs().sort_stats('tottime').print_stats(0.2)
main()

# train_img = np.load('./data/mnist/train_img.npy') #shape(60000, 784)
# # print train_img[1]
# bias_train = np.ones(num_train)
# train_img_bias = np.column_stack((train_img, bias_train))
# train_lbl = np.load('./data/mnist/train_lbl.npy') #shape(60000,)
# one_train_lbl = np.load('./data/mnist/one_train_lbl.npy') #shape(10, 60000)
# tmp = np.load('./result/SGD_mean/no_fault/same_digit/wei0.01_alpha0.0001_sqrt_2data(-100)/theta_li.npy')
# tmp = list(tmp)
# theta_star = tmp[-1]
# max_grad = 0
# batch = 3000
# lam = 0.01
#
# for i in range(num_class):
#     s1 = 'G:\python_code/byzantine/RSGD_multiLR/data/mnist/2/train_img' + str(i) + '.npy'
#     s2 = 'G:\python_code/byzantine/RSGD_multiLR/data/mnist/2/one_train_lbl' + str(i) + '.npy'
#     train = np.load(s1)
#     label = np.load(s2)
#     size = train.shape[0]
#     num1 = size / 2
#     tmp_bias = np.ones(size)
#     train_bias = np.column_stack((train, tmp_bias))
#     if(i < 1):
#         train_bias = -100*train_bias
#     grad1 = cal_total_grad(train_bias[0:num1, :], label[0:num1, :], theta_star, lam)
#     grad2 = cal_total_grad(train_bias[num1:, :], label[num1:, :], theta_star, lam)
#     tmp1 = np.max(np.abs(grad1)) * (float(num1) / num_train)
#     tmp2 = np.max(np.abs(grad2)) * (float(num1) / num_train)
#     tmp = []
#     tmp.append(max_grad)
#     tmp.append(tmp1)
#     tmp.append(tmp2)
#     max_grad = max(tmp)
# print "max:", max_grad
# # max_grad = max_grad / 20.0
# theta_lam = np.max(np.abs(lam * theta_star))
# print "max:", max_grad
# if theta_lam > max_grad:
#     max_grad = theta_lam
# print max_grad
