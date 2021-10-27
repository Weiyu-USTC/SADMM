import numpy as np
import random
import matplotlib.pyplot as plt
import time
#np.set_printoptions(threshold='nan')

num_class = 10
num_feature = 28 * 28
num_train = 60000
num_test = 10000
num_machines = 20
batch_size = 32

num_iter = 6100
exit_byzantine = True
num_byz = 4


def cal_total_grad(x, machine_id):
    """
    :param X:
    :return: total_grad
    """
    total_grad = 2 * (machine_id+1) * x
    return total_grad


def cal_acc(x):
    """
    :param x:
    :return:
    """
    acc = 0
    # for i in range(num_machines):
    #     acc += x*x + i * x
    acc = 210 * x*x + 200
    return acc



class Machine:
    def __init__(self, machine_id):
        self.beta1 = 0
        self.beta0 = 0
        self.machine_id = machine_id

    def update(self, x0, x, alpha, l1_lambda, weight_lambda, delta):
        grad_f = cal_total_grad(x, self.machine_id)
        self.beta1 = self.beta0 +(alpha/2)*(x0-x)
        if(self.beta1 >= l1_lambda):
            self.beta1 = l1_lambda
        elif(self.beta1 <= -l1_lambda):
            self.beta1 = -l1_lambda
        new_x = x + (1/(delta + alpha))*(2*self.beta1 - self.beta0 - grad_f)
        self.beta0 = self.beta1
        return new_x


class Parameter_server:
    def __init__(self):
        """Initializes all machines"""
        self.x0_li = []
        self.x_li = [] #list that stores each theta, grows by one iteration
        self.acc_li = []
        self.beta1 = 0
        self.beta0 = 0
        self.beta = []
        for i in range(num_machines):
            self.beta.append(0)

        self.machines = []
        for i in range(num_machines):
            new_machine = Machine(i)
            self.machines.append(new_machine)

    def broadcast(self, x0, x_li, alpha, l1_lambda, weight_lambda, delta):
        """Broadcast theta
        Accepts theta, a numpy array of shape:(dimension,)
        Return a list of length 'num_machines' containing the updated theta of each machine"""

        new_x_li = []
        for i, mac in enumerate(self.machines):
            new_x_li.append(mac.update(x0, x_li[i], alpha, l1_lambda, weight_lambda, delta))
        same_attack = 1
        self.beta1 = 0
        for i in range(len(new_x_li)):
            # L1 norm
            ### no fault
            #self.beta[i] = self.beta[i] + (alpha/2)*(new_x_li[i] - x0)
            ######### fault
            if (exit_byzantine == True and i < num_machines - num_byz):
                  self.beta[i] = self.beta[i] + (alpha/2)*(new_x_li[i] - x0)
            elif (exit_byzantine == True and i >= num_machines - num_byz):
                  self.beta[i] = self.beta[i] + (alpha/2)*(same_attack - x0)

            if(self.beta[i]>=l1_lambda):
                self.beta[i] = l1_lambda
            elif(self.beta[i]<=-l1_lambda):
                self.beta[i] = -l1_lambda
            self.beta1 = self.beta1 + self.beta[i]



        new_x0 = x0 + (1/(delta + num_machines*alpha))*(2*self.beta1 - self.beta0 - weight_lambda*x0)
        self.beta0 = self.beta1
        # new_theta0 = theta0 - alpha * (l1_lambda * tmp + weight_lambda * theta0)
        return new_x0, new_x_li

    def train(self, init_x0, init_x, alpha, l1_lambda, weight_lambda, d):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""

        self.x0_li.append(init_x0)
        self.x_li.append(init_x)
        k = 0
        for i in range(num_iter):
            delta = d / np.sqrt(2*(i + 1))
            rec_x0, rec_x = self.broadcast(self.x0_li[-1], self.x_li[-1], alpha, l1_lambda, weight_lambda, delta)
            self.x0_li.append(rec_x0)
            self.x_li.append(rec_x)
            if i % 100 == 0:
                acc = cal_acc(rec_x0)
                print(i, acc) #"step:", i, " acc:", acc

        print("train end!")
        # print "var:", self.var_li
        #print "time:", self.time_li


def init():
    server = Parameter_server()
    return server


def main():
    server = init()
    init_x0 = 20 #random.random()
    init_x = []
    for i in range(num_machines):
        init_x.append(20)# random.random())
    alpha = 0.5
    d = 2000
    l1_lambda = 0.1
    weight_lambda = 0.01
    server.train(init_x0, init_x, alpha, l1_lambda, weight_lambda, d)


main()

