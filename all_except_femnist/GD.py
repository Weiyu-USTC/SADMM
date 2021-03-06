import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold='nan')

num_class = 10
num_feature = 28 * 28
num_train = 60000
num_test = 10000


def cal_grad(x, y, theta, weight_lambda):

    x = list(x)
    x.append(1.0)
    x = np.array(x)
    tmp = [int(y == i) for i in range(num_class)]
    indice_y = np.array(tmp)
    indice_y = indice_y.reshape((num_class, 1))
    t = np.dot(theta, x)
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

    loss = 0.0
    m = X.shape[0]
    t = np.dot(theta, X.T)
    t = t - np.max(t, axis=0)
    pro = np.exp(t) / np.sum(np.exp(t), axis=0)
    total_grad = -np.dot((Y.T - pro), X) / (m * 20) # + weight_lambda * theta
    loss = -np.sum(Y.T * np.log(pro)) / m + weight_lambda / 2 * np.sum(theta ** 2)
    return total_grad, loss


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
        pro = np.exp(np.dot(theta, test_x[i]))
        # pro = list(pro)
        # index = pro.index(max(pro))
        index = np.argmax(pro)
        pred.append(index)
        if index == test_y[i]:
            num += 1
    acc = float(num) / m
    return acc, pred


def train():

    """using GD to optimize the MultiLR"""
    num_iter = 5000
    alpha = 0.00005
    weight_lambda = 0.01
    theta = np.zeros((num_class, num_feature + 1))

    grad_norm = []
    acc_li = []
    theta_li = []
    loss_li = []

    train_img = np.load('./data/mnist/train_img.npy') #shape(60000, 784)
    train_lbl = np.load('./data/mnist/train_lbl.npy') #shape(60000,)
    one_train_lbl = np.load('./data/mnist/one_train_lbl.npy') #shape(10, 60000)
    test_img = np.load('./data/mnist/test_img.npy') #shape(10000, 784)
    test_lbl = np.load('./data/mnist/test_lbl.npy') #shape(10000,)

    # train_img = train_img / 255.0
    # test_img = test_img / 255.0

    bias_train = np.ones(num_train)
    train_img_bias = np.column_stack((train_img, bias_train))
    train_img_bias[train_lbl == 0] = train_img_bias[train_lbl == 0] * 100

    bias_test = np.ones(num_test)
    test_img_bias = np.column_stack((test_img, bias_test))

    for step in range(num_iter):

        grad, loss = cal_total_grad(train_img_bias, one_train_lbl, theta, weight_lambda)
        grad_norm.append(np.linalg.norm(grad))
        theta = theta - alpha * grad
        # loss = cal_loss(train_img_bias, one_train_lbl, theta, weight_lambda)
        loss_li.append(loss)
        if (step + 1) > num_iter - 11:
            theta_li.append(theta)
        if (step + 1) % 10 == 0:
            acc, _ = cal_acc(test_img_bias, test_lbl, theta)
            acc_li.append(acc)
            print "step:", step, " acc:", acc
        # print "step:", step, "loss:", loss
        print "step:", step, "grad_norm:", grad_norm[-1]

    s1 = 'wei0.01_alpha0.00005_2data(100)'
    np.save('./result/GD/' + s1 + '/grad_norm.npy', grad_norm)
    np.save('./result/GD/' + s1 + '/acc.npy', acc_li)
    np.save('./result/GD/' + s1 + '/theta_li.npy', theta_li)
    np.save('./result/GD/' + s1 + '/loss_li.npy', loss_li)

    plt.plot(np.arange(len(acc_li)) * 10, acc_li)
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.savefig('./result/GD/' + s1 + '/acc.jpg')
    plt.show()

    plt.semilogy(np.arange(num_iter), grad_norm)
    plt.xlabel('iter')
    plt.ylabel('log||grad||')
    plt.savefig('./result/GD/' + s1 + '/grad_norm.jpg')
    plt.show()

    plt.plot(np.arange(len(loss_li)), loss_li)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.savefig('./result/GD/' + s1 + '/loss.jpg')
    plt.show()


# train()


train_img = np.load('./data/mnist/train_img.npy') #shape(60000, 784)
# print train_img[1]
bias_train = np.ones(num_train)
train_img_bias = np.column_stack((train_img, bias_train))
train_lbl = np.load('./data/mnist/train_lbl.npy') #shape(60000,)
one_train_lbl = np.load('./data/mnist/one_train_lbl.npy') #shape(10, 60000)
tmp = np.load('./result/GD/wei0.01_alpha0.00005_2data(100)/theta_li.npy')
tmp = list(tmp)
theta_star = tmp[-1]
max_grad = 0
batch = 3000
lam = 0.01
# for i in range(20):
#     grad, _ = cal_total_grad(train_img_bias[(i*batch):(i+1)*batch], one_train_lbl[(i*batch):(i+1)*batch], theta_star, lam)
#     # tmp = np.sum(np.abs(grad))
#     tmp = np.linalg.norm(grad)
#     # tmp = np.max(np.abs(grad))
#     if tmp > max_grad:
#         max_grad = tmp

for i in range(num_class):
    s1 = 'G:\python_code/byzantine/RSGD_multiLR/data/mnist/2/train_img' + str(i) + '.npy'
    s2 = 'G:\python_code/byzantine/RSGD_multiLR/data/mnist/2/one_train_lbl' + str(i) + '.npy'
    train = np.load(s1)
    label = np.load(s2)
    size = train.shape[0]
    num1 = size / 2
    print "num1:", num1
    a = float(num1) / num_train
    print "a:", a
    tmp_bias = np.ones(size)
    train_bias = np.column_stack((train, tmp_bias))
    # if(i < 1):
    #     train_bias = 100*train_bias
    grad1, _ = cal_total_grad(train_bias[0:num1, :], label[0:num1, :], theta_star, lam)
    grad2, _ = cal_total_grad(train_bias[num1:, :], label[num1:, :], theta_star, lam)
    tmp1 = np.max(np.abs(grad1)) * (float(num1) / num_train)
    tmp2 = np.max(np.abs(grad2)) * (float(num1) / num_train)
    tmp = []
    tmp.append(max_grad)
    tmp.append(tmp1)
    tmp.append(tmp2)
    max_grad = max(tmp)
print "max:", max_grad
# max_grad = max_grad / 20.0
theta_lam = np.max(np.abs(lam * theta_star))
print "max:", max_grad
if theta_lam > max_grad:
    max_grad = theta_lam
print max_grad

##### 2data(-100)
# L1.alpha=0.00005 , 0.0956786877037
# L1 alpha=0.00001 , 0.115778185949

##### 2data (100)
# L1.alpha=0.00005 , 0.105243683586

######## first time data
#L1 norm: 0.0868313176682
#L2 norm: 1.13441424117
#max norm : 50.3779267272

