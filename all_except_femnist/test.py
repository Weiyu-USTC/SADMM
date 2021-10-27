import numpy as np
# import random
import matplotlib.pyplot as plt

# file1 = './result/RSGD/fault/q8/lam0.5_wei0.01_alpha0.0001_sqrt_2/theta0_li_diff.npy'
# file2 = './result/RSGD/fault/q8/lam0.5_wei0.01_alpha0.0001_sqrt_2/theta_li_diff.npy'
# theta0 = np.load(file1)
# theta_li = np.load(file2)
# num_iter = theta0.shape[0]
#
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.semilogy(np.arange(num_iter), theta_li[:, 1], color='green', label='theta2_diff')
# plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.2), fontsize=14)
# plt.xlabel('iter', fontsize=14)
# plt.ylabel('log||theta2(k)-theta2(k-1)||', fontsize=14)
# plt.show()


########### grad norm compare ###############
# mean_file_no = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/wei0.01_alpha0.0001_sqrt_(2)/grad_norm.npy'
# rsgd_file_no = 'G:\python_code/byzantine/RSGD_multiLR/result\RSGD/no_fault/lam0.1_wei0.01_alpha0.002_sqrt/grad_norm.npy'
# rsgd_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt_(2)/grad_norm.npy'
#
# mean_no = np.load(mean_file_no)
# rsgd_no = np.load(rsgd_file_no)
# rsgd_q8 = np.load(rsgd_file_q8)
#
# print mean_no.shape
# print rsgd_no.shape
#
# num_iter = mean_no.shape[0]
# fontsize = 18
# mark = 50
# size = 8
#
# plt.semilogy(np.arange(num_iter), mean_no, color='blue', linewidth='1.0', label='Mean_without_Byzantine')
# plt.semilogy(np.arange(num_iter), rsgd_no, color='red', linewidth='1.0', label='RSGD_no_fault')
# plt.semilogy(np.arange(num_iter), rsgd_q8, color='green', linewidth='1.0', label='RSGD_q=8')
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.legend(loc='lower right', bbox_to_anchor=(1.01, 0.7), fontsize=fontsize)
# plt.xlabel('iter', fontsize=fontsize)
# plt.ylabel('log||grad||', fontsize=fontsize)
# plt.grid()
# plt.show()

# ############# time  compare same attack q8 #############
mean_no_file = './result/SGD_mean/wei0.01_alpha0.0001_sqrt(time)4/time_li.npy'
rsgd_q8_file = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt(time)4/time_li.npy'
krum_q8_file = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q8/alpha0.001_sqrt_wei0.01(time)2/time_li.npy'
geo_q8_file = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q8/alpha0.00005_wei0.01_sqrt(time)2/time_li.npy'
med_q8_file = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q8/alpha0.00001_wei0.01_sqrt(time)2/time_li.npy'
#
acc_mean_file = './result/SGD_mean/wei0.01_alpha0.0001_sqrt(time)3/acc.npy'
acc_rsgd_file = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt(time)4/acc.npy'
acc_krum_file = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q8/alpha0.001_sqrt_wei0.01(time)2/acc_li.npy'
acc_geo_file = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q8/alpha0.00005_wei0.01_sqrt(time)2/acc_li.npy'
acc_med_file = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q8/alpha0.00001_wei0.01_sqrt(time)2/acc_li.npy'


# ##############  compare time sign4 q8
# mean_no_file = './result/SGD_mean/wei0.01_alpha0.0001_sqrt(time)4/time_li.npy'
# rsgd_q8_file = './result/RSGD/fault/sign_attack2/4/q8/lam0.01_wei0.01_alpha0.0003_sqrt_time/time_li.npy'
# # rsgd_q8_file = './result/RSGD/fault/L2/sign_attack2/4/q8/lam0.01_wei0.01_alpha0.0003_sqrt_time/time_li.npy'
# # krum_q8_file = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.001_sqrt_grad4_wei0.01(time)2/time_li.npy'
# krum_q8_file = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.0001_sqrt_grad4_wei0.01(time)/time_li.npy'
# geo_q8_file = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q8/alpha0.0001_wei0.01_sqrt_time(2)/time_li.npy'
# med_q8_file = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q8/alpha0.00005_wei0.01_sqrt(time)2/time_li.npy'
#
# acc_mean_file = './result/SGD_mean/wei0.01_alpha0.0001_sqrt(time)3/acc.npy'
# acc_rsgd_file = './result/RSGD/fault/sign_attack2/4/q8/lam0.01_wei0.01_alpha0.0003_sqrt_time/acc.npy'
# # acc_rsgd_file = './result/RSGD/fault/L2/sign_attack2/4/q8/lam0.01_wei0.01_alpha0.0003_sqrt_time/acc.npy'
# # acc_krum_file = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.001_sqrt_grad4_wei0.01(time)2/acc_li.npy'
# acc_krum_file = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.0001_sqrt_grad4_wei0.01(time)/acc_li.npy'
# acc_geo_file = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q8/alpha0.0001_wei0.01_sqrt_time(2)/acc_li.npy'
# acc_med_file = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q8/alpha0.00005_wei0.01_sqrt(time)2/acc_li.npy'
#

############ q8 sign1
# mean_no_file = './result/SGD_mean/wei0.01_alpha0.0001_sqrt(time)4/time_li.npy'
# rsgd_q8_file = './result/RSGD/fault/sign_attack2/1/q8/lam0.01_wei0.01_alpha0.0003_sqrt_time/time_li.npy'
# # krum_file_q8  = './result/RSGD/fault/sign_attack2/1/q8/lam0.03_wei0.01_alpha0.0005_sqrt/acc.npy'
# krum_q8_file = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.001_wei0.01_sqrt_time/time_li.npy'
# med_q8_file = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt_time/time_li.npy'
# geo_q8_file = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt_time/time_li.npy'
# # mean_q8_file = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q8/wei0.01_alpha0.0001_sqrt/acc.npy'
#
# acc_mean_file = './result/SGD_mean/wei0.01_alpha0.0001_sqrt(time)3/acc.npy'
# acc_rsgd_file = './result/RSGD/fault/sign_attack2/1/q8/lam0.01_wei0.01_alpha0.0003_sqrt_time/acc.npy'
# # krum_file_q8  = './result/RSGD/fault/sign_attack2/1/q8/lam0.03_wei0.01_alpha0.0005_sqrt/acc.npy'
# acc_krum_file = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.001_wei0.01_sqrt_time/acc_li.npy'
# acc_med_file = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt_time/acc_li.npy'
# acc_geo_file = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt_time/acc_li.npy'
# # mean_q8_file = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q8/wei0.01_alpha0.0001_sqrt/acc.npy'

mean_no = np.load(mean_no_file)
rsgd_q8 = np.load(rsgd_q8_file)
krum_q8 = np.load(krum_q8_file)
geo_q8 = np.load(geo_q8_file)
med_q8 = np.load(med_q8_file)
#
mean_acc = np.load(acc_mean_file)
rsgd_acc = np.load(acc_rsgd_file)
krum_acc = np.load(acc_krum_file)
geo_acc = np.load(acc_geo_file)
med_acc = np.load(acc_med_file)

# print "mean_no:", mean_no[-1]
# print "RSGD:", rsgd_q8[-1]
# print "krum:", krum_q8[-1]
# print "Median:", med_q8[-1]
# print "Geometric:", geo_q8[-1]

# print "sgd:", mean_no[-1]
# print "rsa:", rsgd_q8[-1]
# print "krum:", krum_q8[-1]
# print "geo:", geo_q8[-1]
# print "med:", med_q8[-1]
# fontsize = 18
# leg_size = 22
# size = 10
# plt.plot(mean_no, mean_acc, 'ks-', linewidth='2', label='Ideal SGD', markevery=100, markersize=size)
# plt.plot(geo_q8, geo_acc, 'bd-', linewidth='2', label='GeoMed', markevery=100, markersize=size)
# plt.plot(krum_q8, krum_acc, 'mv-', linewidth='2', label='Krum', markevery=100, markersize=size)
# plt.plot(med_q8, med_acc, 'g>--', linewidth='2', label='Median', markevery=100, markersize=size)
# plt.plot(rsgd_q8, rsgd_acc, 'ro-', linewidth='2', label='RSA', markevery=100, markersize=size)
# #
# plt.legend(loc='lower right', bbox_to_anchor=(1.01, -0.0), fontsize=leg_size)
# plt.xticks(fontsize=fontsize)
# plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95], fontsize=fontsize)
# plt.ylim(0.70, 0.95)
# plt.xlabel('Computation Time (seconds)', fontsize=fontsize)
# plt.ylabel('Top-1 Accuracy', fontsize=fontsize)
# plt.grid()
# plt.show()


import cProfile
import pstats
s1 = 'wei0.01_alpha0.0001_sqrt(time)(test4)'
p = pstats.Stats("./result/SGD_mean/" + s1 + "/result_profile.out")
p.strip_dirs().sort_stats('tottime').print_stats(0.2)

s2 = 'lam0.07_wei0.01_alpha0.001_sqrt_time(test4)'
p = pstats.Stats("./result/RSGD/fault/same_attack/q8/" + s2 + "/result_profile.out")
p.strip_dirs().sort_stats('tottime').print_stats(0.2)

# f1 = './result/RSGD/step_k/x_i_norm/no_fault/lam0.1_wei0.01_alpha0.005_step/acc.npy'
# f2 = './result/RSGD/step_k/x_i_norm/no_fault/lam0.1_wei0.01_alpha0.005_sqrt/acc.npy'
#
# step = np.load(f1)
# sqrt = np.load(f2)
#
# num_iter = step.shape[0]
# plt.plot(np.arange(num_iter) * 10, step, color='red', linewidth='2', label=r"$ \alpha^k=0.005/k $", marker='H', markevery=100, markersize=size)
# # plt.plot(np.arange(num_iter) * 10, sqrt, color='blue', linewidth='2', label=r"$ \alpha^k=0.005/\sqrt{k}$", marker='H', markevery=100, markersize=size)
# plt.legend(loc='lower right', bbox_to_anchor=(1.01, -0.0), fontsize=leg_size)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.xlabel('iter', fontsize=fontsize)
# plt.ylabel('accuracy', fontsize=fontsize)
# plt.grid()
# plt.show()


# def huber_loss_grad(e, d):
#
#     t = (np.abs(e) <= d) * e
#     e[np.abs(e) <= d] = 0
#     grad = t + d * np.sign(e)
#     return grad
#
# a = [[-4, -2], [2, 3]]
# a = np.array(a)
# print "a:", a
# b = huber_loss_grad(a, 2)
# print "grad:", b

############### calculate data distribution  #############
# train_lbl = np.load('./data/mnist/train_lbl.npy')  # shape(60000,)
# print train_lbl.shape
# num_tr = 60000
# num_ma = 20
# sam_per = num_tr / num_ma
# print sam_per
# for i in range(num_ma):
#     y = train_lbl[i * sam_per:(i + 1)*sam_per]
#     key = np.unique(y)
#     result = {}
#     for k in key:
#         mask = (y == k)
#         y_new = y[mask]
#         v = y_new.size
#         result[k] = v
#     print "machine %d: " %(i), result


#################  count l1-norm of the x* #############
# f = './result/GD/wei0.01_alpha0.00001/theta_li.npy'
# # f = './result/GD/wei0.01_alpha0.00001/grad_norm.npy'
# theta = np.load(f)
# # print theta
# # print np.linalg.norm(theta)
# # print np.max(np.abs(theta))
# x_star = theta[-1]
# x_abs = np.sum(np.abs(x_star))
# print x_abs
# #|x*| = 9.24004181831
# # |x*| = 3.8757995999