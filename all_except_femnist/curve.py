import numpy as np
# import random
import matplotlib.pyplot as plt

def file2list(filename):
    """read data from txt file"""
    fr = open(filename)
    arrayOLines = fr.readlines()
    returnMat = []
    for line in arrayOLines:
        line = line.strip()
        returnMat.append(float(line))
    return returnMat

# file = './result/SGD/wei0.01_alpha0.00001/grad_norm.npy'
# # x_star_norm = file2list(file)
# x_star_norm = np.load(file)
# num_iter = x_star_norm.shape[0]
# plt.plot(np.arange(num_iter), x_star_norm)
# plt.xlabel('iter')
# plt.ylabel('||grad||')
# plt.title('wei0.01_alpha0.00001')
# plt.savefig('./result/SGD/wei0.01_alpha0.00001/grad_norm2.jpg')
# plt.show()

##############   compare norm acc q8 same attack
file_mean = './result/SGD_mean/wei0.01_alpha0.0001_sqrt/acc.npy'
# l1_q8_file = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# l2_q8_file = './result/RSGD/fault/L2/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# max_norm_q8_file = './result/RSGD/fault/max_norm/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# huber_q8_file = './result/RSGD/huber_loss/fault/same/q8/delta0.01/lam0.07_wei0.01_alpha0.005_sqrt/acc.npy'

# #### compare  norm no_fault  #############
# L1_no_file = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.003_sqrt/acc.npy'
# L2_no_file = './result/RSGD/no_fault/L2/lam1.2_wei0.01_alpha0.001_sqrt/acc.npy'
# max_norm_no_file = './result/RSGD/no_fault/max_norm/lam51_wei0.01_alpha0.00005_sqrt/acc.npy'
# huber_no_file = './result/RSGD/huber_loss/no_fault/delta0.1/lam0.1_wei0.01_alpha0.05_sqrt/acc.npy'

########### no fault ###############
# RSGD_file_q8 = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.003_sqrt/acc.npy'
# # RSGD_file_no2 = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.005_sqrt_(2)/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/no_fault/alpha0.00001_wei0.01/acc_li.txt'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/no_fault/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/no_fault/alpha0.0001_wei0.01_sqrt/acc_li.npy'

# s = np.load(file_mean)
# r = np.load(RSGD_file_q8)
# m = np.load(med_file_q8)
# g = np.load(geo_file_q8)
#
# np.savetxt('./result/SGD_mean/wei0.01_alpha0.0001_sqrt/acc.txt', s)
# np.savetxt('./result/RSGD/no_fault/lam0.1_wei0.01_alpha0.003_sqrt/acc.txt', r)
# np.savetxt('G:\python_code/byzantine/Median_LR/result/machine20/no_fault/alpha0.0001_wei0.01_sqrt/acc.txt', m)
# np.savetxt('G:\python_code/byzantine/GeoMedian/result/LR/machine20/no_fault/alpha0.0001_wei0.01_sqrt/acc.txt', g)
# print 'over'

################### same attack q8 ################
RSGD_file_q8 = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q8/alpha0.001_sqrt_wei0.01/acc_li.txt'
med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q8/alpha0.00001_wei0.01_sqrt/acc_li.npy'
geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/q8/wei0.01_alpha0.0001_sqrt/acc.npy'

################### q10 same
# RSGD_file_q8 = './result/RSGD/fault/same_attack/q10/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q10/alpha0.001_sqrt_wei0.01/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/q10/wei0.01_alpha0.0001_sqrt/acc.npy'

# ############# q8 sign1
# RSGD_file_q8 = './result/RSGD/fault/sign_attack/q8/lam0.01_wei0.01_alpha0.005_sqrt/acc.npy'
# RSGD_file_q8 = './result/RSGD/fault/sign_attack2/1/q8/lam0.01_wei0.01_alpha0.0003_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.001_sqrt_wei0.01_(2)/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q8/wei0.01_alpha0.0001_sqrt/acc.npy'

# ###########  q8 sign4
# RSGD_file_q8 = './result/RSGD/fault/sign_attack/q8/lam0.07_wei0.01_alpha0.005_sqrt_4/acc.npy'
# RSGD_file_q8 = './result/RSGD/fault/sign_attack2/4/q8/lam0.01_wei0.01_alpha0.0005_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.0001_sqrt_wei0.01_4/acc_li.txt'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q8/wei0.01_alpha0.0001_sqrt_grad4/acc.npy'

# ####################  sign1 q10
# RSGD_file_q8 = './result/RSGD/fault/sign_attack/q10/lam0.01_wei0.01_alpha0.005_sqrt/acc.npy'
# RSGD_file_q8 = './result/RSGD/fault/sign_attack2/1/q10/lam0.0046_wei0.01_alpha0.0005_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q10/alpha0.0001_sqrt_wei0.01/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q10/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q10/wei0.01_alpha0.0001_sqrt/acc.npy'

####################  sign4 q10
# RSGD_file_q8 = './result/RSGD/fault/sign_attack/q10/lam0.07_wei0.01_alpha0.005_sqrt_4/acc.npy'
# RSGD_file_q8 = './result/RSGD/fault/sign_attack2/4/q10/lam0.0046_wei0.01_alpha0.001_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q10/alpha0.0001_sqrt_wei0.01_4/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q10/wei0.01_alpha0.0001_sqrt_grad4/acc.npy'

################## same attack q4 ################
# RSGD_file_q8 = './result/RSGD/fault/same_attack/q4/lam0.01_wei0.01_alpha0.003_sqrt/acc.npy'
# # RSGD_file_q8 = './result/RSGD/fault/same_attack/q4/lam0.01_wei0.01_alpha0.003_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q4/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q4/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q4/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/q4/wei0.01_alpha0.0001_sqrt/acc.npy'

############# q4 sign1
# RSGD_file_q8 = './result/RSGD/fault/sign_attack2/1/q4/lam0.01_wei0.01_alpha0.0005_sqrt/acc.npy'
# # krum_file_q8  = './result/RSGD/fault/sign_attack2/1/q4/lam0.07_wei0.01_alpha0.0001_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q4/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q4/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q4/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q4/wei0.01_alpha0.0001_sqrt/acc.npy'

############# q4 sign4
# RSGD_file_q8 = './result/RSGD/fault/sign_attack2/4/q4/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# # krum_file_q8  = './result/RSGD/fault/sign_attack2/4/q4/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q4/alpha0.00005_wei0.01_sqrt_grad4/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q4/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q4/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q4/wei0.01_alpha0.0001_sqrt_grad4/acc.npy'


# ############ q8 sign1
# RSGD_file_q8 = './result/RSGD/fault/sign_attack2/1/q8/lam0.01_wei0.01_alpha0.0003_sqrt/acc.npy'
# # krum_file_q8  = './result/RSGD/fault/sign_attack2/1/q8/lam0.03_wei0.01_alpha0.0005_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.001_sqrt_wei0.01_(2)/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q8/wei0.01_alpha0.0001_sqrt/acc.npy'

# ###########  q8 sign4
# RSGD_file_q8 = './result/RSGD/fault/sign_attack2/4/q8/lam0.01_wei0.01_alpha0.0003_sqrt/acc.npy'
# krum_file_q8  = './result/RSGD/fault/sign_attack2/4/q8/lam0.01_wei0.01_alpha0.001_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.0001_sqrt_wei0.01_4/acc_li.txt'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# mean_file_q8 = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/sign_attack/q8/wei0.01_alpha0.0001_sqrt_grad4/acc.npy'
#
mean_no = np.load(file_mean)
# # L1_no = np.load(l1_q8_file)
# # L2_no = np.load(l2_q8_file)
# # max_no = np.load(max_norm_q8_file)
# # huber_no = np.load(huber_q8_file)
# # max_norm_no = np.load(max_norm_q8_file)
# # rsgd_no2 = np.load(RSGD_file_no2)
# # krum_no = file2list(krum_file_no)
# # med_no = np.load(med_file_no)
# # geo_no = np.load(geo_file_no)
rsgd_q8 = np.load(RSGD_file_q8)
krum_q8 = file2list(krum_file_q8)
# krum_q8 = np.load(krum_file_q8)
med_q8 = np.load(med_file_q8)
geo_q8 = np.load(geo_file_q8)
# mean_q8 = np.load(mean_file_q8)

# num_iter = mean_no.shape[0]
# line = 2
# fontsize = 18
# leg_size = 20
# mark = 50
# size = 10
# plt.plot(np.arange(num_iter)*10, mean_no, 'ks-', linewidth=1.5, label='Ideal SGD', markevery=50, markersize=size)
# plt.plot(np.arange(num_iter)*10, geo_q8, 'bd-', linewidth=1.5, label='GeoMed', markevery=30, markersize=size)
# plt.plot(np.arange(num_iter)*10, krum_q8, 'mv-', linewidth=line, label='Krum', markevery=60, markersize=size)
# plt.plot(np.arange(num_iter)*10, med_q8, 'g>--', linewidth=line, label='Median', markevery=40, markersize=size)
# plt.plot(np.arange(num_iter)*10, rsgd_q8, 'ro-', linewidth=1.5, label='RSA', markevery=70, markersize=size)
# plt.plot(np.arange(num_iter)*10, mean_q8, 'cp--', linewidth=3, label='SGD', markevery=85, markersize=size)

########### lambda=0.07  q8
# plt.plot(np.arange(num_iter)*10, L1_no, 'ks-', linewidth='2', label='L1_norm', markevery=60, markersize=size)
# plt.plot(np.arange(num_iter)*10, L2_no, 'ro-', linewidth='2', label='L2_norm', markevery=50, markersize=10)
# plt.plot(np.arange(num_iter)*10, max_no, 'mv-', linewidth='2', label='Max_norm', markevery=70, markersize=size)
# plt.plot(np.arange(num_iter)*10, huber_no, 'cd-', linewidth='2', label='huber_loss', markevery=40, markersize=size)
#
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.ylim(0.0, 1.0)
# plt.legend(loc='lower right', bbox_to_anchor=(1.01, -0.0), fontsize=leg_size)
# plt.xlabel('Number of Iterations', fontsize=fontsize)
# plt.ylabel('Top-1 Accuracy', fontsize=fontsize)
# plt.grid()
# plt.show()


# ################ compute var no fault diff norm
# rsa_no_file1 = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.003_sqrt_theta/var_li1.npy'
# rsa_no_file2 = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.003_sqrt_theta/var_li2.npy'
# rsa_var1 = np.load(rsa_no_file1)
# rsa_var2 = np.load(rsa_no_file2)
# rsa_var1 = rsa_var1.tolist()
# rsa_var2 = rsa_var2.tolist()
# rsa_var = rsa_var1 + rsa_var2
#
# l2_no_file1 = './result/RSGD/no_fault/L2/lam1.4_wei0.01_alpha0.001_sqrt_theta/var_li1.npy'
# l2_no_file2 = './result/RSGD/no_fault/L2/lam1.4_wei0.01_alpha0.001_sqrt_theta/var_li2.npy'
# l2_var1 = np.load(l2_no_file1).tolist()
# l2_var2 = np.load(l2_no_file2).tolist()
# l2_var = l2_var1 + l2_var2
#
# max_no_file1 = './result/RSGD/no_fault/max_norm/lam51_wei0.01_alpha0.0001_sqrt_theta2/var_li1.npy'
# max_no_file2 = './result/RSGD/no_fault/max_norm/lam51_wei0.01_alpha0.0001_sqrt_theta2/var_li2.npy'
# max_var1 = np.load(max_no_file1).tolist()
# max_var2 = np.load(max_no_file2).tolist()
# max_var = max_var1 + max_var2
#
# huber_no_file = './result/RSGD/huber_loss/no_fault/delta0.1/lam0.1_wei0.01_alpha0.05_sqrt/var_li.npy'
# huber_var = np.load(huber_no_file)

################# compute acc no fault diff norm
# file_mean = './result/SGD_mean/wei0.01_alpha0.0001_sqrt/acc.npy'
# rsa_no_file1 = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.001_sqrt_theta/acc.npy'
# l2_no_file1 = './result/RSGD/no_fault/L2/lam1.4_wei0.01_alpha0.001_sqrt_theta/acc.npy'
# max_no_file1 = './result/RSGD/no_fault/max_norm/lam51_wei0.01_alpha0.0001_sqrt_theta2/acc.npy'
# huber_no_file = './result/RSGD/huber_loss/no_fault/delta0.1/lam0.1_wei0.01_alpha0.05_sqrt/acc.npy'
#
# mean_acc = np.load(file_mean)
# rsa_acc = np.load(rsa_no_file1)
# l2_acc = np.load(l2_no_file1)
# max_acc = np.load(max_no_file1)
# huber_acc = np.load(huber_no_file)

############### compare norm var same attack q8
# file_mean = './result/SGD_mean/wei0.01_alpha0.0001_sqrt/acc.npy'
# rsa_q8_file1 = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt_theta/var_li1.npy'
# rsa_q8_file2 = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt_theta/var_li2.npy'
# rsa_q8_var1 = np.load(rsa_q8_file1).tolist()
# rsa_q8_var2 = np.load(rsa_q8_file2).tolist()
# rsa_var = rsa_q8_var1 + rsa_q8_var2
#
# l2_file = './result/RSGD/L2/fault/same_attack/q8/lam0.8_wei0.01_alpha0.003_sqrt/var_li.npy'
# l2_var = np.load(l2_file)
#
# max_file = './result/RSGD/fault/max_norm/same_attack/q8/lam20_wei0.01_alpha0.0001_sqrt/var_li.npy'
# max_var = np.load(max_file)
#
# huber_var_file = './result/RSGD/huber_loss/fault/same/q8/delta0.01/lam0.07_wei0.01_alpha0.005_sqrt/var_li.npy'
# huber_var = np.load(huber_var_file)
#
#
# # ################ compare norm acc same attack q=8
# file_mean = './result/SGD_mean/wei0.01_alpha0.0001_sqrt/acc.npy'
# rsa_file = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt_theta/acc.npy'
# # l2_file = './result/RSGD/L2/fault/same_attack/q8/lam1.2_wei0.01_alpha0.001_sqrt/acc.npy'
# l2_file = './result/RSGD/L2/fault/same_attack/q8/lam0.8_wei0.01_alpha0.003_sqrt/acc.npy'
# # max_file = './result/RSGD/fault/max_norm/same_attack/q8/lam50_wei0.01_alpha0.0001_sqrt/acc.npy'
# max_file = './result/RSGD/fault/max_norm/same_attack/q8/lam20_wei0.01_alpha0.0001_sqrt/acc.npy'
# huber_file = './result/RSGD/huber_loss/fault/same/q8/delta0.01/lam0.07_wei0.01_alpha0.005_sqrt/acc.npy'
# # #
# mean_acc = np.load(file_mean)
# rsa_acc = np.load(rsa_file)
# # l2_acc = np.load(l2_file)
# l2_acc = np.load(l2_file)
# # max_acc = np.load(max_file)
# max_acc = np.load(max_file)
# # huber_acc = np.load(huber_file)
# # #
# num_iter = len(rsa_var)
# # num_iter = mean_acc.shape[0]
# fontsize = 18
# leg_size = 20
# mark = 50
# size = 10
# line = 2
# #
# plt.plot(np.arange(num_iter)*10, mean_acc, 'bH-', linewidth=line, label='Ideal SGD', markevery=35, markersize=10)
# plt.plot(np.arange(num_iter)*10, rsa_acc, 'ks-', linewidth=line, label=r"$ \bf{\ell_1-norm} $", markevery=80, markersize=size)
# plt.plot(np.arange(num_iter)*10, l2_acc, 'ro-', linewidth=line, label=r"$ \bf{\ell_2-norm} $", markevery=60, markersize=size)
# plt.plot(np.arange(num_iter)*10, max_acc, 'mv-', linewidth=line, label=r"$ \bf{\ell_\infty-norm} $", markevery=50, markersize=size)
# plt.plot(np.arange(num_iter)*10, huber_acc, 'cd-', linewidth=line, label='huber_loss', markevery=75, markersize=size)
# plt.semilogy(np.arange(num_iter), rsa_var, 'ks-', linewidth=line, label=r"$ \bf{\ell_1-norm} $", markevery=500, markersize=size)
# plt.semilogy(np.arange(num_iter), l2_var, 'ro-', linewidth=line, label=r"$ \bf{\ell_2-norm} $", markevery=500, markersize=size)
# plt.semilogy(np.arange(num_iter), max_var, 'mv-', linewidth=line, label=r"$ \bf{\ell_\infty-norm} $", markevery=500, markersize=size)
# # plt.semilogy(np.arange(num_iter), huber_var, 'cd-', linewidth=line, label='huber_loss', markevery=500, markersize=size)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# # plt.ylim(0.0, 1.0)
# plt.legend(loc='lower right', bbox_to_anchor=(1.01, 0.7), fontsize=fontsize)
# plt.xlabel('Number of Iterations', fontsize=fontsize)
# plt.ylabel('Variance', fontsize=fontsize)
# # plt.ylabel('Top-1 Accuracy', fontsize=fontsize)
# plt.grid()
# plt.show()
#

##############  q8 same different lambda  ###############
# lam0_001_file = './result/RSGD/fault/same_attack/q8/lam0.001_wei0.01_alpha0.0001_sqrt/acc.npy'
lam0_001_file2 = './result/RSGD/fault/same_attack/q8_2/lam0.001_wei0.01_alpha0.001_sqrt/acc.npy'
lam0_01_file = './result/RSGD/fault/same_attack/q8_2/lam0.01_wei0.01_alpha0.0003_sqrt/acc.npy'
lam0_07_file = './result/RSGD/fault/same_attack/q8_2/lam0.07_wei0.01_alpha0.003_sqrt/acc.npy'
lam0_1_file = './result/RSGD/fault/same_attack/q8_2/lam0.1_wei0.01_alpha0.003_sqrt/acc.npy'
lam0_5_file = './result/RSGD/fault/same_attack/q8_2/lam0.5_wei0.01_alpha0.0003_sqrt/acc.npy'
lam0_8_file = './result/RSGD/fault/same_attack/q8_2/lam0.8_wei0.01_alpha0.0001_sqrt/acc.npy'
lam1_0_file = './result/RSGD/fault/same_attack/q8_2/lam1.0_wei0.01_alpha0.0001_sqrt/acc.npy'

####### -x*  same vaule attack q=8/10 compare lambda
# lam0_0001_file = './result/RSGD/fault/same_attack2/1/q10/lam0.0001_wei0.01_alpha0.001_sqrt/acc.npy'
# lam0_001_file = './result/RSGD/fault/same_attack2/1/q10/lam0.001_wei0.01_alpha0.0001_sqrt/acc.npy'
# lam0_0046_file = './result/RSGD/fault/same_attack2/1/q10/lam0.0046_wei0.01_alpha0.0001_sqrt/acc.npy'
# lam0_01_file = './result/RSGD/fault/same_attack2/1/q10/lam0.01_wei0.01_alpha0.0001_sqrt/acc.npy'
# lam0_03_file = './result/RSGD/fault/same_attack2/1/q10/lam0.03_wei0.01_alpha0.0001_sqrt/acc.npy'
# lam0_07_file = './result/RSGD/fault/same_attack2/1/q10/lam0.07_wei0.01_alpha0.0001_sqrt/acc.npy'
#
# lam0_0001 = np.load(lam0_0001_file)
# lam0_001_2 = np.load(lam0_001_file2)
lam0_001 = np.load(lam0_001_file2)
# lam0_0046 = np.load(lam0_0046_file)
lam0_01 = np.load(lam0_01_file)
# lam0_03 = np.load(lam0_03_file)
lam0_07 = np.load(lam0_07_file)
lam0_1 = np.load(lam0_1_file)
lam0_5 = np.load(lam0_5_file)
lam0_8 = np.load(lam0_8_file)
lam1_0 = np.load(lam1_0_file)
#
# point = []
# point.append(lam0_001[-1])
# point.append(lam0_01[-1])
# point.append(lam0_07[-1])
# point.append(lam0_1[-1])
# point.append(lam0_5[-1])
# point.append(lam0_8[-1])
# point.append(lam1_0[-1])
#
# print point
#
# lam = ['0.001', '0.01', '0.07', '0.1', '0.5', '0.8', '1.0']
# len_lam = [1, 2, 3, 4, 5, 6, 7]
# print len_lam
#
num_iter = lam0_001.shape[0]
fontsize = 18
leg_size = 20
mark = 50
size = 10
line = 2
#
# plt.plot(len_lam, point, 'o-', markersize=10)
# plt.xticks(len_lam, lam, fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# # plt.legend(loc='lower right', bbox_to_anchor=(1.01, 0.0), fontsize=fontsize)
# plt.xlabel(r"$ \lambda $ varies", fontsize=fontsize)
# plt.ylabel('Top-1 Accuracy', fontsize=fontsize)
# plt.grid()
# plt.show()


plt.plot(np.arange(num_iter)*10, lam0_001, 'ks-', linewidth=line, label=r"$ \lambda=0.001 $", markevery=35, markersize=10)
# plt.plot(np.arange(num_iter)*10, lam0_01, 'm^-', linewidth=line, label=r"$ \lambda=0.01 $", markevery=60, markersize=size)
plt.plot(np.arange(num_iter)*10, lam0_07, 'ro-', linewidth=line, label=r"$ \lambda=0.07 $", markevery=60, markersize=size)
plt.plot(np.arange(num_iter)*10, lam0_1, 'yH-', linewidth=line, label=r"$ \lambda=0.1 $", markevery=50, markersize=size)
# plt.plot(np.arange(num_iter)*10, lam0_5, 'g>-', linewidth=line, label=r"$ \lambda=0.5 $", markevery=50, markersize=size)
plt.plot(np.arange(num_iter)*10, lam0_8, 'cv-', linewidth=line, label=r"$ \lambda=0.8 $", markevery=50, markersize=size)
plt.plot(np.arange(num_iter)*10, lam1_0, 'b<-', linewidth=line, label=r"$ \lambda=1.0 $", markevery=50, markersize=size)
# plt.plot(np.arange(num_iter)*10, lam1_0, 'yH-', linewidth=line, label=r"$ \lambda=1.0 $", markevery=50, markersize=size)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc='lower right', bbox_to_anchor=(1.01, 0.0), fontsize=fontsize)
plt.xlabel('Number of Iterations', fontsize=fontsize)
plt.ylabel('Top-1 Accuracy', fontsize=fontsize)
plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95], fontsize=fontsize)
plt.ylim(0.70, 0.95)
# plt.title(r"$ q=10, \epsilon=-1, \beta=0.01 $", fontsize=18)
plt.grid()
plt.show()


############  compare q ###############
# no_f = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.003_sqrt/acc.npy'
# q1_f = './result/RSGD/fault/same_attack/q1/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# q3_f = './result/RSGD/fault/same_attack/q3/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# q8_f = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# q10_f = './result/RSGD/fault/same_attack/q10/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# q11_f = './result/RSGD/fault/same_attack/q11/lam0.001_wei0.01_alpha0.001_sqrt/acc.npy'
# q12_f = './result/RSGD/fault/same_attack/q12/lam0.001_wei0.01_alpha0.001_sqrt/acc.npy'
# q15_f = './result/RSGD/fault/same_attack/q15/lam0.001_wei0.01_alpha0.001_sqrt/acc.npy'
#
# no = np.load(no_f)
# q1 = np.load(q1_f)
# q3 = np.load(q3_f)
# q8 = np.load(q8_f)
# q10 = np.load(q10_f)
# q11 = np.load(q11_f)
# q12 = np.load(q12_f)
# q15 = np.load(q15_f)
#
# q_point = []
# q_point.append(no[-1])
# q_point.append(q1[-1])
# q_point.append(q3[-1])
# q_point.append(q8[-1])
# q_point.append(q10[-1])
# q_point.append(q11[-1])
# q_point.append(q12[-1])
# q_point.append(q15[-1])
#
# q = [1, 2, 3, 4, 5, 6, 7, 8]
# q_lab = ['0', '1', '3', '8', '10', '11', '12', '15']
#
# num_iter = no.shape[0]
# fontsize = 18
# leg_size = 20
# mark = 50
# size = 10
# line = 2
#
# plt.plot(q, q_point, 'bo-', linewidth=line, markersize=10)
#
# plt.xticks(q, q_lab, fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.legend(loc='lower right', bbox_to_anchor=(1.01, 0.0), fontsize=fontsize)
# plt.xlabel('Number of Byzantine Workers', fontsize=fontsize)
# plt.ylabel('Top-1 Accuracy', fontsize=fontsize)
# plt.grid()
# plt.show()

# ############### same attack4 same_digit distrit attack=worker_i's transmited value
# ############ q8
# sgd_f = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/same_attack4/q8/wei0.01_alpha0.00005_sqrt/acc.npy'
# rsa_f = './result/RSGD/fault/same_attack4/q8/lam0.5_wei0.01_alpha0.0005_sqrt/acc.npy'
# # geomed_f = './result/RSGD/fault/same_attack4/q8/lam0.5_wei0.01_alpha0.0005_sqrt/acc.npy'
# geomed_f = 'G:/python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack4/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# median_f = 'G:/python_code/byzantine/Median_LR/result/machine20/fault/same_attack4/q8/alpha0.000001_wei0.01_sqrt/acc_li.npy'
# krum_f = 'G:/python_code/byzantine/Krum/result/mnist/machine20/fault/same_attack4/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
#
# ########## q4
# # sgd_f = 'G:\python_code/byzantine/RSGD_multiLR/result/SGD_mean/fault/same_attack4/q4/wei0.01_alpha0.00005_sqrt/acc.npy'
# # rsa_f = './result/RSGD/fault/same_attack4/q4/lam0.5_wei0.01_alpha0.0005_sqrt/acc.npy'
# # # geomed_f = './result/RSGD/fault/same_attack4/q8/lam0.5_wei0.01_alpha0.001_sqrt/acc.npy'
# # geomed_f = 'G:/python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack4/q4/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# # median_f = 'G:/python_code/byzantine/Median_LR/result/machine20/fault/same_attack4/q4/alpha0.000001_wei0.01_sqrt/acc_li.npy'
# # krum_f = 'G:/python_code/byzantine/Krum/result/mnist/machine20/fault/same_attack4/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# sgd = np.load(sgd_f)
# rsa = np.load(rsa_f)
# geomed = np.load(geomed_f)
# median = np.load(median_f)
# krum = np.load(krum_f)
#
# num_iter = rsa.shape[0]
# fontsize = 18
# leg_size = 20
# mark = 50
# size = 10
# line = 2
# plt.plot(np.arange(num_iter)*10, geomed, 'bd-', linewidth=line, label='GeoMde', markevery=30, markersize=size)
# plt.plot(np.arange(num_iter)*10, krum, 'mv-', linewidth=line, label='Krum', markevery=50, markersize=size)
# plt.plot(np.arange(num_iter)*10, median, 'g>--', linewidth=line, label='Median', markevery=40, markersize=size)
# plt.plot(np.arange(num_iter)*10, rsa, 'ro-', linewidth=line, label='RSA', markevery=70, markersize=size)
# plt.plot(np.arange(num_iter)*10, sgd, 'cp--', linewidth=line, label='SGD', markevery=85, markersize=size)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.legend(loc='lower right', bbox_to_anchor=(1.01, 0.0), fontsize=fontsize)
# plt.xlabel('Number of Iterations', fontsize=fontsize)
# plt.ylabel('Top-1 Accuracy', fontsize=fontsize)
# # plt.ylim(0.0, 0.6)
# # plt.title(r"$ q=8,\beta=0.01 $", fontsize=18)
# plt.grid()
# plt.show()
