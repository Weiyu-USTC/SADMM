import  numpy as np
from scipy import  io

def file2list(filename):
    """read data from txt file"""
    fr = open(filename)
    arrayOLines = fr.readlines()
    returnMat = []
    for line in arrayOLines:
        line = line.strip()
        returnMat.append(float(line))
    return returnMat

# krum_file_q10 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q10/alpha0.0001_sqrt_wei0.01_4/acc_li.txt'
# krum_10 = file2list(krum_file_q10)
# krum_10 = np.array(krum_10)
# np.save('G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q10/alpha0.0001_sqrt_wei0.01_4/acc_li.npy', krum_10)

###### no fault
# file_mean = './result/SGD_mean/wei0.01_alpha0.0001_sqrt/acc.npy'
# RSGD_file_no = './result/RSGD/no_fault/lam0.1_wei0.01_alpha0.003_sqrt/acc.npy'
# krum_file_no = 'G:\python_code/byzantine/Krum/result/mnist/machine20/no_fault/alpha0.00001_wei0.01/acc_li.npy'
# med_file_no = 'G:\python_code/byzantine/Median_LR/result/machine20/no_fault/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# geo_file_no = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/no_fault/alpha0.0001_wei0.01_sqrt/acc_li.npy'
#
# mean = np.load(file_mean)
# RSGD_no = np.load(RSGD_file_no)
# krum_no = np.load(krum_file_no)
# med_no = np.load(med_file_no)
# geo_no = np.load(geo_file_no)
#
# io.savemat('./result/SGD_mean/wei0.01_alpha0.0001_sqrt/acc.mat', {'acc': mean})
# io.savemat('./result/RSGD/no_fault/lam0.1_wei0.01_alpha0.005_sqrt/acc.mat', {'acc': RSGD_no})
# io.savemat('G:\python_code/byzantine/Krum/result/mnist/machine20/no_fault/alpha0.00001_wei0.01/acc_li.mat', {'acc': krum_no})
# io.savemat('G:\python_code/byzantine/Median_LR/result/machine20/no_fault/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': med_no})
# io.savemat('G:\python_code/byzantine/GeoMedian/result/LR/machine20/no_fault/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': geo_no})

########################################### q8 same
# RSGD_file_q8 = './result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q8/alpha0.001_sqrt_wei0.01/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q8/alpha0.00001_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
#
# RSGD_q8 = np.load(RSGD_file_q8)
# krum_q8 = np.load(krum_file_q8)
# med_q8 = np.load(med_file_q8)
# geo_q8 = np.load(geo_file_q8)
#
# io.savemat('./result/RSGD/fault/same_attack/q8/lam0.07_wei0.01_alpha0.001_sqrt/acc.mat', {'acc': RSGD_q8})
# io.savemat('G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q8/alpha0.001_sqrt_wei0.01/acc_li.mat', {'acc': krum_q8})
# io.savemat('G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q8/alpha0.00001_wei0.01_sqrt/acc_li.mat', {'acc': med_q8})
# io.savemat('G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q8/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': geo_q8})

#################### q10 same
# RSGD_file_q10 = './result/RSGD/fault/same_attack/q10/lam0.07_wei0.01_alpha0.001_sqrt/acc.npy'
# krum_file_q10 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q10/alpha0.001_sqrt_wei0.01/acc_li.npy'
# med_file_q10 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# geo_file_q10 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
#
# RSGD_q10 = np.load(RSGD_file_q10)
# krum_q10 = np.load(krum_file_q10)
# med_q10 = np.load(med_file_q10)
# geo_q10 = np.load(geo_file_q10)
#
# io.savemat('./result/RSGD/fault/same_attack/q10/lam0.07_wei0.01_alpha0.001_sqrt/acc.mat', {'acc': RSGD_q10})
# io.savemat('G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q10/alpha0.001_sqrt_wei0.01/acc_li.mat', {'acc': krum_q10})
# io.savemat('G:\python_code/byzantine/Median_LR/result/machine20/fault/same_attack/q10/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': med_q10})
# io.savemat('G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/same_attack/q10/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': geo_q10})

############## q8 sign1
# RSGD_file_q8 = './result/RSGD/fault/sign_attack/q8/lam0.01_wei0.01_alpha0.005_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.001_sqrt_wei0.01_(2)/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt/acc_li.npy'
#
# RSGD_q8 = np.load(RSGD_file_q8)
# krum_q8 = np.load(krum_file_q8)
# med_q8 = np.load(med_file_q8)
# geo_q8 = np.load(geo_file_q8)
#
# io.savemat('./result/RSGD/fault/sign_attack/q8/lam0.01_wei0.01_alpha0.005_sqrt/acc.mat', {'acc': RSGD_q8})
# io.savemat('G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.001_sqrt_wei0.01_(2)/acc_li.mat', {'acc': krum_q8})
# io.savemat('G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt/acc_li.mat', {'acc': med_q8})
# io.savemat('G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q8/alpha0.00005_wei0.01_sqrt/acc_li.mat', {'acc': geo_q8})

################## sign4   q8
# RSGD_file_q8 = './result/RSGD/fault/sign_attack/q8/lam0.01_wei0.01_alpha0.005_sqrt/acc.npy'
# krum_file_q8 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.0001_sqrt_wei0.01_4/acc_li.npy'
# med_file_q8 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q8/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q8 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q8/alpha0.0001_wei0.01_sqrt/acc_li.npy'
#
# RSGD_q8 = np.load(RSGD_file_q8)
# krum_q8 = np.load(krum_file_q8)
# med_q8 = np.load(med_file_q8)
# geo_q8 = np.load(geo_file_q8)

# io.savemat('./result/RSGD/fault/sign_attack/q8/lam0.01_wei0.01_alpha0.005_sqrt/acc.mat', {'acc': RSGD_q8})
# io.savemat('G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q8/alpha0.0001_sqrt_wei0.01_4/acc_li.mat', {'acc': krum_q8})
# io.savemat('G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q8/alpha0.00005_wei0.01_sqrt/acc_li.mat', {'acc': med_q8})
# io.savemat('G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q8/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': geo_q8})

#####################  sign1 q10
# RSGD_file_q10 = './result/RSGD/fault/sign_attack/q10/lam0.01_wei0.01_alpha0.005_sqrt/acc.npy'
# krum_file_q10 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q10/alpha0.0001_sqrt_wei0.01/acc_li.npy'
# med_file_q10 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q10/alpha0.00005_wei0.01_sqrt/acc_li.npy'
# geo_file_q10 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
#
# RSGD_q10 = np.load(RSGD_file_q10)
# krum_q10 = np.load(krum_file_q10)
# med_q10 = np.load(med_file_q10)
# geo_q10 = np.load(geo_file_q10)
#
# io.savemat('./result/RSGD/fault/sign_attack/q10/lam0.01_wei0.01_alpha0.005_sqrt/acc.mat', {'acc': RSGD_q10})
# io.savemat('G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q10/alpha0.0001_sqrt_wei0.01/acc_li.mat', {'acc': krum_q10})
# io.savemat('G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/1/q10/alpha0.00005_wei0.01_sqrt/acc_li.mat', {'acc': med_q10})
# io.savemat('G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/1/q10/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': geo_q10})

#####################  sign4 q10
# RSGD_file_q10 = './result/RSGD/fault/sign_attack/q10/lam0.07_wei0.01_alpha0.005_sqrt_4/acc.npy'
# krum_file_q10 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q10/alpha0.0001_sqrt_wei0.01_4/acc_li.npy'
# med_file_q10 = 'G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
# geo_file_q10 = 'G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q10/alpha0.0001_wei0.01_sqrt/acc_li.npy'
#
# RSGD_q10 = np.load(RSGD_file_q10)
# krum_q10 = np.load(krum_file_q10)
# med_q10 = np.load(med_file_q10)
# geo_q10 = np.load(geo_file_q10)
#
# io.savemat('./result/RSGD/fault/sign_attack/q10/lam0.07_wei0.01_alpha0.005_sqrt_4/acc.mat', {'acc': RSGD_q10})
# io.savemat('G:\python_code/byzantine/Krum/result/mnist/machine20/fault/sign_attack/q10/alpha0.0001_sqrt_wei0.01_4/acc_li.mat', {'acc': krum_q10})
# io.savemat('G:\python_code/byzantine/Median_LR/result/machine20/fault/sign_attack/4/q10/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': med_q10})
# io.savemat('G:\python_code/byzantine/GeoMedian/result/LR/machine20/fault/sign_attack/4/q10/alpha0.0001_wei0.01_sqrt/acc_li.mat', {'acc': geo_q10})

########### compare diff norm #################


