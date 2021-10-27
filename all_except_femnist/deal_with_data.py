import numpy as np
#把covertype数据集按类型分到不同文件里

#记录下最短的
def main():
    #导入数据
    train_img = np.load('./data/train_img1.npy')
    train_lbl = np.load('./data/train_lbl1.npy')
    #分训练集
    train_img_list = [[]for i in range(7)]
    one_label_list = [[]for i in range(7)]
    for i in range(len(train_img)):
        temp = train_img[i]
        label = train_lbl[i]
        train_img_list[label].append(temp)
        one_label = []
        for j in range(7):
            if j == label:
                one_label.append(1)
            else:
                one_label.append(0)
        one_label_list[label].append(one_label)

    for i in range(7):
        if len(train_img_list[i]) > 5000:
            a = np.array(train_img_list[i][0:5000])
            s = './data/3/train_img' + str(i) + '.npy'
            np.save(s, a)
            s = './data/3/one_train_lbl' + str(i) + '.npy'
            a = np.array(one_label_list[i][0:5000])
            np.save(s, a)
        else:
            a = np.array(train_img_list[i])
            s = './data/3/train_img' + str(i) + '.npy'
            np.save(s, a)
            s = './data/3/one_train_lbl' + str(i) + '.npy'
            a = np.array(one_label_list[i])
            np.save(s, a)



main()


