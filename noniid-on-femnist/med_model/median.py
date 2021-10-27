import numpy as np

a = np.array([
                [
                [1, 2],
                [3, 4]
                ],
                [
                [5, 6],
                [7, 8]
                ],
                [
                [9, 10],
                [11, 12]
                ],
            ])
a_reshape = np.reshape(a, (3, -1))
med = np.median(a_reshape, axis=0)
print(a_reshape)
print(med)