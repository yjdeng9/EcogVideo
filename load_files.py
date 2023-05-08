
# 使用scripy导入mat文件
import scipy.io as sio
import numpy as np
import os

mat_path = 'ecog/Walk.mat'
mat_contents = sio.loadmat(mat_path)

print(mat_contents.keys())

# for key, value in mat_contents.items():
#     print(key, type(value))
#     if type(value) == np.ndarray:
#         print(value.shape)
#     else:
#         print(value)

print(mat_contents['SR'])
print('---------------------------------')

m_data = mat_contents['y']
time_ = m_data[0,:]
ECOG = m_data[1:161,:]
DI = m_data[161,]
stimCode = m_data[162,]
GroupID = m_data[163,]

print(ECOG.shape)
print(DI.shape)
print(DI[50000:])

print([i for i in range(50000, 51000) if DI[i] == 0])
print([i for i in range(50000, 51000) if DI[i] == 1])

# 统计GroupID的频率
a,b= np.unique(time_, return_counts=True)
print(np.max(b))

import matplotlib.pyplot as plt
# plt.scatter(time_, DI)
# plt.plot(time_, DI)
# # plt.plot(time_, stimCode)
# # plt.plot(time_, GroupID)
# plt.legend(['DI','S', 'GroupID'])
# plt.xlabel('time')
# plt.show()


plt.figure(figsize=(15,4))
for i in range(160):
    plt.plot(time_, ECOG[i,:])
plt.xlabel('time')
plt.ylabel('ECOG_signal')
plt.tight_layout()
plt.show()

# #
# # print(ECOG.shape)
# # print(DI.shape)
#
#
# para_mat_path = 'ecog/Walk_paradigmInfo.mat'
# para_mat_contents = sio.loadmat(para_mat_path)
# print(para_mat_contents.keys())
#
# for key, value in para_mat_contents.items():
#     print(key, type(value))
#     if type(value) == np.ndarray:
#         print(value.shape)
#     else:
#         print(value)
