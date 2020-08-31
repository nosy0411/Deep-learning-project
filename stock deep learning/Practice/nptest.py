import numpy as np
x=np.r_[0:10]
y=np.c_[0:10]
print(x)
print(y)

N = 3
A = np.eye(N)
print('A = ', A)
B = np.c_[A, A[2]]
print('B = ', B)
C = np.r_[A, [A[2]]]
print('C = ', C)

a = np.array([1,2,3])
print('a = ', a)
print('np.c_ = ', np.c_[a])