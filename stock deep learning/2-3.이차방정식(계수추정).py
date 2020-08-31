# x, y 데이터 세트가 있을 때, 이차 방정식 y = w1x^2 + w2x + b를 만족하는
# parameter w1, w2, b를 추정한다.
# -------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# y = 2x^2 + 3x + 5 일 때 dataX, dataY 집합을 생성한다
dataX = list(np.arange(-5, 5, 0.5))
dataY = []
for i in dataX:
    dataY.append(2 * i * i + 3 * i + 5)

#####################################################
# dataX, dataY를 만족하는 w1, w2, b를 찾는다.
# y = w1x^2 + w2x + b
# w1 = 2, w2 = 3, b = 5가 나와야 한다.
    
# 그래프를 생성한다.
tf.reset_default_graph()    # Default 그래프 영역을 리셋한다
w1 = tf.Variable(0.0, dtype=tf.float32, name = "w1")
w2 = tf.Variable(0.0, dtype=tf.float32, name = "w2")
b = tf.Variable(0.0, dtype=tf.float32, name = "bias")
x = tf.placeholder(tf.float32, name = "X")
y = tf.placeholder(tf.float32, name = "Y")

# Cost 함수를 정의하고, Optimizer는 Adaptive Moment (Adam)를 사용한다.
# learning rate = 0.05
cost = tf.reduce_sum(tf.square(w1 * x * x + w2 * x + b - y))
optimizer = tf.train.AdamOptimizer(0.05).minimize(cost)

# 세션을 오픈하고 그래프를 실행한다 (fitting)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

trCost = []
for i in range(1000):
  _, rcost = sess.run([optimizer, cost], { x: dataX, y: dataY })
  trCost.append(rcost)
  
  if i % 20 == 0:
      rw1, rw2, rb, rcost = sess.run([w1, w2, b, cost], { x: dataX, y: dataY })
      print("%d) w1 = %.4f, w2 = %.4f, b = %.4f, cost = %.4f" % (i, rw1, rw2, rb, rcost))

# 실행 결과를 확인한다.
rw1, rw2, rb, rcost = sess.run([w1, w2, b, cost], { x: dataX, y: dataY })
sess.close()

print("\n추정 결과 :")
print("w1 = %.2f" % rw1)
print("w2 = %.2f" % rw2)
print("b = %.2f" % rb)
print("cost = %.4f" % rcost)

plt.plot(trCost[100:], color='red', linewidth=1)
plt.title("Cost function")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()