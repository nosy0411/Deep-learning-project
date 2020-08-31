# Unsupervised Learning 예시 (2층 신경망)
# Competitive Learning (경쟁학습)에 의한 Clustering 예제
#
# 절차 요약 : 은닉층을 추가하고 단층 신경망과 동일한 과정을 2 번 수행한다.
# 1. 입력층 ~ 은닉층 학습을 단층 신경망과 동일하게 수행한다 (은닉층 <-- 출력층)
# 2. 은닉층 ~ 출력층 학습도 단층 신경망과 동일하게 수행한다 (은닉층 <-- 입력층)
#
# 2018.8.29, 아마추어퀀트 (조성현)
# ----------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm

nInput = 2
nOutput = 3
nHidden = 3
ALPHA = 0.1

# 학습용 데이터를 생생한다. 3개의 정규분포로부터 3부류의 데이터를 생성한다.
def createData(n):
    xy = []
    for i in range(n):
        r = np.random.random()
        if r < 0.33:
            x = np.random.normal(0.0, 0.9)
            y = np.random.normal(0.0, 0.9)
        elif r < 0.66:
            x = np.random.normal(1.0, 0.3)
            y = np.random.normal(2.0, 0.3)
        else:
            x = np.random.normal(3.0, 0.6)
            y = np.random.normal(1.0, 0.6)
        xy.append([x, y])
    
    return pd.DataFrame(xy, columns=['x', 'y'])

# Winner neuron을 찾는다.
# 방법 1 : w를 normalization한 후 output이 가장 큰 뉴런을 찾는다.
# 방법 2 : input x와 w와의 거리가 가장 짧은 뉴런을 찾는다. <-- 이 방법을 적용함.
#          K-Means의 중점과의 거리 계산과 유사한 개념임.
def findWinner(w, x):
    distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(w, tf.transpose(x))), 1))
    winner = tf.argmin(distance, axis=0)
    return tf.slice(distance, [winner], [1]), winner

# Winner neuron의 W를 업데이트한다 (Hebbian learning)
# Hebb's rule : W = W + alpha * (a - W)
def updateWeights(w, winner, x):
    subW = tf.gather(w, winner)
    updW = tf.add(subW, tf.multiply(ALPHA, tf.subtract(tf.transpose(x), subW)))
    return tf.scatter_update(w, [winner], updW)

# Winner-takes-all
# Winner neuron의 출력값만 '1'로 설정하고 나머지 neuron의 출력값은 '0'으로 설정한다.
# K-Means의 중점 할당 (assignment)과 유사한 개념임.
def winner_take_all(w, x, n):
    _, winner = findWinner(w, x)
    r = tf.one_hot(winner, n)
    return r, tf.argmax(r, 0)

# input data를 생성한다
n = 1000
inputXY = np.array(createData(n)).T

# 그래프를 생성한다
tf.reset_default_graph()
xi = tf.placeholder(tf.float32, shape=[nInput, None], name='xi')
xh = tf.placeholder(tf.float32, shape=[nHidden, None], name='xh')
Wh = tf.Variable(tf.random_uniform([nHidden, nInput], 0, 1), dtype=tf.float32, name='Wh')
Wo = tf.Variable(tf.random_uniform([nOutput, nHidden], 0, 1), dtype=tf.float32, name='Wo')

# Hidden layer
distIWh, winH = findWinner(Wh, xi)
updateHidden = updateWeights(Wh, winH, xi)
outH, winnerH = winner_take_all(Wh, xi, nHidden)

# Output Layer
distIWo, winO = findWinner(Wo, xh)
updateOutput = updateWeights(Wo, winO, xh)
outO, winnerO = winner_take_all(Wo, xh, nOutput)

# 그래프를 실행한다
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# trainX를 한 개씩 입력하면서 5회 반복 학습한다.
for i in range(5):
    for k in range(n):
        dx = inputXY[:, k]
        dx = dx.reshape([nInput, 1])
        _, outputHidden = sess.run([updateHidden, outH], feed_dict={xi: dx})
        sess.run(updateOutput, feed_dict={xh: np.reshape(outputHidden, [nHidden,1])})
    print("%d done" % (i+1))

# 학습이 완료되었으므로, trainX를 한 개씩 입력하면서 clust를 결정한다.
clust = []
for k in range(n):
    dx = inputXY[:, k]
    dx = dx.reshape([nInput, 1])
    outputHidden = sess.run(outH, feed_dict={xi: dx})
    cluster = sess.run(winnerO, feed_dict={xh: np.reshape(outputHidden, [nHidden,1])})
    clust.append(cluster)

# 학습이 완료된 weight = centroid
centXY = sess.run(Wh)
sess.close()

# 분류 결과를 표시한다
clust = np.array(clust)
dataXY = inputXY.T
color = cm.rainbow(np.linspace(0, 1, nOutput))
plt.figure(figsize=(8, 6))
for i, c in zip(range(nOutput),color):
    plt.scatter(dataXY[clust == i, 0], dataXY[clust == i, 1], s=20, c=c, marker='o', alpha=0.5, label='cluster ' + str(i))
plt.scatter(centXY[:, 0], centXY[:, 1], s=250, marker='*', c='black', label='centroids')
plt.title("Competitive Learning")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
