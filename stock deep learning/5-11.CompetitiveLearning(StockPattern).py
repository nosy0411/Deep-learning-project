# Unsupervised Learning 예시 : 주가 패턴 분석
# Competitive Learning (경쟁학습) 알고리즘으로 주가의 과거 패턴을 분류한다.
#
# 2018.8.29, 아마추어퀀트 (조성현)
# --------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nInput = 20
nOutput = 8
ALPHA = 0.1

# Winner neuron을 찾는다.
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

# input data를 생성한다
ds = pd.read_csv('dataset/4-7.patternDataset.csv')
ds = ds.drop(ds.columns[0], 1)

dsClass = ds['class']               # 'class' column을 저장해 둔다
ds = ds.drop(ds.columns[20], 1)     # 'vol' column을 제거한다
ds = ds.drop(ds.columns[20], 1)     # 'class' column을 제거한다

# data set을 행을 기준으로 랜덤하게 섞는다 (shuffling)
ds = ds.sample(frac=1).reset_index(drop=True)

# 학습용 데이터
trainX = np.array(ds.iloc[0:len(ds), 0:(nInput + 1)]).T

# 그래프를 생성한다
tf.reset_default_graph()
xi = tf.placeholder(tf.float32, shape=[nInput, None], name='xi')
Wo = tf.Variable(tf.random_uniform([nOutput, nInput], 0, 1), dtype=tf.float32, name='Wo')

# Output Layer
distIWo, winO = findWinner(Wo, xi)
updateOutput = updateWeights(Wo, winO, xi)

# 그래프를 실행한다
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# trainX를 한 개씩 입력하면서 5회 반복 학습한다.
for i in range(10):
    for k in range(5790):
        dx = trainX[:, k]
        dx = dx.reshape([nInput, 1])
        sess.run(updateOutput, feed_dict={xi: dx})
    print("%d done" % (i+1))

# 학습이 완료되었으므로, trainX를 한 개씩 입력하면서 clust를 결정한다.
clust = []
for k in range(trainX.shape[1]):
    dx = trainX[:, k]
    dx = dx.reshape([nInput, 1])
    cluster = sess.run(winO, feed_dict={xi: dx})
    clust.append(cluster)

# 학습이 완료된 weight = centroid
centXY = sess.run(Wo)
sess.close()

# Centroid pattern을 그린다
fig = plt.figure(figsize=(10, 6))
colors = "bgrcmykw"
for i in range(nOutput):
    s = 'pattern-' + str(i)
    p = fig.add_subplot(2, (nOutput+1)//2, i+1)
    p.plot(centXY[i], color=colors[np.random.randint(0, 7)], linewidth=1.0)
    p.set_title('Cluster-' + str(i))

plt.tight_layout()
plt.show()

# clust = 0 인 패턴 몇 개만 그려본다
cluster = 4
ds['cluster'] = clust
plt.figure(figsize=(8, 8))
p = ds.loc[ds['cluster'] == cluster]
p = p.sample(frac=1).reset_index(drop=True)
for i in range(20):
    plt.plot(p.iloc[i][0:20], linewidth=1.0)
    
plt.title('Cluster-' + str(cluster))
plt.show()

ds.to_csv('dataset/4-7.patternCluster.csv')

