# TensorFlow를 이용하여 Logistic Regression를 연습한다.
# Loss function은 Binary Cross Entropy를 사용한다.
#
# 2018.08.14, 아마추어퀀트 (조성현)
# ---------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Credit data set을 읽어온다
ds = pd.read_csv('dataset/3-4.credit_data(rand).csv')

# data set을 행을 기준으로 랜덤하게 섞는다 (shuffling)
ds = ds.sample(frac=1).reset_index(drop=True)

# 학습용 데이터와 시험용 데이터로 나눈다. (8 : 2)
trainRate = 0.8
trainLen = int(len(ds) * trainRate)
trainX = np.array(ds.iloc[0:trainLen, 0:6])
trainY = np.array(ds.iloc[0:trainLen, -1])
trainY = trainY.reshape(trainLen, 1)

testX = np.array(ds.iloc[trainLen:, 0:6])
testY = np.array(ds.iloc[trainLen:, -1])
testY = testY.reshape(len(testY), 1)

# 그래프 모델을 생성한다
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, 6], name="X")
y = tf.placeholder(tf.float32, shape=[None, 1], name="Y")
r = tf.random_uniform([6, 1], -1.0, 1.0)
w = tf.Variable(r, name="W")
b = tf.Variable(tf.zeros([1]), name = "Bias")

# Loss function을 정의한다. (Binary Cross Entropy)
tmpX = tf.add(tf.matmul(x, w), b)
predY = tf.div(1., 1. + tf.exp(-tmpX))

# Cross entropy를 계산할 때 log(0)이 나오는 경우를 방지한다.
clipY = tf.clip_by_value(predY, 0.000001, 0.99999)
CE = -tf.reduce_mean(y * tf.log(clipY) + (1 - y) * tf.log(1 - clipY))

# 학습할 optimizer를 정의한다
optimizer = tf.train.AdamOptimizer(0.005)
train = optimizer.minimize(CE)

# 세션을 오픈하고 그래프를 실행한다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

trLoss = []
nBatchCnt = 10       # Mini-batch를 위해 input 데이터를 5개 블록으로 나눈다.
nBatchSize = int(trainX.shape[0] / nBatchCnt)  # 블록 당 Size
for i in range(3000):
    # Mini-batch 방식으로 학습한다
    for n in range(nBatchCnt):
        # input 데이터를 Mini-batch 크기에 맞게 자른다
        nFrom = n * nBatchSize
        nTo = n * nBatchSize + nBatchSize
        
        # 마지막 루프이면 nTo는 input 데이터의 끝까지.
        if n == nBatchCnt - 1:
            nTo = trainX.shape[0]
            
        bx = trainX[nFrom : nTo]
        by = trainY[nFrom : nTo]
        
        # 학습한다
        sess.run(train, feed_dict={x: bx, y: by})
        
    # Cross Entropy를 추적한다.
    resultLoss = sess.run(CE, feed_dict={x: trainX, y:trainY})
    trLoss.append(resultLoss)
    
    if i % 10 == 0:
        print("%d) CE = %f" % (i, resultLoss))

# 시험용 데이터를 이용하여 정확도를 측정한다.
yHat = sess.run(predY, feed_dict={x: testX})
testYhat = np.where(yHat > 0.5, 1, 0)
accuracy = 100 * (testY == testYhat).sum() / len(testY)

sW = sess.run(w)
sb = sess.run(b)
sess.close()

# 결과를 확인한다
fig = plt.figure(figsize=(10, 4))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)

p1.plot(trLoss)
p1.set_title("Loss function")
p1.set_xlabel("epoch")
p1.set_ylabel("loss")

n, bins, patches = p2.hist(yHat, 50, facecolor='blue', alpha=0.5)
p2.set_title("yHat distribution")
plt.show()

print("W = ", sW.T)
print("b = ", sb)
print("\nAccuracy = %.2f %s" % (accuracy, '%'))
print("Class = 0 count : ", len(np.where(ds['class'] == 0.0)[0]))
print("Class = 1 count : ", len(np.where(ds['class'] == 1.0)[0]))