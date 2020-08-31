# 인공신경망을 이용하여 대출 심사 평가 데이터를 학습한다. (이진 분류)
# Loss function은 Binary Cross Entropy를 사용한다
#
# 2018.8.23, 아마추어퀀트 (조성현)
# ---------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nInput = 6        # input layer의 neuron 개수
nHidden1 = 8      # hidden layer-1의 neuron 개수
nHidden2 = 8      # hidden layer-2의 neuron 개수
nOutput = 1       # output layer의 neuron 개수

# Credit data set을 읽어온다
ds = pd.read_csv('dataset/3-4.credit_data(Rand).csv')

# data set을 행을 기준으로 랜덤하게 섞는다 (shuffling)
ds = ds.sample(frac=1).reset_index(drop=True)

# 학습용 데이터와 시험용 데이터로 나눈다. (8 : 2)
trainRate = 0.8
trainLen = int(len(ds) * trainRate)
trainX = np.array(ds.iloc[0:trainLen, 0:nInput])
trainY = np.array(ds.iloc[0:trainLen, -1])
trainY = trainY.reshape(trainLen, 1)

testX = np.array(ds.iloc[trainLen:, 0:nInput])
testY = np.array(ds.iloc[trainLen:, -1])
testY = testY.reshape(len(testY), 1)

# 그래프를 생성한다
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, nInput], name='x')
y = tf.placeholder(tf.float32, shape=[None, nOutput], name='y')

# hidden layer-1의 Weight (Wh)와 Bias (Bh)
Wh1 = tf.Variable(tf.truncated_normal([nInput, nHidden1]), dtype=tf.float32, name='Wh1')
Bh1 = tf.Variable(tf.zeros(shape=[nHidden1]), dtype=tf.float32, name='Bh1')

# hidden layer-2의 Weight (Wh)와 Bias (Bh)
Wh2 = tf.Variable(tf.truncated_normal([nHidden1, nHidden2]), dtype=tf.float32, name='Wh2')
Bh2 = tf.Variable(tf.zeros(shape=[nHidden2]), dtype=tf.float32, name='Bh2')

# output layer의 Weight (Wo)와 Bias (Bo)
Wo = tf.Variable(tf.truncated_normal([nHidden2, nOutput]), dtype=tf.float32, name='Wo')
Bo = tf.Variable(tf.zeros(shape=[nOutput]), dtype=tf.float32, name='Bo')

# Hidden layer-1, 2의 출력값. activation function은 sigmoid
H1 = tf.sigmoid(tf.matmul(x, Wh1) + Bh1, name='H1')
H2 = tf.sigmoid(tf.matmul(H1, Wh2) + Bh2, name='H2')

# Output layer의 출력값. activation function은 sigmoid
predY = tf.sigmoid(tf.matmul(H2, Wo) + Bo, name='predY')

# Cost function 정의. cross-entropy 사용
clipY = tf.clip_by_value(predY, 0.000001, 0.99999)  # log(0)를 방지한다
cost = -tf.reduce_mean(y * tf.log(clipY) + (1-y) * tf.log(1-clipY))

# 학습
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(cost)

# 그래프를 실행한다
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 인공신경망에 trainX, trainY를 1000번 집어 넣어서 학습 시킨다. (Batch update)
trLoss = []
for i in range(10000):
    sess.run(train, feed_dict={x: trainX, y: trainY})
    
    # Cost (Cross Entropy)를 추적한다.
    resultLoss = sess.run(cost, feed_dict={x: trainX, y:trainY})
    trLoss.append(resultLoss)
    
    if i % 100 == 0:
        print("%d done, loss = %.4f" % (i+1, trLoss[-1]))

# 학습이 완료되면, Wh, Bh, Wo, Bo 이 모두 업데이트 되었으면, testX를 넣어서 출력값을 확인한다.
# textX의 출력값 (추정값)과 testY (실제값)를 이용하여 정확도를 측정한다.
yHat = sess.run(predY, feed_dict={x: testX})
testYhat = np.where(yHat > 0.5, 1, 0)
accuracy = 100 * (testY == testYhat).sum() / len(testY)
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

print("Accuracy = %.2f" % accuracy)

