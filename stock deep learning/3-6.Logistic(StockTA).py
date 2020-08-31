# Logistic Regression으로 기술적분석 지표들과 변동성을 Feature로 향후 20일 동안
# 목표 수익률을 달성할 가능성이 있는지를 추정한다.
#
# 2018.08.15, 아마추어퀀트 (조성현)
# -------------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MyUtil import YahooData, TaFeatureSet

stocks = {'005380':'현대차', '000660':'SK하이닉스', '015760':'한국전력', '034220':'LG디스플레이',
          '005490':'POSCO', '035420':'NAVER', '017670':'SK텔레콤', '012330':'현대모비스',
          '055550':'신한지주', '000270':'기아차', '105560':'KB금융', '051910':'LG화학'}

# Yahoo site로부터 위에 정의된 주가 데이터를 수집하여 ./stockData 폴더에 저장해 둔다.
def getData(stockList=stocks, start='2007-01-01'):
    YahooData.getStockDataList(stockList, start=start)

# 데이터 세트를 생성하고, ./dataset/3-6.TaDataset.csv 에 저장해 둔다.
# 시간이 오래 걸린다.
def createDataSet(stockList=stocks, u=0.5, d=-0.5, nFuture=20, binary=True):
    n = 1
    for code in stockList.keys():
        # 저장된 파일을 읽어온다
        data = pd.read_csv('stockData/' + code + '.csv', index_col=0, parse_dates=True)
        
        # 과거 20일의 종가 패턴과 변동성이 어느 수준일 때 매수하면 손실인지 아닌지를 class로 부여하여
        # 목표 수익률 달성 여부를 학습한다. 목표 수익률을 달성한다는 것은 향후 주가가 상승하는 것을 의미함.
        # Binary classification을 위해 class = 0, 1로 구분하였음.
        # u = 0.5 : 수익률 표준편차의 0.5 배 이상이면 수익 (class = 1)
        # d = -0.5 : 수익률 표준편차의 -0.5배 이하이면 손실 (class = 0)
        # 중간이면 주가 횡보 (classs = 0)
        if n == 1:
            ft = TaFeatureSet.getTaFeatureSet(data, u, d, nFuture, binary)
        else:
            ft = ft.append(TaFeatureSet.getTaFeatureSet(data, u, d, nFuture, binary))
        print("%d) %s (%s)를 처리하였습니다." % (n, stockList[code], code))
        n += 1
    
    ft.to_csv('dataset/3-6.TaDataset.csv')

# 데이터 세트를 읽어온다
ds = pd.read_csv('dataset/3-6.TaDataset.csv')
ds = ds.drop(ds.columns[0], 1)

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
optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(CE)

# 세션을 오픈하고 그래프를 실행한다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

trLoss = []
nBatchCnt = 5       # Mini-batch를 위해 input 데이터를 5개 블록으로 나눈다.
nBatchSize = int(trainX.shape[0] / nBatchCnt)  # 블록 당 Size
for i in range(1000):
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
print("Class = 0 : ", len(np.where(ds['class'] == 0.0)[0]))
print("Class = 1 : ", len(np.where(ds['class'] == 1.0)[0]))
print("\nAccuracy = %.2f %s" % (accuracy, '%'))