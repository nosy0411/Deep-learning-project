# 인공신경망을 이용하여 주가의 기술적분석 지표와 변동성 지표를 학습하여 목표 수익률을 달성 여부를 학습한다. (이진 분류)
#
# 2018.8.23, 아마추어퀀트 (조성현)
# --------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MyUtil import YahooData, TaFeatureSet

stocks = {'005380':'현대차', '000660':'SK하이닉스', '015760':'한국전력', '034220':'LG디스플레이',
          '005490':'POSCO', '035420':'NAVER', '017670':'SK텔레콤', '012330':'현대모비스',
          '055550':'신한지주', '000270':'기아차', '105560':'KB금융', '051910':'LG화학'}

nInput = 6          # input layer의 neuron 개수
nHidden1 = 20       # hidden layer-1의 neuron 개수
nHidden2 = 20       # hidden layer-2의 neuron 개수
nOutput = 1         # output layer의 neuron 개수
saveDir = './Saver/5-5.save'   # 학습 결과를 저정할 폴더
saveFile = saveDir + '/save'   # 학습 결과를 저장할 파일

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
H1 = tf.nn.relu(tf.matmul(x, Wh1) + Bh1, name='H1')
H2 = tf.nn.relu(tf.matmul(H1, Wh2) + Bh2, name='H2')

# Output layer의 출력값. activation function은 sigmoid
predY = tf.sigmoid(tf.matmul(H2, Wo) + Bo, name='predY')

# Cost function 정의. cross-entropy 사용
clipY = tf.clip_by_value(predY, 0.000001, 0.99999)  # log(0)를 방지한다
cost = -tf.reduce_mean(y * tf.log(clipY) + (1-y) * tf.log(1-clipY))

# 학습
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(cost)

# 저장
saver = tf.train.Saver()

# 그래프를 실행한다
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 기존 학습 결과를 적용한 후 추가로 학습한다
if tf.train.checkpoint_exists(saveDir):
    saver.restore(sess, saveFile)

# 인공신경망에 trainX, trainY를 1000번 집어 넣어서 학습 시킨다. (Mini-Batch update)
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
    resultLoss = sess.run(cost, feed_dict={x: trainX, y:trainY})
    trLoss.append(resultLoss)
    
    if i % 10 == 0:
        print("%d done, loss = %.4f" % (i+1, resultLoss))

# 학습이 완료되면, Wh, Bh, Wo, Bo 이 모두 업데이트 되었으면, testX를 넣어서 출력값을 확인한다.
# textX의 출력값 (추정값)과 testY (실제값)를 이용하여 정확도를 측정한다.
yHat = sess.run(predY, feed_dict={x: testX})
testYhat = np.where(yHat > 0.5, 1, 0)
accuracy = 100 * (testY == testYhat).sum() / len(testY)

# 학습 결과를 저장해 둔다
saver.save(sess, saveFile)
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

