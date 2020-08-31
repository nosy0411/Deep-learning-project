# KNN으로 기술적분석 지표들과 변동성을 Feature로 향후 20일 동안
# 목표 수익률을 달성할 가능성이 있는지를 추정한다.
#
# 2018.08.20, 아마추어퀀트 (조성현)
# --------------------------------------------------------
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
tx = tf.placeholder(tf.float32, shape=[None, 6], name="X")
parK = tf.placeholder(tf.int32, shape=[], name="K")

# test point와 모든 x와의 거리를 측정한다 (Euclidean distance)
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, tx)), 1))

# test point와 모든 x와의 거리중 거리가 짧은 K개의 class (trainY)를 찾는다.
# 이 classes에서 다수결로 test point의 class를 판정한다.
# ex : tf.gather(trainY, indices) = [1, 1, 0, 1, 0, 1, 1] --> class = 1로 판정한다.
#      class 들의 평균 = 5/7 = 0.714 --> 0.5 보다 크므로 class = 1로 판정함.
values, indices = tf.nn.top_k(tf.negative(distance), k=parK, sorted=False)
classMean = tf.reduce_mean(tf.cast(tf.gather(trainY, indices), tf.float32))

# 그래프를 실행한다.
sess = tf.Session()
accuracy = []       # K = [1,2,3..]에 따른 정확도를 기록할 list
yHatK = []          # 최적 K일 때 testX의 class를 추정한 list
optK = 0            # 최적 K
optAccuracy = 0.0   # 최적 K일 때의 정확도
minK = 50
maxK = 100          # K = minK ~ maxK 까지 변화시키면서 정확도를 측정한다
for k in range(minK, maxK+1):
    yHat = []
    for dx in testX:
        # testX를 하나씩 읽어가면서 k-개의 가까운 거리를 찾는다
        dx = dx.reshape(1, 6)
        yHat.append(sess.run(classMean, feed_dict = {x: trainX, tx: dx, parK: k}))
    
    # test data의 class를 추정한다.
    yHat = np.array(yHat)
    yHat = yHat.reshape(len(yHat), 1)
    testYhat = np.where(yHat > 0.5, 1, 0)
    
    # test data의 실제 class와 추정 class를 비교하여 정확도를 측정한다.
    accK = 100 * (testY == testYhat).sum() / len(testY)
    
    # 정확도가 가장 높은 최적 K, yHatK, optAccuracy를 기록해 둔다
    if accK > optAccuracy:
        yHatK = yHat.copy()
        optK = k
        optAccuracy = accK
    
    # K에 따라 달라지는 정확도를 추적하기위해 history를 기록해 둔다
    accuracy.append(accK)
    print("k = %d done" % k)
sess.close()

# 결과를 확인한다
fig = plt.figure(figsize=(10, 4))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)

xK = np.arange(minK, maxK+1)
p1.plot(xK, accuracy)
p1.set_title("Accuracy (optimal K = " + str(optK) + ")")
p1.set_xlabel("K")
p1.set_ylabel("accuracy")

n, bins, patches = p2.hist(yHatK, 50, facecolor='blue', alpha=0.5)
p2.set_title("yHat distribution")
plt.show()

print("\nAccuracy = %.2f %s" % (optAccuracy, '%'))

# 2개 Feature를 선택하여 2-차원 상에서 각 Feature에 대한 class를 육안으로 확인한다
# 전체 Feature의 6-차원 공간의 확인은 불가하므로 2-차원으로 확인한다
dsX = 0     # x-축은 첫 번째 feature
dsY = 1     # y-축은 두 번째 feature
cnt = 300   # 300개만 그린다
class0 = ds[ds['class'] == 0].iloc[0:cnt, [dsX, dsY]]
colX = class0.columns[0]
colY = class0.columns[1]

plt.figure(figsize=(8, 7))
plt.scatter(class0[colX], class0[colY], color='blue', marker='o', s=50, alpha=0.5, label='class=0')

class1 = ds[ds['class'] == 1].iloc[0:cnt, [dsX, dsY]]
plt.scatter(class1[colX], class1[colY], color='red', marker='x', s=50, alpha=0.7, label='class=1')

plt.xlabel(colX)
plt.ylabel(colY)
plt.legend()
plt.show()
