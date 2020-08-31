# K-Means clustering 알고리즘으로 주가의 과거 패턴을 분류한다.
# 참고 자료 : Jordi Torres, First contact with tensorflow (page 27)
#
# 2018.08.22, 아마추어퀀트 (조성현)
# ----------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MyUtil import YahooData, PatternFeatureSet

stocks = {'005380':'현대차', '000660':'SK하이닉스', '015760':'한국전력', '034220':'LG디스플레이',
          '005490':'POSCO', '035420':'NAVER', '017670':'SK텔레콤', '012330':'현대모비스',
          '055550':'신한지주', '000270':'기아차', '105560':'KB금융', '051910':'LG화학'}

# Yahoo site로부터 위에 정의된 주가 데이터를 수집하여 ./stockData 폴더에 저장해 둔다.
def getData(stockList=stocks, start='2007-01-01'):
    YahooData.getStockDataList(stockList, start=start)

# 데이터 세트를 생성하고, ./dataset/4-7.patternDataset.csv 에 저장해 둔다.
# 시간이 오래 걸린다.
def createDataSet(stockList=stocks, u=0.5, d=-0.5, nPast=20, nHop=5, nFuture=20, binary=True):
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
            ft = PatternFeatureSet.getPatternFeatureSet(data, u, d, nPast, nHop, nFuture, binary)
        else:
            ft = ft.append(PatternFeatureSet.getPatternFeatureSet(data, u, d, nPast, nHop, nFuture, binary))
        print("%d) %s (%s)를 처리하였습니다." % (n, stockList[code], code))
        n += 1
    
    ft.to_csv('dataset/4-7.patternDataset.csv')

# 데이터 세트를 읽어온다
ds = pd.read_csv('dataset/4-7.patternDataset.csv')
ds = ds.drop(ds.columns[0], 1)

dsClass = ds['class']               # 'class' column을 저장해 둔다
ds = ds.drop(ds.columns[20], 1)     # 'vol' column을 제거한다
ds = ds.drop(ds.columns[20], 1)     # 'class' column을 제거한다

# input data tensor를 생성한다
tf.reset_default_graph()
inputXY = tf.constant(ds)

# 초기 중심 좌표를 설정한다.
k = 8
randXY = tf.random_uniform([k], 0, len(ds)-1, dtype=tf.int32) # 초기 중점으로 사용할 좌표 k개를 random sampling한다
tmpCent = tf.gather(inputXY, randXY)
initCent = tf.Variable(tmpCent)               # 초기 중심좌표 텐서

# 데이터 ~ 중심좌표 사이의 거래 계산을 위해 dimension을 맞춘다. 3-차원 텐서.
#expXY.get_shape()   --> (D0, D1, D2) = (1, n, 2) : D0을 확장함
#expCent.get_shape() --> (D0, D1, D2) = (k, 1, 2) : D1을 확장함
expXY = tf.expand_dims(inputXY, 0)      # D0-축을 확장한다
expCent = tf.expand_dims(initCent, 1)   # D1-축을 확장한다

# 데이터와 중삼좌표 사이의 거리를 계산한다
tmpDist = tf.square(tf.subtract(expXY, expCent))
dist = tf.sqrt(tf.reduce_sum(tmpDist, 2))   # D2 축으로 합침
error = tf.reduce_sum(dist)                 # 거리의 총합을 error로 정의한다

# 각 데이터를 거리가 작은 중점에 assign 한다.
# assignment = [0 1 3 2 1 0 0 0 1 3] --> 첫 번째 데이터는 중점 0, 두 번째 데이터는 중점 1 에 assign
assignment = tf.argmin(dist)

# 중점의 좌표를 update한다
for c in range(k):
    # tfEqual
    # c=0 : [ True False False False False True True True False False ]
    tfEqual = tf.equal(assignment, c)
    
    # tfWhere : [[0] [5] [6] [7]] <-- True가 몇 번째에 있는지
    tfWhere = tf.where(tfEqual)
    
    # tfReshape : [0 5 6 7] <-- [-1]은 flatten 하라는 의미임
    tfReshape = tf.reshape(tfWhere, [-1])
        
    # tfGather : inputXY의 0, 5, 6, 7 번째 좌표를 모음
    tfGather = tf.gather(inputXY, tfReshape)
    
    # 평균좌표를 계산한다
    # tfMean : [[0.03167805 0.29030013]]
    tfMean = tf.reduce_mean(tfGather, 0)    # row 방향으로 reduce
    
    # k-개의 평균 좌표를 모은다
    if c == 0:
        tfConcat = tf.concat(tfMean, 0)
    else:
        tfConcat = tf.concat([tfConcat, tfMean], 0)

# 평균 좌표 : meanXY
meanXY = tf.reshape(tfConcat, [k, 20])
    
# 새로운 중심 좌표를 초기 좌표로 assign 한다 (반복 계산을 위해)
newCent = tf.assign(initCent, meanXY)

# 세션을 돌려서 반복적으로 중심 좌표를 업데이트한다
sess = tf.Session()

# 100회 반복하고 거리의 총합으로 측정한 error가 최소가 되는 결과를 채택한다.
# 초깃값에 따라 E-M 알고리즘의 Local optimization 문제점을 해소하기 위함.
finalErr = np.inf
trcErr = []
for i in range(100):
    sess.run(tf.global_variables_initializer())
    prevErr = 0
    for j in range(200):
        _, centXY, clust, err = sess.run([newCent, initCent, assignment, error])
        if np.abs(err - prevErr) < 0.001:
            break
        prevErr = err

    # error가 최소인 결과를 저장해 둔다
    if err < finalErr:
        finalCentXY = centXY.copy()
        finalClust = clust.copy()
        finalErr = err
    
    trcErr.append(err)
    print("%d done" % i)

ds['cluster'] = clust
ds['class'] = dsClass

# Centroid pattern을 그린다
fig = plt.figure(figsize=(10, 6))
colors = "bgrcmykw"
for i in range(k):
    s = 'pattern-' + str(i)
    p = fig.add_subplot(2, (k+1)//2, i+1)
    p.plot(centXY[i], 'b-o', markersize=3, color=colors[np.random.randint(0, 7)], linewidth=1.0)
    p.set_title('Cluster-' + str(i))

plt.tight_layout()
plt.show()

# clust = 0 인 패턴 몇 개만 그려본다
cluster = 0
plt.figure(figsize=(8, 7))
p = ds.loc[ds['cluster'] == cluster]
p = p.sample(frac=1).reset_index(drop=True)
for i in range(10):
    plt.plot(p.iloc[i][0:20])
    
plt.title('Cluster-' + str(cluster))
plt.show()

# 거리의 총합으로 정의한 error의 history를 관찰한다.
plt.plot(trcErr)
plt.title("Error (by total distance)")
plt.show()

print("min Error = %.2f" % finalErr)

# 분석 결과를 저장해 둔다.
ds.to_csv('dataset/4-7.patternResult.csv')
sess.close()

