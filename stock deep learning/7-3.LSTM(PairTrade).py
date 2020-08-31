# LSTM(GRU) 예시 : 출력값이 한 개인 경우로 페어트레이딩의 스프레드를 예측해 본다.
# 스프레드와, 스프레드의 20일, 60일 이동평균을 이용하여 향후 10일 동안의 스프레드를 예측해 본다.
# 과거 30일 (step = 30) 스프레드와 이동평균 패턴을 학습하여 예측한다.
#
# 실제로는 스프레드의 노이즈를 줄이기 위해 스프레드의 2일 이동평균을 학습하고 예측하는 것으로 한다.
# Features :
# diffNPI : 정규화된 두 주가의 차이
# spread : diffNPI 시계열의 2일 이동평균 (학습, 예측 대상)
# shortMa : diffNPI 시계열의 단기 이동평균 (학습 보조용)
# longMa : diffNPI 시계열의 장기 이동평균 (학습 보조용)
#
# Inputs : [spread, shortMa, longMa]
# Outputs :[spread]
#
# 2018.11.22, 아마추어퀀트 (조성현)
# -------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MyUtil import YahooData

nInput = 3
nOutput = 1
nStep = 30
nNeuron = 100
MASHORT = 20
MALONG = 60
saveDir = './Saver/7-3.save'   # 학습 결과를 저정할 폴더
saveFile = saveDir + '/save'   # 학습 결과를 저장할 파일

# 2차원 배열의 시계열 데이터로 학습용 배치 파일을 만든다.
# return : xBatch - RNN 입력 : 3 개
#          yBatch - RNN 출력 : 1 개
#
# step = 2, ni = 3, no = 1 이라면,
# xData = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], ...]
# xBatch = [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]], ...]
# yBatch = [[[4], [7]], [[10], [13]], ...]
def createTrainData(xData, step, ni=nInput, no=nOutput):
    m = np.arange(len(xData) - step)
    #np.random.shuffle(m)
    
    x, y = [], []
    for i in m:
        a = xData[i:(i+step)]
        x.append(a)
    xBatch = np.reshape(np.array(x), (len(m), step, ni))
    
    for i in m+1:
        a = xData[i:(i+step)]
        y.append(a[:, 0])
    yBatch = np.reshape(np.array(y), (len(m), step, no))
    
    return xBatch, yBatch

# 페어 트레이딩용 데이터를 생성한다. (스프레드 및 스프레드의 이동평균)
#df = YahooData.getStockDataYahoo('105560.KS', start='2010-01-01')  # Yahoo 사이트에서 데이터 수집
df = pd.read_csv('StockData/105560.csv', index_col=0, parse_dates=True)                            # 이미 수집된 데이터를 읽음
df = pd.DataFrame(df['Close'])
#tmp = YahooData.getStockDataYahoo('055550.KS', start='2010-01-01')
tmp = pd.read_csv('StockData/055550.csv', index_col=0, parse_dates=True)
df['Close2'] = pd.DataFrame(tmp['Close'])
df = df.dropna()

df = (df - df.mean()) / df.std()    # Normalized price (NPI)
df['diffNPI'] = df['Close'] - df['Close2']
df['spread'] = pd.DataFrame(df['diffNPI']).rolling(window=2).mean()
df['shortMa'] = pd.DataFrame(df['diffNPI']).rolling(window=MASHORT).mean()
df['longMa'] = pd.DataFrame(df['diffNPI']).rolling(window=MALONG).mean()
df = df.dropna()

# Normalized 주가 차르를 그려본다.
plt.figure(figsize=(8, 3))
plt.plot(df['Close'], color='red', linewidth=1)
plt.plot(df['Close2'], color='blue', linewidth=1)
plt.title("Pair Stock price")
plt.show()

# 스프레드 차트를 그려본다.
plt.figure(figsize=(8, 3))
plt.plot(df['spread'], color='blue', linewidth=1)
#plt.plot(df['shortMa'], color='green', linewidth=1)
plt.plot(df['longMa'], color='red', linewidth=1)
plt.axhline(y=0, color='black', linewidth=1)
plt.axhline(y= 2* df['spread'].std(), color='blue', linestyle='--', linewidth=1)
plt.axhline(y= -2 * df['spread'].std(), color='blue', linestyle='--', linewidth=1)
plt.title("Pair Spread")
plt.show()

# 학습 데이터를 생성한다.
df = df.drop('diffNPI', 1)
df = df.drop('Close', 1)
df = df.drop('Close2', 1)
data = np.array(df)
xBatch, yBatch = createTrainData(data, nStep)

# RNN 그래프를 생성한다 (Wx, Wh). xBatch를 RNN에 입력한다.
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, nStep, nInput])  
rnn = tf.nn.rnn_cell.LSTMCell(nNeuron)
#rnn = tf.nn.rnn_cell.GRUCell(nNeuron)
output, state = tf.nn.dynamic_rnn(rnn, x, dtype=tf.float32)

# RNN의 출력값을 입력으로 받아 1개의 y가 출력되도록 하는 feed-forward network를 생성한다. (Wy)
y = tf.placeholder(tf.float32, [None, nStep, nOutput])
inFC = tf.reshape(output, [-1, nNeuron])          
outFC = tf.layers.dense(inFC, nOutput)       
predY = tf.reshape(outFC, [-1, nStep, nOutput])

# Mean square error (MSE)로 Loss를 정의한다. xBatch가 입력되면 yBatch가 출력되도록 함.
loss = tf.reduce_sum(tf.square(predY - y))    
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)         
minLoss = optimizer.minimize(loss)

# 저장
saver = tf.train.Saver()

# 그래프를 실행한다. 학습한다. (Wx, Wh, Wy를 업데이트함)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 기존 학습 결과를 적용한 후 추가로 학습한다
if tf.train.checkpoint_exists(saveDir):
    saver.restore(sess, saveFile)
    
lossHist = []
for i in range(100):
    sess.run(minLoss, feed_dict={x: xBatch, y: yBatch})
    
    if i % 5 == 0:
        ploss = sess.run(loss, feed_dict={x: xBatch, y: yBatch})
        lossHist.append(ploss)
        print(i, "\tLoss:", ploss)

# 학습 결과를 저장해 둔다
saver.save(sess, saveFile)

# 향후 10 기간 데이터를 예측한다. 향후 1 기간을 예측하고, 예측값을 다시 입력하여 2 기간을 예측한다.
# 이런 방식으로 10 기간까지 예측한다.
nFuture = 5
if len(data) > 100:
    lastData = np.copy(data[-100:])  # 원 데이터의 마지막 100개만 그려본다
else:
    lastData = np.copy(data)
dx = np.copy(lastData)
estimate = [dx[-1]]
for i in range(nFuture):
    # 마지막 nStep 만큼 입력데이로 다음 값을 예측한다
    px = dx[-nStep:,]
    px = np.reshape(px, (1, nStep, nInput))
    
    # 다음 값을 예측한다.
    ySpread = sess.run(predY, feed_dict={x: px})[0][-1]
    
    # ySpread를 이용하여 다음 입력값으로 사용할 이동평균을 계산한다.
    shortMa = np.mean(np.append(dx[-MASHORT+1:, 0], ySpread))
    longMa = np.mean(np.append(dx[-MALONG+1:, 0], ySpread))
    newInput = [ySpread[0], shortMa, longMa]
    
    # 예측값을 저장해 둔다
    estimate.append(newInput)
    
    # 이전 예측값을 포함하여 또 다음 값을 예측하기위해 예측한 값을 저장해 둔다
    dx = np.vstack([dx, np.array(newInput)])

# Loss history를 그린다
plt.figure(figsize=(8, 3))
plt.plot(lossHist, color='red')
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 원 시계열과 예측된 시계열을 그린다
SPREAD = 0       # 스프레드를 예측한다
estimate = np.array(estimate)
ax1 = np.arange(1, len(lastData[:, SPREAD]) + 1)
ax2 = np.arange(len(lastData), len(lastData) + len(estimate))
plt.figure(figsize=(8, 6))
plt.plot(ax1, lastData[:, SPREAD], color='blue', label='Spread', linewidth=1)
plt.plot(ax1, lastData[:, 1], color='black', label='Short MA', linewidth=1)
plt.plot(ax1, lastData[:, 2], color='red', label='Long MA', linewidth=1)
plt.plot(ax2, estimate[:, SPREAD], color='red')
plt.plot(ax2, estimate[:, 1], color='blue')
plt.plot(ax2, estimate[:, 2], color='blue')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.title("Pair spread  prediction")
plt.show()


