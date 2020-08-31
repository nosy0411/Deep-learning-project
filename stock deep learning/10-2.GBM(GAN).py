# Generative Adversarial Network을 활용한 GBM 시뮬레이션
#
# GBM 모형은 W를 샘플링할 때 표준정규분포로 가정하지만, GAN을 활용하면 실제 주가 수익률의
# 분포를 generation하여 샘플링하므로 실제 주가의 특성이 반영된 시뮬레이션을 만들어 볼 수 있다.
# GBM에서 주가 수익률의 변동성을 상수로 가정하였고, 이 부분은 GAN에서도 동일하다. (향후 보완해야할 숙제임.)
#
# 원 논문 : Ian J. Goodfellow, et, al., 2014, Generative Adversarial Nets.
# 위 논문의 Psedudo code (Section 4. Algorithm 1)를 위주로 하였음.
# 참고 : Original GAN은 unstable, mode collapse 등의 문제가 있으며, 최근 변형된 알고리즘이
#       많이 개발되어 있다. 여기서는 원 알고리즘을 사용하였으므로 결과가 개선의 여지가 있음.
#
# 2018.9.11, 아마추어퀀트 (조성현)
# -------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MyUtil import YahooData

slim = tf.contrib.slim
stockCode = '005380'
saveDir = './Saver/10-2.' + stockCode            # 학습 결과를 저정할 폴더
saveFile = saveDir + '/10-2.' + stockCode        # 학습 결과를 저장할 파일

# Yahoo site로부터 주가 데이터를 수집하여 ./stockData 폴더에 저장해 둔다.
def getStockData(stockCode=stockCode, start='2007-01-01'):
    YahooData.getStockDataYahoo(stockCode + '.KS', start=start)
    
# 주가 데이터를 읽어서 수익률을 계산한다.
def createDataSet(code):
    data = pd.read_csv('StockData/' + code + '.csv', index_col=0, parse_dates=True)
    
    # 종가 기준으로 수익률을 계산한다
    data['rtn'] = pd.DataFrame(data['Close']).apply(lambda x: np.log(x) - np.log(x.shift(1)))
    data = data.dropna()

    # 수익률과 last price를 리턴한다.
    volatility = np.std(data['rtn'])
    lastPrice = data['Close'][len(data)-1]
    return np.array(data['rtn']), lastPrice, volatility

# 데이터 P, Q에 대한 KL divergence를 계산한다.
def KL(P, Q):
    # 두 데이터의 분포를 계산한다
    histP, binsP = np.histogram(P, bins=150)
    histQ, binsQ = np.histogram(Q, bins=binsP)
    
    # 두 분포를 pdf로 만들기 위해 normalization한다.
    histP = histP / np.sum(histP) + 1e-8
    histQ = histQ / np.sum(histQ) + 1e-8

    # KL divergence를 계산한다
    kld = np.sum(histP * np.log(histP / histQ))
    return histP, histQ, kld

# 학습데이터 : 주가 수익률 데이터를 생성한다
realData, lastPrice, volatility = createDataSet(stockCode)
realData = 10 * (realData - np.mean(realData)) / volatility
realData = realData.reshape(realData.shape[0], 1)
nDataRow = realData.shape[0]
nDataCol = realData.shape[1]

nGInput = 20
nGHidden = 128
nDHidden = 128

tf.reset_default_graph()
def Generator(z, nOutput=nDataCol, nHidden=nGHidden, nLayer=1):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [nHidden] * nLayer, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, nOutput, activation_fn=None)
    return x

def Discriminator(x, nOutput=1, nHidden=nDHidden, nLayer=1, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h = slim.stack(x, slim.fully_connected, [nHidden] * nLayer, activation_fn=tf.nn.relu)
        d = slim.fully_connected(h, nOutput, activation_fn=None)
    return d

def getNoise(m, n=nGInput):
    z = np.random.uniform(-1., 1., size=[m, n])
    return z

# 각 네트워크의 출력값
with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
    x = tf.placeholder(tf.float32, shape=[None, nDataCol], name='x')
    z = tf.placeholder(tf.float32, shape=[None, nGInput], name='z')
    Gz = Generator(z, nOutput=nDataCol)
    Dx = Discriminator(x)
    DGz = Discriminator(Gz, reuse=True)
    
# D-loss function
# Binary cross entropy를 이용한다
# labels * -log(sigmoid(logits)) + (1 - labels) * -log(1 - sigmoid(logits))
D_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits=DGz, labels=tf.zeros_like(DGz)))
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DGz, labels=tf.ones_like(DGz)))

thetaG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
thetaD = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

trainD = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list = thetaD)
trainG = tf.train.AdamOptimizer(0.00001).minimize(G_loss, var_list = thetaG)

# 학습 결과 저장용 변수
saver = tf.train.Saver()

# 그래프를 실행한다
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 기존 학습 결과를 적용한 후 추가로 학습한다
if tf.train.checkpoint_exists(saveDir):
    saver.restore(sess, saveFile)
    
histLossD = []      # Discriminator loss history 저장용 변수
histLossG = []      # Generator loss history 저장용 변수
histKL = []         # KL divergence history 저장용 변수
nBatchCnt = 10      # Mini-batch를 위해 input 데이터를 n개 블록으로 나눈다.
nBatchSize = int(realData.shape[0] / nBatchCnt)  # 블록 당 Size
nK = 1              # Discriminator 학습 횟수 (위 논문에서는 nK = 1을 사용하였음)
k = 0
for i in range(1000):
    for n in range(nBatchCnt):
        # input 데이터를 Mini-batch 크기에 맞게 자른다
        nFrom = n * nBatchSize
        nTo = n * nBatchSize + nBatchSize
        
        # 마지막 루프이면 nTo는 input 데이터의 끝까지.
        if n == nBatchCnt - 1:
            nTo = realData.shape[0]
               
        # 학습 데이터를 준비한다
        bx = realData[nFrom : nTo]
        bz = getNoise(m=bx.shape[0])

        if k < nK:
            # Discriminator를 nK-번 학습한다.
            _, lossDHist = sess.run([trainD, D_loss], feed_dict={x: bx, z : bz})
            k += 1
        else:
            # Generator를 1-번 학습한다.
            _, lossGHist = sess.run([trainG, G_loss], feed_dict={x: bx, z : bz})
            k = 0
    
    # 100번 학습할 때마다 Loss, KL의 history를 보관해 둔다
    if i % 100 == 0:
        p, q, kld = KL(bx, sess.run(Gz, feed_dict={z : bz}))
        histKL.append(kld)
        histLossD.append(lossDHist)
        histLossG.append(lossGHist)
        print("%d) D-loss = %.4f, G-loss = %.4f, KL = %.4f" % (i, lossDHist, lossGHist, kld))

# 학습 결과를 저장해 둔다
saver.save(sess, saveFile)

plt.figure(figsize=(6, 3))
plt.plot(histLossD, label='Loss-D')
plt.plot(histLossG, label='Loss-G')
plt.legend()
plt.title("Loss history")
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(histKL)
plt.title("KL divergence")
plt.show()

# real data 분포 (p)와 fake data 분포 (q)를 그려본다
plt.figure(figsize=(8,4))
fakeData = sess.run(Gz, feed_dict={z : getNoise(m=realData.shape[0])})
p, q, kld = KL(realData, fakeData)
x = np.linspace(-3, 3, 150)
plt.plot(x, p, color='blue', linewidth=2.0, alpha=0.7, label='Real Data')
plt.plot(x, q, color='red', linewidth=2.0, alpha=0.7, label='Fake Data')

# normal 분포를 그려본다
h, b = np.histogram(np.random.normal(0, 1, 5000), bins=150)
h = h / np.sum(h) + 1e-8
plt.plot(x, h, color='gray', linewidth=2.0, alpha=0.7, label='Normal Distribution')
plt.legend()
plt.title("Distibution of Real and Fake Data")
plt.show()
print("KL divergence = %.4f" % kld)

# Fake Data를 Discriminator에 넣었을 때 출력값을 확인해 본다.
r = 0.015 / 365
nSample = 250
plt.figure(figsize=(8,4))
for i in range(100):
    fakeData = sess.run(Gz, feed_dict={z : getNoise(m=nSample)})
    fakeData /= 10
    fakeData = np.reshape(fakeData, (nSample))
    S0 = lastPrice
    stockPrice = [S0]
    for W in fakeData:
        S1 = S0 * np.exp(volatility * W + (r - (volatility ** 2) / 2))
        stockPrice.append(S1)
        S0 = S1

    plt.plot(stockPrice, linewidth=1)
plt.show()

#sess.close()
