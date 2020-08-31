# GBM과 GAN을 이용하여 연관관계를 갖는 두 주가를 생성한다.
#
# GAN을 이용하여 실제 두 주가의 수익률 연관관계 분포를 모방한 후 두 주가를 생성한다.
#
# 참고 : Original GAN을 사용한 결과 mode collapse 현상이 발생하고 있다. Unrolled GAN 등
#        변형된 GAN을 사용하여 추가 보완할 필요가 있다.
#
# 2018.12.19, 아마추어퀀트 (조성현)
# ---------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
slim = tf.contrib.slim

# 데이터 세트를 생성한다
stockA = pd.read_csv('stockData/105560.csv', index_col=0, parse_dates=True)
stockB = pd.read_csv('stockData/055550.csv', index_col=0, parse_dates=True)
ds = pd.DataFrame(stockA['Close'])
ds.columns = ['stockA']
ds['stockB'] = pd.DataFrame(stockB['Close'])
ds = ds.dropna()

# 두 주가의 그래프를 육안으로 확인하기 위해 표준화한다.
ds['normA'] = (ds['stockA'] - ds['stockA'].mean()) / ds['stockA'].std()
ds['normB'] = (ds['stockB'] - ds['stockB'].mean()) / ds['stockB'].std()
ds['spread'] = ds['normA'] - ds['normB']

# 두 주가의 수익률을 계산하고 GBM 모형을 위해 변동성을 측정해 둔다
ds['rtnA'] = np.log(ds['stockA']) - np.log(ds['stockA'].shift(1))
ds['rtnB'] = np.log(ds['stockB']) - np.log(ds['stockB'].shift(1))
ds = ds.dropna()
volA = ds['rtnA'].std()
volB = ds['rtnB'].std()

# 수익률을 표준화 시킨다.
ds['rtnA'] = (ds['rtnA'] - ds['rtnA'].mean()) / ds['rtnA'].std()
ds['rtnB'] = (ds['rtnB'] - ds['rtnB'].mean()) / ds['rtnB'].std()

# 두 주가와 상관관계, 스프레드 차트, 스프레드 분포를 육안으로 확인해 본다
fig = plt.figure(figsize=(10, 8))
p1 = fig.add_subplot(2,2,1)
p2 = fig.add_subplot(2,2,2)
p3 = fig.add_subplot(2,2,3)
p4 = fig.add_subplot(2,2,4)

p1.plot(ds['normA'], color='red', linewidth=1, label='Stock-A')
p1.plot(ds['normB'], color='blue', linewidth=1, label='Stock-B')
p1.legend()
p1.set_title("Stock Price Chart")

p2.set_xlim(-4, 4)
p2.set_ylim(-4, 4)
p2.scatter(ds['rtnA'], ds['rtnB'], c='green', s=3)
p2.set_title("Correlation")

boundSpread = 2 * ds['spread'].std()
p3.plot(ds['spread'], color='blue', linewidth=1)
p3.axhline(y=0, color='black', linewidth=1)
p3.axhline(y=boundSpread, linestyle='--', color='red', linewidth=1)
p3.axhline(y=-boundSpread, linestyle='--', color='red', linewidth=1)
p3.axhline(y=0, color='black', linewidth=1)
p3.set_title("Pair-Spread Chart")

r = np.copy(ds['spread'])
r.sort()
pdf = stats.norm.pdf(r, np.mean(r), np.std(r))
p4.plot(r,pdf)
p4.hist(r, density=True, bins=100)
p4.set_title("Distribution")
plt.show()

# GAN 학습
realData = np.array(ds.loc[:, ['rtnA', 'rtnB']])
nDataRow = realData.shape[0]
nDataCol = realData.shape[1]

nGInput = 16
nGHidden = 32
nDHidden = 32
saveDir = './Saver/10-4.save'   # 학습 결과를 저정할 폴더
saveFile = saveDir + '/save'    # 학습 결과를 저장할 파일

tf.reset_default_graph()
def Generator(z, nOutput=nDataCol, nHidden=nGHidden, nLayer=2):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [nHidden] * nLayer, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, nOutput, activation_fn=None)
    return x

def Discriminator(x, nOutput=1, nHidden=nDHidden, nLayer=2, reuse=False):
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

# 저장
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 기존 학습 결과를 적용한 후 추가로 학습한다
if tf.train.checkpoint_exists(saveDir):
    saver.restore(sess, saveFile)
    
histLossD = []      # Discriminator loss history 저장용 변수
histLossG = []      # Generator loss history 저장용 변수
nBatchCnt = 5       # Mini-batch를 위해 input 데이터를 n개 블록으로 나눈다.
nBatchSize = int(realData.shape[0] / nBatchCnt)  # 블록 당 Size
nK = 1              # Discriminator 학습 횟수 (위 논문에서는 nK = 1을 사용하였음)
k = 0
for i in range(100):
    # Mini-batch 방식으로 학습한다
    np.random.shuffle(realData)
    
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
        histLossD.append(lossDHist)
        histLossG.append(lossGHist)
        print("%d) D-loss = %.4f, G-loss = %.4f" % (i, lossDHist, lossGHist))

# 학습 결과를 저장해 둔다
saver.save(sess, saveFile)

plt.figure(figsize=(6, 3))
plt.plot(histLossD, label='Loss-D')
plt.plot(histLossG, label='Loss-G')
plt.legend()
plt.title("Loss history")
plt.show()

# 두 주가 수익률의 산포도를 그린다
plt.figure(figsize=(8,7))
fakeData = sess.run(Gz, feed_dict={z : getNoise(m=1000)})
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.scatter(realData[:, 0], realData[:, 1], c='blue', s=5, label="Real Data")
plt.scatter(fakeData[:, 0], fakeData[:, 1], c='red', s=5, label="Fake Data")
plt.xlabel('Stock-A')
plt.ylabel('Stock-B')
plt.legend()
plt.title("Correlation")
plt.show()
print("Corr-Real Data = ", np.corrcoef(realData[:, 0], realData[:, 1])[0,1])
print("Corr-Fake Data = ", np.corrcoef(fakeData[:, 0], fakeData[:, 1])[0,1])

# GBM/GAN 모형을 이용하여 두 주가를 두 번 생성한다.
S1 = S2 = 1 
mu = 0.0175 / 252     # daily drift rate
for i in range(2):
    stockA = []
    stockB = []
    for k in range(500):
        w1, w2 = sess.run(Gz, feed_dict={z : getNoise(m=1)})[0]
        S1 = S1 * np.exp((mu - 0.5 * volA**2) + volA * w1)
        S2 = S2 * np.exp((mu - 0.5 * volB**2) + volB * w2)
        stockA.append(S1)
        stockB.append(S2)
    
    # 두 주가와 상관관계, 스프레드 차트, 스프레드 분포를 육안으로 확인해 본다
    fig = plt.figure(figsize=(10, 8))
    p1 = fig.add_subplot(2,2,1)
    p2 = fig.add_subplot(2,2,2)
    p3 = fig.add_subplot(2,2,3)
    p4 = fig.add_subplot(2,2,4)
    
    normA = (stockA - np.mean(stockA)) / np.std(stockA)
    normB = (stockB - np.mean(stockB)) / np.std(stockB)
    p1.plot(normA, color='red', linewidth=1, label='Stock-A')
    p1.plot(normB, color='blue', linewidth=1, label='Stock-B')
    p1.legend()
    p1.set_title("Stock Price Chart")
    
    rtnA = np.diff(stockA)
    rtnB = np.diff(stockB)
    rtnA = (rtnA - np.mean(rtnA)) / np.std(rtnA)
    rtnB = (rtnB - np.mean(rtnB)) / np.std(rtnB)
    p2.set_xlim(-4, 4)
    p2.set_ylim(-4, 4)
    p2.scatter(rtnA, rtnB, c='green', s=3)
    p2.set_title("Correlation")
    
    spread = normA - normB
    boundSpread = 2 * np.std(spread)
    p3.plot(spread, color='blue', linewidth=1)
    p3.axhline(y=0, color='black', linewidth=1)
    p3.axhline(y=boundSpread, linestyle='--', color='red', linewidth=1)
    p3.axhline(y=-boundSpread, linestyle='--', color='red', linewidth=1)
    p3.axhline(y=0, color='black', linewidth=1)
    p3.set_title("Pair-Spread Chart")
    
    r = np.copy(spread)
    r.sort()
    pdf = stats.norm.pdf(r, np.mean(r), np.std(r))
    p4.plot(r,pdf)
    p4.hist(r, density=True, bins=100)
    p4.set_title("Distribution")
    plt.show()