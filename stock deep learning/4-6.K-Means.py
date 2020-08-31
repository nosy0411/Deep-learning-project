# K-Means clustering
# 참고 자료 : Jordi Torres, First contact with tensorflow (page 27)
#
# 2018.08.21, 아마추어퀀트 (조성현)
# ----------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

def createData(n):
    xy = []
    for i in range(n):
        if np.random.random() > 0.5:
            x = np.random.normal(0.0, 0.9)
            y = np.random.normal(0.0, 0.9)
        else:
            x = np.random.normal(3.0, 0.6)
            y = np.random.normal(1.0, 0.6)
        xy.append([x, y])
    
    return xy

# input data tensor를 생성한다
n = 1000
tf.reset_default_graph()
inputXY = tf.constant(createData(n))

# 초기 중심 좌표를 설정한다.
k = 5
tmpXY = tf.random_shuffle(inputXY)          # input data를 shuffling 한다
tmpCent = tf.slice(tmpXY, [0, 0], [k, -1])  # 앞의 k개를 중심좌표로 설정한다
initCent = tf.Variable(tmpCent)             # 초기 중심좌표 텐서

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
    # [[ 0.35267928 -0.34095716]
    #  [-0.03551814  0.5964925 ]
    #  [-0.38566497  0.6562558 ]
    #  [ 0.19521604  0.24940939]]
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
#[[ 0.03167805  0.29030013]
# [ 2.5208025   1.274969  ]
# [ 1.158408   -1.8294015 ]
# [ 3.8284082   1.1019807 ]]
meanXY = tf.reshape(tfConcat, [k, 2])
    
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

# 분류 결과를 표시한다
dataXY = sess.run(inputXY)
color = cm.rainbow(np.linspace(0, 1, k))
plt.figure(figsize=(8, 6))
for i, c in zip(range(k),color):
    plt.scatter(dataXY[finalClust == i, 0], dataXY[finalClust == i, 1], s=20, c=c, marker='o', alpha=0.5, label='cluster ' + str(i))

plt.scatter(finalCentXY[:,0], finalCentXY[:,1], s=250, marker='*', c='black', label='centroids')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.plot(trcErr)
plt.title("Error (by total distance)")
plt.show()

print("min Error = %.2f" % finalErr)
#sess.close()