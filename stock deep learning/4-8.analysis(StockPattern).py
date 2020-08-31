# K-Means clustering 알고리즘으로 주가의 과거 패턴을 분류한다.
# 참고 자료 : Jordi Torres, First contact with tensorflow (page 27)
# ----------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# K-Means 분석 결과를 읽어온다
ds = pd.read_csv('dataset/4-7.patternResult.csv')
ds = ds.drop(ds.columns[0], 1)
k = 8

# 분석 (1)
# 동일 clust 뒤에 나오는 class를 관찰한다. 예를 들어, clust-0 의 class들이 주로 0이나 1이
# 나오는 특성이 있는지 확인해 본다. 만약 clust-0의 class가 주로 1 이라면, clust-0 패턴
# 이후에는 주가가 오르는 특성이 있다고 볼 수 있다.
clustClass = []
for c in range(k):
    clustClass.append(np.mean(ds[ds['cluster'] == c]['class']))

fig = plt.figure(figsize=(6, 2.5))
x = np.arange(k)
plt.bar(x, clustClass)
plt.title("Clust vs. class")
plt.xlabel("clust")
plt.ylabel("class")
plt.show()

# 분석 (2)
# 패턴 발생 빈도를 관찰한다.
classFreq = np.unique(ds['cluster'], return_counts=True)[1]
fig = plt.figure(figsize=(6, 2.5))
plt.bar(x, classFreq, color='brown', alpha=0.7)
plt.title("Clust Frequency")
plt.xlabel("clust")
plt.ylabel("frequency")
plt.show()

# 분석 (3)
# 특정 패턴 후에 특정 패턴이 발생하는 경향이 있는지 관찰한다.
# 예를 들어, clust-0 패턴 다음에 clust-1 패턴이 얼마나 자주 나오는지 관찰해 본다.
# nHop = 3으로 분석한 것이므로, shift=-6 으로 해야, 다음에 이어지는 패턴이 된다. (하루 차이는 발생함)
ds['next'] = ds['cluster'].shift(-6)
ds = ds.dropna()

fig = plt.figure(figsize=(10, 6))
for i in range(k):
    s = 'pattern-' + str(i)
    
    # 다음에 이어지는 패턴의 발생 빈도를 측정한다.
    nextClass = ds[ds['cluster'] == i]['next']
    nextCount = np.unique(nextClass, return_counts=True)[1]
    
    # 패턴 별 총 발생 빈도가 다르므로, 총 발생 빈도로 나눠준다.
    nextFreq = nextCount / classFreq
    p = fig.add_subplot(2, (k+1)//2, i+1)
    p.bar(x, nextFreq, color='purple', alpha=0.7)
    p.set_title('Cluster-' + str(i))

plt.tight_layout()
plt.show()

# clust = 2 인 패턴 몇 개만 그려본다
cluster = 2
plt.figure(figsize=(8, 3.5))
p = ds.loc[ds['cluster'] == cluster]
p = p.sample(frac=1).reset_index(drop=True)
for i in range(10):
    plt.plot(p.iloc[i][0:20])
    
plt.title('Cluster-' + str(cluster))
plt.show()

# clust = 3 인 패턴 몇 개만 그려본다
cluster = 3
plt.figure(figsize=(8, 3.5))
p = ds.loc[ds['cluster'] == cluster]
p = p.sample(frac=1).reset_index(drop=True)
for i in range(10):
    plt.plot(p.iloc[i][0:20])
    
plt.title('Cluster-' + str(cluster))
plt.show()
