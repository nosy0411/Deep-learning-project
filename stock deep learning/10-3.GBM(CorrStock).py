# GBM 모형과 Cholesky 분해식과 GBM을 이용하여 상관관계를 갖는 두 개의 주가를 생성한다
#
# 2017.8.9, 아마추어퀀트 (조성현)
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 초기값 S0 에서 다음 GBM값 1개를 계산한다. drift, vol은 연간 단위
def GBM(w, drift, vol, S0=1):
    mu = drift / 252             # daily drift rate
    sigma = vol / np.sqrt(252) 	# daily volatlity 
    
    # Monte Carlo simulation
    S = S0 * np.exp((mu - 0.5 * sigma**2) + sigma * w)
    return S

# n-기간 동안의 가상 주가 2개를 생성한다
def CorStockPrice(n, corr, drift, vol, S0):
    sA = []
    sB = []
    S0A = S0
    S0B = S0
    for i in range(0, n):
        wA = np.random.normal(0, 1, 1)[0]
        wB = np.random.normal(0, 1, 1)[0]
        zA = wA
        zB = wA * corr + wB * np.sqrt(1 - corr ** 2)
        
        pA = GBM(zA, drift, vol, S0A)
        pB = GBM(zB, drift, vol, S0B)
        
        sA.append(pA)
        sB.append(pB)
        
        S0A = pA
        S0B = pB
        
    s = pd.DataFrame(sA, columns = ['A'])
    s['B'] = sB
    return s

# n-일 동안의 주가를 생성하고 차트를 그려본다.
def simulate(corr, n):
    # 상관관계 주가를 생성한다
    s = CorStockPrice(n, corr, drift = 1.25 / 100, vol = 20 / 100, S0=2000)
    
    # s1과 s2의 실제 수익률 상관계수를 계산한다
    s['rtnA'] = pd.DataFrame(s['A']).apply(lambda x: x - x.shift(1))
    s['rtnB'] = pd.DataFrame(s['B']).apply(lambda x: x - x.shift(1))
    s = s.dropna()
    s.index = range(len(s.index))
    realCorr = np.corrcoef(s['rtnA'], s['rtnB'])[1,0]
    
    # 두 주가를 그린다.
    plt.figure(1, figsize=(8,3))
    plt.title("Correlated Stocks (Corr = %.2f)" % realCorr)
    plt.plot(s['A'], linewidth=1, color='blue', label = 'Stock-A')
    plt.plot(s['B'], linewidth=1, color='red', label = 'Stock-B')
    plt.xlabel('days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    # 수익률 산포도를 그린다.
    plt.figure(figsize=(8,3))
    plt.scatter(s['rtnA'], s['rtnB'], c='brown', s=10)
    plt.xlabel('Stock Return-A')
    plt.ylabel('Stock Return-B')
    plt.show()
    
