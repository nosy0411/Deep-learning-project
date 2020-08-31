# GBM 모형 실습 : 주어진 drift와 volatility로 가상의 주가를 생성한다.
# 
# 2018.12.14 아마추어 퀀트 (blog.naver.com/chunjein)
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 초기값 S0 에서 다음 GBM값 1개를 계산한다. drift, vol은 연간 단위
def GBM(drift, vol, S0=1):
    mu = drift / 252             # daily drift rate
    sigma = vol / np.sqrt(252)   # daily volatlity 
    
    # Monte Carlo simulation
    w = np.random.normal(0, 1, 1)[0]
    S = S0 * np.exp((mu - 0.5 * sigma**2) + sigma * w)
    return S

# n-기간 동안의 가상 주가를 생성한다
def StockPrice(n, drift, vol, S0):
    s = []
    for i in range(0, n):
        price = GBM(drift, vol, S0)
        s.append(GBM(drift, vol, S0))
        S0 = price
    return s

drift = 1.25 / 100
volatility = 50 / 100
S0 = 2000

def Chart_1():
    # 252일 동안의 주가를 생성하고 차트를 그려본다.
    # Drift = 1.25%/year, volatility = 20%/year, S0=2000
    s = StockPrice(252, drift, volatility, S0)
    plt.figure(1, figsize=(7,3.5))
    plt.plot(s, linewidth=1, color='blue')
    plt.xlabel('days')
    plt.ylabel('Stock Price')
    plt.show()
    
def Chart_2(m):
    # 252일 동안의 주가를 여러 번 생성하고 차트를 그려본다.
    # Drift = 1.25%/year, volatility = 20%/year, S0=2000
    for i in range(0, m):
        s = StockPrice(252, drift, volatility, S0)
        plt.figure(1, figsize=(7,3.5))
        plt.plot(s, linewidth=1)
    plt.xlabel('days')
    plt.ylabel('Stock Price')
    plt.show()
    
Chart_1()
Chart_2(100)