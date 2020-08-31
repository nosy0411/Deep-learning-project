# n-기간 변동성과 수익률을 이용하여 Label (HOLD, BUY, SELL)을 부여한다.
# 
# 1. period : 기간
# 2. upper bound : period 기간 동안 수익률이 여기 이상이면 --> BUY (UP)
# 3. lower bound : period 기간 동안 수익률이 여기 이하이면 --> SELL (DOWN)
# 4. upper bound ~ lower bound 사이에 있으면 --> HOLD (FLAT)
#
# bound는 변동성을 고려하여 설정한다.
# Usage : df = StockLabel.returnLabel(df, upper=0.2, lower=-0.2, period=5)
#
# 2018.12.03, 아마추어퀀트 (조성현)
# --------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MyUtil import YahooData, StockLabel

HOLD = 0
BUY = 1
SELL = 2
    
# 주가 데이터를 읽어온다
#df = YahooData.getStockDataYahoo('105560.KS', start='2010-01-01')
df = pd.read_csv('StockData/105560.csv')    # date를 indexer로 사용하지 않음.

# n-기간 변동성과 수익률로 Label을 부여한다.
df = StockLabel.returnLabel(df, upper=0.2, lower=-0.2, period=5)

# 주가와 label을 차트로 확인해 본다
action = np.array(df['label'])
data = np.array(df['Close'])

# 주가 차트에 optimal action을 표시한다
buyMark = list(np.where(action == BUY)[0])
sellMark = list(np.where(action == SELL)[0])
plt.figure(figsize=(10, 4))
ax = np.arange(0, len(data))
plt.plot(ax, data, color='green', linewidth=1)
plt.plot(buyMark, data[buyMark], '^', markersize=8, markerfacecolor='red', alpha=0.5, markeredgecolor='gray', label="Buy")
plt.plot(sellMark, data[sellMark], 's', markersize=8, markerfacecolor='blue', alpha=0.5, markeredgecolor='gray', label="Sell")
plt.legend()
plt.show()

# 주가 차트 뒷 부분을 크게 그려본다
n = 200
lastAct = np.copy(action[-n:])
lastPrc = np.copy(data[-n:])
buyMark = list(np.where(lastAct == BUY)[0])
sellMark = list(np.where(lastAct == SELL)[0])
plt.figure(figsize=(10, 6))
ax = np.arange(0, len(lastPrc))
plt.plot(ax, lastPrc, color='green', linewidth=1)
plt.plot(buyMark, lastPrc[buyMark], '^', markersize=8, markerfacecolor='red', alpha=0.5, markeredgecolor='gray', label="Buy")
plt.plot(sellMark, lastPrc[sellMark], 's', markersize=8, markerfacecolor='blue', alpha=0.5, markeredgecolor='gray', label="Sell")
plt.legend()
plt.show()

