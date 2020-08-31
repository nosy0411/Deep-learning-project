# Trade action을 이용하여 저점 매수, 고점 매도 방식으로 Label (HOLD, BUY, SELL)을 부여한다.
# 
# Usage : df = StockLabel.actionLabel(df, window=20, optimize=True, neighbor=1, verbose=False)
#
# 1. window : 고점과 저점을 판단하기 위한 sliding window
# 2. optimize : window를 변화해 가면서 누적 수익이 최대가 되는 window를 적용함
# 3. neighbor : 매수 (매도) 지점의 인근 지점도 매수 (매도)로 지정함. 분할 매수 (매도)
# 4. verboae : optimize 과정을 출력함
#
# 2018.12.03, 아마추어퀀트 (조성현)
# --------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MyUtil import YahooData, StockLabel

HOLD = 0
BUY = 1
SELL = 2
    
# 주가 데이터를 읽어온다
#df = YahooData.getStockDataYahoo('069500.KS', start='2010-01-01')
df = pd.read_csv('StockData/105560.csv')    # date를 indexer로 사용하지 않음.

# Trade action을 이용하여 최적 Label을 부여한다.
df, profitProfile = StockLabel.tradeLabel(df, optimize=True, neighbor=1)

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
n = 300
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

# window size에 대한 profit profile을 그려 본다 
maxProfit = np.max(profitProfile)
maxIdx = np.where(profitProfile == maxProfit)
plt.figure(figsize=(10, 6))
plt.plot(profitProfile)
plt.title("Profit profile")
plt.xlabel("Window size")
plt.ylabel("Cumulative Profit")
plt.axvline(x=maxIdx,  linestyle='dashed', color = 'red', linewidth=1)
plt.show()

print("Max Profit = %.4f" % maxProfit)
print("Optimal window size = %d" % maxIdx)
