# DQN 실습 : 주가의 기술적 지표를 이용한 강화학습 예시
# 
# 아래 MDP로 최적 action을 찾아본다.
# 개별 주가는 랜덤 성향이 크므로 실무적으로 적용할 수준의 결과를 기대할 수 없으며,
# 이 예시는 단지 DQN의 동작 원리를 연습하기 위한 것이다.
# 
# State : 시가, 고가, 저가, 종가, 변동성, 단기이평, 장기이평, 모멘텀, 
#         MACD, RSI, Holding Time, 재고수량
# Action : BUY, SELL, HOLD
# Reward : profit = sell price - buy price
#
#
# 2018.12.14 아마추어 퀀트 (blog.naver.com/chunjein)
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import random

ALPHA =  0.0001
GAMMA = 0.9
EPSILON = 0.1
nAction = 3
nState = 12

ACTBUY = 0
ACTSELL = 1
ACTHOLD = 2

# Inventory
NOINVEN = 0.01      # 0을 피하기 위해 작은 수로 사용함
BUYINVEN = +0.5
SELLINVEN = -0.5

nBatch = 5
nStartReplay = 60
nMaxReplay = 100
wSaver = "Saver/11-1.DQN(stock).h5"
dataFile = "dataset/11-1.DQN(stock_data).csv"   # 학습용 데이터 (KB금융)
testFile = "dataset/11-1.DQN(stock_test).csv"   # 시험용 데이터 (신한지주)

# Experiance replay memory를 확보한다
replayMemory = deque(maxlen = nMaxReplay)

# DQN model을 생성한다
def buildDQN():
    dqn = Sequential()
    dqn.add(Dense(64, input_dim=nState, activation='relu'))
    dqn.add(Dense(64, activation='relu'))
    dqn.add(Dense(nAction, activation='linear'))
    adam = optimizers.Adam(lr = ALPHA)
    dqn.compile(loss='mse', optimizer=adam)
    return dqn

# DQN에 state를 입력하고 output을 측정한다 (인출, 회상)
def recallDQN(dqn, state):
    state = np.reshape(state, [1, nState])
    return dqn.predict(state)[0]

def myArgmax(d):
    maxValue = np.max(d)
    indices = np.where(np.asarray(d) == maxValue)[0]
    return np.random.choice(indices)

# Action을 선택한다
def ChooseAction(dqn, state, e):
    nInventory = state[10]

    # Exploration
    if (np.random.rand() < e):
        if nInventory > 0.3:
            # 매수 재고가 있으면
            return np.random.choice([ACTSELL, ACTHOLD])
        elif nInventory < -0.3:
            # 매도 재고가 있으면
            return np.random.choice([ACTBUY, ACTHOLD])
        else:
            # 재고가 없으면
            return np.random.choice([ACTBUY, ACTSELL, ACTHOLD])
    
    # Exploitation
    q = recallDQN(dqn, state)
    if nInventory > 0.3:    # 매수 재고 = +0.5 임. 안전하게 0.3 이상이면 매수 재고임.
        # 매수 재고가 있으면, SELL or HOLD (추가 매수는 금지)
        if q[ACTSELL] > q[ACTHOLD]:
            return ACTSELL
        else:
            return ACTHOLD
    elif nInventory < -0.3: # 매도 재고 = -0.5 임. 안전하게 -0.3 이하이면 매도 재고임.
        # 매도 재고가 있으면, BUY or HOLD (추가 매도는 금지)
        if q[ACTBUY] > q[ACTHOLD]:
            return ACTBUY
        else:
            return ACTHOLD
    else:
        # 재고가 없으면, BUY, SELL, or HOLD
        return myArgmax(q)

def Training(dqn1, dqn2, data):
    # 초기 상태를 설정한다.
    buyPrice = 0
    sellPrice = 0
    holdTime = 1
    trajHoldTime = []
    
    # 매일 매일 거래하지 않고 일정기간을 건너뛰면서 거래한다.
    start = int(np.random.rand() * 100) + 1
    step = int(np.random.normal(20, 5)) + 1      # 평균 20일 표준편차 5일
    step = np.max([1, step])
    
    currInventory = NOINVEN
    currState = list(data.iloc[start-1])
    currState.append(currInventory)
    currState.append(np.tanh(holdTime / 50.0))  # Holding Time : 0 ~ 1 사잇값으로 표준화
    currAction = ChooseAction(dqn1, currState, EPSILON)
    
    for i in range(start, len(data), step):
        terminal = False
        reward = 0
        if currInventory > 0.3:
            # 매수 재고가 있으면 (SELL이면 청산, HOLD이면 아무거도 안함)
            if currAction == ACTSELL:
                # 청산한다
                sellPrice = data['Close'][i]
                reward = sellPrice - buyPrice
                terminal = True
            else:
                nextInventory = currInventory
        elif currInventory < -0.3:
            # 매도 재고가 있으면 (BUY이면 청산, HOLD이면 아무거도 안함)
            if currAction == ACTBUY:
                # 청산한다
                buyPrice = data['Close'][i]
                reward = sellPrice - buyPrice
                terminal = True
            else:
                nextInventory = currInventory
        else:
            # 재고가 없으면
            if currAction == ACTBUY:
                # 매수한다
                nextInventory = BUYINVEN
                buyPrice = data['Close'][i]
                holdTime = 2
            elif currAction == ACTSELL:
                nextInventory = SELLINVEN
                sellPrice = data['Close'][i]
                holdTime = 2
            else:
                nextInventory = currInventory
        
        if currAction == ACTHOLD and nextInventory != NOINVEN:
            # 재고 보유상태에서 Holding하면 reward를 얼마를 줘야할까?
            # 보유 기간이 길어질수록 보유 위험에 대한 penalty를 부여해야할 것 같은데...
            # 현재는 reward = 0으로 처리함. 개선이 필요함.
            holdTime += step
        
        if terminal:
            trajHoldTime.append(holdTime)
            holdTime = 1
            nextInventory = NOINVEN
            
        nextState = list(data.iloc[i])
        nextState.append(nextInventory)
        nextState.append(np.tanh(holdTime / 50.0))
        nextAction = ChooseAction(dqn1, nextState, EPSILON)

        if terminal:
            G = reward
        else:
            nextQ = recallDQN(dqn2, nextState)[nextAction]  # Target network을 이용한다
            G = reward + GAMMA * nextQ
        
        # 업데이트할 current State, action, G를 experience memory에 저장해 둔다
        replayMemory.append((currState, currAction, G))
        
        # replayMemory에 nStartReplay 이상 경험치가 쌓여있으면, 여기서 nBatch개만 sampling하여 학습한다.
        if len(replayMemory) > nStartReplay:
            sampleData = random.sample(replayMemory, nBatch)
            
            # sampleData 에서 currState, action, G를 꺼낸다
            sampleState = np.zeros((nBatch, nState))
            sampleAction = []
            sampleG = []
            for k in range(nBatch):
                sampleState[k] = sampleData[k][0]
                sampleAction.append(sampleData[k][1])
                sampleG.append(sampleData[k][2])
        
            # dqn1 출력층 value의 target (desired output)을 측정한다
            target = dqn1.predict(sampleState)

            # sampleAction에 해당하는 targar을 변경한다. --> w가 업데이트 될 것임.
            # 나머지 action에 대해서는 기존 값을 유지한다. --> w가 업데이트 되지 않음.
            for k in range(nBatch):
                target[k][sampleAction[k]] = sampleG[k]

            # 출력층 value가 target이 나오도록 dqn1의 weight를 update한다
            dqn1.fit(sampleState, target, epochs=1, batch_size=nBatch, verbose=False)

        currState = np.copy(nextState)
        currAction = nextAction
        currInventory = nextInventory
        
        if terminal:
            # 한 에피소드가 끝나면 dqn2의 weight을 dqn1과 동일하게 맞춘다
            dqn2.set_weights(dqn1.get_weights())
         
    return trajHoldTime

# 학습을 시작한다
def learn(n):
    dqn1 = buildDQN()
    dqn2 = buildDQN()   # Target Network
    try:
        dqn1.load_weights(wSaver)
        dqn2.load_weights(wSaver)
        print("# 기존 학습 결과 Weight를 적용하였습니다.")
    except:
        print("# DQN Weight을 랜덤 초기화 하였습니다.")
    
    # 저장된 학습 데이터를 읽어와서 학습한다.
    fs = pd.read_csv(dataFile)
    for i in range(0, n):
        trajHoldTime = Training(dqn1, dqn2, fs)        
        print("%d) 학습 완료 (%.2f)" % (i+1, np.mean(trajHoldTime)))
    
    # 학습 결과 Weight를 저장해 둔다
    dqn1.save_weights(wSaver)

# weight를 dump한다. 혹시 발산하는지 등을 육안으로 확인한다.
def dumpWeight():
    dqn = buildDQN()
    try:
        dqn.load_weights(wSaver)
        print("# 기존 학습 결과 Weight를 적용하였습니다.")
    except:
        print("# DQN Weight을 랜덤 초기화 하였습니다.")
    
    for w in dqn.layers:
        print(w.get_weights())
        
# 학습 결과대로 가상 거래를 수행한다.
# --------------------------------
def trade():
    dqn = buildDQN()
    try:
        dqn.load_weights(wSaver)
        print("# 기존 학습 결과 Weight를 적용하였습니다.")
    except:
        print("# DQN Weight을 랜덤 초기화 하였습니다.")
    
    # 시험용 데이터를 읽어와서 가상 거래를 수행해 본다.
    fs = pd.read_csv(testFile)
    
    # 초기 상태
    buyPrice = 0
    sellPrice = 0
    profit = 0
    reward = []
    holdTime = 1
    trajHoldTime = []
    tradeCnt = 1
    buyMark = []
    sellMark = []
    
    # 매일 매일 거래하지 않고 일정기간을 건너뛰면서 거래한다.
    start = int(np.random.rand() * 100) + 1
    step = int(np.random.normal(5, 3)) + 1      # 평균 5일 표준편차 3일
    step = np.max([1, step])
    
    currInventory = NOINVEN
    currState = list(fs.iloc[0])
    currState.append(currInventory)
    currState.append(np.tanh(holdTime / 50.0))  # Holding Time : 0 ~ 1 사잇값으로 표준화
    action = ChooseAction(dqn, currState, 0)
    
    for i in range(start, len(fs), step):
#        print(recallDQN(dqn, currState))
        if currInventory > 0.3:
            # 매수 재고가 있으면 (SELL이면 청산, HOLD이면 아무거도 안함)
            if action == ACTSELL:
                # 청산한다
                currInventory = NOINVEN
                sellPrice = fs['Close'][i]
                profit += sellPrice - buyPrice
                reward.append(profit)
                trajHoldTime.append(holdTime)
                holdTime = 1
                tradeCnt += 1
                sellMark.append(i)
        elif currInventory < -0.3:
            # 매도 재고가 있으면 (BUY이면 청산, HOLD이면 아무거도 안함)
            if action == ACTBUY:
                # 청산한다
                currInventory = NOINVEN
                buyPrice = fs['Close'][i]
                profit += sellPrice - buyPrice
                reward.append(profit)
                trajHoldTime.append(holdTime)
                holdTime = 1
                tradeCnt += 1
                buyMark.append(i)
        else:
            # 재고가 없으면
            if action == ACTBUY:
                # 매수한다
                currInventory = BUYINVEN
                buyPrice = fs['Close'][i]
                holdTime = 2
                buyMark.append(i)
            elif action == ACTSELL:
                currInventory = SELLINVEN
                sellPrice = fs['Close'][i]
                holdTime = 2
                sellMark.append(i)
        
        if action == ACTHOLD and currInventory != NOINVEN:
            holdTime += step
            
        currState = list(fs.iloc[i])
        currState.append(currInventory)
        currState.append(np.tanh(holdTime / 50.0))
        action = ChooseAction(dqn, currState, 0)
            
    # 누적 profit을 확인한다
    plt.figure(figsize=(8, 6))
    plt.plot(reward, linewidth=1, color='red', marker='o', markersize = 4)
    plt.title('Accumulated Profit')
    plt.show()
    
    print("# Profit = %.4f, Trade count = %d" % (profit, tradeCnt))
    print("# 평균 보유 기간 = %.2f days" % (np.mean(trajHoldTime)))

    # optimal action을 종가 차트에 표시해 본다.
    plt.figure(figsize=(14, 8))
    ax = np.arange(0, len(fs))
    plt.plot(ax, fs['Close'], color='lightgreen', linewidth=1, label='Stock Price')
    plt.plot(buyMark, fs.iloc[buyMark]['Close'], '^', markersize=8, markerfacecolor='red', alpha=0.7, markeredgecolor='gray', label="Buy")
    plt.plot(sellMark, fs.iloc[sellMark]['Close'], 's', markersize=8, markerfacecolor='blue', alpha=0.7, markeredgecolor='gray', label="Sell")
    plt.legend()
    plt.show()

#지금은 학습시켜놓은 파일이 있기 때문에 100정도만 학습시켜도 되지 처음하는 경우에는 큰 값을 줘야함.
learn(100)
trade()