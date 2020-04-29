#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import time
import numpy as np
import pandas as pd
import talib as ta
from talib import MA_Type
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
from stock_reader import reader
import mxnet.contrib.tensorboard
import sklearn.preprocessing
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import warnings


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


context = mx.cpu(); model_ctx=mx.cpu()
mx.random.seed(1719)


# In[4]:


def parser(x):
    return datetime.datetime.strptime(x,'%m/%d/%Y')


# In[5]:


#  load data
dataset_ex_df = reader()


# In[6]:


def get_technical_indicators(dataset):
    
    #이동평균선
    dataset['ma3'] = ta.MA(dataset_ex_df['close'], timeperiod = 3)
    dataset['ma5'] = ta.MA(dataset_ex_df['close'], timeperiod = 5)
    dataset['ma10'] = ta.MA(dataset_ex_df['close'], timeperiod = 10)
    dataset['ma20'] = ta.MA(dataset_ex_df['close'], timeperiod = 20)
    dataset['ma60'] = ta.MA(dataset_ex_df['close'], timeperiod = 60)


#     #지수이평선
    dataset['ema5'] = ta.EMA(dataset_ex_df['close'],timeperiod = 5)
    dataset['ema10'] = ta.EMA(dataset_ex_df['close'],timeperiod = 10)
    dataset['12ema'] = dataset['close'].ewm(span=12).mean()
    dataset['ema20'] = ta.EMA(dataset_ex_df['close'],timeperiod = 20)
    dataset['26ema'] = dataset['close'].ewm(span=26).mean()
    dataset['ema60'] = ta.EMA(dataset_ex_df['close'],timeperiod = 60)

    
#     #거래량 이평선
    dataset['vma5'] = ta.MA(dataset_ex_df['volume'], timeperiod=5, matype=0)
    dataset['vma10'] = ta.MA(dataset_ex_df['volume'], timeperiod=10, matype=0)
    dataset['vma20'] = ta.MA(dataset_ex_df['volume'], timeperiod=20, matype=0)
    dataset['vma60'] = ta.MA(dataset_ex_df['volume'], timeperiod=60, matype=0)

    
#     #이격도
    dataset['dis5'] = ((dataset_ex_df['close'] - (dataset_ex_df['close'].rolling(window=5).mean()))/(dataset_ex_df['close'].rolling(window=5).mean()))*100
    dataset['dis10'] = ((dataset_ex_df['close'] - (dataset_ex_df['close'].rolling(window=10).mean()))/(dataset_ex_df['close'].rolling(window=10).mean()))*100
    dataset['dis20'] = ((dataset_ex_df['close'] - (dataset_ex_df['close'].rolling(window=20).mean()))/(dataset_ex_df['close'].rolling(window=20).mean()))*100


#     #볼린져 벤드
    dataset['20sd'] = dataset['close'].rolling(window = 20).std()
    dataset['upper'] = (dataset['close'].rolling(window = 20).mean()) + (dataset['20sd']*2)
    dataset['lower'] = (dataset['close'].rolling(window = 20).mean()) - (dataset['20sd']*2)


#     #ATR 
    dataset['atr5'] = ta.ATR(dataset_ex_df['high'], dataset_ex_df['low'], dataset_ex_df['close'], timeperiod=5)
    dataset['atr10'] = ta.ATR(dataset_ex_df['high'], dataset_ex_df['low'], dataset_ex_df['close'], timeperiod=10)
    dataset['atr20'] = ta.ATR(dataset_ex_df['high'], dataset_ex_df['low'], dataset_ex_df['close'], timeperiod=20) 
    dataset['atr60'] = ta.ATR(dataset_ex_df['high'], dataset_ex_df['low'], dataset_ex_df['close'], timeperiod=60) 
    

#     #MACD 
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    
    dataset['momentum'] = (dataset['close']/100)-1
    
    return dataset


# In[7]:


dataset_TI_df = get_technical_indicators(dataset_ex_df).reset_index()


# In[8]:


dataset_TI_df.tail()


# In[9]:


def plot_technical_indicators(dataset, last_days): #기술적지표 그래프
    plt.figure(figsize=(16, 10), dpi=150)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)
    
    # Plot first subplot
    plt.subplot(311)
    plt.plot(dataset['ma5'],label='MA 5', color='gold',linestyle='--')
    plt.plot(dataset['close'],label='Closing Price', color='b')
    plt.plot(dataset['ma20'],label='MA 20', color='r',linestyle='--')
    plt.plot(dataset['upper'],label='Upper Band', color='c')
    plt.plot(dataset['lower'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower'], dataset['upper'], alpha=0.35)
    plt.title('Technical indicators - last {} days.'.format(last_days))
    plt.ylabel('원or달러')
    plt.xlabel('시간')
    plt.legend()

    # Plot second subplot
    plt.subplot(312)
    plt.title('MACD')
    plt.plot(dataset['MACD'],label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['momentum'],label='Momentum', color='b',linestyle='-')
    plt.plot(dataset['20sd'],label='20일표준편차', color='black')
    plt.ylabel('macd')
    plt.xlabel('시간')
    plt.legend()

    
    plt.subplot(313)
    plt.title('이격도')
    plt.plot(dataset['dis5'],label='dis5', color='green')
    plt.plot(dataset['dis10'],label='dis10', color='b')
    plt.plot(dataset['dis20'],label='dis20', color='r')
    plt.ylabel('%')
    plt.xlabel('시간')
    plt.legend()
    
    plt.show()


# In[10]:


plot_technical_indicators(dataset_TI_df, 400)


# In[11]:


data_FT = dataset_ex_df.reset_index()[['Date', 'close']]
close_fft = np.fft.fft(np.asarray(data_FT['close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))


# In[12]:


plt.figure(figsize=(14, 7), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 100]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
plt.plot(data_FT['close'],  label='Real')
plt.xlabel('Days')
plt.ylabel('원or달러')
plt.title('Figure 3: close stock prices & Fourier transforms')
plt.legend()
plt.show()


# In[13]:


def get_fourier(dataset):
    data_FT = dataset[['Date', 'close']]
    close_fft = np.fft.fft(np.asarray(data_FT['close'].tolist()))
    close_fft = np.fft.ifft(close_fft)
    close_fft
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_list_m10= np.copy(fft_list); fft_list_m10[100:-100]=0
    dataset['Fourier'] = pd.DataFrame(fft_list_m10).apply(lambda x: np.abs(x))
    #dataset['absolute'] = dataset['Fourier'].apply(lambda x: np.abs(x))
    return dataset


# In[14]:


dataset_TI_df = get_fourier(dataset_ex_df.reset_index())


# In[15]:


dataset_TI_df.head(100)


# In[16]:


from collections import deque
items = deque(np.asarray(fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(fft_df)/2)))
plt.figure(figsize=(10, 7), dpi=80)
plt.stem(items)
plt.title('Figure 4: Components of Fourier transforms')
plt.show()


# In[17]:


from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime

series = data_FT['close']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[18]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
plt.figure(figsize=(10, 7), dpi=80)
plt.show()


# In[19]:


from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[20]:


X = series.values
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

dataset_TI_df['ARIMA'] = pd.DataFrame(predictions)


# In[21]:


error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# In[22]:


# Plot the predicted (from ARIMA) and real prices

plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test, color='blue', label='Real')
plt.plot(predictions, color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 5: ARIMA model on stock')
plt.legend()
plt.show()


# In[23]:


dataset_TI_df.head(8)


# In[24]:


print('Total dataset has {} samples, and {} features.'.format(dataset_TI_df.shape[0],                                                               dataset_TI_df.shape[1]))


# In[25]:


def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['close']
    X = data.iloc[:,1:19]
    
    train_samples = int(X.shape[0] * 0.65)
 
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]
    
    return (X_train, y_train), (X_test, y_test)


# In[26]:


# Get training and test data
(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df)


# In[27]:


regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=200,base_score=0.7,colsample_bytree=1,learning_rate=0.05)


# In[28]:


xgbModel = regressor.fit(X_train_FI,y_train_FI,                          eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],                          verbose=False)


# In[29]:


eval_result = regressor.evals_result()


# In[30]:


training_rounds = range(len(eval_result['validation_0']['rmse']))


# In[31]:


plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training Vs Validation Error')
plt.legend()
plt.show()


# In[32]:


fig = plt.figure(figsize=(8,8))
plt.xticks(rotation='vertical')
plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
plt.title('Figure 6: Feature importance of the technical indicators.')
plt.show()


# In[33]:


dataset_lstm_df = dataset_TI_df.drop(columns='Date')
dataset_lstm_df.head(7)


# In[34]:


print('Total dataset has {} samples, and {} features.'.format(dataset_lstm_df.shape[0],                                                               dataset_lstm_df.shape[1]))


# # 우선 ohlc 를 이용하여 lstm 모델을 구축해봄

# In[35]:


# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import Flatten


# In[36]:


dataset = dataset_ex_df[['open', 'close','high','low']]


# In[37]:


dataset.tail()


# In[38]:


# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM


# In[39]:


#FOR REPRODUCIBILITY
np.random.seed(7)


# In[40]:


# IMPORTING DATASET 
dataset = dataset.reset_index()
dataset


# In[41]:


# dataset = dataset.reindex(index = dataset.index[::-1])
dataset


# In[42]:


# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)


# In[43]:


#TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset.mean(axis = 1)
HLC_avg = dataset[['high', 'low', 'close']].mean(axis = 1)
close_val = dataset[['close']]


# In[44]:


# PLOTTING All INDICATORS IN PLOT
plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
plt.plot(obs, close_val, 'b', label = 'Closing price')
plt.plot(obs, HLC_avg,   'g', label = 'HLC avg')
plt.xlabel('Days')
plt.ylabel('OHLC average')
plt.show()


# In[45]:


# PREPARATION OF TIME SERIES DATASE
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) 
print("일평균 데이터 : {},\n {}".format( OHLC_avg[:5], type(OHLC_avg)))
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)
print("정규화 데이터 : {}, {}".format(type(OHLC_avg), OHLC_avg[:5]) )


# In[46]:


def new_dataset(dataset, step_size):
    import numpy as np 
    data_X, data_Y = [], []
    for i in range(len(dataset) - step_size - 1):
        data_X.append(dataset[i : (i+step_size), 0])  
        data_Y.append(dataset[i + step_size,     0])
    return np.array(data_X), np.array(data_Y)

# TRAIN-TEST SPLIT
train_OHLC = int(len(OHLC_avg) * 0.75)
test_OHLC  = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY   = new_dataset(test_OHLC, 1)
print(trainX.shape, trainY.shape)  
print(trainX[1], trainY[1])

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX  = np.reshape(testX,  (testX.shape[0],  1, testX.shape[1]))
trainX.shape, testX.shape


# # LSTM 모델 (전모델로) 돌려보는 구간
# 

# In[47]:


# LSTM MODEL
step_size = 1
model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))


# In[48]:


# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae']) # Try mae, adam, adagrad and compare!!!
model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=2)


# In[49]:


# MAE : 예측과 타깃 사이 거리의 절댓값으로 여기서 MAE가 0.5면 정규처리된 가격이 0.5 -> 예를들어 1000이 정규화되어서 0.01.정도가 된거면 0.5 0> 50000


# In[50]:


mae = model.evaluate(testX, testY, batch_size=16)
print('Mean Absolute Error for Y:', mae)


# In[51]:


# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[52]:


# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[53]:


# TRAINING rmse
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train : %.2f' % (trainScore))


# In[54]:


# mean_squared_error(trainY[0], trainPredict[:,0])


# In[55]:


# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))


# In[56]:


trainPredictPlot          = np.empty_like(OHLC_avg)
trainPredictPlot[ : , : ] = np.nan
trainPredictPlot[step_size : len(trainPredict) + step_size , : ] = trainPredict

testPredictPlot       = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (step_size*2) + 1 : len(OHLC_avg) - 1 , : ] = testPredict


# In[57]:


OHLC_avg = scaler.inverse_transform(OHLC_avg)

plt.figure(figsize=(12,6))
plt.plot(OHLC_avg,         'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'train predicted stock price')
plt.plot(testPredictPlot,  'b', label = 'test predicted stock price')
plt.xlabel('Time in Days')
plt.ylabel('OHLC Value of Apple Stocks')
plt.legend(loc = 'upper left')
plt.grid(); plt.show()


# In[58]:


# PREDICT FUTURE VALUES
last_val        = testPredict[-1:]
last_val_scaled = last_val / last_val
next_val        = model.predict(np.reshape(last_val_scaled, (1,1,1)))
last_price      = np.asscalar(last_val)
next_price      = np.asscalar(last_val * next_val)
print ("마지막 예측가격 : {} \n다음날 예측가격 : {} \n증감률 : {}".format(
        last_price, 
        next_price, 
        (next_price - last_val) / last_val) )

