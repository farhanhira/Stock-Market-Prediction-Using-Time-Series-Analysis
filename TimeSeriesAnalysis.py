#!/usr/bin/env python
# coding: utf-8

# In[135]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[136]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
from pandas import DataFrame
import matplotlib.pyplot as plt


# In[137]:


from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# In[138]:


df = pd.read_csv('aapl.csv', parse_dates=['Date'])
df.head(3)


# In[139]:


print (df.describe())
print ("=============================================================")
print (df.dtypes)


# In[140]:


df1 = df[['Date','Close']]
df1.head(3)


# In[141]:


# Setting the Date as Index
df_ts = df1.set_index('Date')
df_ts.sort_index(inplace=True)
print (type(df_ts))
print (df_ts.head(3))
print ("========================")
print (df_ts.tail(3))


# In[142]:


df_ts[df_ts.isnull()]


# In[143]:


len(df_ts[df_ts.isnull()])


# In[144]:


df_ts = df_ts.sort_index()
df_ts.index


# In[145]:


df_ts.Close.fillna(method='pad', inplace=True)


# In[146]:


df_ts[df_ts.Close.isnull()]
len(df_ts[df_ts.Close.isnull()])


# In[147]:


df_ts.plot()


# In[148]:


# Dickey Fuller Test Function
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    from statsmodels.tsa.stattools import adfuller
    print('Results of Dickey-Fuller Test:')
    print ("==============================================")
    
    dftest = adfuller(timeseries, autolag='AIC')
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    print(dfoutput)


# In[149]:


# Stationarity Check - Lets do a quick check on Stationarity with Dickey Fuller Test 
# Convert the DF to series first
ts = df_ts['Close']
test_stationarity(ts)


# In[152]:


rolmean = ts.rolling(window=365).mean()
rolvar = ts.rolling(window=365).std()

plt.plot(ts, label='Original')
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolvar, label='Rolling Standard Variance')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# In[151]:


orig=plt.plot(ts, color='blue', label='Original')
mean=plt.plot(rolmean, color='red', label='rMean')
std=plt.plot(rolstd, color='black', label='rStd')
plt.legend(loc='best')
plt.title('Rolling Mean & STD')
plt.show(block=False)


# In[75]:


ts.dropna(inplace=True)
ts.head(5)
from statsmodels.tsa.stattools import adfuller


# In[77]:


print('results of dikey-fuller test:')
dftest=adfuller(ts, autolag='AIC')


# In[78]:


dfoutput=pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', '#observations'])
for key,value in dftest[4].items():
    dfoutput['Critical value (%s)'%key]=value
    
print (dfoutput)


# In[79]:


ts_logScale=np.log(ts)
plt.plot(ts_logScale)


# In[80]:


movingAverage=ts_logScale.rolling(window=12).mean()
movingSTD=ts_logScale.rolling(window=12).std()
plt.plot(ts_logScale)
plt.plot(movingAverage, color='red')


# In[81]:


ts_LogScaleMinusMA=ts_logScale-movingAverage
ts_LogScaleMinusMA.head(12)
ts_LogScaleMinusMA.dropna(inplace=True)
ts_LogScaleMinusMA.head(10)


# In[84]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD=timeseries.rolling(window=12).std()
    orig=plt.plot(timeseries, color='blue', label='Original')
    mean=plt.plot(movingAverage, color='red', label='rMean')
    std=plt.plot(movingSTD, color='black', label='rStd')
    plt.legend(loc='best')
    plt.title('Rolling Mean & STD')
    plt.show(block=False)
    print('results of dikey-fuller test:')
    dftest=adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', '#observations'])
    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)'%key]=value
    
    print (dfoutput)


# In[85]:


test_stationarity(ts_LogScaleMinusMA)


# In[86]:


exponentialDecayWeightedAverage=ts_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(ts_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')


# In[88]:


ts_LogScaleMinusExponentialDecayAverage= ts-exponentialDecayWeightedAverage
test_stationarity(ts_LogScaleMinusExponentialDecayAverage)


# In[89]:


ts_LogDiffShifting= ts_logScale-ts_logScale.shift()
plt.plot(ts_LogDiffShifting)


# In[90]:


ts_LogDiffShifting.dropna(inplace=True)
test_stationarity(ts_LogDiffShifting)


# In[92]:


ts_logScale.head()


# In[95]:


from statsmodels.tsa.seasonal import seasonal_decompose
ts_logScale.dropna(inplace=True)
decomposition = seasonal_decompose(ts_logScale, freq=30)
trend =decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid


# In[96]:


plt.subplot(411)
plt.plot(ts_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(ts_logScale, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(ts_logScale, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(ts_logScale, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[97]:


decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[100]:


from statsmodels.tsa.stattools import acf, pacf

lag_acf=acf(ts_LogDiffShifting, nlags=20)
lag_pacf=pacf(ts_LogDiffShifting, nlags=20, method='ols')


# In[101]:


plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_LogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_LogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_LogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_LogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[114]:


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(ts_logScale, order=(1,1,1))
results_AR = model.fit (disp=-1)
plt.plot(ts_LogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_LogDiffShifting)**2))
print('Plotting AR Model')


# In[115]:


model=ARIMA(ts_logScale, order=(1,1,1))
results_ARIMA = model.fit (disp=-1)
plt.plot(ts_LogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_LogDiffShifting)**2))


# In[117]:


predicitons_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predicitons_ARIMA_diff.head())


# In[118]:


predicitons_ARIMA_diff_cumsum=predicitons_ARIMA_diff.cumsum()
print(predicitons_ARIMA_diff_cumsum)


# In[121]:


predictions_ARIMA_log=pd.Series(ts_logScale.ix[0], index=ts_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predicitons_ARIMA_diff_cumsum, fill_value=0)


# In[123]:


predictions_ARIMA_log.head()


# In[124]:


predicitons_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predicitons_ARIMA)


# In[125]:


ts_logScale


# In[127]:


results_ARIMA.plot_predict(1,9575)


# In[132]:


results_ARIMA.forecast(14)


# In[131]:





# In[ ]:




