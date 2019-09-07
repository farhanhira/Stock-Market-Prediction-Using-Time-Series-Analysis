#!/usr/bin/env python
# coding: utf-8

# In[166]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6


# In[225]:


dataset=pd.read_csv("SPEd500.csv")


# In[226]:


dataset['DATE']= pd.to_datetime(dataset['DATE'], infer_datetime_format=True)
indexedDataset=dataset.set_index(['DATE'])


# In[227]:


from datetime import datetime
indexedDataset.dtypes


# In[229]:


pd.to_numeric(indexedDataset['SP500'])


# In[230]:


plt.xlabel("Date")
plt.ylabel("SP500")
plt.plot(indexedDataset)


# In[231]:


rolmean=indexedDataset.rolling(window=365).mean()
rolstd=indexedDataset.rolling(window=365).std()
print(rolmean, rolstd)


# In[232]:


orig=plt.plot(indexedDataset, color='blue', label='Original')
mean=plt.plot(rolmean, color='red', label='rMean')
std=plt.plot(rolstd, color='black', label='rStd')
plt.legend(loc='best')
plt.title('Rolling Mean & STD')
plt.show(block=False)


# In[233]:


indexedDataset.dropna(inplace=True)
indexedDataset.head(5)
from statsmodels.tsa.stattools import adfuller


# In[234]:


print('results of dikey-fuller test:')
dftest=adfuller(indexedDataset['SP500'], autolag='AIC')


# In[235]:


dfoutput=pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', '#observations'])
for key,value in dftest[4].items():
    dfoutput['Critical value (%s)'%key]=value
    
print (dfoutput)


# In[236]:


indexedDataset_logScale=np.log(indexedDataset)
plt.plot(indexedDataset_logScale)


# In[237]:


movingAverage=indexedDataset_logScale.rolling(window=12).mean()
movingSTD=indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage, color='red')


# In[238]:


datasetLogScaleMinusMA=indexedDataset_logScale-movingAverage
datasetLogScaleMinusMA.head(12)
datasetLogScaleMinusMA.dropna(inplace=True)
datasetLogScaleMinusMA.head(10)


# In[239]:


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
    dftest=adfuller(timeseries['SP500'], autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', '#observations'])
    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)'%key]=value
    
    print (dfoutput)


# In[240]:


test_stationarity(datasetLogScaleMinusMA)


# In[241]:


exponentialDecayWeightedAverage=indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')


# In[242]:


datasetLogScaleMinusExponentialDecayAverage= indexedDataset-exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusExponentialDecayAverage)


# In[243]:


datasetLogDiffShifting= indexedDataset_logScale-indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[244]:


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)


# In[245]:


from statsmodels.tsa.seasonal import seasonal_decompose
indexedDataset_logScale.dropna(inplace=True)
decomposition =seasonal_decompose(indexedDataset_logScale)
trend =decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid


# In[246]:


plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(indexedDataset_logScale, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(indexedDataset_logScale, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(indexedDataset_logScale, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[247]:


decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[262]:


from statsmodels.tsa.stattools import acf, pacf

lag_acf=acf(datasetLogDiffShifting, nlags=20)


# In[263]:


lag_pacf=pacf(datasetLogDiffShifting, nlags=20, method='ols')


# In[264]:


plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[278]:


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(indexedDataset_logScale, order=(2,1,0))
results_AR = model.fit (disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting['SP500'])**2))
print('Plotting AR Model')


# In[279]:


model=ARIMA(indexedDataset_logScale, order=(0,1,2))
results_MA = model.fit (disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting['SP500'])**2))
print('Plotting AR Model')


# In[281]:


model=ARIMA(indexedDataset_logScale, order=(2,1,0))
results_ARIMA = model.fit (disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting['SP500'])**2))


# In[284]:


predicitons_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues, copy=True


# In[ ]:




