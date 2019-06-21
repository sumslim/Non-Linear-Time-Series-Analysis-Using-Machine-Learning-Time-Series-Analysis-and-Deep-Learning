#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/home/sim06/Downloads/cer3.csv')


# In[136]:


time = pd.date_range(start='1961',periods=56,freq='Y')
df.set_index(time,inplace=True)
df.head()


# In[137]:


x = df.drop(df[['Cereals']],axis=1)
y = df[['Cereals']]


# In[144]:


y_train = y[0:45]
y_test = y[45:57]
x_train = x[0:45]
x_test = x[45:57]


# In[145]:


y_train.plot()
plt.show()


# In[146]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(y_train, lags=15, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(y_train, lags=15, ax=ax2)
plt.show()


# In[147]:


import statsmodels
print(statsmodels.tsa.stattools.adfuller(y_train.Cereals))


# In[148]:

plt.plot(y_train.diff())
plt.show()
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(y_train.diff(), lags=15, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(y_train.diff(), lags=15, ax=ax2)
plt.show()


# In[149]:
P=[0,1,2,3,4]
Q=[0,1,12]
for p in P:
    for q in Q:
        try:
            model=sm.tsa.ARIMA(endog=y_train,order=(p,1,q))
            results=model.fit()
            print(results.summary())
        except:
            pass

# In[150]:

model_selected=sm.tsa.ARIMA(endog=y_train,order=(0,1,1))
result_produced=model_selected.fit()
print(statsmodels.tsa.stattools.adfuller(result_produced.resid))
result_produced.plot_predict(1,50)
plt.show()
# In[151]:


result_produced.resid.plot()
plt.show()
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(result_produced.resid, lags=15, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(result_produced.resid, lags=15, ax=ax2)
plt.show()


# In[158]:


forecast,std,conf=result_produced.forecast(11)


# In[159]:


import numpy as np
s = np.asarray(forecast)


# In[160]:


m = []
for i in y_test.Cereals:
    m.append(i)


# In[165]:


plt.plot(s,color='green',label='Predicted')
plt.plot(m,color='red',label='True')
plt.legend()
plt.show()


# In[166]:


diff = s-m
print(diff)


# In[12]:


diff.mean()


# In[ ]:




