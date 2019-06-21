#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[35]:


df = pd.read_csv('/home/sim06/Downloads/cer3.csv')
time = pd.date_range(start='1961',periods=56,freq='Y')
df.set_index(time,inplace=True)
df = df.drop(df[['No','month','day','year']],axis=1)


# In[36]:


plt.plot(df[['Cereals']])
plt.show()


# In[37]:


df_train = df[0:int((0.8)*(len(df)))]
df_test = df[int((0.8)*(len(df))):len(df)]


# In[38]:


model = VAR(endog=df_train)


# In[39]:


model_fit = model.fit()


# In[40]:


prediction = model_fit.forecast(model_fit.y,steps=len(df_test))


# In[41]:


cols = df_train.columns
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,14):
    for i in range(len(prediction)):
        pred.iloc[i][j] = prediction[i][j]
        
print(pred)


# In[44]:


for i in cols:
    print('rmse value for', i, 'is : ', np.sqrt(mean_squared_error(pred[i], df_test[i])))


# In[45]:


model = VAR(endog=df)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)


# In[47]:


time1 = pd.date_range(start='2005',periods=12,freq='Y')
pred.set_index(time1,inplace=True)


# In[50]:


plt.plot(pred['Cereals'],color='green',label='Predicted_train')
plt.plot(df_test['Cereals'],color='red',label="True_train")
plt.legend()
plt.show()

