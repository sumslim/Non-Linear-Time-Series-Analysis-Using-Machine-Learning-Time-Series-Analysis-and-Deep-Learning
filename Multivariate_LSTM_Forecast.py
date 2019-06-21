#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM


# In[2]:


df = pd.read_csv('/home/sim06/Downloads/cer3.csv')
time = pd.date_range(start='1961',periods=56,freq='Y')
df.set_index(time,inplace=True)


# In[5]:


df = df.drop(df[['No','month','day','year']],axis=1)


# In[7]:


feature = df.iloc[:,:].values
feature = feature.astype(float)


# In[8]:


scale = MinMaxScaler(feature_range=(0,1))
feature_scaled = scale.fit_transform(feature)


# In[9]:


x_train = []
y_train = []
for i in range(10,len(feature_scaled)):
    x_train.append(feature_scaled[i-10:i,:])
    y_train.append(feature_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)


# In[10]:


model = Sequential()


# In[11]:


from keras.layers import Dropout
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))


# In[12]:


model.compile(loss='mean_squared_error',optimizer='adam')


# In[13]:


history = model.fit(x_train,y_train,epochs=100,batch_size=10)


# In[14]:


plt.xlabel('Iterations')
plt.ylabel('Error')
plt.plot(history.history['loss'],label='Error')
plt.show()


# In[15]:


predicted_stock_price = model.predict(x_train)


# In[16]:


print(predicted_stock_price)


# In[17]:


plt.xlabel('Dataset')
plt.ylabel('Scaled Output')
plt.plot(y_train,color='red',label='True')
plt.plot(predicted_stock_price,color='green',label='Predicted')
plt.legend()
plt.show()


# In[ ]:




