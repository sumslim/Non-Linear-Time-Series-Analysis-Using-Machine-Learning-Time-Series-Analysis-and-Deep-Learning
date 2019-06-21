#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import operator
from pandas.plotting import register_matplotlib_converters


# In[57]:


df = pd.read_csv('/home/sim06/Downloads/cer3.csv')
time = pd.date_range(start='1961',periods=56,freq='Y')
df.set_index(time,inplace=True)
df = df.drop(df[['No','month','day','year']],axis=1)


# In[58]:


X = df.drop(df[['Cereals']],axis=1)
Y = df[['Cereals']]


# In[59]:


register_matplotlib_converters(explicit=True)


# In[60]:


poly_model = PolynomialFeatures(degree=1)
x_poly = poly_model.fit_transform(X)
poly_model.fit(x_poly,Y['Cereals'])
MR_model = LinearRegression()
MR_model.fit(x_poly,Y['Cereals'])
X_predicted = MR_model.predict(x_poly)
rmse = np.sqrt(mean_squared_error(Y['Cereals'],X_predicted))
r2 = r2_score(Y['Cereals'],X_predicted)
print("Root Mean Square Error : ",rmse)
print("r2 score : ",r2)
s = np.asarray(Y.Cereals)
diff = X_predicted - s
#for i in range(len(diff)):
    #print(diff[i])
ans = MR_model.predict(poly_model.fit_transform(X))
time1 = pd.date_range(start='1961',periods=56,freq='Y')
predictions = pd.DataFrame(ans,columns=['Cereals'])
predictions.set_index(time1,inplace=True)
plt.plot(predictions[['Cereals']],color='green',label='Predicted')
plt.plot(df[['Cereals']],color='red',label='Real')
plt.title("Polynomial of degree 1")
plt.legend()
plt.show()
#print('The predicted and true plots are not same this means that polynomial of degree 1 does not fits the given data.')


# In[61]:


poly_model = PolynomialFeatures(degree=2)
x_poly = poly_model.fit_transform(X)
poly_model.fit(x_poly,Y['Cereals'])
MR_model = LinearRegression()
MR_model.fit(x_poly,Y['Cereals'])
X_predicted = MR_model.predict(x_poly)
rmse = np.sqrt(mean_squared_error(Y['Cereals'],X_predicted))
r2 = r2_score(Y['Cereals'],X_predicted)
print("Root Mean Square Error : ",rmse)
print("r2 score : ",r2)
s = np.asarray(Y.Cereals)
diff = X_predicted - s
#for i in range(len(diff)):
    #print(diff[i])
ans = MR_model.predict(poly_model.fit_transform(X))
time1 = pd.date_range(start='1961',periods=56,freq='Y')
predictions = pd.DataFrame(ans,columns=['Cereals'])
predictions.set_index(time1,inplace=True)
plt.plot(predictions[['Cereals']],color='green',label='Predicted')
plt.plot(df[['Cereals']],color='red',label='Real')
plt.title("Polynomial of degree 2")
plt.legend()
plt.show()


# In[62]:


poly_model = PolynomialFeatures(degree=3)
x_poly = poly_model.fit_transform(X)
poly_model.fit(x_poly,Y['Cereals'])
MR_model = LinearRegression()
MR_model.fit(x_poly,Y['Cereals'])
X_predicted = MR_model.predict(x_poly)
rmse = np.sqrt(mean_squared_error(Y['Cereals'],X_predicted))
r2 = r2_score(Y['Cereals'],X_predicted)
print("Root Mean Square Error : ",rmse)
print("r2 score : ",r2)
s = np.asarray(Y.Cereals)
diff = X_predicted - s
#for i in range(len(diff)):
    #print(diff[i])
ans = MR_model.predict(poly_model.fit_transform(X))
time1 = pd.date_range(start='1961',periods=56,freq='Y')
predictions = pd.DataFrame(ans,columns=['Cereals'])
predictions.set_index(time1,inplace=True)
plt.plot(predictions[['Cereals']],color='green',label='Predicted')
plt.plot(df[['Cereals']],color='red',label='Real')
plt.title("Polynomial of degree 3")
plt.legend()
plt.show()


# In[ ]:




