#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


# In[3]:


df=pd.read_csv("C:/Users/Revati Kore/OneDrive/Desktop/MLABS/temperatures.csv")


# In[4]:


df


# In[6]:


#input data
x=df['YEAR']
y=df['ANNUAL']


# In[16]:


import matplotlib.pyplot as plt
plt.title('Temperature plot of India')
plt.xlabel('Year')
plt.ylabel('Anuual Average Temp')
plt.scatter(x,y)


# In[15]:


import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
plt.title('Temperature plot of India')
plt.xlabel('Year')
plt.ylabel('Annual Average Temp')
plt.scatter(x,y)


# Rest of your plotting code goes here

plt.show()


# In[17]:


df.shape


# In[18]:


x.shape


# In[19]:


x=x.values
x


# In[22]:


x=x.reshape(117,1)
x.shape


# In[23]:


from sklearn.linear_model import LinearRegression


# In[24]:


regressor= LinearRegression()


# In[25]:


regressor.fit(x,y)


# In[27]:


regressor.coef_


# In[28]:


regressor.intercept_


# In[29]:


regressor.predict([[2024]])


# In[31]:


predicted=regressor.predict(x)


# In[32]:


predicted


# In[33]:


#actual values
y


# In[34]:


#mean absolute error
import numpy as np
np.mean(abs(y-predicted))


# In[39]:


#by function
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y,predicted)


# In[36]:





# In[37]:


#mean sq error
np.mean(abs(y-predicted)**2)


# In[40]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y,predicted)


# In[43]:


#R-sq matrix
from sklearn.metrics import r2_score
r2_score(y,predicted)


# In[44]:


regressor.score(x,y)


# In[46]:


plt.title('Temperature plot of India')
plt.xlabel('Year')
plt.ylabel('Anuual Average Temp')
plt.scatter(x,y, label='actual', color='r')
plt.plot(x,predicted,label='predicted',color='g')
plt.legend()


# In[48]:


import seaborn as sns
sns.regplot(x='YEAR',y='ANNUAL',data=df)


# In[ ]:




