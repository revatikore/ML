#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


df=pd.read_csv("C:/Users/Revati Kore/OneDrive/Desktop/MLABS/Heart (1).csv")
df


# In[6]:


#find no of rows and clouns 
df.shape


# In[40]:


#missing values, here false means not having null values.
df.isnull()


# In[8]:


#sum of null values in each columns
df.isnull().sum()


# In[9]:


df.count()


# In[11]:


#data types of each coluns , dtypes is an attritubte
df.dtypes


# In[12]:


#find out 0's , here true value represnt 0
df==0


# In[13]:


#highlight 0
df[df==0]


# In[14]:


#count 0 in each columns
df[df==0].count()


# In[15]:


df.columns


# In[16]:


#find maen of age
df['Age'].mean()


# In[18]:


#subset
df1=df[['Age','Sex','ChestPain','RestBP','Chol']]
df1


# In[19]:


#cross validation
from sklearn.model_selection import train_test_split


# In[45]:


train, test = train_test_split(df, random_state=0, test_size=0.25)


# In[46]:


train.shape


# In[22]:


test.shape


# In[23]:


import numpy as np


# In[26]:


actual=list(np.ones(45))+list(np.zeros(55))
np.array(actual)


# In[29]:


predicted=list(np.ones(40))+list(np.zeros(52))+list(np.ones(8))
np.array(predicted)


# In[30]:


#confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(actual,predicted)


# In[36]:


from sklearn.metrics import classification_report 
classification_report(actual,predicted)




# In[37]:


from sklearn.metrics import accuracy_score


# In[38]:


accuracy_score(actual,predicted)


# In[39]:


from sklearn.metrics import classification_report 
classification_report(actual,predicted)


# In[41]:


df. dropna( inplace = True )


# In[42]:


df


# In[43]:


df.isnull().sum()


# In[44]:


df.count()


# In[ ]:





# In[ ]:





# In[ ]:




