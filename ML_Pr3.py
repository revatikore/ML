#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[2]:


df=pd.read_csv("C:/Users/Revati Kore/Downloads/archive/Admission_Predict.csv")
df


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.shape


# In[6]:


#thresold value
from sklearn.preprocessing import Binarizer
bi=Binarizer(threshold=0.75)
df[ 'Chance of Admit ']=bi.fit_transform(df[[ 'Chance of Admit ']])


# In[7]:


df


# In[9]:


df.head()


# In[10]:


x=df.drop( 'Chance of Admit ', axis=1)
y=df[ 'Chance of Admit ']


# In[11]:


x


# In[14]:


y=y.astype('int')


# In[16]:


sns.countplot(x=y);


# In[17]:


#model / cross validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, test_size=0.25)


# In[19]:


x_train.shape


# In[20]:


x_test.shape


# In[24]:


#model making, decision_tree
#import class
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state=0)

classifier.fit(x_train, y_train) #model 


# In[25]:


#plotying graph

y_pred=classifier.predict(x_test)

result=pd.DataFram({
    'acutal':y_test,
    'predicted':y_pred
})


# In[26]:


from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


# In[28]:


accuracy_score(y_test, y_pred)


# In[29]:


print(classification_report(y_test, y_pred))


# In[ ]:




