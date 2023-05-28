#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Clustering Algo

import pandas as pd 
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/Revati Kore/Downloads/archive (2)/Mall_Customers.csv")
df


# In[4]:


x=df.iloc[:,3:]
x


# In[7]:


#forming unclustered data
import matplotlib.pyplot as plt
plt.title('Unclustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'],x['Spending Score (1-100)'])


# In[10]:


from sklearn.cluster import KMeans, AgglomerativeClustering
km=KMeans(n_clusters=3) #0,1,2
km.fit_predict(x)


# In[12]:


#SSE sum of inertia
km.inertia_


# In[15]:


sse = []
for k in range(1,16):
    km=KMeans(n_clusters=k)
    km.fit_predict(x)
    sse.append(km.inertia_)
    sse


# In[16]:


sse


# In[51]:


#Elbow Method

plt.title('Elbow Method')
plt.xlabel('Value of K')
plt.ylabel('SSE')
plt.grid()
plt.xticks(range(1,16))
plt.plot(range(1,16), sse, marker='.')


# # silhouette method

# In[24]:


from sklearn.metrics import silhouette_score
silh=[]
for k in range(2,16):
    km=KMeans (n_clusters=k)
    labels=km.fit_predict(x)
    score=silhouette_score(x, labels)
    silh.append(score)
    silh


# In[25]:


silh


# In[29]:


plt.title('Silhouette Method')
plt.xlabel('Value of K')
plt.ylabel('Silhoutte Score')
plt.grid()
plt.xticks(range(2,16))
plt.bar(range(2,16), silh, color='red')


# # Clustered data
# 
# 

# In[31]:


#c_data
km=KMeans (n_clusters=5 , random_state=0
labels=km.fit_predict(x)
labels


# In[39]:


import matplotlib.pyplot as plt


plt.title('clustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'],x['Spending Score (1-100)'],
           c=labels)


# In[45]:





# In[40]:





# In[46]:


cent=km.cluster_centers_


# In[47]:


plt.title('clustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'],x['Spending Score (1-100)'],
           c=labels)
plt.scatter(cent[:,0], cent[:,1], s=100, color='k')


# In[50]:





# In[ ]:




