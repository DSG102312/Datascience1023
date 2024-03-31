#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


# In[28]:


data =pd.read_csv('https://raw.githubusercontent.com/training-ml/Files/main/heart_disease.csv')
data.head()


# In[29]:


data.shape


# In[30]:


data.describe()


# In[31]:


data.isnull().sum()


# In[32]:


plt.figure(figsize=(20,25), facecolor='yellow')
plotnumber = 1

for column in data:
    if plotnumber<=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
       
    plotnumber+=1
plt.show()


# In[35]:


plt.figure(figsize=(20,25))
graph = 1

for column in data:
    if graph<=9:
        plt.subplot(3,3,graph)
        ax=sns.boxplot(data= data[column])
        plt.xlabel(column,fontsize=15)
    graph+=1
plt.show()


# In[36]:


data.shape


# In[37]:


q1 = data.quantile(0.25)

q3 = data.quantile(0.75)

iqr = q3 - q1


# In[38]:


q1


# In[39]:


trestbps_high = (q3.trestbps + (1.5 * iqr.trestbps))
trestbps_high


# In[40]:


np_index = np.where(data['trestbps'] > trestbps_high)
np_index


# In[41]:


data = data.drop(data.index[np_index])
data.shape


# In[42]:


data.reset_index()


# In[43]:


chol_high = (q3.chol + (1.5 * iqr.chol))
chol_high


# In[44]:


np_index = np.where(data['chol'] > chol_high)
np_index


# In[45]:


data = data.drop(data.index[np_index])
data.shape


# In[46]:


data.reset_index()


# In[ ]:




