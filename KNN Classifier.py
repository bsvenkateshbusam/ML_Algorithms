#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required Libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# In[2]:


# Loading Data and checking the keys
cancer = load_breast_cancer()
cancer.keys()


# In[3]:


# Converting into DataFrame 
df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
df['Target'] = cancer.target


# In[4]:


# Checking the first 5 colums
df.head()


# In[5]:


# Checking the tail 5 colums
df.tail()


# In[6]:


# Checking the shape of the data 
df.shape


# In[7]:


# Columns of the data 
df.columns


# In[8]:


# checking the class value counts of the data 
df['Target'].value_counts()


# In[9]:


# Dividing the Independent and Depending features 
X = df.drop('Target',axis = 1)
y = df['Target']


# In[10]:


# Splitting the data into Training and Testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 21)


# In[11]:


# Importing the K-Neighbors Classifier 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[12]:


from sklearn.metrics import accuracy_score, f1_score


# In[13]:


for i in range(3,12):
    #print(i)
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_pred,y_test)
    f1 = f1_score(y_pred,y_test)
    #print(acc, f1)
    print('The accuracy is {} and f1score is {} for {} neighbors'.format(acc, f1,i))
    print('************************************************************************************************')


# The best accuracy of the K Neighbors classifier is 95.37 with 9 neighbors.
