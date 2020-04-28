#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('C:/Users/markus/Desktop/ckdisease/kidney_disease.csv')
dataset


# In[2]:


# removing row that have empty or no values for specific columns
dataset=dataset[pd.notnull(dataset['age'])]
dataset=dataset[pd.notnull(dataset['pcc'])]
dataset=dataset[pd.notnull(dataset['appet'])]
dataset=dataset[pd.notnull(dataset['cad'])]

dataset


# In[3]:


# replacing null values with string normal 
dataset['rbc'].fillna('normal',inplace=True)
dataset['pc'].fillna('normal',inplace=True)
dataset.fillna(dataset.mean(),inplace=True)


# In[4]:


# removing some entries in the classification column that had a tab pressed unintentionally while entering data
dataset=dataset[~dataset.classification.str.contains('ckd\t')]
dataset


# In[5]:


# splitting of data into X - variables & attributes and y - output (whether chronic kidney disease or not)
X = dataset.iloc[:, 1:24].values
y = dataset.iloc[:, 25].values
y


# In[6]:


X


# In[7]:


#Encoding categorical data, changing string to numeric

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
lb =LabelEncoder()
lb_y =LabelEncoder()
y=lb_y.fit_transform(y)
X[:,5]=lb.fit_transform(X[:,5])
X[:,6]=lb.fit_transform(X[:,6])
X[:,7]=lb.fit_transform(X[:,7])
X[:,8]=lb.fit_transform(X[:,8])
X[:,18]=lb.fit_transform(X[:,18])
X[:,19]=lb.fit_transform(X[:,19])
X[:,20]=lb.fit_transform(X[:,20])
X[:,21]=lb.fit_transform(X[:,21])
X[:,22]=lb.fit_transform(X[:,22])
X[:,23]=lb.fit_transform(X[:,23])


# In[8]:


X


# In[9]:


y


# In[10]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[11]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[12]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[14]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[15]:


# Making the Confusion Matrix and Plotting AUC graph to see how well the model fits
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

