#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import string
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 


# In[3]:


datatwitter= pd.read_csv("https://raw.githubusercontent.com/SathanaR/capstone-project/main/processed_twitter_data.csv")
datatwitter.head()


# In[4]:


datatwitter.shape


# In[5]:


datatwitter.isnull().sum()


# In[6]:


datatwitter.tweet=datatwitter.tweet.interpolate(method='pad')


# In[7]:


datatwitter.isnull().sum()


# In[8]:


datatwitter.shape


# GRAPHS

# In[9]:


sns.countplot(datatwitter.labels)


# WORD CLOUD

# In[10]:


text = " ".join(cat for cat in datatwitter.tweet)
word_cloud = WordCloud(
    width=3000,
    height=2000,
    random_state=1,
    background_color="white",
    colormap="Pastel1",
    collocations=False,
    stopwords=STOPWORDS,
    ).generate(text)


# In[11]:


plt.figure(figsize=(12,12))
plt.imshow(word_cloud)
plt.axis("off")
plt.show()


# INDEPENDENT AND  DEPENDENT VARIABLES

# In[12]:


x = np.array(datatwitter ["tweet"])
Y = np.array(datatwitter ["labels"])


# In[13]:


x.shape


# In[14]:


Y.shape


# In[15]:


cntvec = CountVectorizer()
X = cntvec.fit_transform(x) 


# Cross Validation

# In[16]:


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[17]:


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5)
scores_dectree = []
scores_rf = []
scores_grad= []
scores_svm=[]
scores_knn=[]


# DecisionTreeClassifier

# In[18]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    scores_dectree.append(get_score(DecisionTreeClassifier(), X_train, X_test, y_train, y_test))


# In[19]:


res_dectree=np.average(scores_dectree )


# RandomForestClassifier

# In[20]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))


# In[21]:


res_rf=np.average(scores_rf )


# GradientBoostingClassifier

# In[22]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    for learning_rate in lr_list:
        scores_grad.append(get_score(GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0),X_train, X_test, y_train, y_test))


# In[23]:


res_grad=np.average(scores_grad)


#  Support Vector Machines

# In[ ]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    scores_svm.append(get_score(svm.SVC(), X_train, X_test, y_train, y_test))


# In[ ]:


res_svm=np.average(scores_svm)


# KNeighborsClassifier  

# In[ ]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    scores_knn.append(get_score(KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2 ), X_train, X_test, y_train, y_test))


# In[ ]:


res_knn=np.average(scores_knn)


# TABLE

# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=["MODEL NAME", "SCORE"]),cells=dict(values=[["DECISION TREE", "RANDOM FOREST","GRADIENT BOOST", "SUPPORT VECTOR MACHINE","K NEAREST NEIGHBOR"],[res_dectree,res_rf,res_grad,res_svm,res_knn]]))])
fig.show()


# SVM has the highest score and it is selected as final model

# In[ ]:


model_final=svm.SVC()


# In[ ]:


model_final.fit(X,Y)


# INPUT

# In[ ]:


user = input("Enter a Text: ")
data = cntvec.transform([user]).toarray()
output = model_final.predict(data)
print(output)


# In[ ]:


import pickle
from flask import Flask, render_template, request


# In[ ]:


file=open('Capstone_svm.pkl','wb')


# In[ ]:


pickle.dump(model_final,file)


# In[ ]:




