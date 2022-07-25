#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


datatwitter= pd.read_csv("https://raw.githubusercontent.com/SathanaR/capstone-project/main/twitter.csv")
datatwitter.head()


# In[4]:


datatwitter.shape


# In[5]:


datatwitter.isnull().sum()


# In[6]:


datatwitter["labels"] = datatwitter["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
datatwitter.head()


# In[7]:


datatwitter= datatwitter[["tweet", "labels"]]
datatwitter .head()


# In[8]:


nltk.download('stopwords')
stopword=set(stopwords.words('english'))


# In[9]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


# In[10]:


datatwitter["tweet"] =datatwitter ["tweet"].apply(clean)
datatwitter.head()


# GRAPHS

# In[11]:


sns.countplot(datatwitter.labels)


# WORD CLOUD

# In[12]:


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


# In[13]:


plt.figure(figsize=(12,12))
plt.imshow(word_cloud)
plt.axis("off")
plt.show()


# INDEPENDENT AND  DEPENDENT VARIABLES

# In[14]:


x = np.array(datatwitter ["tweet"])
Y = np.array(datatwitter ["labels"])


# In[15]:


x.shape


# In[16]:


Y.shape


# In[17]:


cntvec = CountVectorizer()
X = cntvec.fit_transform(x) 


# Cross Validation

# In[18]:


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[19]:


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5)
scores_dectree = []
scores_rf = []
scores_grad= []
scores_svm=[]
scores_knn=[]


# DecisionTreeClassifier

# In[20]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    scores_dectree.append(get_score(DecisionTreeClassifier(), X_train, X_test, y_train, y_test))


# In[21]:


res_dectree=np.average(scores_dectree )


# RandomForestClassifier

# In[22]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))


# In[23]:


res_rf=np.average(scores_rf )


# GradientBoostingClassifier

# In[24]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    for learning_rate in lr_list:
        scores_grad.append(get_score(GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0),X_train, X_test, y_train, y_test))


# In[25]:


res_grad=np.average(scores_grad)


#  Support Vector Machines

# In[26]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    scores_svm.append(get_score(svm.SVC(), X_train, X_test, y_train, y_test))


# In[27]:


res_svm=np.average(scores_svm)


# KNeighborsClassifier  

# In[28]:


for train_index, test_index in folds.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],                                        Y[train_index],Y[test_index]
    scores_knn.append(get_score(KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2 ), X_train, X_test, y_train, y_test))


# In[29]:


res_knn=np.average(scores_knn)


# TABLE

# In[30]:


import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=["MODEL NAME", "SCORE"]),cells=dict(values=[["DECISION TREE", "RANDOM FOREST","GRADIENT BOOST", "SUPPORT VECTOR MACHINE","K NEAREST NEIGHBOR"],[res_dectree,res_rf,res_grad,res_svm,res_knn]]))])
fig.show()


# SVM has the highest score and it is selected as final model

# In[31]:


model_final=svm.SVC()


# In[32]:


model_final.fit(X,Y)


# INPUT

# In[33]:


user = input("Enter a Text: ")
data = cntvec.transform([user]).toarray()
output = model_final.predict(data)
print(output)


# In[34]:


import pickle


# In[35]:


file=open('Capstone_svm.pkl','wb')


# In[36]:


pickle.dump(model_final,file)


# In[37]:


import streamlit as st
from PIL import Image
classifier = pickle.load(open('Capstone_svm.pkl', 'rb'))
def welcome():
    return 'welcome all'

def prediction(tweet):  
   
    prediction = classifier.predict(
        [[tweet]])
    print(prediction)
    return prediction
      
  
# this is the main function in which we define our webpage 
def main():
    global result
      # giving the webpage a title
    st.title("Tweet type Prediction")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    tweet= st.text_input("tweet", "Type Here")
    if st.button("Predict"):
        st.success('The output is {}'.format( prediction(tweet)))
     
    if __name__=='__main__':
        main()


# In[ ]:


streamlit run app.py


# In[ ]:


def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = model_final.predict(data)
        st.title(a)
hate_speech_detection()


# In[ ]:




