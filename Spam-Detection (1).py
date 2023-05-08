#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud


# In[2]:


# https://www.kaggle.com/uciml/sms-spam-collection-dataset
get_ipython().system('wget https://lazyprogrammer.me/course_files/spam.csv')


# In[3]:


df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
#find contains some invalid chars
#depending on which version of pandas you have you may face an error
#so the encoding is not neccessary but the thing is being safe rather than sorry -Kiavash qoutes


# In[4]:


df.head()


# In[5]:


df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)


# In[6]:


df.head()


# In[7]:


df.columns = ['labels', 'data']


# In[8]:


df.head()


# In[9]:


df['labels'].hist()


# In[10]:


df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
Y = df['b_labels'].to_numpy()
Y


# In[11]:


df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size = 0.33)


# In[12]:


#brute forcing through the errors, why database bad? :'(
featurizer = TfidfVectorizer(decode_error='ignore')
Xtrain = featurizer.fit_transform(df_train)
Xtest = featurizer.transform(df_test)


# In[13]:


Xtrain


# In[14]:


model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train acc: ",model.score(Xtrain, Ytrain))
print("test acc: ",model.score(Xtest,Ytest))


# In[15]:


Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)
print("train F1: ", f1_score(Ytrain, Ptrain))
print("test F1: ", f1_score(Ytest, Ptest))


# In[16]:


Prob_train = model.predict_proba(Xtrain)[:,1]
Prob_test = model.predict_proba(Xtest)[:,1]
print("train AUC:", roc_auc_score(Ytrain, Prob_train))
print("test AUC:", roc_auc_score(Ytest, Prob_test))


# In[17]:


cm = confusion_matrix(Ytrain, Ptrain)
cm


# In[18]:


#This is for representation only - ignore if you don't understand (no big deal)
def plot_cm(cm):
    classes = ['ham', 'spam']
    df_cm = pd.DataFrame(cm, index = classes, columns = classes)
    ax = sn.heatmap(df_cm, annot = True, fmt = 'g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
plot_cm(cm)


# In[19]:


cm_test = confusion_matrix(Ytest, Ptest)
plot_cm(cm_test)


# In[20]:


def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
    #see what i did? :))
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[21]:


visualize('spam')


# In[22]:


visualize('ham')

