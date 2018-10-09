import numpy as np
import pandas as pd
import itertools
import gensim
import keras
import nltk
import re
import codecs
import os

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer

os.chdir("/Users/nitya/childsplay/data")
df = pd.read_csv('peppa_dataframe_with_labels')
df = df.drop_duplicates('videoId')

def ttsplit(thing):
    ''''
    Train-test splitting with indices
    ''''
    list_corpus = df[str(thing)].tolist()
    list_labels = df['class_label'].tolist()
    indices = np.arange(len(list_labels))
    x_train, x_test, y_train, y_test, i1, i2 = train_test_split(list_corpus,
                                                                list_labels,
                                                                indices,
                                                                test_size=0.2,
                                                                random_state=40)
    return x_train,x_test, y_train, y_test, i1, i2

x1_ti, x2_ti, y1_ti, y2_ti, i1, i2 = ttsplit('title')
x1_des, x2_des, y1_des, y2_des, i1, i2 = ttsplit('description')



def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

x1ti_tfidf, tfidf_vectorizer = tfidf(x1_ti)
x2ti_tfidf = tfidf_vectorizer.transform(x2_ti)

clf_tfidf_ti = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_tfidf_ti.fit(x1ti_tfidf, y1_ti)


x1des_tfidf, tfidf_vectorizer = tfidf(x1_des)
x2des_tfidf = tfidf_vectorizer.transform(x2_des)

clf_tfidf_des = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_tfidf_des.fit(x1des_tfidf, y1_des)
