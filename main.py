import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
os.path.dirname(os.path.abspath(__file__))


def text_to_wordlist(text, remove_stopwords=True):
    """Function to convert a document to a sequence of words and returns a list"""
    text = re.sub("[^a-zA-Z]"," ", text)
    words = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b=[]
    stemmer = english_stemmer
    for word in words:
        b.append(stemmer.stem(word))
    return(b)

# inputting the data
df = pd.read_csv('LabelledData (1).txt', sep=r',,,',header=None, engine = 'python')

# removing the spaces from the target class
df[1] = df[1].map(lambda x: x.strip())

# encoding the target class
le = LabelEncoder()
le.fit(df[1])
cat = le.transform(df[1]) 


corpus = []
for question in df[0]:
    corpus.append( " ".join(text_to_wordlist(question,remove_stopwords=False)))


vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 5000, ngram_range = ( 1, 4 ),
                              sublinear_tf = True )

vectorizer = vectorizer.fit(corpus)

train_features = vectorizer.transform(corpus)

fselect = SelectKBest(chi2 , k=1000)
train_features = fselect.fit_transform(train_features, cat)


# using XGboost to train our model
params = {
    'objective':'multi:softmax',
    'eval_metric':'merror',
    'eta':0.025,
    'max_depth':9,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':5,
    'num_class':5,
    'silent':1
}


import xgboost as xgb
dtrain = xgb.DMatrix(data=train_features, label = cat)
print ("starting the cross validation")
bst = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=40,nfold=5,verbose_eval=10)
print ("starting the training...")
bst_train = xgb.train(params, dtrain, num_boost_round=1000)


def testing(ques):
    test_ques = [ques]
    test_list = []
    for question in test_ques:
        test_list.append( " ".join(text_to_wordlist(question,remove_stopwords=False)))
    test_features = vectorizer.transform(test_list)
    test_features = fselect.transform(test_features)
    dtest = xgb.DMatrix(data=test_features)
    pred = bst_train.predict(dtest)
    pred = int(pred[0])
    return (list(le.inverse_transform([pred])))


if __name__ == '__main__':
    print ("press y to continue or n to quit")
    cont = input(str)
    while cont == 'y':
        if cont == 'y':
            print ("Enter your Question")
            quest = input(str)
            print ("Class : %s"%(testing(quest)))
            print ("press y to continue or n to quit")
            cont = input(str)