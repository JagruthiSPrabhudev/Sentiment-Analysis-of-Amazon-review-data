import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import  MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import joblib

#read the data from csv
reviews =  pd.read_csv('reviews.csv',error_bad_lines=False,sep='|')
print("Totalreviews :",len(reviews))
#get the length of the reviews
reviews['text length'] = reviews['text'].apply(len)
X = reviews['label']
Y = reviews['text']
reviews['label'] = np.where(reviews['label']=='positive' , 1, 0)

#change index of review to start from 1
reviews.index = reviews.index+1

#separate train data and test data
test_data = reviews[reviews.index % 5 == 0]
train_data = reviews[reviews.index % 5 != 0]
print("test data",len(test_data))
print("train data",len(train_data))
train_data_reviews = train_data['text']
train_data_label = train_data['label']
test_data_reviews = test_data['text']
test_data_label = test_data['label']
#Convert a raw review to a cleaned review

def cleanText(raw_text, remove_stopwords=False, stemming=False, split_text=False):
    '''
    Convert a raw review to a cleaned review
    '''

    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)  # remove non-character
    words = letters_only.lower().split() # convert to lower case

    if remove_stopwords: # remove stopword
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if stemming==True: # stemming
#         stemmer = PorterStemmer()
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]

    if split_text==True:  # split text
        return (words)


    return( " ".join(words))


X_train_cleaned = []
X_test_cleaned = []

for d in train_data_reviews:
    X_train_cleaned.append(cleanText(d))

for d in test_data_reviews:
    X_test_cleaned.append(cleanText(d))

countVect = CountVectorizer()
X_train_countVect = countVect.fit_transform(X_train_cleaned)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_countVect)
joblib.dump(countVect.vocabulary_,open("feature.pkl","wb"))

X_new_counts = countVect.transform(X_test_cleaned)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


#Neural networks train
print("Neural Network")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(7, 2), random_state=1)
clf.fit(X_train_tfidf,train_data_label)
predictedNN = clf.predict(X_new_tfidf)
print(np.mean(predictedNN == test_data_label))

#using grid search and pipeline for logistic regression
filename = 'finalized_model.sav'
joblib.dump(clf, filename)
