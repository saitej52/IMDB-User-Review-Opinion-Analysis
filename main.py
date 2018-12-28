# -*- coding: utf-8 -*-

"""

Data Science

author@Sunkara

"""


from TeamCode_Extracter import extract as ex
ex.extraction()
import pandas as pd
#import numpy as np
df = pd.read_csv('movie_data.csv',encoding='utf-8')
df.head(5)
#docs = np.array(['Sai Tej is a good boy','Charan is a Good boy','Surya is an intelligent and good boy'])
import TeamCode_feature_vectorization.vectors as vectors_creation
bag_of_words,vocabulary = vectors_creation.vc(df)
import TeamCode_tfidf.tras as ts
Vectorized_array = ts.Transform(df)

from TeamCode_cleaning import clean
df = clean.clean_data(df)

# Tokenizing the data

def tokenizer(text):
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]
    tokenizer_porter(text)

#tokenizer('Runners like running thus run in ground')
 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
#[w for w in tokenizer('a runner likes running and runs a lot')[-10:] if w not in stop]

X_train = df.loc[:2500,'review'].values
Y_train = df.loc[:2500,'sentiment'].values
print("Enter the Text : ")
testing_text = input()
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents = None,
                        lowercase = False,
                        preprocessor = None)
param_grid = [{'vect__ngram_range': [(1,1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer,
    tokenizer],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]},
    {'vect__ngram_range': [(1,1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer,
    tokenizer],
    'vect__use_idf':[False],
    'vect__norm':[None],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]}
    ]
lr_tfidf = Pipeline([('vect', tfidf),
    ('clf',
    LogisticRegression(random_state=0))])
model = GridSearchCV(lr_tfidf, param_grid,
    scoring='accuracy',
    cv=5, verbose=1,
    n_jobs=-1)
model.fit(X_train,Y_train)
