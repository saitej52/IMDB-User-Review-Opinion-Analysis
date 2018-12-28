# -*- coding: utf-8 -*-

'''
Feature Vectorization
author@Sunkara
'''

def vc(docs):
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer()
    #docs = np.array(['Sai Tej is a good boy','Charan is a Good boy','Surya is an intelligent and good boy'])
    print("### Vectorizing the words ###")
    bag = count.fit_transform(docs)
    print("### Words are vectorized ###")
    return bag,count.vocabulary_