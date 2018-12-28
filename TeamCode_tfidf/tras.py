# -*- coding: utf-8 -*-

def Transform(docs):
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    tfidf = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
    np.set_printoptions(precision=2)
    count = CountVectorizer()
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())