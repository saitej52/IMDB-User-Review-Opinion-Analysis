# -*- coding: utf-8 -*-

def extraction():
    import tarfile
    print("### Extracting the IMDB Data ###")
    with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
        tar.extractall()
    print("### IMDB Data Extraction Sucessfully Completed ###")
    # Data preperation
    import pyprind
    import pandas as pd
    import os
    basepath = "aclImdb"
    labels = {'pos':1,'neg':0}
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()
    for s in ('test','train'):
        for l in ('pos','neg'):
            path = os.path.join(basepath,s,l)
            for file in os.listdir(path):
                with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                    txt = infile.read()
                df = df.append([[txt,labels[l]]],ignore_index=True)
                pbar.update()
    df.columns=['review','sentiment']
    def shuffle(df):
        import numpy as np
        print("### Data was being Shuffled ###")
        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))
        df.to_csv('movie_data.csv',index=False,encoding='utf-8')
        print("### Data was Shuffled Successfully ###")
        return df
    df = shuffle(df)