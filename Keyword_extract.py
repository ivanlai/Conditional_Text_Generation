#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Download news articles from the urls given in News Aggregator Dataset
    ref: https://archive.ics.uci.edu/ml/datasets/News+Aggregator
"""

import io, os, sys
import pandas as pd
import numpy as np
import requests
import zipfile
import time
import csv
import re
import multiprocessing
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

import en_core_web_lg

#------------------------------------------------------------------------------#
# Configuration
#------------------------------------------------------------------------------#
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip'

NEWS_AGGREGATOR_DIR = 'news_aggregator'
ARTICLES_DIR        = 'articles'

# DEBUG               = True
DEBUG               = False
RE_D                = re.compile('\d')
TOP_N               = 20
TFIDF_THRESHOLD     = 0.1

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def load_articles(df):
    data = {'id': [],
            'category': [],
            'title': [],
            'text': []}

    for root, dirs, files in os.walk(ARTICLES_DIR, topdown=True):
        t0 = time.time()

        for i, f in enumerate(files):
            #id, category, title, keywords, text
            id = int(f[:-4])
            tmp = df[['CATEGORY', 'TITLE']][df.ID==id].values
            category, title = tmp[0][0], tmp[0][1]

            with open(f'{ARTICLES_DIR}/{f}', "r") as infile:
                text = infile.read()

            data['id'].append(id)
            data['category'].append(category)
            data['title'].append(title)
            data['text'].append(text)

            if DEBUG and i >= 10:
                break

            if i%1000==0 and i>0:
                print(f"({os.getpid()}) Items processed: {i :,}/{len(files):,}; {(time.time()-t0)/60 :.1f} minutes")


    print(f"Number of articles: {len(data['text']) :,}")

    return data

#--------------------------------------------------------------------#

def has_numbers(string, regex_pattern=RE_D):
    return bool(regex_pattern.search(string))

#--------------------------------------------------------------------#

def get_keywords(corpus, nlp, top_n=TOP_N, threshold=TFIDF_THRESHOLD):
    t0 = time.time()

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=STOPWORDS)
    X = vec.fit_transform(corpus)
    print(f"X.shape: {X.shape}")

    terms = np.array(vec.get_feature_names())

    tfidfs, keywords = [], []
    all_keywords = set()
    N = len(corpus)

    for i, text in enumerate(corpus):
        D = X.getrow(i)
        D = np.squeeze(D.toarray())
        ind = np.argsort(D)[::-1]
        ind = ind[:top_n]

        D = D[ind]
        D = D[D > threshold]
        kw = terms[ind][:len(D)]

        doc = nlp(text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split())>1]

        D  = [d for d, w in zip(D, kw) if (w in noun_phrases or len(w.split())==1) and not has_numbers(w)]
        kw = [w for w in kw if (w in noun_phrases or len(w.split())==1) and not has_numbers(w)]

        tfidfs.append(D)
        keywords.append(kw)

        for word in kw:
            all_keywords.add(word)

        if i%1000==0 and i>0:
            print(f"({os.getpid()}) Items processed: {i :,}/{N:,}; {(time.time()-t0)/60 :.1f} minutes")

    return terms, keywords, tfidfs, all_keywords

#--------------------------------------------------------------------#

def run():
    # Load News Aggregator Data with URLs --------------#
    columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'Alphanumeric ID',
               'HOSTNAME Url', 'TIMESTAMP']

    df = pd.read_csv(f"{NEWS_AGGREGATOR_DIR}/newsCorpora.csv",
                     sep='\t', header=None, names=columns)

    data = load_articles(df)

    nlp = en_core_web_lg.load()
    terms, keywords, tfidfs, all_keywords = get_keywords(data['text'], nlp)

    with open('keywords.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',',  quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for id, kw in zip(data['id'], keywords):
            line = [id] + kw
            writer.writerow(line)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

if __name__ == '__main__':
    T0 = time.time()
    run()
    print(f"Done in {(time.time()-T0)/60 :.1f} minutes")
