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
import multiprocessing
from multiprocessing import Pool
from newspaper import Article

#------------------------------------------------------------------------------#
# Configuration
#------------------------------------------------------------------------------#
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip'

NEWS_AGGREGATOR_DIR = 'news_aggregator'
ARTICLES_DIR        = 'articles'

PUBLISHER_THRESHOLD = 500
SAMPLE_SIZE         = 10000
# SAMPLE_SIZE         = 10

#------------------------------------------------------------------------#

def download_news_aggregator_data():
    r = requests.get(URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(NEWS_AGGREGATOR_DIR)

#------------------------------------------------------------------------#
def download_articles(df):
    t0 = time.time()

    print(f"{os.getpid()}: Number of rows in df {len(df) :,}.")

    count, success_count, fail_count = 0, 0, 0
    for id, url in zip(df.ID, df.URL):
        article = Article(url, language="en")
        try:
            article.download()
            article.parse()
            text = article.text
            if len(text) > 100:
                with open(f"{ARTICLES_DIR}/{id}.txt", "w") as f:
                    f.write(text)
                success_count += 1
            else:
                fail_count += 1
        except:
            fail_count += 1

        #-----------------------------------#
        count += 1
        if count > 0 and count%200 == 0:
            # clear_output(wait=True)
            line = (f"Pid: {os.getpid()}; "
                    f"Success: {success_count :,}; Fail: {fail_count :,}; "
                    f"{(time.time()-t0)/60 :.1f} minutes")
            print(line)

        #-----------------------------------#
        if success_count >= SAMPLE_SIZE or fail_count >= 3*SAMPLE_SIZE:
            break

    print(f"{os.getpid()} exit with {success_count :,} successes and {fail_count :,} fails.")

#------------------------------------------------------------------------#

def run():
    # Load News Aggregator Data with URLs --------------#
    # News category (b = business, t = science and technology,
    #                e = entertainment, m = health)
    columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'Alphanumeric ID',
               'HOSTNAME Url', 'TIMESTAMP']

    df = pd.read_csv(f"{NEWS_AGGREGATOR_DIR}/newsCorpora.csv",
                     sep='\t', header=None, names=columns)

    #Select only publishers with a count > threshold
    pubshare = df['PUBLISHER'].value_counts()
    filtered_publishers = pubshare[pubshare>PUBLISHER_THRESHOLD].index
    print(f"Filtered for publishers which has at least {PUBLISHER_THRESHOLD} articles")

    # Split by categories ------------------------------#
    categories = ['b', 't', 'e', 'm']
    cat = dict()
    for c in categories:
        mask = (df.CATEGORY==c) & df.PUBLISHER.isin(filtered_publishers)
        cat[c] = df[mask]
        cat[c].reset_index(drop=True, inplace=True)
        cat[c] = cat[c].sample(frac=1).reset_index(drop=True) #Shuffle rows
        print(f"Categort '{c}' has {len(cat[c]) :,} entries")

    # Download articles --------------------------------#
    # We can use more workers than the number of cores since the process is
    # io bound
    input = []
    for c in categories:
        s = int(SAMPLE_SIZE/2)
        input.append(cat[c][:s])
        input.append(cat[c][s:SAMPLE_SIZE])

    print(f"Start downloading articles on {len(input)} workers.")
    with Pool(len(input)) as p:
        p.map(download_articles, input)

    print(f"Number of articles: {len(os.listdir(f'{ARTICLES_DIR}')) :,}")

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

if __name__ == '__main__':
    T0 = time.time()

    os.makedirs(ARTICLES_DIR, exist_ok=True)
    if not os.path.isdir(NEWS_AGGREGATOR_DIR):
        download_news_aggregator_data()
    else:
        print('News Aggregator Data already exist.')

    run()

    print(f"Done in {(time.time()-T0)/60 :.1f} minutes")
