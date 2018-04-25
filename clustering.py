# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:58:05 2017

@author: Impact
"""

import pandas as pd
from sklearn.cluster import KMeans
data = pd.read_csv('C:/Users/Impact/Documents/Python Scripts/DeepLearning/clusterdata.csv')
km = KMeans(n_clusters=10, init='k-means++', random_state=1)
km.fit(data.transpose())
predict = km.predict(data.transpose())
col_list = data.columns.tolist()
cluster = dict(zip(col_list, predict))
clust = {}
for k, v in cluster.items():
    clust.setdefault(v, []).append(k)