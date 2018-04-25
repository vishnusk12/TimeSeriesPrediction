# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:56:17 2017

@author: Impact
"""

import pandas as pd
df = pd.read_csv('C:/Users/Impact/Documents/Python Scripts/DeepLearning/data.csv', usecols=[8])
df = df[:10000]
df.to_csv('C:/Users/Impact/Documents/Python Scripts/DeepLearning/data_new.csv', index=False, header=False)