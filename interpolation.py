# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:16:52 2017

@author: Impact
"""

import pandas as pd
import numpy as np
date = pd.date_range('1/1/2011', periods=673, freq='M')
data = pd.read_csv("C:/Users/Impact/Documents/Python Scripts/DeepLearning/preprocessed.csv", header=None)
data = data.set_index(date)
list_ = []
for i in data.columns:
#    list_.append(data[i].resample('D').interpolate(method='linear'))
    list_.append(data[i].resample('D').interpolate(method='spline', order=2))
new_list = np.array(list_).T.tolist()
df = pd.DataFrame(new_list, index=None)
df.to_csv('C:/Users/Impact/Documents/Python Scripts/DeepLearning/interpolated_data.csv', index=False, header=False)

