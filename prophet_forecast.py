# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
from fbprophet import Prophet
date = pd.date_range('1/1/2011', periods=673, freq='D')

data = pd.read_csv("C:/Users/Impact/Documents/Python Scripts/DeepLearning/LSTM Time Series/preprocessed_head.csv")

data['Day'] = date

df = data[['Day', 'Product_1469']]

df.columns = ['ds', 'y']


m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=10)

forecast = m.predict(future)

pred = np.round(forecast['yhat'])

