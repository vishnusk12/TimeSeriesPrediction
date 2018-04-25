# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:59:12 2017

@author: Impact
"""

import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/Impact/Documents/Python Scripts/DeepLearning/LSTM Time Series/data.csv")
df = data['Product_d'].value_counts()
max_count = (df).max()
min_count = df.min()
threshold = np.round(max_count+min_count)/2
valid_products = []
df_new = pd.DataFrame()
for i, row in df.iteritems():
    if row >= threshold:
        valid_products.append(i)
list_pro = []
list_qua = []
for index, row in data.iterrows():
     if row['Product_d'] in valid_products:
         list_pro.append(row['Quantity'])
         list_qua.append(row['Product_d'])
df_new = pd.DataFrame(list_pro)
df_new_ = pd.DataFrame(list_qua)
frames = [df_new, df_new_]
result = pd.concat(frames,axis=1)
result.columns = [ "quantity", "Product"]
list_ = []
for i in valid_products[:20]:
    res = result[(result['Product']==i)]
    list_.append(res['quantity'])
new_list = np.array(list_).T.tolist()
new_data = pd.DataFrame(new_list)
new_data.columns = valid_products[:20]
#new_data['parser'] = range(len(new_data))
#new_data.to_csv("C:/Users/Impact/Documents/Python Scripts/DeepLearning/interpolated_data.csv", index=False)

new_data.to_csv("C:/Users/Impact/Documents/Python Scripts/DeepLearning/LSTM Time Series/preprocessed_head.csv", index=False)


from dateutil import rrule
from datetime import datetime, timedelta
now = datetime.now()
length = now + timedelta(days=20480)
list = []
for dt in rrule.rrule(rrule.MONTHLY, dtstart=now, until=length):
    list.append(dt.date())
df = pd.DataFrame({'Month': list})
new_df = new_data.join(df)
new_df.to_csv("C:/Users/Impact/Documents/Python Scripts/DeepLearning/interpolated_data.csv", index=False)


from datetime import *
from dateutil.relativedelta import *
TODAY = date.today()
list = []
for i in range(max_count):
    list.append(TODAY + relativedelta(months=+i))
df = pd.DataFrame({'Month': list})