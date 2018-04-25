# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:05:03 2017

@author: Impact
"""

import numpy as np
import pandas as pd
import os
import warnings
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from numpy import newaxis

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

df = pd.read_csv('C:/Users/Impact/Documents/Python Scripts/DeepLearning/data_new.csv', usecols=[0])
df = df.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)

def create_dataset(dataset, time_steps=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_steps-1):
		a = dataset[i:(i+time_steps), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_steps, 0])
	return np.array(dataX), np.array(dataY)

def Predictor(trainX, trainY):
    model = Sequential()
    model.add(LSTM(
            input_dim=time_steps ,
            output_dim=10,
            return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(trainX, trainY, 
              batch_size=128,
              epochs=100)
    return model

time_steps = 1
X, Y = create_dataset(df, time_steps)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
model = Predictor(X, Y)
prediction_length = 28

reference = range(1, len(X) + 1)
reference = np.asarray(reference)
x = reference[:, newaxis]
predict_result = np.array([v for v in range(max(x[0])+1, max(x[0])+(prediction_length+1))])[:, np.newaxis]
predict_result = np.reshape(predict_result, (predict_result.shape[0], predict_result.shape[1], time_steps))
forecast = model.predict(predict_result)
forecast = scaler.inverse_transform(forecast)

#reference = range(1, len(X) + 1)
#
#reference = np.asarray(reference)
#
#x = reference[:, newaxis]
#
#predict_result = range(max(x[0])+1, max(x[0])+(time_steps+1)+(prediction_length+1))
#
#predict_result = np.asarray(predict_result)
#
#y = predict_result[:, newaxis]
#
#predict_result1 = create_dataset(y, time_steps)
#
#predict_result1 = np.reshape(predict_result1[0], (predict_result1[0].shape[0], 1, time_steps))
#
#forecast = model.predict(predict_result1)
#
#forecast = scaler.inverse_transform(forecast)


#def load_data(filename, seq_len):
#    f = open(filename, 'rb').read()
#    data = f.decode().split('\n')
#    sequence_length = seq_len + 1
#    result = []
#    for index in range(len(data) - sequence_length):
#        result.append(data[index: index + sequence_length])
#    new_result = []
#    scalar = MinMaxScaler(feature_range=(0, 1))
#    for i in  range(len(result)):
#        new_result.append(scalar.fit_transform(result[i]))
#    result = np.array(new_result)
#    row = round(0.8 * result.shape[0])
#    train = result[:int(row), :]
#    np.random.shuffle(train)
#    x_train = train[:, :-1]
#    y_train = train[:, -1]
#    x_test = result[int(row):, :-1]
#    y_test = result[int(row):, -1]
#    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#    return [x_train, y_train, x_test, y_test, scalar]




