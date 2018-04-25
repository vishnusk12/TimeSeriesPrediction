# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:20:55 2017

@author: Impact
"""

import numpy as np
import pandas as pd
import os
import warnings
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.models import Sequential
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


#def load_data(filename, seq_len):
#    df = pd.read_csv(filename, header=None)
#    df = df.fillna(df.mean())
#    X_Train = []
#    Y_Train = []
#    X_Test = []
#    Y_Test = []
#    for i in range(len(df.columns[:2])):
#        data = df[i].tolist()
#        sequence_length = seq_len + 1
#        result = []
#        for index in range(len(data) - sequence_length):
#            result.append(data[index: index + sequence_length])
#        new_result = []
#        scalar = MinMaxScaler(feature_range=(0, 1))
#        for i in  range(len(result)):
#            new_result.append(scalar.fit_transform(result[i]))
#        result = np.array(new_result)
#        row = round(0.9 * result.shape[0])
#        train = result[:int(row), :]
#        np.random.shuffle(train)
#        x_train = train[:, :-1]
#        y_train = train[:, -1]
#        x_test = result[int(row):, :-1]
#        y_test = result[int(row):, -1]
#        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#        X_Train.append(x_train)
#        Y_Train.append(y_train)
#        X_Test.append(x_test)
#        Y_Test.append(y_test)
#    return [X_Train, Y_Train, X_Test, Y_Test, scalar]


#def build_model(layers):
#    model = Sequential()
#    model.add(LSTM(
#            input_dim=layers[0],
#            output_dim=layers[1],
#            return_sequences=False))
##    model.add(Dropout(0.2))
##    model.add(LSTM(
##        layers[2],
##        return_sequences=False))
#    model.add(Dropout(0.5))
#    model.add(Dense(
#            output_dim=layers[3]))
#    model.add(Activation("linear"))
#    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
#    return model


def load_data(filename,orig_file, seq_len):
    df = pd.read_csv(filename, header=None)
    df = df.fillna(df.mean())
    df_orig=pd.read_csv(orig_file,header=None)
    df_orig = df_orig[int(np.round((len(df_orig))*.60)):]
    X_Train = []
    Y_Train = []
    X_Test = []
    Y_Test = []
    for i in range(len(df.columns[:5])):
        data = df[i].tolist()
        orig_data=df_orig[i].tolist()
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        new_result = []
        scalar = MinMaxScaler(feature_range=(0, 1))
        for i in  range(len(result)):
            new_result.append(scalar.fit_transform(result[i]))
        result = np.array(new_result)
        result_orig=[]
        for index in range(len(orig_data) - sequence_length):
            result_orig.append(orig_data[index: index + sequence_length])
        
        new_result_orig = []
        scalar = MinMaxScaler(feature_range=(0, 1))
        for i in  range(len(result_orig)):
            new_result_orig.append(scalar.fit_transform(result_orig[i]))
        result_orig=np.array(new_result_orig)
        row = round(0.9 * result.shape[0])
        train = result[:int(row), :]
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test=result_orig[:, :-1]
        y_test=result_orig[:,-1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        X_Train.append(x_train)
        Y_Train.append(y_train)
        X_Test.append(x_test)
        Y_Test.append(y_test)
    return [X_Train, Y_Train, X_Test, Y_Test, scalar]


def build_model(layers):
    model = Sequential()
    model.add(LSTM(
        input_dim=layers[0], 
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05,seed=None),
        output_dim=layers[1],
        return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(
            output_dim=layers[3]))
    model.add(Activation("relu"))
    sgd = optimizers.sgd(lr=0.0025, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd, metrics=['mae'])
    return model

def predict_sequences_multiple(model, data, time_steps, prediction_len):
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        X = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(X[newaxis,:,:])[0,0])
            X = X[1:]
            X = np.insert(X, [time_steps-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
