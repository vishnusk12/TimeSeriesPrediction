# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:46:27 2017

@author: Impact
"""

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import pandas as pd
import os
import warnings
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import initializers
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

loss = ['categorical_crossentropy', 'mean_squared_error', 
        'mean_absolute_error', 'mean_squared_logarithmic_error',
        'squared_hinge', 'hinge', 'poisson', 'categorical_hinge', 
        'logcosh', 'cosine_proximity', 'sparse_categorical_crossentropy', 
        'binary_crossentropy', 'kullback_leibler_divergence']
    
dropout = [0.1, 0.2, 0.3, 0.4, 0.5]

optimizer = ['rmsprop', 'adam', 'sgd', 
             'adagrad', 'adadelta', 
             'adamax', 'nadam']

activation = ['relu', 'sigmoid', 'tanh', 'linear', 
              'softplus', 'softsign', 'elu', 'selu']


def load_data(filename,orig_file, seq_len):
    df = pd.read_csv(filename, header=None)
    df = df.fillna(df.mean())
    df_orig=pd.read_csv(orig_file,header=None)
    df_orig=df_orig[int(np.round((len(df_orig))*.60)):]
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

def build_model(loss, dropout, optimizer, activation):
    model = Sequential()
    model.add(LSTM(
        input_dim=1, 
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05,seed=None),
        output_dim=50,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
            output_dim=1))
    model.add(Activation(activation))
    model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['mae'])
    return model


time_steps = 5
X_train, y_train, X_test, y_test, scalar = load_data(
            'C:/Users/Impact/Documents/Python Scripts/DeepLearning/LSTM Time Series/interpolatednew_data.csv', 
            'C:/Users/Impact/Documents/Python Scripts/DeepLearning/LSTM Time Series/Preprocessed_data.csv', 
            time_steps)

for i in range(1):
    model = KerasRegressor(build_fn=build_model, epochs=10, batch_size=10, verbose=0)
    param_grid = dict(activation=activation, loss=loss, optimizer=optimizer, dropout=dropout)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train[i], y_train[i])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))