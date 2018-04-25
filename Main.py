# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:22:03 2017

@author: Impact
"""

import LSTM_Model_Builder
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_absolute_error
import  matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv('C:/Users/Impact/Documents/Python Scripts/DeepLearning/LSTM Time Series/data.csv')
    df = data['Product_d'].value_counts()
    max_count = df.max()
    min_count = df.min()
    threshold = np.round(max_count+min_count)/2
    valid_products = []
    df_new = pd.DataFrame()
    for i, row in df.iteritems():
        if row >= threshold:
            valid_products.append(i)
    prod = valid_products[:5]
    global_start_time = time.time()
    epochs  = 140
    time_steps = 50
    prediction_length = 56
    X_train, y_train, X_test, y_test, scalar = LSTM_Model_Builder.load_data(
            'C:/Users/Impact/Documents/Python Scripts/DeepLearning/LSTM Time Series/interpolatednew_data.csv', 
            'C:/Users/Impact/Documents/Python Scripts/DeepLearning/LSTM Time Series/Preprocessed_data.csv', 
            time_steps)
    forecast = []
    for i in range(len(prod)):
        model = LSTM_Model_Builder.build_model([1, 50, 100, 1])
        history = model.fit(X_train[i], 
                            y_train[i],
                            batch_size=512,
                            epochs=epochs,
                            validation_split=0.05)
        predictions = LSTM_Model_Builder.predict_sequences_multiple(model, X_test[i], time_steps, prediction_length)
        actual_predictions = (scalar.inverse_transform(np.array(predictions[len(predictions) - 1]).reshape(-1, 1)))
        forecast.append(actual_predictions)
        x_test = model.predict(X_test[i])
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('Model Error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('C:/Users/Impact/Documents/Python Scripts/DeepLearning/plots/Accuracy' + str(i) + '.png')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('C:/Users/Impact/Documents/Python Scripts/DeepLearning/plots/Loss' + str(i) + '.png')
        plt.show()
        plt.plot(y_test[i])
        plt.legend(['Actual'], loc='upper left')
        plt.savefig('C:/Users/Impact/Documents/Python Scripts/DeepLearning/plots/Actual' + str(i) + '.png')
        plt.show()
        plt.plot(x_test)
        plt.legend(['Predicted'], loc='upper left')
        plt.savefig('C:/Users/Impact/Documents/Python Scripts/DeepLearning/plots/Predicted' + str(i) + '.png')
        plt.show()
        plt.title('Comparison')
        plt.ylabel('Values')
        plt.xlabel('Count')
        plt.plot(y_test[i])
        plt.plot(x_test)
        plt.legend(['Actual', 'Predicted'], loc='upper left')
        plt.savefig('C:/Users/Impact/Documents/Python Scripts/DeepLearning/plots/Comparison' + str(i) + '.png')
        plt.show()
        y_true = np.reshape(y_test[i],(len(y_test[i]),1))
        error = mean_absolute_error(y_true, x_test)
        print (error)
    print('Training duration (s) : ', time.time() - global_start_time)
    actual_forecast = [np.around(abs(elem)) for elem in forecast]
    prod_forecast = dict(zip(prod, actual_forecast))  
