#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:44:45 2018

Name: khalednakhleh
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

def main():
    dataset = pd.read_csv("clean.csv")

    X_train = dataset.iloc[:,0:9]
    y_train = dataset.iloc[:, 10]
    
    y_train = keras.utils.to_categorical(y_train, dtype = "float32")

    model = Sequential()
    model.add(Dense(512, input_dim = 9, activation='relu'))
    model.add(Dense(384, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(256, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(6, kernel_initializer='uniform', activation='softmax'))
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs = 5, batch_size = 256, verbose = 2)
    
    score = model.evaluate(X_train, y_train)

    print("\nModel accuracy: " + str(100 * round(score[1], 3)) + "%\n")

    model.save('model.h5')
    
if __name__ == "__main__":
    print("\nFile intended as read-only. Please use start.sh")
    exit