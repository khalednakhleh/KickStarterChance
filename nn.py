a#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:44:45 2018

Name: khalednakhleh
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras import regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy as np

def main():
    
    data = pd.read_csv("clean.csv")
    data.astype("float32")
    
    # Normalizing the data per label
    data.iloc[:, 0] /= max(data.iloc[:, 0])
    data.iloc[:, 1] /= max(data.iloc[:, 1])
    data.iloc[:, 2] /= max(data.iloc[:, 2])
    data.iloc[:, 3] /= max(data.iloc[:, 3])
    data.iloc[:, 4] /= max(data.iloc[:, 4])
    data.iloc[:, 5] /= max(data.iloc[:, 5])
    data.iloc[:, 6] /= max(data.iloc[:, 6])
    data.iloc[:, 7] /= max(data.iloc[:, 7])
    data.iloc[:, 8] /= max(data.iloc[:, 8])
    data.iloc[:, 9] /= max(data.iloc[:, 9])

    X_train = (data.iloc[:, 0:10].values)
    y_train = (data.iloc[:, 10].values)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25)
        
    print ("Training data dimensions {} {}" .format(X_train.shape, y_train.shape))
    print ("Validation data dimensions {} {}\n\n" .format(X_test.shape, y_test.shape))

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    model = Sequential()
    model.add(Dense(60, input_dim = 10, activation='relu'))
    model.add(Dense(120, kernel_initializer='uniform',  kernel_regularizer=regularizers.l2(0.3), activation='relu'))
    model.add(Dense(120, kernel_initializer='uniform',  kernel_regularizer=regularizers.l2(0.15), activation='relu'))
    model.add(Dense(250, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(250, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(100, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(3, kernel_initializer='uniform', activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,
              epochs = 5,  
              batch_size= 500)
    
    loss, accuracy  = model.evaluate(X_test, y_test)

    print (model.summary())
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.grid()
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.savefig("model.png")
    plt.show()
    
    plt.figure(2)
    
    plt.plot(history.history['loss'])
    plt.grid()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.savefig("error.png")
    plt.show()

    model.save("model.h5")
    
if __name__ == "__main__":
    main()
