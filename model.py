#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:44:45 2018

Name: khalednakhleh
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D, MaxPooling2D
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

    df = data.iloc[:, 0:10]
    y = data.iloc[:, 10]
    
    #df = np.expand_dims(df, axis=2)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2)


    print ("Training data dimensions {} {}" .format(X_train.shape, y_train.shape))
    print ("Validation data dimensions {} {}\n\n" .format(X_test.shape, y_test.shape))
    

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    model = Sequential()
    model.add(Dense(512, input_dim = 10, activation='relu'))
    model.add(Dense(256, kernel_initializer='uniform',  kernel_regularizer=regularizers.l2(0.3), activation='relu'))
    model.add(Dense(128, kernel_initializer='uniform',  kernel_regularizer=regularizers.l2(0.15), activation='relu'))
    model.add(Dense(64, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(32, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(16, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(6, kernel_initializer='uniform', activation='softmax'))
    
    """
    model = Sequential()
    model.add(Conv1D(32, 1, input_shape = (5,5,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(16, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(6, kernel_initializer='uniform', activation='softmax'))
    """
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs = 5,
              batch_size = 1024,
              validation_data = (X_test,y_test),
              verbose = 2)
    
    loss, accuracy  = model.evaluate(X_test, y_test, verbose = False)

    print (model.summary())
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.grid()
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.savefig("model.png")
    plt.show()
    
    plt.figure(2)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
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
