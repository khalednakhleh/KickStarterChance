#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:44:18 2018

Name: khalednakhleh
"""

from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf

def log_reg(x, y, t, q):
    """ This function is an amalgamation of different minute tasks that 
    I just gatherd into a singal call function to ease work."""
    
    pred = lr(solver = "saga", tol = 0.001, max_iter = 600, n_jobs = -1, fit_intercept = True)
    pred.fit(x,y)                              # Predictor training
    g = pred.score(t,q)           # Predictor test
    pred = pred.predict(t)                     # Predicting correct labels
    
    # Printing some information for user
    print("------------------------------------------")
    print("accuracy rate is %{}" .format(round(g * 100 , 3)))
    print("Error rate is %{}" .format(round((1 - g) * 100 , 3)))
    
    return pred

def rand_forest(x, y, t, q):
    clf = rf(n_estimators=100, max_depth=6,random_state=0)
    clf.fit(x, y)
    g=clf.score(t, q)
    pred_rf=clf.predict(t)
    
    # Printing some information for user
    print("------------------------------------------")
    print("accuracy rate in random forest is %{}" .format(round(g * 100 , 3)))
    print("Error rate in random forest is %{}" .format(round((1 - g) * 100 , 3)))
    
    return pred_rf

def calc(name,country,currency,goal,adj_goal,raised,adj_raised,backers):
    
    curr_map = {'GBP': 0, 'USD': 1, 'CAD': 2, 'AUD': 3, 'NOK': 4, 
                          'EUR': 5, 'MXN': 6, 'SEK': 7, 'NZD': 8, 'CHF': 9,
                          'DKK': 10, 'HKD': 11, 'SGD': 12, 'JPY': 13}
     
    count_map = {'GB': 0, 'US': 1, 'CA': 2, 'AU': 3, 'NO': 4, 'IT': 5, 
                           'DE': 6, 'IE': 7, 'MX': 8, 'ES': 9, 'N,0"': 10, 
                           'SE': 11, 'FR': 12, 'NL': 13, 'NZ': 14, 'CH': 15, 
                           'AT': 16, 'DK': 17, 'BE': 18, 'HK': 19, 'LU': 20,
                           'SG': 21, 'JP': 22}

    # cleaning name
    name_len = (len(name))
    upper_len = (sum(1 for c in name if c.isupper()))
    lower_len = (sum(1 for c in name if c.islower()))
    
    # mapping country and currency to integer values
    country = count_map.get(country)
    currency = curr_map.get(currency)
    
    input_value = np.array([name_len, upper_len, lower_len, country, currency,
                            goal, adj_goal, raised, adj_raised, backers])
    return input_value


def main():
    
    data = pd.read_csv("clean.csv")
    #data.astype("float32")
    
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

    X_train = (data.iloc[:, 0:10])
    y_train = (data.iloc[:, 10])
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25)
    pred = log_reg(X_train, y_train, X_test, y_test)
    pred_rf = rand_forest(X_train, y_train, X_test, y_test)
    
    print(pred)
    print("------------------------------------------\n")
    print(pred_rf)

if __name__ == "__main__":
    main()