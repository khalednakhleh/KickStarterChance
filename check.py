#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:44:18 2018

Name: khalednakhleh
"""
import keras
from keras.models import load_model
import numpy as np

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
    
    input_value = np.array([name_len, upper_len, lower_len, country, currency, goal, adj_goal, raised, adj_raised, backers])
    return input_value
    
def prediction(y_test):
    
    model = load_model("model.h5")
    prediction = model.predict_classes(y_test)
    
    return prediction
    
def interface():
    
    # Defining the success dictionary for mapping values
    success_map = {'failed': 0, 'canceled': 1, 'successful': 2,
                       'live': 3, 'undefined': 4, 'suspended': 5}
        
    print("\n\t\t-----------------------------\n\t\tKickStarterChance.\
Version 1.0.\n\t\t-----------------------------")
    
    print("\n\nThis program tries to guess whether a KickStarter project\
would succeed, fail, get canceled, live, get suspended, or be undefined.")
    
    name = input("Enter project name(default = Cool Gadgets): ") or "Cool Gadgets"
    country = input("Country were project was started (default = US): ") or "US"
    currency = input("Currency used for funding (default = USD): ") or "USD"
    goal = input("Funding goal (default = 42069): ") or 42069
    adj_goal = input("Adjusted funding goal (default = 69420): ") or 69420
    raised = input("Projected raised amount (default = 4242): ") or 4242
    adj_raised = input("Adjusted raised amount (default = 2424): ") or 2424
    backers = input("Projected number of backers (default = 99): ") or 99

    y_test = calc(name,country,currency,goal,adj_goal,raised,adj_raised,backers)
    y_test = np.reshape(y_test, (1, 10))
    
    pred = prediction(y_test)

    print("\n----------------------------------\n")
 
    for outcome, value in success_map.items():
        if value == pred[0]:
            print("Project is expected to: {} ".format(outcome))
    
    exit
    
if __name__ == "__main__":
    interface()