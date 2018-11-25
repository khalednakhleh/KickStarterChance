#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:13:59 2018

Name: khalednakhleh
"""

import pandas as pd

def mapping_strings(filename):
    curr_map = {'GBP': 0, 'USD': 1, 'CAD': 2, 'AUD': 3, 'NOK': 4, 
                         'EUR': 5, 'MXN': 6, 'SEK': 7, 'NZD': 8, 'CHF': 9,
                         'DKK': 10, 'HKD': 11, 'SGD': 12, 'JPY': 13}
    
    count_map = {'GB': 0, 'US': 1, 'CA': 2, 'AU': 3, 'NO': 4, 'IT': 5, 
                          'DE': 6, 'IE': 7, 'MX': 8, 'ES': 9, 'N,0"': 10, 
                          'SE': 11, 'FR': 12, 'NL': 13, 'NZ': 14, 'CH': 15, 
                          'AT': 16, 'DK': 17, 'BE': 18, 'HK': 19, 'LU': 20,
                          'SG': 21, 'JP': 22}
    
    df = pd.read_csv(filename)
    print(count_map)
    print(curr_map)
        
    df["Country"] = df["Country"].map(count_map)
    df["Currency"] = df["Currency"].map(curr_map)
    
    df.to_csv("final.csv", index = False)
    
    
