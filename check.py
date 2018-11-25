#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:44:18 2018

Name: khalednakhleh
"""

from keras.models import load_model
import sys

def interface():
    
    name = sys.argv[1]
    country = sys.argv[2]
    currency = sys.argv[3]
    goal = sys.argv[4]
    adj_goal = sys.argv[5]
    raised = sys.argv[6]
    adj_raised = sys.argv[7]
    backers = sys.argv[8]


    print("\n\t\t-----------------------------\n\t\tKickStarterChance.\
Version 1.0.\n\t\t-----------------------------")
    
    print("\n\nThis program tries to guess whether a KickStarter project\
would succeed, fail, get canceled, live, get suspended, or be undefined.")
    
    
    #model = load_model('model.h5')

    

if __name__ == "__main__":
    interface()