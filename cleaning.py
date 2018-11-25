#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:05:40 2018

Name: khalednakhleh
"""
import pandas as pd
import numpy as np

class Cleaning(object):
    
    def __init__(self, name):
        
        self.df = pd.read_csv(name)
        self.clean = pd.DataFrame()
        
        self.curr_map = {}
        self.count_map = {}
        self.succ_map = {}
        currency = self.df.currency.unique()
        countries = self.df.country.unique()
        funded = self.df.state.unique()

        i = 0
        while ( i < currency.shape[0]):
            self.curr_map[currency[i]] = i
            i += 1
          
        i = 0
        while ( i < countries.shape[0]):
            self.count_map[countries[i]] = i
            i += 1
            
        i = 0
        while ( i < funded.shape[0]):
            self.succ_map[funded[i]] = i
            i += 1
        
        print("Dictionaires to map countries, currinces, and success outcome:\n")
        print("Countries dictionary:\n" + str(self.count_map) + "\n\n")
        print("currinces dictionary:\n" + str(self.curr_map)  + "\n\n")
        print("success dictionary:\n" + str(self.succ_map)  + "\n\n")
        
        
    def copying(self):
        
        print("\nCopying columns to the clean file in preparation...\n")
        self.clean["Country"] = self.df["country"]
        self.clean["Currency"] = self.df["currency"]
        #self.clean["Start"] = self.df["launched"]
        #self.clean["End"] = self.df["deadline"]
        self.clean["Goal"] = self.df["goal"]
        self.clean["GoalAdjusted"] = self.df["usd_goal_real"]
        self.clean["Raised"] = self.df["usd pledged"]
        self.clean["RaisedAdjusted"] = self.df["usd_pledged_real"]
        self.clean["Backers"] = self.df["backers"]
        self.clean["State"] = self.df["state"]
    
    def add(self):
        
        print("\nPrinting info and filling empty elements with 0...\n")
        if (self.df.isnull().values.any()):
            print("\nData is not clean. NaN detected.\n--------------------------------\n")
            print("Data dimensions: " + str(self.df.shape) + "\n")
            
            # Missing values calcuation
            ZeroValues = self.df.isnull().sum()
            print("Number of non-defined values in each column:\n\n" + str(ZeroValues) + "\n")
            TotalCells = np.product(self.df.shape)
            Total = ZeroValues.sum()
            print("Missing data percentage: %" +str(round((Total/TotalCells) * 100, 4)))
            
            # Filling missing values with zero
            self.df = self.df.fillna(0)
            
        else:
            print("data is clean")
            
    def str_int(self):
        
        print("\nCalculating number of characters,uppercases, and lowercases in name column...\n")
        self.clean["NameCount"] = (self.df["name"].str.len())
        self.clean['UpperCase'] = self.df["name"].str.findall(r"[A-Z]").str.len()
        self.clean['LowerCase'] = self.df["name"].str.findall(r"[a-z]").str.len()


    def quant_values(self):
        
        print("\nConverting strings to numbers for model calculation...\n")
        self.clean["Country"] = self.clean["Country"].map(self.count_map)
        self.clean["Currency"] = self.clean["Currency"].map(self.curr_map)
        self.clean["State"] = self.clean["State"].map(self.succ_map)
        
    def saveit(self):
        
        print("\nSaving file...\n")
        self.clean.to_csv("clean.csv", index = False)
        print("Cleaned data, and placed it in file 'clean.csv'")



if __name__ == "__main__":
    print("\nFile intended as read-only. Please use start.sh")
    exit
        
        
        
        
        
        
        
        
    