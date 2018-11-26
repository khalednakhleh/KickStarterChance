#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:30:15 2018

Name: khalednakhleh

ECEN 649 Project: KickStarterChance
"""

# My own scripts
from cleaning import Cleaning
   
def main():
    
    print("\nInitializing main.py...\n")
    FileName = "ks2018.csv"
    
    clean = Cleaning(FileName)
    clean.str_int()
    clean.copying()
    clean.add()
    clean.quant_values()
    clean.saveit()       
    
if __name__ == "__main__":
    main() 
