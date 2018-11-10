#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:30:15 2018

Name: khalednakhleh

ECEN 649 Project: KickStarterChance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    df = pd.read_csv("ks2018.csv")
    print("Data dimensions: " + str(df.shape))

if __name__ == "__main__":
    main()