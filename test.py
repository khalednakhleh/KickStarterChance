#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 18:57:10 2018

Name: khalednakhleh
"""

import pandas as pd
import numpy as np

df = pd.read_csv("clean.csv")


print(np.shape(df.iloc[0,:].values))