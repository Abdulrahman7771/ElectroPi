# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 01:15:28 2023

@author: 3ndalib
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras import Sequential

pd.set_option('display.max_columns', None)

df = pd.read_csv("bank.csv",sep = ';')

print(df)

print(df.describe())

print(df.info())

















