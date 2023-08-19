# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:07:21 2023

@author: 3ndalib
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.options.display.max_columns = None

df = pd.read_csv("boston-housing-dataset.csv")

x = df.drop(columns=[df.columns[0],"MEDV"])
y = df["MEDV"]

scaler = StandardScaler()
columns = x.columns
x = scaler.fit_transform(x)
x = pd.DataFrame(x,columns=columns)

XTrain,XTest,YTrain,YTest= train_test_split(x,y, test_size=0.3)

print(XTrain.shape)
print(XTest.shape)
print(YTrain.shape)
print(YTest.shape)

print(XTrain.head())

















