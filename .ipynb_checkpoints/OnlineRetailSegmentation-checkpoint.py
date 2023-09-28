# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:49:08 2023

@author: 3ndalib
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

df = pd.read_excel(r"E:\Projects\ElectroPi\UnSuperVised\OnlineRetail.xlsx")
print("DataHead")
print(df.head())

df = df.drop(["StockCode","Description"],axis=1)
print("Dropping ['StockCode','Description'] columns")
print(df.head())

print(df.info())

df = df.drop(["CustomerID"],axis=1)
print("Dropping ['CustomerID'] columns")
print(df.head())
plt.figure()
sns.pairplot(df)
plt.show()




























