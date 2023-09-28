# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:49:08 2023

@author: 3ndalib
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder as LE
pd.set_option('display.max_columns', None)

df = pd.read_excel(r"E:\Projects\ElectroPi\UnSuperVised\OnlineRetail.xlsx")

df = df.drop(["StockCode","Description","CustomerID"],axis=1)
#df=pd.get_dummies(df,drop_first=True)
df["InvoiceNo"] = df["InvoiceNo"].astype('str')
df["InvoiceNo"] = LE().fit_transform(df["InvoiceNo"])
df["InvoiceDate"] = df["InvoiceDate"].astype('str')
df["InvoiceDate"] = LE().fit_transform(df["InvoiceDate"])
df["Country"] = df["Country"].astype('str')
df["Country"] = LE().fit_transform(df["Country"])

plt.figure()
sns.pairplot(df)
plt.show()




























