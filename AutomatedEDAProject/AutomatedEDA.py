# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:24:46 2023

@author: 3ndalib
"""
import sys
import pandas as pd
from pandas.api.types import is_object_dtype as ObjectDtype
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists    


def ReadingFile():
    read = False
    path = input("Please enter the path of the file: ")
    path = CheckExistence(path)
    while(not read):
        extension = path.split(".")[-1]
        match(extension):
            case("csv"):
                read = True
                return (pd.read_csv(path))
            case("xls"):
                read = True
                return (pd.read_excel(path))
            case("sql"):
                read = True
                return (pd.read_sql(path))
            case(_):
                path = input("Please enter a supported file type 'csv' or 'xlc' or 'sql': ")
                path = CheckExistence(path)

def CheckExistence(path):
    while(not exists(path)):
        path = input("Please enter a valid file path like 'Drive:\folder1\folder2\example.csv': ")
    return path        

def DataInfo(df):
     print("--------------- DataHead ---------------")
     print(df.head(10))
     print("--------------- DataDescribtion ---------------")
     print(df.describe())
     print("--------------- DataInfo ---------------")
     print(df.info())
     print("--------------- DataNunique ---------------")
     print(df.nunique())

def DataPreProcess(df):
    df = HandleNan(df)
    return df

def HandleNan(df):
    if(df.isnull().any(axis=1).sum() <= (df.shape[0]/10)):
        return df.dropna()
    else:
        return df.fillna(df.mean())

def Visualize(df,col):
     if(ObjectDtype(df[col])): 
         plt.figure()
         plt.hist(df[col], rwidth=0.9)     
         plt.title(col)
         plt.show()
         plt.figure()
         plt.scatter(pd.DataFrame(df[col].value_counts()).T.columns, df[col].value_counts())
         plt.title(col)
         plt.show()     
         plt.figure()
         sns.countplot(x=col,data=df,lw=0)
         plt.title(col)
         plt.show()
         plt.figure()
         plt.pie(df[col].value_counts(),labels=pd.DataFrame(df[col].value_counts()).T.columns)
         plt.title(col)
         plt.show()
     else:
         plt.figure()
         plt.hist(df[col], rwidth=0.9)     
         plt.title(col)
         plt.legend()
         plt.show()
         plt.figure()
         df[col].plot(kind="density")
         plt.title(col)
         plt.legend()
         plt.show()
         plt.figure()
         sns.boxplot(df[col])
         plt.title(col)
         plt.legend()
         plt.show()
         plt.figure()
         sns.violinplot(df[col])
         plt.title(col)
         plt.legend()
         plt.show()
         df_z_scaled = df.copy()
         df_z_scaled[col] = (df_z_scaled[col] - df_z_scaled[col].mean()) / df_z_scaled[col].std()    
         print("-------Normalized column-------")
         print(df_z_scaled[col])
         plt.figure()
         plt.hist(df_z_scaled[col], rwidth=0.9)     
         plt.title("Normalized"+" "+col)
         plt.legend()
         plt.show()
         plt.figure()
         df_z_scaled[col].plot(kind="density")
         plt.title("Normalized"+" "+col)
         plt.legend()
         plt.show()
         plt.figure()
         sns.boxplot(df_z_scaled[col])
         plt.title("Normalized"+" "+col)
         plt.legend()
         plt.show()
         plt.figure()
         sns.violinplot(df_z_scaled[col])
         plt.title("Normalized"+" "+col)
         plt.legend()
         plt.show()

pd.set_option('display.max_columns', None)



df = ReadingFile()
df = DataPreProcess(df)
DataInfo(df)

column = input("Please choose a column to visualize or Type 'exit' to close the program: ")

while(column!="exit"):
    while (column!="exit"):
        try:
            print(df[column])
        except:
            column = input("Please enter a valid column name: ")
        else:
            break
    if(column!="exit"):
        Visualize(df,column)
        column = input("Please choose a column to visualize or Type 'exit' to close the program: ")

sys.exit()

        






















