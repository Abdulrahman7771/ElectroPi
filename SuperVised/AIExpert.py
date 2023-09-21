# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:06:53 2023

@author: 3ndalib
"""

import pandas as pd

from os.path import exists    
import streamlit as st
from sklearn.linear_model import LinearRegression as LNR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import Ridge as ridge
from sklearn.linear_model import SGDRegressor as sgdr
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import PolynomialFeatures as polynomial
from sklearn.metrics import ConfusionMatrixDisplay

import streamlit as st
from sklearn.metrics import mean_absolute_error,accuracy_score,precision_score,recall_score,r2_score,mean_squared_error


pd.set_option('display.max_columns', None)

# def GetData(x):
#     read = False
#     path = x
#     path = CheckExistence(path)
#     while(not read):
#         extension = path.split(".")[-1]
#         if(extension=="csv"):
#             read = True
#             print(pd.read_csv(path))
#             return (pd.read_csv(path))
#         elif(extension=="xls"):
#             read = True
#             print(pd.read_excel(path))
#             return (pd.read_excel(path))
#         elif(extension=="sql"):
#                 read = True
#                 print(pd.read_sql(path))
#                 return (pd.read_sql(path))
#         else:
#             path = input("Please enter a supported file type: ")
            
    
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

file = st.file_uploader(
    "Upload your data file",
    accept_multiple_files=False,
    key=st.session_state["file_uploader_key"],
)

if file:
    st.session_state["uploaded_files"] = file

# if st.button("Clear uploaded files"):
#     st.session_state["file_uploader_key"] += 1
#     st.experimental_rerun()



if file is not None:
        extension = file.name.split(".")[-1]
        if(extension=="csv"):
            df = pd.read_csv(file)
            st.toast("File uploaded succesfuly")
        elif(extension=="xls" or extension=="xlsx"):
            df = pd.read_excel(file)
            st.toast("File uploaded succesfuly")
        elif(extension=="sql"):
            df = pd.read_sql(file)
            st.toast("File uploaded succesfuly")
        else:
            st.session_state["file_uploader_key"] += 1
            st.experimental_rerun()  
            

def TaskType(df,target):
    if(not (df[target].dtype.kind in 'biufc')):
        return "classification"
    else:
        if(df[target].nunique()>5):
            return "regression"
        else:
            return "classification"

def HandleNan(df,numerical,categorical):
    for i in range(df.shape[1]):
        if( df.iloc[:,i].dtype.kind in 'biufc'):
            if(numerical=="mean"):
                df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].mean)
            else:
                df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].mode)
        else:
            if(categorical=="mode"):
                df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].mode)
            else:
                df.iloc[:,i] = df.iloc[:,i].fillna(categorical)
    return df
def LabelEncoding(df):
    for i in range(df.shape[1]):
        if(not (df.iloc[:,i].dtype.kind in 'biufc')):
            df.iloc[:,i] = LE().fit_transform(df.iloc[:,i])
    return df

def StandardScaling(df,target):
    for i in range(df.shape[1]):
        if(df.iloc[:,i].dtype.kind in 'biufc' and df.columns[i]!=target):
            df.iloc[:,i] = SS().fit_transform(df[[df.columns[i]]]).flatten()
    return df
def PreProcess(df,target,numerical,categorical):
    df = HandleNan(df,numerical,categorical)
    df = StandardScaling(df, target)
    df = LabelEncoding(df)
    return(df)    

    
class ClassificationExpert:
    def setup(self,data,target,numerical,categorical,TestSize=0.3):
        self.data = data
        self.target = target
        self.numerical = numerical
        self.categorical = categorical
        self.TestSize = TestSize
        self.PPData = PreProcess(self.data,self.target,self.numerical,self.categorical)
        print(self.PPData)
        self.x,self.y=self.PPData.drop(target,axis=1),self.PPData[target]
        print(self.x)
        print(self.y)
        self.XTrain,self.XTest,self.YTrain,self.YTest =  tts(self.x,self.y,test_size=TestSize)
        self.info = pd.DataFrame({"Description":["Target",
                                                  "Original Shape",
                                                  "Shape after preprocessing",
                                                  "XTrainShape",
                                                  "YTrainShape",
                                                  "XTestShape",
                                                  "YTestShape"],
                                  "Value":[target
                                            , data.shape,
                                            self.PPData.shape,
                                            self.XTrain.shape,
                                            self.YTrain.shape,
                                            self.XTest.shape,
                                            self.YTest.shape]})
    def CompareModels(self):
        self.lrclf = LR()
        self.lrclf.fit(self.XTrain, self.YTrain)
        self.lrpreds = self.lrclf.predict(self.XTrain)
        self.lrtestpreds = self.lrclf.predict(self.XTest)
       
        self.csvclf = SVC()
        self.csvclf.fit(self.XTrain, self.YTrain)
        self.csvpreds = self.csvclf.predict(self.XTrain)
        self.csvtestpreds = self.csvclf.predict(self.XTest)
        
        self.knclf = KNC()
        self.knclf.fit(self.XTrain, self.YTrain)
        self.knpreds = self.knclf.predict(self.XTrain)
        self.kntestpreds = self.knclf.predict(self.XTest)
        
        self.dtclf = DTC(max_depth=5)
        self.dtclf.fit(self.XTrain, self.YTrain)
        self.dtpreds = self.dtclf.predict(self.XTrain)
        self.dttestpreds = self.dtclf.predict(self.XTest)
        
        self.rfclf = RFC(n_estimators=20)
        self.rfclf.fit(self.XTrain, self.YTrain)
        self.rfpreds = self.rfclf.predict(self.XTrain)
        self.rftestpreds = self.rfclf.predict(self.XTest)
        
        self.ModelsPerformance = pd.DataFrame(
            {"Model":[
                "LogisticRegression",
                "SuppportVectorMachine",
                "KNeighrestNeighbor",
                "DecisionTree",
                "RandomForest",],
             "TrainAccuracy":[ 
                 accuracy_score(self.YTrain, self.lrpreds),
                 accuracy_score(self.YTrain, self.csvpreds),
                 accuracy_score(self.YTrain, self.knpreds),
                 accuracy_score(self.YTrain, self.dtpreds),
                 accuracy_score(self.YTrain, self.rfpreds),],
             "TestAccuracy":[ 
                 accuracy_score(self.YTest, self.lrtestpreds),
                 accuracy_score(self.YTest, self.csvtestpreds),
                 accuracy_score(self.YTest, self.kntestpreds),
                 accuracy_score(self.YTest, self.dttestpreds),
                 accuracy_score(self.YTest, self.rftestpreds),],
             "Precision":[
                 precision_score(self.YTrain, self.lrpreds),
                 precision_score(self.YTrain, self.csvpreds),
                 precision_score(self.YTrain, self.knpreds),
                 precision_score(self.YTrain, self.dtpreds),
                 precision_score(self.YTrain, self.rfpreds),],
             "Recall":[                                            
                 recall_score(self.YTrain, self.lrpreds),
                 recall_score(self.YTrain, self.csvpreds),
                 recall_score(self.YTrain, self.knpreds),
                 recall_score(self.YTrain, self.dtpreds),
                 recall_score(self.YTrain, self.rfpreds),]})
        print(self.ModelsPerformance)
        self.best = self.ModelsPerformance.query('TestAccuracy == TestAccuracy.max()')
        self.best = self.ModelsPerformance.query('TrainAccuracy == TrainAccuracy.max()')
        print(self.best)
    def plot(self):
        disp = ConfusionMatrixDisplay.from_estimator(self.rfclf,self.XTrain,self.YTrain)
        disp = ConfusionMatrixDisplay.from_estimator(self.rfclf,self.XTest,self.YTest)

class RegressionExpert:
    def setup(self,data,target,numerical,categorical,TestSize=0.3):
        self.data = data
        self.target = target
        self.numerical = numerical
        self.categorical = categorical
        self.TestSize = TestSize
        self.PPData = PreProcess(self.data,self.target,self.numerical,self.categorical)
        print(self.PPData)
        self.x,self.y=self.PPData.drop(target,axis=1),self.PPData[target]
        print(self.x)
        print(self.y)
        self.XTrain,self.XTest,self.YTrain,self.YTest =  tts(self.x,self.y,test_size=TestSize)
        self.info = pd.DataFrame({"Description":["Target",
                                                  "Original Shape",
                                                  "Shape after preprocessing",
                                                  "XTrainShape",
                                                  "YTrainShape",
                                                  "XTestShape",
                                                  "YTestShape"],
                                  "Value":[target
                                            , data.shape,
                                            self.PPData.shape,
                                            self.XTrain.shape,
                                            self.YTrain.shape,
                                            self.XTest.shape,
                                            self.YTest.shape]})
    def CompareModels(self):
        self.lnr = LNR()
        self.lnr.fit(self.XTrain, self.YTrain)
        self.lnrpreds = self.lnr.predict(self.XTrain)
        self.lnrtestpreds = self.lnr.predict(self.XTest)
       
        self.poly = polynomial(degree=2)
        self.XTrainPoly = self.poly.fit_transform(self.XTrain)
        self.XTestPoly = self.poly.fit_transform(self.XTest)
        
        self.lrpoly = LNR()
        self.lrpoly.fit(self.XTrainPoly, self.YTrain)
        self.polypreds = self.lrpoly.predict(self.XTrainPoly)
        self.polytestpreds = self.lrpoly.predict(self.XTestPoly)
        
        self.lassopoly = lasso(0.0001)
        self.lassopoly.fit(self.XTrainPoly, self.YTrain)
        self.polylassopreds = self.lassopoly.predict(self.XTrainPoly)
        self.polylassotestpreds = self.lassopoly.predict(self.XTestPoly)
        
        self.ridgepoly = ridge(0.0001)
        self.ridgepoly.fit(self.XTrainPoly, self.YTrain)
        self.polyridgepreds = self.ridgepoly.predict(self.XTrainPoly)
        self.polyridgetestpreds = self.ridgepoly.predict(self.XTestPoly)
        
        self.sgdpoly = sgdr(max_iter=100)
        self.sgdpoly.fit(self.XTrainPoly, self.YTrain)
        self.polysgdpreds = self.sgdpoly.predict(self.XTrainPoly)
        self.polysgdtestpreds = self.sgdpoly.predict(self.XTestPoly)
        
        self.ModelsPerformance = pd.DataFrame(
            {"Model":[
                "LinearRegression",
                "PolyLinearRegression",
                "Lasso",
                "ridge",
                "GradientDescent",
                ],
             "Trainr2":[ 
                 r2_score(self.YTrain, self.lnrpreds),
                 r2_score(self.YTrain, self.polypreds),
                 r2_score(self.YTrain, self.polylassopreds),
                 r2_score(self.YTrain, self.polyridgepreds),
                 r2_score(self.YTrain, self.polysgdpreds),
                 ],
             "Testr2":[ 
                 r2_score(self.YTest, self.lnrtestpreds),
                 r2_score(self.YTest, self.polytestpreds),
                 r2_score(self.YTest, self.polylassotestpreds),
                 r2_score(self.YTest, self.polyridgetestpreds),
                 r2_score(self.YTest, self.polysgdtestpreds),
                 ],
              "MeanSquareError":[
                 mean_squared_error(self.YTest, self.lnrtestpreds),
                 mean_squared_error(self.YTest, self.polytestpreds),
                 mean_squared_error(self.YTest, self.polylassotestpreds),
                 mean_squared_error(self.YTest, self.polyridgetestpreds),
                 mean_squared_error(self.YTest, self.polysgdtestpreds),
             ],
              "MeanAbsoluteError":[                                            
                 mean_absolute_error(self.YTest, self.lnrtestpreds),
                 mean_absolute_error(self.YTest, self.polytestpreds),
                 mean_absolute_error(self.YTest, self.polylassotestpreds),
                 mean_absolute_error(self.YTest, self.polyridgetestpreds),
                 mean_absolute_error(self.YTest, self.polysgdtestpreds),
             ]
             })
        print(self.ModelsPerformance)
        # self.best = self.ModelsPerformance.query('TestAccuracy == TestAccuracy.max()')
        # self.best = self.ModelsPerformance.query('TrainAccuracy == TrainAccuracy.max()')
        # print(self.best)
    # def plot(self):
    #     disp = ConfusionMatrixDisplay.from_estimator(self.rfclf,self.XTrain,self.YTrain)
    #     disp = ConfusionMatrixDisplay.from_estimator(self.rfclf,self.XTest,self.YTest)

if file is not None:
    st.caption("Data head")
    st.write(df.head())
    target = st.selectbox('Choose target column to predict', df.columns)
    d = df.columns.values.tolist()
    d.remove(target)
    dropcolumns = st.multiselect('Select Columns to drop'
                , d)
    df = df.drop(dropcolumns,axis=1)
    st.toast("Columns dropped succesfuly")
    st.write(df.head())
    match TaskType(df,target):
        case 'regression':
            expert = RegressionExpert()
        case 'classification':
            expert = ClassificationExpert()
    HandleNumericNan = st.selectbox('Choose how to handle numeric null values', ["mean","mode"])
    HandleCategoricalNan = st.selectbox('Choose how to handle categorical null values', ["mode","add a new category"])
    if HandleCategoricalNan == "add a new category":
        HandleCategoricalNan = st.text_input("Enter the category")
    TestSize = st.slider('choose test data size', min_value=0.1, max_value=0.5)
    expert.setup(df, target, HandleNumericNan, HandleCategoricalNan,TestSize)
# s = ClassificationExpert()
# s.setup(df, "Purchased")
# print(s.info)
# s.CompareModels()
# s.plot()






















