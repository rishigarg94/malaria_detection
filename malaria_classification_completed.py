# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:03:38 2020

@author: rishi
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib

##Step1: Load Dataset

dataframe1= pd.read_csv("csv/dataset1.csv")
dataframe=pd.read_csv("csv/dataset.csv")
#print(dataframe.head())
#print(dataframe1.head())
label=dataframe.index
area_0=dataframe["Label"]
area_1=dataframe["area_0"]
area_2=dataframe["area_1"]
area_3=dataframe["area_2"]
area_4=dataframe["area_3"]
dataframe["Label"]=label
dataframe["area_0"]=area_0
dataframe["area_1"]=area_1
dataframe["area_2"]=area_2
dataframe["area_3"]=area_3
dataframe["area_4"]=area_4
dataframe=dataframe.set_index(["Label"])
dataframe=dataframe.reset_index()
dataframe=dataframe.fillna(0)
#dataframe.set_index(["Label"],inplace=True,append=True,drop=True)
#Label=dataframe.index
#dataframe.reset_index
#dataframe["label"]=Label
print(dataframe)

#Step2: Split into training and test data
x = dataframe.drop(["Label"],axis=1)
#x=dataframe.iloc[0]
#dataframe.columns =dataframe.iloc[0]
#dataframe=dataframe[1:]
#print(x)
#print("here end x")
y = dataframe["Label"]
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2)

##Step4: Build a model

model = RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(x_train,y_train)
joblib.dump(model,"rf_malaria_100_5")


##Step5: Make predictions and get classification report

predictions = model.predict(x_test)

print(metrics.classification_report(predictions,y_test))
print(model.score(x_test,y_test))