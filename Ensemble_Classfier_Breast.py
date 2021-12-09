# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 23:37:01 2021
@author: Maher
"""
# importing utility modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# importing machine learning models for prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import xgboost as xgboost

data= pd.read_csv('breast-cancer-wisconsin-data\data.csv',header=0)
#%%
#delete unused columns
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

#change label M = malignant = 1 dan B = Benign = 0
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
m_data = data.loc[data['diagnosis'] == 1] 
b_data = data.loc[data['diagnosis'] == 0]

b_data.head(10)


x = data.iloc[:, 1:]
y = data['diagnosis'].tolist()

#share test and train data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# #change label M = malignant = 1 dan B = Benign = 0
# breast.Class = [1 if each == "yes" else 0 for each in breast.Class]
# m_data = data.loc[data['diagnosis'] == 1] 
# b_data = data.loc[data['diagnosis'] == 0]
# Splitting between train data into training and validation dataset
# initializing all the model objects with default parameters
model_1 = SVC(kernel='poly', degree=8)
model_2 = xgboost.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')
model_3 = RandomForestClassifier()

# training all the model on the training dataset
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)

# # predicting the output on the validation dataset
pred_1 = model_1.predict(X_test)
pred_2 = model_2.predict(X_test)
pred_3 = model_3.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("------------------  SVM Classfier----------------")
# print(confusion_matrix(y_test, pred_1))
print(classification_report(y_test, pred_1))

print("------------------  xgboost Classfier----------------")
# print(confusion_matrix(y_test, pred_2))
print(classification_report(y_test, pred_2))

print("------------------  RandomForest Classfier----------------")
# print(confusion_matrix(y_test, pred_3))
print(classification_report(y_test, pred_3))

#------------------ Ensemble Classfier ------------------------
# Making the final model using voting classifier
final_model = VotingClassifier(
    estimators=[('lr', model_1), ('xgb', model_2), ('rf', model_3)], voting='hard')
 # training all the model on the train dataset
final_model.fit(X_train, y_train)
 # predicting the output on the test dataset
pred_final = final_model.predict(X_test)

print("------------------  Ensemble Classfier----------------")
print(confusion_matrix(y_test, pred_final))
print(classification_report(y_test, pred_final))
