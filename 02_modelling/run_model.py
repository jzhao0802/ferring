# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:58:39 2017

@author: TReason
"""
import os
import plotly
import pandas as pd
import numpy as np
from sklearn import metrics
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import tree, neighbors, ensemble, metrics, svm, preprocessing, linear_model, tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

path = "C:/Users/treason/Desktop/Projects/Ferring/Data/initial_analysis"

os.chdir(path)

from random_forest_classifier_project import RandomForestClassifier
import pandas as pd

df = pd.read_csv('PROCESSED_FLATFILE.csv')

delivery_dict = {'CESAREAN SECTION': 0, 'VAGINAL': 1}

df = df[['LABEL', 'AGE', 'BMI', 'BS_BASELINE', 
          'RACE_dummy_ASIAN','RACE_dummy_BLACK_OR_AFRICAN_AMERICAN', 
          'RACE_dummy_HISPANIC', 'RACE_dummy_OTHER', 'RACE_dummy_WHITE',
          'GESTATIONAL_AGE_DAYS','MHTERM_GESTATIONAL_DIABETES_FLAG',
          'MHTERM_GESTATIONAL_DIABETES_PREV_PREG_FLAG']].dropna()

#df[df['MODE_OF_DELIVERY']=='VAGINAL'] = 0
#df[df['MODE_OF_DELIVERY']=='CESAREAN SECTION'] = 1 


rf_classifier = RandomForestClassifier(data_frame=df)

preds = rf_classifier.run_pipeline(
    criteria_to_include=[True, True], #['gini', 'entropy']
    max_features_to_include=[True, True],#['sqrt', 'log']
    estimators_range_to_include=range(1, 20, 5), 
    number_of_splits=10, 
    scoring_metric='roc_auc'
)

data_dict = rf_classifier.split_data()

X_train = data_dict["training_predictors"]
        
y_train = data_dict["training_outcome"]

X_test = data_dict["test_predictors"]
        
y_test = data_dict["test_outcome"]

print(preds)

print(metrics.classification_report(preds, y_test))

print(metrics.confusion_matrix(preds, y_test))
#preds = model.predict(X_test)
#
#conf_matrix = metrics.confusion_matrix(preds, y_test) 
#
#print(conf_matrix)
#
#print(metrics.classification_report(preds, y_test))


'''
Scikit-Learn confusion matrix reads as:
TP  FN
FP  TN    
'''
