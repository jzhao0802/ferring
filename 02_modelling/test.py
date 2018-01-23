# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:45:04 2018

@author: TReason
"""

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

df_covars = df[['AGE', 'BMI', 'BS_BASELINE', 
          'RACE_dummy_ASIAN','RACE_dummy_BLACK_OR_AFRICAN_AMERICAN', 
          'RACE_dummy_HISPANIC', 'RACE_dummy_OTHER', 'RACE_dummy_WHITE',
          'GESTATIONAL_AGE_DAYS','MHTERM_GESTATIONAL_DIABETES_FLAG',
          'MHTERM_GESTATIONAL_DIABETES_PREV_PREG_FLAG']].dropna()

clf = ensemble.RandomForestClassifier()

pipeline_config = {"n_estimators": [1, 2, 3, 4, 5, 10, 20, 50],
                   "criterion": ['gini', 'entropy'],
                   "max_features": ['auto', 'sqrt', 'log2']}

kf = StratifiedKFold(n_splits=5, shuffle=True) 
gs = GridSearchCV(          
    estimator = clf,
    param_grid = pipeline_config,
    scoring = 'f1',
    cv = kf
)

X_train, X_test, y_train, y_test = train_test_split(df_covars, df['LABEL'], 
                                                            test_size = 0.2)
gs.fit(X_train, y_train)

gs.best_estimator_.fit(X_train, y_train)

preds = gs.predict(X_test)

conf_matrix = metrics.confusion_matrix(preds, y_test)

print(metrics.classification_report(preds, y_test))
