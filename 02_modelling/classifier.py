# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:10:41 2017

@author: TReason
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

from sklearn import tree, neighbors, ensemble, metrics, svm, preprocessing, linear_model, tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


        

class Classifier():
    """
    classifier is a class designed to avoid repetition of code when calling
    individual classifiers, running pipelines, plotting confusion matrices etc
    all individual classifiers are subclassed off this classifier

    """
    
    def __init__(self):

                



    
    def pred_summary(self, tp_summary, tn_summary, fp_summary, fn_summary, conf_matrix):
        
        '''
        Slightly nicer textual summary of a confusion matrix
        '''
        
        true_positives = conf_matrix[1][1]
        false_positives = conf_matrix[0][1]
        true_negatives = conf_matrix[0][0]
        false_negatives = conf_matrix[1][0]
        
        total_positives = true_positives + false_positives
        total_negatives = true_negatives + false_negatives
        
        positive_predictive_value =  "%.2f" % round((true_positives/total_positives)*100,2)
        negative_predictive_value =  "%.2f" % round((true_negatives/total_negatives)*100,2)
        
        
        print(tp_summary + " " + str(true_positives) + "/" + str(total_positives) + 
        " = " + positive_predictive_value + "%")
        
        print(tn_summary + " " + str(true_negatives) + "/" + str(total_negatives) + 
        " = " + negative_predictive_value + "%")



    def get_best_estimator(
            self, predictors, response, pipeline_config, n_splits, 
            scoring_metric, method
            ):
        
        '''
        Function to run k fold cross validation using pipeline and return 'best' estimator
        
        Commented out code is an alternative to the series of if statements and will be 
        implemented in a later version to make code more concise
        
        All of the individual classifiers use this code to run pipelines
        
        response = response variable
        predictors = "features" of interest
        pipelineConfig = parameters to try when fitting model
        nSplits = number of splits to use in cross validation
        scoringMetric = metric used to assess model accuracy
        
        '''
        if method == "Support Vector Machine":
            est = Pipeline([
                ('scale', preprocessing.StandardScaler()),
                ('svc', svm.SVC())
                ])
    
        elif method == "Random Forest":
            est = ensemble.RandomForestClassifier()

        
        elif method == "K Nearest Neighbours":
            est = neighbors.KNeighborsClassifier()
            
        elif method == 'Stochastic Gradient Descent':
            est = linear_model.SGDClassifier()
        
        elif method == 'Gradient Tree Boosting':
            est = ensemble.GradientBoostingClassifier()
        
        elif method == 'Decision Tree':
            est = tree.DecisionTreeClassifier()
        '''
        methods = {
                   "Support Vector Machine": lambda: Pipeline([
                   ('scale', preprocessing.StandardScaler()),
                   ('svc', svm.SVC())
                    ]),
                    "Random Forest" : ensemble.RandomForestClassifier,
                    "Decision Tree" : tree.DecisionTreeClassifier,
                    }
        
        est = methods[method]()
        '''
        
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True) 
        gs = GridSearchCV(          
            estimator = est,
            param_grid = pipeline_config,
            scoring = scoring_metric,
            cv = kf
        )
        
        
        gs.fit(predictors, np.ravel(response))
        
   
        return gs.best_estimator_
    
    
    def plot_feature_importance(self, features, estimator):
        f = estimator.feature_importances_
        plt.figure(figsize=(12,6))
        plt.title("Feature importances")
        plt.bar(range(features.shape[1]), f[np.argsort(f)[::-1]],color="r", align="center")
        plt.xticks(range(features.shape[1]), features, rotation = 75)
        plt.tight_layout()
       # plt.savefig(featureImportanceFileName)


