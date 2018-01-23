
import time
import pandas as pd
import numpy as np
from sklearn import metrics
from classifier import Classifier
import plotly.plotly as py


class RandomForestClassifier(Classifier):
    """
    Subclass of classifier to run classification via random forest
    """



    def run_pipeline(
            self, 
            criteria_to_include, 
            max_features_to_include,
            estimators_range_to_include, 
            number_of_splits, 
            scoring_metric
            ):
        """
        Function to call train/test data split and run pipeline
        
        Takes in responses from user:
            
        criteria_to_include is a list of bools to indicate whether to include
        the gini and entropy split measurement criteria
        
        max_features_to_include is a list of bools to indicate whether to 
        include the log2 and sqrt options for number of features to consider
        for a split
        
        estimators_range_to_include is a range containing the number
        of trees in the forest to try
        
        """
        
        split_data = self.split_data()
        
        X_train = split_data["training_predictors"]
                
        y_train = split_data["training_outcome"]
        
        X_test = split_data["test_predictors"]
                
        y_test = split_data["test_outcome"]
        
        all_criteria = ['gini', 'entropy']
    
        all_max_features = ['log2', 'sqrt']
        
        pipeline_config = {            
            "criterion" : Classifier.bool_filter(
                    all_criteria, criteria_to_include
                    ),
            "n_estimators" : estimators_range_to_include,
            "max_features" : Classifier.bool_filter(
                    all_max_features, max_features_to_include
                    ),
        }
        
        pipeline_config = {k: v for k, v in pipeline_config.items() if v}
        
        if not pipeline_config:
            raise ValueError("Invalid pipeline")
                                
        n_splits = number_of_splits
                
        scoring_metric = scoring_metric
        
        method = "Random Forest"

        model = self.get_best_estimator(predictors = X_train, response = y_train, 
                                  pipeline_config = pipeline_config,
                               n_splits = n_splits, scoring_metric = scoring_metric, 
                               method = method
                              )
        
        preds = model.predict(X_test)
        
        conf_matrix_lr = metrics.confusion_matrix(preds, y_test) 
        
        print(model)
        
        print(conf_matrix_lr)
                        
        self.plot_feature_importance(features = X_train, estimator = model)

        return preds

    def normalise_confusion_matrix(self, cm):
        normalised_matrix = np.ndarray(shape=(2,2), dtype=float, order='F')
        
        normalised_matrix[0,0] = cm[0,0]/(cm[0,0] + cm[0,1])
        normalised_matrix[0,1] = cm[0,1]/(cm[0,0] + cm[0,1])
        
        normalised_matrix[1,0] = cm[1,0]/(cm[1,0] + cm[1,1])
        normalised_matrix[1,1] = cm[1,1]/(cm[1,0] + cm[1,1])
       
        return normalised_matrix
    
    def plotly_confusion_matrix(
        self,
        conf_matrix,        
        title,
        zero_label,
        one_label
        ):
        
        '''
        Function that takes in a 2x2 confusion matrix, associated labels 
        for reference and non-reference categories and title, produces
        confusion matrix plot and publishes to plotly
        '''
        
        normalised_matrix = self.normalise_confusion_matrix(conf_matrix)
    
        plotly_json = {
          "data": [
            {
              "y": [
                one_label, # y=0
                zero_label,# y=1
              ],
              "x": [
                zero_label, #x=0
                one_label, #x=1
              ],
              "z": [
                [
                  float("{0:.3f}".format(normalised_matrix[1][0])),# x=0, y=0
                  float("{0:.3f}".format(normalised_matrix[1][1])),# x=1, y=0
                ],
                [
                  float("{0:.3f}".format(normalised_matrix[0][0])),# x=0, y=1
                  float("{0:.3f}".format(normalised_matrix[0][1])),# x=1, y=1
                ],
              ],
              "type": "heatmap",
              "colorscale": "Portland"
            }
          ],
          "layout": {
            "autosize": False,
            "yaxis": {
              "autotick": False,
              "ticks": "inside"
            },
            "title": title,
            "annotations": [
              {
                "xref": "x1",
                "yref": "y1",
                "text": str(conf_matrix[0][0]),# true negatives
                "y": zero_label,
                "x": zero_label,
                "font": {
                  "color": "white"
                },
                "showarrow": False
              },
              {
                "xref": "x1",
                "yref": "y1",
                "text": str(conf_matrix[0][1]),# false positives
                "y": zero_label,
                "x": one_label,
                "font": {
                  "color": "white"
                },
                "showarrow": False
              },
                {
                "xref": "x1",
                "yref": "y1",
                "text": str(conf_matrix[1][0]),# false negatives
                "y": one_label,
                "x": zero_label,
                "font": {
                  "color": "white"
                },
                "showarrow": False
              },
               {
                "xref": "x1",
                "yref": "y1",
                "text": str(conf_matrix[1][1]),# true positives
                "y": one_label,
                "x": one_label,
                "font": {
                  "color": "white"
                },
                "showarrow": False
              }],
            "height": 800,
            "width": 800,
            "xaxis": {
              "autotick": False,
              "ticks": "inside"
                  }   
                }
            }
        
        py.plot(plotly_json, filename='confusion matrix')


        
