import os
import plotly
import pandas as pd
import set_lib_paths
import numpy as np
from sklearn import metrics
import plotly.plotly as py
import plotly.graph_objs as go
from xgboost import XGBClassifier
from sklearn import tree, neighbors, ensemble, metrics, svm, preprocessing, linear_model, tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
import modelling
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
#Set random seed to be fixed to allow reproducability
__SEED__ = 1234
np.random.seed(__SEED__)
random.seed(__SEED__)


if __name__ == '__main__':
    dataset_cleaner = modelling.ModellingDatasetCleaner()

    #Load data
    data_dir = 'F:/Projects/Ferring/data/pre_modelling/'
    df = pd.DataFrame.from_csv(os.path.join('%s/merged_data/PROCESSED_FLATFILE.csv' % (data_dir)))

    #delivery_dict = {'CESAREAN SECTION': 0, 'VAGINAL': 1}

    #Clean dataset for modelling
    df_cleaned, reference_dummies, removed_vars = dataset_cleaner.clean_data_for_modelling(df)
    df_cleaned = df_cleaned.reset_index(drop=True)

    print(list(df_cleaned.keys()))
    print (removed_vars)
    print(reference_dummies)


    #Create test ensembl/grid search
    clf = linear_model.LogisticRegression()


    kf = StratifiedKFold(n_splits=5, shuffle=True)

    label_col = 'LABEL'

    model_features = {
            'ALL': list(df_cleaned.keys()),
            'OBRISK': ['^RACE', 'GESTATIONAL_DIABETES', 'AGE', 'BMI', 'BS']
        }

    eval_metrics = [metrics.precision_recall_curve, metrics.roc_curve, metrics.roc_auc_score, metrics.accuracy_score,
                    metrics.confusion_matrix, metrics.classification_report, modelling.LR_model_sum]
    parameters = {'C': 1e90}
    grid_search_metric = 'ROC_AUC'

    y = df_cleaned.pop('LABEL')

    output_metrics = {}

    for model_name,feature_set in model_features.items():
        #for origin in model_origins:
        modelling_data = df_cleaned.filter(regex='|'.join(feature_set))


        cv_outputs = modelling.run_CV(df_cleaned, y, linear_model.LogisticRegression, kf, parameters, flatten=True)
        output_metrics[model_name] = modelling.calc_CV_metrics(**cv_outputs['predictions']['test'],
                                                               metrics=eval_metrics, models=cv_outputs['models'],
                                                               feature_names=list(modelling_data.columns.values))

#    coefficient_comparison = modelling.LR_model_sum()



                #gs = GridSearchCV(
                #    estimator=clf,
                #    param_grid=pipeline_config,
                #    scoring='precision',
                #    cv=kf,
                #    n_jobs=10
                #)





    spreadsheet = pd.ExcelWriter('F:/Projects/Ferring/results/modelling/OBRISK_comparison.xlsx')
    modelling.add_metrics_to_spreadsheet(spreadsheet, output_metrics)
    spreadsheet.save()
    spreadsheet.close()

        #print (train_indicies, test_indicies)
    #print(list(map(len, list(CV_splits)[0])))

    sys.exit()


    #fit
    gs.best_estimator_.fit(X_train, y_train)

    print (gs.best_params_)
    #gs
    preds = gs.predict(X_test)

    '''
    Scikit-Learn confusion matrix reads as:
    TP  FN
    FP  TN    
    '''
    conf_matrix = metrics.confusion_matrix(y_test, preds)
    print(conf_matrix)
    print(metrics.classification_report(y_test, preds))

    probs = gs.decision_function(X_test)
    #probs = gs.predict_proba(X_test)
    #probs_pos = [x[1] for x in probs]
    #print(probs)
    #print(preds)
    precision, recall, threshold = precision_recall_curve(y_test, probs)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    #coefficients = gs.best_estimator_.coef_
    #print (precision, recall, threshold)
    #[print(list(X_train)[i], np.exp(coefficients[0][i])) for i in range(len(list(X_train.keys())))]

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()