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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
import modelling
import matplotlib.pyplot as plt
import numpy as np
import random

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

    print(list(df_cleaned.keys()))
    print (removed_vars)
    print(reference_dummies)
    #df = df[['LABEL', 'AGE', 'BMI', 'BS_BASELINE',
    #          'RACE_dummy_ASIAN','RACE_dummy_BLACK_OR_AFRICAN_AMERICAN',
    #          'RACE_dummy_HISPANIC', 'RACE_dummy_OTHER', 'RACE_dummy_WHITE',
    #          'GESTATIONAL_AGE_DAYS','MHTERM_GESTATIONAL_DIABETES_FLAG',
    #          'MHTERM_GESTATIONAL_DIABETES_PREV_PREG_FLAG']].dropna()
    #
    #df_covars = df[['AGE', 'BMI', 'BS_BASELINE',
    #          'RACE_dummy_ASIAN','RACE_dummy_BLACK_OR_AFRICAN_AMERICAN',
    #          'RACE_dummy_HISPANIC', 'RACE_dummy_OTHER', 'RACE_dummy_WHITE',
    #          'GESTATIONAL_AGE_DAYS','MHTERM_GESTATIONAL_DIABETES_FLAG',
    #          'MHTERM_GESTATIONAL_DIABETES_PREV_PREG_FLAG']].dropna()

    #Create test ensembl/grid search
    #clf = ensemble.RandomForestClassifier()
    clf = linear_model.LogisticRegression()
    #clf = XGBClassifier()
    #pipeline_config = {"n_estimators": [1, 2, 3, 4, 5, 10, 20, 50],
    #                   "criterion": ['gini', 'entropy'],
    #                   "max_features": ['auto', 'sqrt', 'log2']}

    pipeline_config = {
       'penalty': ['l2', 'l1'],
        'tol': [1e-4, 1e-3, 1e-2, 1e-1],
       # 'C': [1, 0.8, 0.6, 0.4, 0.2, 0.01, 0.001, 0.0001]
        'C': [1e42]
    }

    #pipeline_config = {

    #}
    #pipeline_config = {}
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    gs = GridSearchCV(
        estimator = clf,
        param_grid = pipeline_config,
        scoring = 'precision',
        cv = kf,
        n_jobs=10
    )

    label = df_cleaned['LABEL']
    del df_cleaned['LABEL']

    #print(train_test_split(df_cleaned, label, test_size = 0.2))

    #create hold out testing set
    X_train, X_test, y_train, y_test = train_test_split(df_cleaned, label, test_size=0.2, random_state=__SEED__)

    X_train.reset_index(drop=True)
    X_test.reset_index(drop=True)
    y_test.reset_index(drop=True)
    y_train.reset_index(drop=True)

    print (X_train.shape)

    #fit
    gs.fit(X_train, y_train)



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