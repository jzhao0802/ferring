'''
This module contains code to compare sklearn logistic regression with high C (analog for no regularization), with
stats model logistic regression with no regularization
'''


import os
import pandas as pd
import set_lib_paths
import numpy as np
from sklearn import tree, neighbors, ensemble, metrics, svm, preprocessing, linear_model, tree
import modelling
import matplotlib.pyplot as plt
import numpy as np
import random
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
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

    #Get label col
    label = df_cleaned['LABEL']
    del df_cleaned['LABEL']


    #Run statsmodel linear regression with no regulariation (seems to be using newton solver)
    logit = sm.Logit(label, df_cleaned).fit()

    eval_metrics = [metrics.precision_recall_curve, metrics.roc_curve, metrics.roc_auc_score, metrics.accuracy_score,
                    metrics.confusion_matrix, metrics.classification_report]

    #Run CV to get performance estimate
    output_metrics = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    logit_cv = modelling.run_CV(df_cleaned, label, sm.Logit, cv, {}, statsmodel=True, flatten=True)
    output_metrics['statsmodel'] = modelling.calc_CV_metrics(**(logit_cv['predictions']['test']), metrics=eval_metrics)
    #Get metrics


    #Compare coefficients with different magnitudes of C
    C = [1e42, 1e70, 1e80, 1e90]
    df = None
    for c in C:
        clf = linear_model.LogisticRegression(C=c)
        clf.fit(df_cleaned, label)
        coefficients = clf.coef_
        print('\n')
        coef = dict([(list(df_cleaned)[i], np.exp(coefficients[0][i])) for i in range(len(list(df_cleaned.keys())))])
        if df is None:
            df = pd.DataFrame.from_dict(coef, orient='index').reset_index()
            df = df.rename(columns={0: str(c)})
        else:
            df_coef =  pd.DataFrame.from_dict(coef, orient='index').reset_index()
            df_coef = df_coef.rename(columns={0: str(c)})
            df = pd.merge(df, df_coef, on='index')
        if c == 1e90:
            cv_outputs = modelling.run_CV(df_cleaned, label, linear_model.LogisticRegression, cv, {'C':c}, flatten=True)
            output_metrics['sklearn_1e90'] = modelling.calc_CV_metrics(**cv_outputs['predictions']['test'], metrics=eval_metrics)


    df.insert(0, 'var', df['index'])
    del df['index']
    statsmodel_params = np.exp(logit.params).reset_index().rename(columns={'index': 'var', 0: 'statsmodel'})
    df = pd.merge(df, statsmodel_params, on='var')

    #print(df)

    #Save output to excel
    spreadsheet = pd.ExcelWriter('F:/Projects/Ferring/results/modelling/regularization_param_comparison.xlsx')
    df.to_excel(spreadsheet, 'coefficient_comparison', index=False)
   # spreadsheet.save()



    #Write out performance metrics
    #metrics_to_add = ['roc_auc_score', 'accuracy_score']
    #models = ['statsmodel', 'sklearn_1e90']
    #metric_df_dict = {'model': models}
    #metric_df_dict.update({name:[output_metrics[model][name] for model in models] for name in metrics_to_add})
    #df_metrics = pd.DataFrame.from_dict(metric_df_dict, orient='columns')
    #df_metrics.to_excel(spreadsheet, 'metric_comparison', index=False)
    ##spreadsheet.save()

    #curves = ['precision_recall_curve', 'roc_curve']

    #for curve in curves:
    #    for model in models:
    #        print ([len(output_metrics[model][curve][i]) for i in output_metrics[model][curve]])
    #        df =  pd.DataFrame.from_dict(output_metrics[model][curve], orient='columns')
    #        df.to_excel(spreadsheet, '%s_curve_%s'%(curve.split('_')[0], model), index=False)
    modelling.add_metrics_to_spreadsheet(spreadsheet, output_metrics)
    spreadsheet.save()
    spreadsheet.close()
