import os
import plotly
import pandas as pd
import set_lib_paths
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn import tree, neighbors, ensemble, metrics, svm, preprocessing, linear_model, tree
from sklearn import metrics
import modelling
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import random
import statsmodels.api as sm
import sys
import sys
from functools import partial, update_wrapper
import pickle
from xgboost import XGBClassifier
#Set random seed to be fixed to allow reproducability
__SEED__ = 1234
np.random.seed(__SEED__)
random.seed(__SEED__)


def main(use_statsmodel=False, grid_search=False, grid_search_metric='ROC_AUC'):
    dataset_cleaner = modelling.ModellingDatasetCleaner()

    #Load data
    data_dir = 'F:/Projects/Ferring/data/pre_modelling/'
    results_dir = 'F:/Projects/Ferring/results/modelling/03_initial_XGB/'
    if grid_search: results_dir+='grid_search/'
    df = pd.DataFrame.from_csv(os.path.join('%s/merged_data/PROCESSED_FLATFILE.csv' % (data_dir)))

    #delivery_dict = {'CESAREAN SECTION': 0, 'VAGINAL': 1}

    #Clean dataset for modelling
    #Original with gs_age in days and no merging of race cats
    #df_cleaned, reference_dummies, removed_vars = dataset_cleaner.clean_data_for_modelling(df, gs_age_weeks=True)
    #GS age in weeks
    df_cleaned, reference_dummies, removed_vars = dataset_cleaner.clean_data_for_modelling(df, gs_age_weeks=True)
    #GS_age in weeks, merged asian and other
    #]df_cleaned, reference_dummies, removed_vars = dataset_cleaner.clean_data_for_modelling(df, gs_age_weeks=True, merge_race_dummies=['OTHER', 'ASIAN'])
    #GS age in weeks, converted race to binary - WHITE or OTHER
    #df_cleaned, reference_dummies, removed_vars = dataset_cleaner.clean_data_for_modelling(df, gs_age_weeks=True, merge_race_dummies=['OTHER', 'ASIAN', 'BLACK_OR_AFRICAN_AMERICAN', 'HISPANIC'])
    df_cleaned = df_cleaned.reset_index(drop=True)

    print (len(df_cleaned.keys()))
    print(list(df_cleaned.keys()))
  #  sys.exit()
    print (removed_vars)
    print(reference_dummies)


    #Create test ensembl/grid search


    outer_kf = StratifiedKFold(n_splits=3, shuffle=True)

    label_col = 'LABEL'

    model_types = {
        'DT': tree.DecisionTreeClassifier,
        'RF': ensemble.RandomForestClassifier,
        'XGB': XGBClassifier
    }

    model_parameters = {
        'DT': {'max_depth': 5},
        'RF': {'n_estimators': 100, 'max_depth': 5},
        'XGB': {'n_estimators': 100, 'max_depth': 5}
    }

    model_features = {
           'ALL': list(df_cleaned.keys()),
            'OBRISK': ['^RACE', '^AGE', '^GESTATIONAL_AGE', 'BMI', '^BS_BASELINE$', 'PREGTYPE', 'MHTERM_DIABETES_FLAG']
            #'OBRISK': ['^RACE', '^AGE', '^GESTATIONAL_AGE', 'BMI', '^BS_BASELINE$', 'PREGTYPE']
    #        'OBRISK': ['^RACE', 'GESTATIONAL_DIABETES', 'AGE', 'BMI', 'BS']
    }


    model_sum = partial(modelling.model_sum, statsmodel=use_statsmodel, type='XGB', coeff_var='feature_importances_')
    update_wrapper(model_sum, modelling.model_sum)

    eval_metrics = [sklearn.metrics.precision_recall_curve, sklearn.metrics.roc_curve, sklearn.metrics.roc_auc_score, sklearn.metrics.accuracy_score,
                    sklearn.metrics.confusion_matrix, sklearn.metrics.classification_report, model_sum]

    y = df_cleaned.pop('LABEL')

    output_metrics = {}
    pipeline_configs = {

        'DT':         {'max_depth': [3,4,5,6,7,8,9,10]},
        'RF':        {"n_estimators": [20, 30, 40, 50, 70, 100],
                       'max_depth': [3,4,5,6,7,8,9,10],
                        'n_jobs': [10]
    #                   'learning_rate': [0.001, 0.01, 0.1, 1]
                    },
        'XGB':     {"n_estimators": [20, 30, 40, 50, 70, 100],
                       'max_depth': [3,4,5,6,7,8,9,10],
                    'n_jobs': [10]
    #                   'learning_rate': [0.001, 0.01, 0.1, 1]
                },

    }



    #clf_class = linear_model.LogisticRegression if not use_statsmodel else sm.Logit
    clf_class = XGBClassifier
    best_params = {}
    for model_type, clf_class  in model_types.items():
        for variable_set,feature_set in model_features.items():
            model_name = '%s_%s'%(model_type, variable_set)
            #print(feature_set)
            print(len(feature_set))
            #for origin in model_origins:
            modelling_data = df_cleaned.filter(regex='|'.join(feature_set))

            if grid_search:
                inner_kf = StratifiedKFold(n_splits=5, shuffle=True)

                gs = GridSearchCV(
                    estimator=clf_class(),
                    param_grid=pipeline_configs[model_type],
                    scoring='roc_auc',
                    cv=inner_kf,
                    n_jobs=10
                )
                cv_outputs = modelling.run_CV(modelling_data, y, clf_class, outer_kf, model_parameters[model_type], flatten=True, statsmodel=use_statsmodel, grid_search=gs)
                best_params[model_name] = cv_outputs['best_params']
            else:
                cv_outputs = modelling.run_CV(modelling_data, y, clf_class, outer_kf, model_parameters[model_type], flatten=True, statsmodel=use_statsmodel)

            output_metrics[model_name] = modelling.calc_CV_metrics(**cv_outputs['predictions']['test'],
                                                                   metrics=eval_metrics, models=cv_outputs['models'],
                                                                   feature_names=list(modelling_data.columns.values))


    #print(output_metrics['OBRISK']['confusion_matrix'])
    suffix = '_statsmodel' if use_statsmodel else ''
    for model, metrics in output_metrics.items():
        spreadsheet = pd.ExcelWriter('%s/%s_results_%s.xlsx'%(results_dir, model, suffix))
        modelling.add_metrics_to_spreadsheet(spreadsheet, output_metrics)
        spreadsheet.save()
        spreadsheet.close()

    with open('%s/cv_outputs%s.pickle'%(results_dir, suffix), 'wb') as fd:
        pickle.dump([cv_outputs, output_metrics], fd)

    #Plot ROC curve
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set()
    for model in output_metrics.keys():
        pd.DataFrame.from_dict(output_metrics[model]['roc_curve']).plot(x='false_pos_rate', y='true_pos_rate', legend=False, title='ROC Curve (AUC = %.2f%%)'%(output_metrics[model]['roc_auc_score']*100))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('%s/%s_ROC_curve.svg'%(results_dir, model), format='svg')

    print (best_params)
    plt.show()
if __name__ == '__main__':
    use_statsmodel = '--statsmodel' in sys.argv
    grid_search = '--gridSearch' in sys.argv
    main(use_statsmodel=use_statsmodel, grid_search=grid_search)
