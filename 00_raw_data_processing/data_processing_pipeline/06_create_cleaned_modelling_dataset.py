import os
import pandas as pd
import set_lib_paths
import modelling


def main():
    dataset_cleaner = modelling.ModellingDatasetCleaner()

    #Load data
    data_dir = 'F:/Projects/Ferring/data/pre_modelling/'
    results_dir = 'F:/Projects/Ferring/results/modelling/03_initial_XGB/'
    df = pd.DataFrame.from_csv(os.path.join('%s/merged_data/PROCESSED_FLATFILE.csv' % (data_dir)))


    df_cleaned, reference_dummies, removed_vars = dataset_cleaner.clean_data_for_modelling(df, gs_age_weeks=True)

    df_cleaned.to_csv(os.path.join(data_dir, 'merged_data', 'PROESSSED_MODELLING_FLATFILE.csv'), index=False)


if __name__ == '__main__':

    main()