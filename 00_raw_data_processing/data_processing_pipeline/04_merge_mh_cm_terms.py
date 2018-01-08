# -*- coding: utf-8 -*-
import pandas as pd
import set_lib_paths
from functools import reduce

import os


def main(data_dir):

    df = pd.DataFrame.from_csv(os.path.join('%s/merged_data/MERGED_COUNT_DATE.csv' % (data_dir)))
    #Get MH columns


    prevalences = (df.filter(regex="^MHTERM").filter(regex='COUNT$') > 0).sum().sort_values(ascending=False).reset_index()
    prevalences = prevalences.reset_index().rename(columns={0: 'Count', 'index': 'Term'})
    prevalences['Term'] = prevalences['Term'].str.replace('MHTERM_', '').str.replace('_COUNT', '')
    prevalences['Prevalence'] = (prevalences['Count']*100)/len(df)

    spreadsheet = pd.ExcelWriter(os.path.join(data_dir, 'merged_data', 'prevalence_change_MH_CM_MERGE.xlsx'))
    prevalences.to_excel(spreadsheet, 'BEFORE MERGE')



    df_MH = df.filter(regex='^MHTERM|USUBJID').set_index(['USUBJID'])
    df_CM = df.filter(regex='^CMINDC|USUBJID').rename(columns=lambda x: x.replace('CMINDC_', 'MHTERM_')).set_index(['USUBJID'])

    df_MH_counts = df_MH.filter(regex='COUNT$')
    df_CM_counts = df_CM.filter(regex='COUNT$')

    df_MH_counts[df_MH_counts.isnull()] = 0
    df_CM_counts[df_CM_counts.isnull()] = 0
    cm_keys_add = set(df_MH_counts.keys()) - set(df_CM_counts.keys())
    print (cm_keys_add)
    for k in cm_keys_add: df_CM_counts[k] = 0
    df_combined_counts = (df_MH_counts + df_CM_counts)

    #Convert counts to flags
    df_combined_counts[df_combined_counts > 0] = 1
    df_combined_counts.columns = df_combined_counts.columns.map(lambda x: x.replace('_COUNT', '_FLAG'))

    df_combined_counts = df_combined_counts.reset_index()

    #For now remove all medical history related variables. Include only flags for now instead of actual counts
    df = df.filter(regex='(?=^((?!MHTERM).)*$)').filter(regex='(?=^((?!CMINDC).)*$)')
    df = pd.DataFrame.merge(df, df_combined_counts, how='outer')

    prevalences = (df.filter(regex="^MHTERM").filter(regex='COUNT$') > 0).sum().sort_values(ascending=False).reset_index()
    prevalences = prevalences.reset_index().rename(columns={0: 'Count', 'index': 'Term'})
    prevalences['Term'] = prevalences['Term'].str.replace('MHTERM_', '').str.replace('_COUNT', '')
    prevalences['Prevalence'] = (prevalences['Count']*100)/len(df)
    prevalences.to_excel(spreadsheet, 'AFTER MERGE')
    spreadsheet.save()
    spreadsheet.close()


    #Write out flatfile
    pd.DataFrame.to_csv(df, os.path.join('%s/merged_data/MERGED_CM_MH_FLATFILE.csv' % (data_dir)))

if __name__ == '__main__':
    output_base_dir = 'F:/Projects/Ferring/data/pre_modelling/'

    main(output_base_dir)
