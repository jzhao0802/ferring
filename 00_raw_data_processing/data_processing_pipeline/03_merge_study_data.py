# -*- coding: utf-8 -*-


import pandas as pd
import set_lib_paths
from functools import reduce

import os

def main(data_dir, case_study_codes):

    dfs = [pd.DataFrame.from_csv(os.path.join('%s/Miso-Obs-%s/'%(data_dir, study_code),\
                                              'MERGED_COUNT_DATE.csv')) for study_code in case_study_codes]
    
    #merged_df = pd.concat(dfs)
    
    for i in range(len(dfs)):
        dfs[i]['STUDY_CODE'] = '0'
        dfs[i]['STUDY_CODE'] = dfs[i]['STUDY_CODE'].astype(str)
        dfs[i]['STUDY_CODE'] = case_study_codes[i]
        
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    #merged_df = pd.DataFrame.from_dict(list(map(dict,dfs)))
    pd.DataFrame.to_csv(merged_df, os.path.join('%s/merged_data/MERGED_COUNT_DATE.csv'%(data_dir)))
    
if __name__ == '__main__':
    output_base_dir = 'F:/Projects/Ferring/data/pre_modelling/'

    case_study_codes = ['303', '004']
    
    output_dir = '%s%s/'%(output_base_dir, case_study_codes)
    
    main(output_base_dir, case_study_codes)
