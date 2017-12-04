# -*- coding: utf-8 -*-


import pandas as pd
import set_lib_paths
from functools import reduce

import os

def main(data_dir, study_code):
    dfs = [pd.read_csv(os.path.join(data_dir, filename)) for filename in os.listdir(data_dir)]
    merged_df = reduce(lambda x,y: pd.merge(x, y, on='USUBJID'), dfs)
    pd.DataFrame.to_csv(merged_df, os.path.join(data_dir, 'MERGED_COUNT_DATE.csv'))
    
if __name__ == '__main__':
    output_base_dir = 'F:/Projects/Ferring/data/pre_modelling/Miso-Obs-'

    case_study_codes = ['303', '004']
    
    for current_case_study_code in case_study_codes:
    #case_study_index = 1
    
        #current_case_study_code = case_study_codes[case_study_index]
        output_dir = '%s%s/'%(output_base_dir, current_case_study_code)
    
        main(output_dir, current_case_study_code)
