
import pandas as pd
import set_lib_paths
import raw_data_explorer
import table_parsers
#from pyarrow import feather
import os

def main(processed_data_dir, raw_data_dir, output_dir, tables_to_parse, study_code):
    processed_explorer = raw_data_explorer.RawDataExplorer()
    processed_explorer.set_base_dir(processed_data_dir)
    processed_explorer.load_data()
    
    raw_explorer = raw_data_explorer.RawDataExplorer()
    raw_explorer.set_base_dir(raw_data_dir)
    raw_explorer.load_data()
    
    #Parse medical history
    for table_name in tables_to_parse:
        processed_table = processed_explorer.get_data(filename='%s.sas7bdat'%table_name)
        raw_table = raw_explorer.get_data(filename='%s.sas7bdat'%table_name)
        parser = table_parsers.table_parser_map[table_name](study_code)
        parsed_table = parser.process_table(raw_table, processed_table, pre_process=True)
        #print (parsed_table)
        if parsed_table is None: continue
    #pd.DataFrame.to_feather(medical_history_table_processed, os.path.join(output_dir, 'MH_COUNT_DATE.feather'))
        pd.DataFrame.to_csv(parsed_table, os.path.join(output_dir, '%s_COUNT_DATE.csv'%(table_name.upper())))
    
if __name__ == '__main__':
    base_dir = 'F:/Projects/Ferring/data/raw_data/DK0-Propess Predictability project/Miso-Obs-'
    output_base_dir = 'F:/Projects/Ferring/data/pre_modelling/Miso-Obs-'

    #case_study_codes = ['303']
    case_study_codes = ['303', '004']
    
    processed_folder_prefix = 'sdtm_data_Miso_Obs_'
    raw_folder_prefix = 'raw_data_Miso_Obs_'
    tables_to_parse = ['mh', 'dd', 'ex', 'bs', 'dm', 'vs', 'oh', 'oa', 'ds']
    #tables_to_parse = ['oa']
    for case_study_index in range(len(case_study_codes)):
        current_case_study_code = case_study_codes[case_study_index]
        processed_data_dir = '%s%s/%s%s'%(base_dir, current_case_study_code, processed_folder_prefix, current_case_study_code)
        raw_data_dir = '%s%s/%s%s'%(base_dir, current_case_study_code, raw_folder_prefix, current_case_study_code)
        output_dir = '%s%s/'%(output_base_dir, current_case_study_code)
        #os.makedirs(output_dir)
        main(processed_data_dir, raw_data_dir, output_dir, tables_to_parse, current_case_study_code)
