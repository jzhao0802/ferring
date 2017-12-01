
import pandas as pd
import set_lib_paths
import raw_data_explorer
import table_parsers.medical_history_parser
#from pyarrow import feather
import os

def main(base_dir, output_dir):
    explorer = raw_data_explorer.RawDataExplorer()
    explorer.set_base_dir(base_dir)
    explorer.load_data()
    
    #Parse medical history
    medical_history_table = explorer.get_data(filename='mh.sas7bdat')
    mh_parser = table_parsers.medical_history_parser.MedicalHistoryParser()
    medical_history_table_processed = mh_parser.process_table(medical_history_table, pre_process=True)
    
    #pd.DataFrame.to_feather(medical_history_table_processed, os.path.join(output_dir, 'MH_COUNT_DATE.feather'))
    pd.DataFrame.to_csv(medical_history_table_processed, os.path.join(output_dir, 'MH_COUNT_DATE.csv'))
    
if __name__ == '__main__':
    base_dir = 'F:/Projects/Ferring/data/raw_data/DK0-Propess Predictability project/Miso-Obs-'
    output_base_dir = 'F:/Projects/Ferring/data/pre_modelling/Miso-Obs-'

    case_study_codes = ['303', '004']
    
    folder_prefix = 'sdtm_data_Miso_Obs_'

    case_study_index = 0
    current_case_study_code = case_study_codes[case_study_index]
    current_dir = '%s%s/%s%s'%(base_dir, current_case_study_code, folder_prefix, current_case_study_code)
    output_dir = '%s%s/'%(output_base_dir, current_case_study_code)
    #os.makedirs(output_dir)
    main(current_dir, output_dir)
