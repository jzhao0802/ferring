library('tidyverse')
library('palab')
library('stringr')

#Import dataframe

base_dir <- 'F:/Projects/Ferring/data/pre_modelling/merged_data/'
output_dir <- 'F:/Projects/Ferring/results/pre_modelling/'


#########################################################
#####Load Data
#########################################################
df <- read_csv(paste0(base_dir, 'PROCESSED_FLATFILE.csv'))

#########################################################
#####Remove variables not relevant to modelling
#########################################################
df <- df %>%
  select(-starts_with('X1'))  %>%
  select(-starts_with('APGAR')) %>%
  select(-one_of('BS_12', 'BS_18', 'BS_24', 'BS_6', 'COMPLETED', 'DD_DELIVERY_TIME', 
                 'DISCHARGE_TIME', 'EX_END_TIME', 'TIME_DELTA_EX_START_OXYTOCIN_ADMIN',
                 'TIME_DELTA_EX_START_ONSET_LABOUR', 'STUDY_CODE', 'SUBJID', 'MODE_OF_DELIVERY',
                 'OXYTOCIN_ADMINISTERED', 'OXYTOCIN_ADMINISTRATION_TIME', 'OXYTOCIN_DOSAGE',
                 'RACE', 'ACTIVE_LABOUR_TIME', 'COUNTRY', 'RACE_RAW', 'RACE_PROCESSED',
                 'EX_START_TIME'))

#########################################################
#####Create var config
#########################################################

config <- palab::var_config_generator(input_df=df)
config[config$Column == 'USUBJID',]$Type <- 'key'
config[grepl('RACE|PREGTYPE|MHTERM|LABEL|COUNTRY', config$Column), ]$Type <- 'categorical'
readr::write_csv(config, paste0(base_dir, 'processed_flatfile_var_config.csv'))


#########################################################
#####Run bivar stats
#########################################################
stats <- palab::bivar_stats_y_flag(df, paste0(base_dir, 'processed_flatfile_var_config.csv'), outcome_var='LABEL')
stats_cat <- palab::bivariate_stats_cat(df, paste0(base_dir, 'processed_flatfile_var_config.csv'), outcome_var='LABEL')
readr::write_csv(stats, paste0(output_dir, 'bivar_stats_flag_prelim.csv'))

write.xlsx(stats_cat, paste0(output_dir, 'bivar_stats_cat_prelim.xlsx'))
#readr::write_csv(stats, paste0(output_dir, 'bivar_stats_cat_cat_prelim.csv'))
