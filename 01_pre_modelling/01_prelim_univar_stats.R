library('tidyverse')
library('palab')
library('stringr')

#Import dataframe

base_dir <- 'F:/Projects/Ferring/data/pre_modelling/merged_data/'
output_dir <- 'F:/Projects/Ferring/results/pre_modelling/'
  
df <- read_csv(paste0(base_dir, 'PROCESSED_FLATFILE.csv'))
df <- df %>%
  select(-starts_with('Unnamed')) %>%
  select(-one_of('X1')) %>%
  select(-starts_with('MHTERM'), -starts_with('RACE'), -starts_with('COUNTRY'), -starts_with('EX'), -starts_with('DD'), -starts_with('PREGTYPE'), -one_of('OXYTOCIN_ADMINISTERED', 'SUBJID'))


config <- palab::var_config_generator(input_df=df)
config[config$Column == 'USUBJID',]$Type <- 'key'
readr::write_csv(config, paste0(base_dir, 'var_config.csv'))

stats <- palab::univariate_stats(df, paste0(base_dir, 'var_config.csv'))
readr::write_csv(stats$numerical, paste0(output_dir, 'univar_stats_prelim.csv'))
