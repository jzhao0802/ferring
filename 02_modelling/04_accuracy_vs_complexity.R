library('palabmod')
library('tidyverse')
library('mlr')
library('openxlsx')

# Set the random seed
random_seed <- 123
set.seed(random_seed, "L'Ecuyer")

data_dir <- 'F:/Projects/Ferring/data/pre_modelling/merged_data/'

df <- readr::read_csv(paste0(data_dir, 'PROESSSED_MODELLING_FLATFILE.csv'))
#colnames(df)[grepl('RACE', colnames(df))] <- paste0(colnames(df)[grepl('RACE', colnames(df))], '_RACE')

race_suffixes <- sub('RACE_dummy', '', colnames(df)[grepl('RACE', colnames(df))])

task <- makeClassifTask(data = df, target = 'LABEL', positive = 1)
lrn <- makeLearner("classif.xgboost", predict.type = "prob", nrounds = 20)
#rdesc <- makeResampleDesc("Holdout", split = 0.7)
rdesc <- makeResampleDesc("CV", iters=5)

rin <- makeResampleInstance(rdesc, task)

ls_measures <- list(mlr::auc, mlr::acc)
ls_n_pred <- length(colnames(df)):1
info <- plotting_acc_vs_comp(lrn, task, rin, ls_n_pred = ls_n_pred, ls_measures = ls_measures, 
                     linear_x_axis = TRUE, models = TRUE, show.info = FALSE, 
                     list_suffixes_stems = race_suffixes)

results_dir <- 'F:/Projects/Ferring/results/modelling/04_accuracy_vs_complexity/'
#Save plot
#dev.copy(png,paste0(results_dir, 'accuracy_complexity.png'))
ggsave(paste0(results_dir, 'accuracy_complexity.jpg') )

#Save list of parameters for each model to excel spreadsheet
openxlsx::write.xlsx(lapply(info$stem_importances, function(df){
    df %>% 
    reshape2::melt() %>%
    add_rownames("feature") %>%
    dplyr::rename(variable_importance=value) %>%
    dplyr::arrange(desc(variable_importance))
  }), paste0(results_dir, 'features.xlsx'))


