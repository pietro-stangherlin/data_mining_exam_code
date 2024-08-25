# data wrangling
library(dplyr)

source("lift_roc.R")

# parallel computing
library(snowfall)

# number of cores
N_CORES = parallel::detectCores()


load("result_preprocessing.Rdata")

#////////////////////////////////////////////////////////////////////////////

# Metrics and data.frame --------------------------------------------------
#////////////////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Quantitative response ---------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

source("loss_functions.R")

# °°°°°°°°°°°°°°°°°°°°°°° Warning: °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
USED.Metrics = function(y.pred, y.test, weights){
  return(tabella.sommario(y.pred, y.test, weights = weights))
}


# anche qua
df_metrics = data.frame(name = NA,
                        misclassification = NA,
                        fp = NA,
                        fn = NA,
                        f_score = NA)

METRICS_NAMES = colnames(df_metrics[,-1])

N_METRICS = length(METRICS_NAMES)

# names used to extract the metric added to df_metrics
# change based on the specific problem
METRIC_VALUES_NAME = "metric_values"
METRIC_CHOSEN_NAME = "f_score"

# names used for accessing list CV matrix (actual metrics and metrics se)
LIST_METRICS_ACCESS_NAME = "metrics"
LIST_SD_ACCESS_NAME = "se"


# metrics names + USED.Loss
# WARNING: the order should be same as in df_metrics
MY_USED_METRICS = c("USED.Metrics", "tabella.sommario")


MY_THRESHOLD = 0.3
# /////////////////////////////////////////////////////////////////
#------------------------ Sottocampionamento ----------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter tuning: cross validation on train: building cv folds  -------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

K_FOLDS = 10

NROW_dati = NROW(dati)

SHUFFLED_ID = sample(1:NROW_dati, NROW_dati)

# NOTE: if the row number of sss is not a multiple of K_FOLDS
# the last fold repeats some ids from the first
# this is fixed in the code below
id_matrix_cv = matrix(SHUFFLED_ID, ncol = K_FOLDS)


# conversion of matrix in list of elements: each element contains a subset of ids
ID_CV_LIST_UNBALANCED = list()

for(j in 1:ncol(id_matrix_cv)){
  ID_CV_LIST_UNBALANCED[[j]] = id_matrix_cv[,j]
}

rm(id_matrix_cv)
gc()


# repeated ids fixing
integer_division_cv = NROW_dati %/% K_FOLDS
modulo_cv = NROW_dati %% K_FOLDS

if(modulo_cv != 0){
  ID_CV_LIST_UNBALANCED[[K_FOLDS]] = ID_CV_LIST_UNBALANCED[[K_FOLDS]][1:integer_division_cv]
}


# Balanced ID CV creation --------------

CV_PROP = 0.5


ID_CV_LIST_BALANCED = list()

for(i in 1:length(ID_CV_LIST_UNBALANCED)){
  # cambiare valori all'occorrenza
  ids_few =ID_CV_LIST_UNBALANCED[[i]][which(dati$y[ID_CV_LIST_UNBALANCED[[i]]] == 1)]
  ids_lot =ID_CV_LIST_UNBALANCED[[i]][which(dati$y[ID_CV_LIST_UNBALANCED[[i]]] == 0)]
  
  tot = round(length(ids_few)/CV_PROP)
  
  ID_CV_LIST_BALANCED[[i]] = c(ids_few,
                               sample(ids_lot, size = tot - length(ids_few), replace = FALSE))
}





source("cv_functions.R")

BALANCED_ID_vector = unlist(ID_CV_LIST_BALANCED)
UNBALANCED_ID_vector = unlist(ID_CV_LIST_UNBALANCED)

MY_WEIGHTS = rep(1, NROW_dati)

USE_ONLY_FIRST_FOLD = FALSE

# //////////////////////////# ////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Classificazione Casuale --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# modello classificazione casuale

df_metrics = Add_Test_Metric(df_metrics,
                             "sss threshold",
                             USED.Metrics(y.pred = rbinom(dati$y[UNBALANCED_ID_vector], 1, MY_THRESHOLD),
                                          y.test = dati$y[UNBALANCED_ID_vector],
                                          weights = MY_WEIGHTS))

df_metrics = na.omit(df_metrics)

df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step linear model --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# se y non è già numerica
# sss$y = as.numeric(sss$y)
# vvv$y = as.numeric(vvv$y)

# AIC criterion is used for model selection

lm0 = lm(y ~ 1, data = dati[BALANCED_ID_vector,])

# NO Interaction -----------
lm_step_no_interaction = step(lm0, scope = formula_no_interaction_yes_intercept,
                              direction = "forward")

lm_step_no_int_cv_metrics = ManualCvLmGlmGam(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                             my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                             my_metric_names = METRICS_NAMES,
                                             my_data = dati,
                                             my_formula = formula(lm_step_no_interaction),
                                             my_model_type = "lm",
                                             my_weights = MY_WEIGHTS,
                                             use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                             is_classification = TRUE,
                                             my_threshold = MY_THRESHOLD)

df_metrics = Add_Test_Metric(df_metrics,
                             "lm_step_no_interaction",
                             lm_step_no_int_cv_metrics)
df_metrics


# save the model as .Rdata
# then remove it from main memory

file_name_lm_step_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                         "lm_step_no_interaction",
                                         ".Rdata", collapse = "", sep = "")

save(lm_step_no_interaction, file = file_name_lm_step_no_interaction)

rm(lm_step_no_interaction)
gc()

# YES Interaction -----------

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: slow°°°°°°°°°°°°°°°°°°°°°°°°°°°°
lm_step_yes_interaction = step(lm0, scope = formula_yes_interaction_yes_intercept,
                               direction = "forward")


lm_step_yes_int_cv_metrics = ManualCvLmGlmGam(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                             my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                             my_metric_names = METRICS_NAMES,
                                             my_data = dati,
                                             my_formula = formula(lm_step_yes_interaction),
                                             my_model_type = "lm",
                                             my_weights = MY_WEIGHTS,
                                             use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                             is_classification = TRUE,
                                             my_threshold = MY_THRESHOLD)


df_metrics = Add_Test_Metric(df_metrics,
                             "lm_step_yes_interaction",
                             lm_step_yes_int_cv_metrics)
df_metrics

# save the model as .Rdata
# then remove it from main memory
file_name_lm_step_yes_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                          "lm_step_yes_interaction",
                                          ".Rdata", collapse = "", sep = "")

save(lm_step_yes_interaction, file = file_name_lm_step_yes_interaction)

rm(lm_step_yes_interaction)
rm(lm0)
gc()


# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step GLM --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# AIC criterion is used for model selection

glm0 = glm(y ~ 1, data = dati[BALANCED_ID_vector,], family = "binomial")

# NO Interaction -----------
glm_step_no_interaction = step(glm0, scope = formula_no_interaction_yes_intercept,
                               direction = "forward")

glm_step_no_int_cv_metrics = ManualCvLmGlmGam(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                               my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                               my_metric_names = METRICS_NAMES,
                                               my_data = dati,
                                               my_formula = formula(glm_step_no_interaction),
                                               my_model_type = "glm",
                                               my_weights = MY_WEIGHTS,
                                               use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                               is_classification = TRUE,
                                               my_threshold = MY_THRESHOLD)

df_metrics = Add_Test_Metric(df_metrics,
                             "glm_step_no_interaction",
                             glm_step_no_int_cv_metrics)
df_metrics


# save the model as .Rdata
# then remove it from main memory

file_name_glm_step_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                          "glm_step_no_interaction",
                                          ".Rdata", collapse = "", sep = "")

save(glm_step_no_interaction, file = file_name_glm_step_no_interaction)

rm(glm_step_no_interaction)
gc()

# YES Interaction -----------

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: slow°°°°°°°°°°°°°°°°°°°°°°°°°°°°
glm_step_yes_interaction = step(glm0, scope = formula_yes_interaction_yes_intercept,
                                direction = "forward")


glm_step_yes_int_cv_metrics = ManualCvLmGlmGam(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                              my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                              my_metric_names = METRICS_NAMES,
                                              my_data = dati,
                                              my_formula = formula(glm_step_yes_interaction),
                                              my_model_type = "glm",
                                              my_weights = MY_WEIGHTS,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                              is_classification = TRUE,
                                              my_threshold = MY_THRESHOLD)

df_metrics = Add_Test_Metric(df_metrics,
                             "glm_step_yes_interaction",
                             glm_step_yes_int_cv_metrics)
df_metrics


# save the model as .Rdata
# then remove it from main memory
file_name_glm_step_yes_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                           "glm_step_yes_interaction",
                                           ".Rdata", collapse = "", sep = "")

save(glm_step_yes_interaction, file = file_name_glm_step_yes_interaction)

rm(glm_step_yes_interaction)
rm(glm0)
gc()


# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge & Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# sparse is preferred if there are many categorical predictors (sparse matrix)
library(Matrix)
X_mm_no_interaction =  sparse.model.matrix(formula_no_interaction_no_intercept, data = dati)

# computational heavy
X_mm_yes_interaction =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = dati)

# default
# X_mm_no_interaction = model.matrix(formula_no_interaction_no_intercept, data = dati)

# X_mm_yes_interaction_dati = model.matrix(formula_yes_interaction_no_intercept, data = dati)

library(glmnet)


# Ridge ------

# NO Interaction -----------

lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

ridge_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                              my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction,
                                              my_y = dati$y,
                                              my_alpha = 0,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                              is_classification = TRUE,
                                              my_threshold = MY_THRESHOLD)

# ridge_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
#                                                       my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
#                                                       my_metric_names = METRICS_NAMES,
#                                                       my_x = X_mm_no_interaction,
#                                                       my_y = dati$y,
#                                                       my_alpha = 0,
#                                                       my_lambda_vals = lambda_vals,
#                                                       my_weights = MY_WEIGHTS,
#                                                       my_metrics_functions = MY_USED_METRICS,
#                                                       my_ncores = N_CORES,
#                                                       use_only_first_fold = USE_ONLY_FIRST_FOLD,
#                                                       is_classification = TRUE,
#                                                       my_threshold = MY_THRESHOLD)

ridge_no_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = ridge_no_interaction_metrics[["metrics"]],
                                         my_one_se_best = TRUE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = ridge_no_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES,
                                         indexes_metric_max = 4) # f_score: higher better


PlotAndSave(function()(
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = ridge_no_interaction_metrics[["metrics"]],
                my_se_matrix = ridge_no_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(ridge_no_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge no interaction CV metrics",
                my_xlab = " log lambda")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "ridge_no_int_metrics_plot.jpeg",
                       collapse = ""))

print("ridge_no_int_best_summary")
ridge_no_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "ridge_no_interaction",
                             ridge_no_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

ridge_no_interaction = glmnet(x = X_mm_no_interaction[BALANCED_ID_vector,],
                              y = dati$y[BALANCED_ID_vector],
                              alpha = 0,
                              lambda = ridge_no_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])

file_name_ridge_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                       "ridge_no_interaction",
                                       ".Rdata", collapse = "", sep = "")

save(ridge_no_interaction, file = file_name_ridge_no_interaction)

rm(ridge_no_interaction)
gc()

# YES Interaction -----------
lambda_vals = glmnet(x = X_mm_yes_interaction, y = dati$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda


ridge_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                              my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_yes_interaction,
                                              my_y = dati$y,
                                              my_alpha = 0,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                              is_classification = TRUE,
                                              my_threshold = MY_THRESHOLD)

# ridge_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
#                                                        my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
#                                                       my_metric_names = METRICS_NAMES,
#                                                       my_x = X_mm_yes_interaction,
#                                                       my_y = dati$y,
#                                                       my_alpha = 0,
#                                                       my_lambda_vals = lambda_vals,
#                                                       my_weights = MY_WEIGHTS,
#                                                       my_metrics_functions = MY_USED_METRICS,
#                                                       my_ncores = N_CORES,
#                                                       is_classification = TRUE,
#                                                       use_only_first_fold = USE_ONLY_FIRST_FOLD,
#                                                       my_threshold = MY_THRESHOLD)

ridge_yes_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                                         my_one_se_best = TRUE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = ridge_yes_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES,
                                         indexes_metric_max = 4) # f_score: higher better


PlotAndSave(function()(
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                my_se_matrix = ridge_yes_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(ridge_yes_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge yes interaction CV metrics",
                my_xlab = " log lambda")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "ridge_yes_int_metrics_plot.jpeg",
                       collapse = ""))

print("ridge_yes_int_best_summary")
ridge_yes_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "ridge_yes_interaction",
                             ridge_yes_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

ridge_yes_interaction = glmnet(x = X_mm_yes_interaction[BALANCED_ID_vector,],
                              y = dati$y[BALANCED_ID_vector],
                              alpha = 0,
                              lambda = ridge_yes_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])

file_name_ridge_yes_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                       "ridge_yes_interaction",
                                       ".Rdata", collapse = "", sep = "")

save(ridge_yes_interaction, file = file_name_ridge_yes_interaction)

rm(ridge_yes_interaction)
gc()

save(df_metrics, file = "df_metrics.Rdata")

# Lasso ------
# NO Interaction -----------

lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

lasso_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                              my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction,
                                              my_y = dati$y,
                                              my_alpha = 1,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                              is_classification = TRUE,
                                              my_threshold = MY_THRESHOLD)

# lasso_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
#                                                       my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
#                                                       my_metric_names = METRICS_NAMES,
#                                                       my_x = X_mm_no_interaction,
#                                                       my_y = dati$y,
#                                                       my_alpha = 1,
#                                                       my_lambda_vals = lambda_vals,
#                                                       my_weights = MY_WEIGHTS,
#                                                       my_metrics_functions = MY_USED_METRICS,
#                                                       my_ncores = N_CORES,
#                                                       use_only_first_fold = USE_ONLY_FIRST_FOLD,
#                                                       is_classification = TRUE,
#                                                       my_threshold = MY_THRESHOLD)

lasso_no_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                                         my_one_se_best = TRUE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = lasso_no_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES,
                                         indexes_metric_max = 4) # f_score: higher better


PlotAndSave(function()(
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                my_se_matrix = lasso_no_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(lasso_no_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso no interaction CV metrics",
                my_xlab = " log lambda")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "lasso_no_int_metrics_plot.jpeg",
                       collapse = ""))

print("lasso_no_int_best_summary")
lasso_no_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_no_interaction",
                             lasso_no_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

lasso_no_interaction = glmnet(x = X_mm_no_interaction[BALANCED_ID_vector,],
                              y = dati$y[BALANCED_ID_vector],
                              alpha = 1,
                              lambda = lasso_no_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])

file_name_lasso_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                       "lasso_no_interaction",
                                       ".Rdata", collapse = "", sep = "")

save(lasso_no_interaction, file = file_name_lasso_no_interaction)

rm(lasso_no_interaction)
gc()

# YES Interaction -----------
lambda_vals = glmnet(x = X_mm_yes_interaction, y = dati$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda


lasso_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                               my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                               my_metric_names = METRICS_NAMES,
                                               my_x = X_mm_yes_interaction,
                                               my_y = dati$y,
                                               my_alpha = 1,
                                               my_lambda_vals = lambda_vals,
                                               my_weights = MY_WEIGHTS,
                                               use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                               is_classification = TRUE,
                                               my_threshold = MY_THRESHOLD)

# lasso_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
#                                                        my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
#                                                        my_metric_names = METRICS_NAMES,
#                                                        my_x = X_mm_yes_interaction,
#                                                        my_y = dati$y,
#                                                        my_alpha = 1,
#                                                        my_lambda_vals = lambda_vals,
#                                                        my_weights = MY_WEIGHTS,
#                                                        my_metrics_functions = MY_USED_METRICS,
#                                                        my_ncores = N_CORES,
#                                                        is_classification = TRUE,
#                                                        use_only_first_fold = USE_ONLY_FIRST_FOLD,
#                                                        my_threshold = MY_THRESHOLD)

lasso_yes_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                          my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                                          my_one_se_best = TRUE,
                                          my_higher_more_complex = FALSE,
                                          my_se_matrix = lasso_yes_interaction_metrics[["se"]],
                                          my_metric_names = METRICS_NAMES,
                                          indexes_metric_max = 4) # f_score: higher better


PlotAndSave(function()(
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                my_se_matrix = lasso_yes_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(lasso_yes_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso yes interaction CV metrics",
                my_xlab = " log lambda")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "lasso_yes_int_metrics_plot.jpeg",
                       collapse = ""))

print("lasso_yes_int_best_summary")
lasso_yes_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_yes_interaction",
                             lasso_yes_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

lasso_yes_interaction = glmnet(x = X_mm_yes_interaction[BALANCED_ID_vector,],
                               y = dati$y[BALANCED_ID_vector],
                               alpha = 1,
                               lambda = lasso_yes_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])

file_name_lasso_yes_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                        "lasso_yes_interaction",
                                        ".Rdata", collapse = "", sep = "")

save(lasso_yes_interaction, file = file_name_lasso_yes_interaction)

rm(lasso_yes_interaction)
gc()

save(df_metrics, file = "df_metrics.Rdata")



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Tree -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

# Model selection

# 0) Full tree to be pruned ------

# default: overfit
tree_full = tree(factor(y) ~.,
                 data = dati,
                 control = tree.control(nobs = nrow(dati),
                                        mindev = 1e-03,
                                        minsize = 10))



# check overfitting
plot(tree_full)


# Selection of size parameter, we have two possible ways

# 1.a) Size: CV ----------
# Selection of size parameter

TREE_MAX_SIZE = 50


tree_cv_metrics = ManualCvTree(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                               my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                               my_metric_names = METRICS_NAMES,
                               my_data = dati,
                               my_max_size = TREE_MAX_SIZE,
                               my_weights = MY_WEIGHTS,
                               my_mindev = 1e-03,
                               my_minsize = 10,
                               is_classification = TRUE,
                               my_threshold = MY_THRESHOLD,
                               use_only_first_fold = USE_ONLY_FIRST_FOLD)

# if parallel shows problems use the non parallel version
# tree_cv_metrics = ManualCvTreeParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
#                                        my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
#                                        my_metric_names = METRICS_NAMES,
#                                        my_data = dati,
#                                        my_max_size = TREE_MAX_SIZE,
#                                        my_metrics_functions = MY_USED_METRICS,
#                                        my_ncores = N_CORES,
#                                        my_weights = MY_WEIGHTS,
#                                        my_mindev = 1e-03,
#                                        my_minsize = 10,
#                                        use_only_first_fold = USE_ONLY_FIRST_FOLD,
#                                        is_classification = TRUE,
#                                        my_threshold = MY_THRESHOLD)

tree_best_summary = CvMetricBest(my_param_values = 2:TREE_MAX_SIZE,
                                 my_metric_matrix = tree_cv_metrics[["metrics"]],
                                 my_one_se_best = TRUE,
                                 my_higher_more_complex = TRUE,
                                 my_se_matrix = tree_cv_metrics[["se"]],
                                 my_metric_names = METRICS_NAMES,
                                 indexes_metric_max = 4)


PlotAndSave(function()(
  PlotCvMetrics(my_param_values = 2:TREE_MAX_SIZE,
                my_metric_matrix = tree_cv_metrics[["metrics"]],
                my_se_matrix = tree_cv_metrics[["se"]],
                my_best_param_values = ExtractBestParams(tree_best_summary),
                my_metric_names = METRICS_NAMES,
                my_main = "Tree CV metrics",
                my_xlab = "size",
                my_legend_coords = "bottomright")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "tree_cv_metrics_plot.jpeg",
                       collapse = ""))


tree_best_size = tree_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]]

print("tree best size")
tree_best_size


# 2) tree : final model --------------------- 

final_tree_pruned = prune.tree(tree_full,
                               best = tree_best_size)



temp_plot_function = function(){
  plot(final_tree_pruned)
  text(final_tree_pruned, cex = 0.7)
}

PlotAndSave(my_plotting_function = temp_plot_function,
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "tree_pruned_plot.jpeg",
                                 collapse = ""))

df_metrics = Add_Test_Metric(df_metrics,
                             "tree_pruned best",
                             tree_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])


df_metrics



rm(temp_pred)

file_name_final_tree_pruned = paste(MODELS_FOLDER_RELATIVE_PATH,
                                    "final_tree_pruned",
                                    ".Rdata", collapse = "", sep = "")

save(final_tree_pruned, file = file_name_final_tree_pruned)


rm(final_tree_pruned)
rm(tree_full)
gc()


# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello Additivo ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(gam)

# step selection via GCV

# stepwise forward: AIC based on generalized df
gam0 = gam(y ~ 1, data = dati[BALANCED_ID_vector,], family = "binomial")

# gam recognizes factor predictors
my_gam_scope = gam.scope(dati[BALANCED_ID_vector,-y_index], arg = c("df=2", "df=3", "df=4", "df=5", "df=6"))

# try parallel (linux only)
# require(doMC)
# registerDoMC(cores= N_CORES)
# step.Gam(gam0, my_gam_scope, parallel=TRUE)

gam_step = step.Gam(gam0, scope = my_gam_scope)

gam_cv_metrics = ManualCvLmGlmGam(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                              my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                              my_metric_names = METRICS_NAMES,
                                              my_data = dati,
                                              my_formula = formula(gam_step),
                                              my_model_type = "gam",
                                              my_weights = MY_WEIGHTS,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                              is_classification = TRUE,
                                              my_threshold = MY_THRESHOLD)

df_metrics = Add_Test_Metric(df_metrics,
                             "gam_step",
                             gam_cv_metrics)

df_metrics

file_name_gam_step = paste(MODELS_FOLDER_RELATIVE_PATH,
                           "gam_step",
                           ".Rdata", collapse = "", sep = "")

save(gam_step, file = file_name_gam_step)



rm(gam_step)
rm(gam0)
gc()

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MARS ---------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# factor predictors indexes are needed
# since in the model.matrix quantitative predictors don't change colum names
# (opposite to factors -> indicator matrix for each except one value)
# we first get the quantitative predictor indexes
# and then we do a set difference

# matrix case
# num_index = which(colnames(X_mm_no_interaction_sss) %in% var_num_names)
# factor_index = setdiff(1:NCOL(X_mm_no_interaction_sss), num_index)

# data.frame case
num_index = which(colnames(dati[,-y_index]) %in% var_num_names)
factor_index = setdiff(1:NCOL(dati[,-y_index]), num_index)

library(polspline)

# step selection via GCV
# only interaction of two terms are admitted 
# (computational and time constraint)

# if problems: -> but usually give problems
# weights = MY_WEIGHTS_sss

mars_step = polymars(responses = dati$y[BALANCED_ID_vector],
                     predictors = dati[BALANCED_ID_vector,-y_index],
                     gcv = 1,
                     factors = factor_index,
                     maxsize = 10,
                     classify = TRUE)


print("mars min size gcv")
min_size_mars = mars_step$fitting$size[which.min(mars_step$fitting$GCV)]
min_size_mars

temp_plot_function = function(){
  plot(mars_step$fitting$size, mars_step$fitting$GCV,
       col = as.factor(mars_step$fitting$`0/1`),
       pch = 16,
       xlab = "basis number",
       ylab = "GCV",
       main = "MARS step GCV")
  legend(c("topright"),
         legend = c("growing", "pruning"),
         col = c("black","red"),
         pch = 16)
  
  
  abline(v = min_size_mars)
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                     "mars_gcv_plot.jpeg",
                                                     collapse = ""))

mars_design = design.polymars(mars_step, dati[,-y_index])

# pay attention to the column
mars_cv_metrics = ManualCvMARS(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                               my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                               my_metric_names = METRICS_NAMES,
                               my_y = dati$y,
                               my_design_matrix = mars_design,
                               my_weights = MY_WEIGHTS,
                               use_only_first_fold = USE_ONLY_FIRST_FOLD,
                               is_classification = TRUE,
                               my_threshold = MY_THRESHOLD)

df_metrics = Add_Test_Metric(df_metrics,
                             "MARS",
                             mars_cv_metrics)

df_metrics

rm(mars_design)

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

mars_names = colnames(data[,-y_index])

file_name_mars_step = paste(MODELS_FOLDER_RELATIVE_PATH,
                            "mars_step",
                            ".Rdata", collapse = "", sep = "")

save(mars_step,
     mars_names,
     file = file_name_mars_step)



rm(mars_step)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PPR ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# max number of ridge functions
PPR_MAX_RIDGE_FUNCTIONS = 4

# possible spline degrees of freedom
PPR_DF_SM = 2:6

# ppr_metrics = PPRRegulationCV(my_data = sss,
#                               my_id_list_cv_train = ID_CV_LIST,
#                               my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
#                               my_spline_df = PPR_DF_SM,
#                               my_metrics_names = METRICS_NAMES,
#                               my_weights = MY_WEIGHTS_sss,
#                               use_only_first_fold = TRUE,
#                               is_classification = TRUE,
#                               my_threshold = MY_THRESHOLD)




# 1.b) Regulation: CV -------

ppr_metrics = PPRRegulationCVParallel(my_data = dati,
                                      my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                      my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                      my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                      my_spline_df = PPR_DF_SM,
                                      my_metrics_names = METRICS_NAMES,
                                      my_weights = MY_WEIGHTS,
                                      my_metrics_functions = MY_USED_METRICS,
                                      my_ncores = N_CORES,
                                      use_only_first_fold = TRUE,
                                      is_classification = TRUE,
                                      my_threshold = MY_THRESHOLD)

ppr_best_summary =  


# 2) final model -------

ppr_best_params = PPRExtractBestParams(ppr_metrics)

ppr_n_ridges_best = ppr_best_params[[METRIC_CHOSEN_NAME]][[1]]
ppr_df_best = ppr_best_params[[METRIC_CHOSEN_NAME]][[2]]


print("ppr best params")
ppr_best_params

ppr_model = ppr(y ~ .,
                data = dati,
                nterms = ppr_best_params[[METRIC_CHOSEN_NAME]][["n_ridge_functions"]],
                sm.method = "spline",
                df = ppr_best_params[[METRIC_CHOSEN_NAME]][["spline_df"]]) 


df_metrics = Add_Test_Metric(df_metrics,
                             "PPR",
                             ppr_metrics[ppr_n_ridges_best,ppr_df_best,])

df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

file_name_ppr_model = paste(MODELS_FOLDER_RELATIVE_PATH,
                            "ppr_model",
                            ".Rdata", collapse = "", sep = "")

save(ppr_model, file = file_name_ppr_model)

rm(ppr_model)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(ranger)

# Nota: se manca il tempo eseguo prima la RandomForest del Bagging
# visto che quest'ultimo è un sotto caso particolare 
# della RandomForest (selezione di tutte le variabili per ogni split)


# massimo numero di esplicative presenti
RF_MAX_VARIABLES = NCOL(dati) - 2 # sottraggo 1 per la variabile risposta
# ridurlo per considerazioni computazionali

RF_ITER = 400

RF_TREE_NUMBER_SEQ = seq(10, 400, 10)

rf_cv_metrics = ManualCvRF(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                           my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                   my_metric_names = METRICS_NAMES,
                                   my_data = dati,
                                   my_n_variables = 2:RF_MAX_VARIABLES,
                                   my_n_bs_trees = RF_ITER,
                                   fix_trees_bool = TRUE,
                                   my_weights = MY_WEIGHTS,
                                   use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                   is_classification = TRUE,
                                   my_threshold = MY_THRESHOLD,
                                   is_multiclass = FALSE)

# rf_cv_metrics = ManualCvRFParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
#                                    my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
#                            my_metric_names = METRICS_NAMES,
#                            my_data = dati,
#                            my_n_variables = 2:RF_MAX_VARIABLES,
#                            my_n_bs_trees = RF_ITER,
#                            my_ncores = N_CORES,
#                            my_metrics_functions = MY_USED_METRICS,
#                            fix_trees_bool = TRUE,
#                            my_weights = MY_WEIGHTS,
#                            use_only_first_fold = USE_ONLY_FIRST_FOLD,
#                            is_classification = TRUE,
#                            my_threshold = MY_THRESHOLD,
#                            is_multiclass = FALSE)


rf_cv_metrics_best = CvMetricBest(my_param_values = 2:RF_MAX_VARIABLES,
                                    my_metric_matrix = rf_cv_metrics[["metrics"]],
                                    my_one_se_best = TRUE,
                                    my_higher_more_complex = TRUE,
                                    my_se_matrix = rf_cv_metrics[["se"]],
                                    my_metric_names = METRICS_NAMES,
                                    indexes_metric_max = 4) #f_score

PlotAndSave(function()(
  PlotCvMetrics(my_param_values = 2:RF_MAX_VARIABLES,
                my_metric_matrix = rf_cv_metrics[["metrics"]],
                my_se_matrix = rf_cv_metrics[["se"]],
                my_best_param_values = ExtractBestParams(rf_cv_metrics_best),
                my_metric_names = METRICS_NAMES,
                my_main = "RF CV metrics",
                my_xlab = "mtry",
                my_legend_coords = "bottomright")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "rf_mtry_cv_metrics_plot.jpeg",
                       collapse = ""))


best_mtry = rf_cv_metrics_best[[METRIC_CHOSEN_NAME]][["best_param_value"]]

print("tree best size")
tree_best_size

# check convergence

rf_n_tree_metrics = ManualCvRFParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                       my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                       my_metric_names = METRICS_NAMES,
                                       my_data = dati,
                                       my_n_variables = best_mtry,
                                       my_n_bs_trees = RF_TREE_NUMBER_SEQ,
                                       my_ncores = N_CORES,
                                       my_metrics_functions = MY_USED_METRICS,
                                       fix_trees_bool = FALSE,
                                       my_weights = MY_WEIGHTS,
                                       use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                       is_classification = TRUE,
                                       my_threshold = MY_THRESHOLD,
                                       is_multiclass = FALSE)

PlotAndSave(function()(
  PlotCvMetrics(my_param_values = RF_TREE_NUMBER_SEQ,
                my_metric_matrix = rf_n_tree_metrics[["metrics"]],
                my_se_matrix =rf_n_tree_metrics[["se"]],
                my_best_param_values = 0,
                my_metric_names = METRICS_NAMES,
                my_main = "RF CV metrics n tree",
                my_xlab = "ntree")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "rf_n_tree_cv_metrics_plot.jpeg",
                       collapse = ""))


# modello finale e previsioni
random_forest_model = ranger(factor(y) ~., dati,
                             mtry = best_mtry,
                             num.trees = 400,
                             oob.error = TRUE,
                             probability = TRUE,
                             importance = "permutation")


df_metrics = Add_Test_Metric(df_metrics,
                             "Random Forest",
                             rf_cv_metrics_best[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")


# Importanza delle variabili
vimp = importance(random_forest_model)

PlotAndSave(my_plotting_function =  function() dotchart(vimp[order(vimp)],
                                                        pch = 16,
                                                        main = "Random Forest Variable Importance Permutation",
                                                        xlab = "error increase"),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "random_forest_importance_plot.jpeg",
                                 collapse = ""))

# save metrics and model
file_name_random_forest = paste(MODELS_FOLDER_RELATIVE_PATH,
                                "random_forests",
                                ".Rdata", collapse = "", sep = "")

save(random_forest_model, file = file_name_random_forest)



rm(random_forest_model)
gc()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Bagging ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

bagging_n_tree_metrics = ManualCvRFParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                       my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                       my_metric_names = METRICS_NAMES,
                                       my_data = dati,
                                       my_n_variables = NCOL(dati) - 1,
                                       my_n_bs_trees = RF_TREE_NUMBER_SEQ,
                                       my_ncores = N_CORES,
                                       my_metrics_functions = MY_USED_METRICS,
                                       fix_trees_bool = FALSE,
                                       my_weights = MY_WEIGHTS,
                                       use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                       is_classification = TRUE,
                                       my_threshold = MY_THRESHOLD,
                                       is_multiclass = FALSE)

bagging_best_summary = CvMetricBest(my_param_values = RF_TREE_NUMBER_SEQ,
                                    my_metric_matrix = bagging_n_tree_metrics[["metrics"]],
                                    my_one_se_best = TRUE,
                                    my_higher_more_complex = TRUE,
                                    my_se_matrix = bagging_n_tree_metrics[["se"]],
                                    my_metric_names = METRICS_NAMES,
                                    indexes_metric_max = 4)

PlotAndSave(function()(
  PlotCvMetrics(my_param_values = RF_TREE_NUMBER_SEQ,
                my_metric_matrix = bagging_n_tree_metrics[["metrics"]],
                my_se_matrix = bagging_n_tree_metrics[["se"]],
                my_best_param_values = 0,
                my_metric_names = METRICS_NAMES,
                my_main = "Bagging metrics n tree",
                my_xlab = "ntree")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "bagging_n_tree_cv_metrics_plot.jpeg",
                       collapse = ""))


df_metrics = Add_Test_Metric(df_metrics,
                             "Bagging",
                             bagging_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])


df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

rm(bagging_model)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Boosting ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(ada)


# Procedura Sub-ottimale per scegliere il numero di split di ciascun albero
# Fisso il numero di iterazioni e scelgo il modello con l'errore minore sulla convalida
# Poi sul modello scelto controllo la convergenza rispetto al numero di iterazioni
# NOTA: fissiamo la complessità massima degli alberi bassa dato che
# l'algoritmo di boosting non funziona correttamente per alberi troppo complessi
# (pochi o nessun errore commesso)

ITER_BOOST_START = 200
ADA_TREE_DEPTHS = 1:3

ada_boost_metrics = ManualCvADAFixedIterations(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                               my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                               my_metric_names = METRICS_NAMES,
                                               my_data = dati,
                                               my_tree_depths = ADA_TREE_DEPTHS,
                                               my_n_iterations = ITER_BOOST_START,
                                               my_weights = MY_WEIGHTS,
                                               use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                               my_threshold = MY_THRESHOLD)

ada_boost_best_metrics =  CvMetricBest(my_param_values = ADA_TREE_DEPTHS,
                                       my_metric_matrix = ada_boost_metrics[["metrics"]],
                                       my_one_se_best = TRUE,
                                       my_higher_more_complex = TRUE,
                                       my_se_matrix = ada_boost_metrics[["se"]],
                                       my_metric_names = METRICS_NAMES,
                                       indexes_metric_max = 4) # f_score: higher better

PlotAndSave(function()(
  PlotCvMetrics(my_param_values = ADA_TREE_DEPTHS,
                my_metric_matrix = ada_boost_metrics[["metrics"]],
                my_se_matrix = ada_boost_metrics[["se"]],
                my_best_param_values =ExtractBestParams(ada_boost_best_metrics),
                my_metric_names = METRICS_NAMES,
                my_main = "ADA tree depth CV metrics",
                my_xlab = " tree depth")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "ada_depth_cv.jpeg",
                       collapse = ""))

# best depth
print("ada boost best tree depth")
ada_boost_best_metrics[[METRIC_CHOSEN_NAME]][["best_param_value"]]

ADA_ITER_VALUES = seq(40, 500, by = 30)

# Iteration Convergence
ada_boost_iter_metrics = ManualCvADAFixedDepth(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                               my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                               my_metric_names = METRICS_NAMES,
                                               my_data = dati,
                                               my_tree_depth = ada_boost_best_metrics[[METRIC_CHOSEN_NAME]][["best_param_value"]],
                                               my_n_iterations = ADA_ITER_VALUES,
                                               my_weights = MY_WEIGHTS,
                                               use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                               my_threshold = MY_THRESHOLD)

PlotAndSave(function()(
  PlotCvMetrics(my_param_values = ADA_ITER_VALUES,
                my_metric_matrix = ada_boost_iter_metrics[["metrics"]],
                my_se_matrix = ada_boost_iter_metrics[["se"]],
                my_best_param_values = 0,
                my_metric_names = METRICS_NAMES,
                my_main = "ADA iter CV metrics",
                my_xlab = " iter")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "ada_iter_cv.jpeg",
                       collapse = ""))


df_metrics= Add_Test_Metric(df_metrics,
                            "ada_boost",
                            bagging_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])


df_metrics
save(df_metrics, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Support Vector Machine ---------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(e1071)


# Stima Convalida
# Kernel Radiale

SVM_COST_VALUES = seq(2,30,by=0.5)

svm_cv_metrics = ManualCvSVMParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                     my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                     my_metric_names = METRICS_NAMES,
                                     my_data = dati,
                                     my_params_vector = SVM_COST_VALUES,
                                     my_kernel_name = "radial",
                                     my_ncores = N_CORES,
                                     my_metrics_functions = MY_USED_METRICS,
                                     my_weights = MY_WEIGHTS,
                                     use_only_first_fold = TRUE)

svm_best_metrics = CvMetricBest(my_param_values = SVM_COST_VALUES,
                                my_metric_matrix = svm_cv_metrics[["metrics"]],
                                my_one_se_best = TRUE,
                                my_higher_more_complex = FALSE,
                                my_se_matrix = svm_cv_metrics[["se"]],
                                my_metric_names = METRICS_NAMES,
                                indexes_metric_max = 4) # f_score: higher better


PlotAndSave(function()(
  PlotCvMetrics(my_param_values = SVM_COST_VALUES,
                my_metric_matrix = svm_cv_metrics[["metrics"]],
                my_se_matrix = svm_cv_metrics[["se"]],
                my_best_param_values =ExtractBestParams(svm_best_metrics),
                my_metric_names = METRICS_NAMES,
                my_main = "SVM CV metrics",
                my_xlab = " cost")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "svm_cost_cv.jpeg",
                       collapse = ""))

print("svm_best_metrics")
svm_best_metrics

df_metrics = Add_Test_Metric(df_metrics,
                             "SVM radial",
                             svm_best_metrics[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics


# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

# /////////////////////////////////////////////////////////////////
#------------------------ Sintesi Finale -------------------------
# /////////////////////////////////////////////////////////////////

rounded_df = cbind(df_metrics[,1],
                   apply(df_metrics[,2:NCOL(df_metrics)], 2, function(col) round(as.numeric(col), 2)))

rounded_df

# Ridge - Lasso ----------------

load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "ridge_no_interaction",
           ".Rdata", collapse = "", sep = ""))

temp_glmnet_object = predict(ridge_no_interaction, type = "coef") %>% as.matrix()
temp_coef = temp_glmnet_object[,1]

temp_main = "(abs) greatest ridge coefficients no interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -1) | (temp_coef > 0.8)) ] %>% sort()

PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_ridge_no_int_plot.jpeg",
                                 collapse = ""))



load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "ridge_yes_interaction",
           ".Rdata", collapse = "", sep = ""))

temp_glmnet_object = predict(ridge_yes_interaction, type = "coef") %>% as.matrix()
temp_coef = temp_glmnet_object[,1]

temp_main = "(abs) greatest ridge coefficients yes interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -0.8) | (temp_coef > 0.5)) ] %>% sort()

PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_ridge_yes_int_plot.jpeg",
                                 collapse = ""))

load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "lasso_no_interaction",
           ".Rdata", collapse = "", sep = ""))

temp_glmnet_object = predict(lasso_no_interaction, type = "coef") %>% as.matrix()
temp_coef = temp_glmnet_object[,1]

temp_main = "(abs) greatest lasso coefficients no interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -2) | (temp_coef > 0.8)) ] %>% sort()

PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_lasso_no_int_plot.jpeg",
                                 collapse = ""))


load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "lasso_yes_interaction",
           ".Rdata", collapse = "", sep = ""))

temp_glmnet_object = predict(lasso_yes_interaction, type = "coef") %>% as.matrix()
temp_coef = temp_glmnet_object[,1]

temp_main = "(abs) greatest lasso coefficients yes interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -0.8) | (temp_coef > 0.5)) ] %>% sort()

PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_lasso_yes_int_plot.jpeg",
                                 collapse = ""))



# Tree -----------------
load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "final_tree_pruned",
           ".Rdata", collapse = "", sep = ""))



tree_temp_plot_fun = function(){
  plot(final_tree_pruned)
  text(final_tree_pruned, cex = 0.7)
}

PlotAndSave(my_plotting_function = tree_temp_plot_fun ,
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "tree_pruned_plot.jpeg",
                                 collapse = ""))


