# data wrangling
library(dplyr)

source("lift_roc.R")

# parallel computing
library(snowfall)

# number of cores
N_CORES = parallel::detectCores()


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



# lista previsioni (punteggi) (se stima - verifica)
# per LIFT e ROC
pred_list = list() 

METRICS_NAMES = colnames(df_metrics[,-1])

N_METRICS = length(METRICS_NAMES)

# names used to extract the metric added to df_metrics
# change based on the specific problem
METRIC_VALUES_NAME = "metric_values"
METRIC_CHOSEN_NAME = "f_score"

# names used for accessing list CV matrix (actual metrics and metrics se)
LIST_METRICS_ACCESS_NAME = "metrics"
LIST_SD_ACCESS_NAME = "se"

USE_ONLY_FIRST_FOLD = FALSE

# metrics names + USED.Loss
# WARNING: the order should be same as in df_metrics
MY_USED_METRICS = c("USED.Metrics", "tabella.sommario")

# /////////////////////////////////////////////////////////////////
#------------------------ Train & Test ------------------------
# /////////////////////////////////////////////////////////////////


# eventually change the proportion
id_stima = sample(1:NROW(dati), 0.75 * NROW(dati))

sss = dati[id_stima,]
vvv = dati[-id_stima,]


# lista dei predittori (se stima - verifica)
# per LIFT e ROC
pred_list = list() 

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter tuning: Train & Test on Train subset  --------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

id_cb1 = sample(1:NROW(sss), 0.8 * NROW(sss))

# delete original data.frame from main memory
rm(dati)
gc()


# ///////////////////////////////////
# Weights ---------------
# //////////////////////////////////

# weights used for each metric function
# default 1
MY_WEIGHTS_sss = rep(1, NROW(sss)) 
MY_WEIGHTS_vvv = rep(1, NROW(vvv))

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter tuning: cross validation on train: building cv folds  -------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

K_FOLDS = 10

NROW_sss = NROW(sss)

SHUFFLED_ID = sample(1:NROW_sss, NROW_sss)

# NOTE: if the row number of sss is not a multiple of K_FOLDS
# the last fold repeats some ids from the first
# this is fixed in the code below
id_matrix_cv = matrix(SHUFFLED_ID, ncol = K_FOLDS)


# conversion of matrix in list of elements: each element contains a subset of ids
ID_CV_LIST = list()

for(j in 1:ncol(id_matrix_cv)){
  ID_CV_LIST[[j]] = id_matrix_cv[,j]
}

rm(id_matrix_cv)
gc()


# repeated ids fixing
integer_division_cv = NROW_sss %/% K_FOLDS
modulo_cv = NROW_sss %% K_FOLDS

if(modulo_cv != 0){
  ID_CV_LIST[[K_FOLDS]] = ID_CV_LIST[[K_FOLDS]][1:integer_division_cv]
}

source("cv_functions.R")

# FALSE = traditional CV on all folds
# TRUE -> use only first fold to test and all other to fit
USE_ONLY_FIRST_FOLD = FALSE
# /////////////////////////////////////////////////////////////////
#------------------------ Analisi esplorative ---------------------
# /////////////////////////////////////////////////////////////////

# Analisi esplorativa sulla stima 
# eventuali inflazioni di zeri

# valutiamo se è sbilanciata 
# ed eventualmente se è ragionevole cambiare la solita soglia a 0.5
table(sss$y)

# soglia di classificazione: cambia eventualmente con
# table(sss$y)[2] / NROW(sss)

MY_THRESHOLD = 0.2

# /////////////////////////////////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Classificazione Casuale --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# modello classificazione casuale

df_metrics = Add_Test_Metric(df_metrics,
                              "sss threshold",
                             USED.Metrics(y.pred = rbinom(nrow(vvv), 1, MY_THRESHOLD),
                                        y.test = vvv$y,
                                        weights = MY_WEIGHTS_vvv))

df_metrics = na.omit(df_metrics)

df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step linear model --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# se y non è già numerica
# sss$y = as.numeric(sss$y)
# vvv$y = as.numeric(vvv$y)

# AIC criterion is used for model selection

lm0 = lm(y ~ 1, data = sss)

# NO Interaction -----------
lm_step_no_interaction = step(lm0, scope = formula_no_interaction_yes_intercept,
                              direction = "forward")

temp_pred = predict(lm_step_no_interaction, newdata = vvv)
pred_list$lm_no_interaction = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "lm_step_no_interaction",
                             USED.Metrics(temp_pred > MY_THRESHOLD %>% as.numeric,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))
df_metrics

rm(temp_pred)


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


temp_pred = predict(lm_step_yes_interaction, newdata = vvv)
pred_list$lm_yes_interaction = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "lm_step_yes_interaction",
                             USED.Metrics(temp_pred > MY_THRESHOLD %>% as.numeric,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))
df_metrics

rm(temp_pred)

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
save(df_metrics, pred_list, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step GLM --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# AIC criterion is used for model selection

glm0 = glm(y ~ 1, data = sss, family = "binomial")

# NO Interaction -----------
glm_step_no_interaction = step(glm0, scope = formula_no_interaction_yes_intercept,
                              direction = "forward")

temp_pred = predict(glm_step_no_interaction, newdata = vvv, type = "response")
pred_list$glm_no_interaction = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "glm_step_no_interaction",
                             USED.Metrics(temp_pred > MY_THRESHOLD %>% as.numeric,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))
df_metrics

rm(temp_pred)

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


temp_pred = predict(glm_step_yes_interaction, newdata = vvv, type = "response")
pred_list$glm_yes_interaction = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "glm_step_yes_interaction",
                             USED.Metrics(temp_pred > MY_THRESHOLD %>% as.numeric,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))
df_metrics

rm(temp_pred)

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
save(df_metrics, pred_list, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge & Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# sparse is preferred if there are many categorical predictors (sparse matrix)
library(Matrix)
X_mm_no_interaction_sss =  sparse.model.matrix(formula_no_interaction_no_intercept, data = sss)
X_mm_no_interaction_vvv =  sparse.model.matrix(formula_no_interaction_no_intercept, data = vvv)

# computational heavy
X_mm_yes_interaction_sss =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = sss)
X_mm_yes_interaction_vvv =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = vvv)

# default
# X_mm_no_interaction_sss = model.matrix(formula_no_interaction_no_intercept, data = sss)
# X_mm_no_interaction_vvv = model.matrix(formula_no_interaction_no_intercept, data = vvv)

# X_mm_yes_interaction_sss = model.matrix(formula_yes_interaction_no_intercept, data = sss)
# X_mm_yes_interaction_vvv = model.matrix(formula_yes_interaction_no_intercept, data = vvv)

library(glmnet)

# criterion to choose the model: "1se" or "lmin"
cv_criterion = "lambda.1se"

# Ridge ------

# NO Interaction -----------

lambda_vals = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

ridge_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction_sss,
                                              my_y = sss$y,
                                              my_alpha = 0,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS_sss,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                              is_classification = TRUE,
                                              my_threshold = MY_THRESHOLD)

# ridge_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST,
#                                                       my_metric_names = METRICS_NAMES,
#                                                       my_x = X_mm_no_interaction_sss,
#                                                       my_y = sss$y,
#                                                       my_alpha = 0,
#                                                       my_lambda_vals = lambda_vals,
#                                                       my_weights = MY_WEIGHTS_sss,
#                                                       my_metrics_functions = MY_USED_METRICS,
#                                                       my_ncores = N_CORES,
#                                                       use_only_first_fold = USE_ONLY_FIRST_FOLD)

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

ridge_no_interaction = glmnet(x = X_mm_no_interaction_sss,
                              y = sss$y,
                              alpha = 0,
                              lambda = ridge_no_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])



# Plan B using default R library 
# ridge_cv_no_interaction = cv.glmnet(x = X_mm_no_interaction_sss, y = sss$y,
#                                     alpha = 0, nfols = K_FOLDS,
#                                     lambda.min.ratio = 1e-07)
# 
# plot(ridge_cv_no_interaction)
# 
# # define plotting function
# temp_plotting_fun = function(){
#   plot(ridge_cv_no_interaction, main = "ridge no interaction")
# }
# 
# PlotAndSave(temp_plotting_fun,
#             my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
#                                  "ridge_no_int_plot.jpeg",
#                                  collapse = ""))
# 
# 
# print(paste("ridge_cv_no_interaction ", cv_criterion, collapse = ""))
# ridge_cv_no_interaction[[cv_criterion]]
# 
# 
# ridge_no_interaction = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
#                                    alpha = 0,
#                                    lambda = ridge_cv_no_interaction[[cv_criterion]])

# rm(ridge_cv_no_interaction)

temp_pred = predict(ridge_no_interaction, newx = X_mm_no_interaction_vvv)
pred_list$ridge_no_interaction = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "ridge_no_interaction",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")

file_name_ridge_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                       "ridge_no_interaction",
                                       ".Rdata", collapse = "", sep = "")

save(ridge_no_interaction, file = file_name_ridge_no_interaction)

rm(ridge_no_interaction)
gc()

# YES Interaction -----------
lambda_vals = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

ridge_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_yes_interaction_sss,
                                              my_y = sss$y,
                                              my_alpha = 0,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS_sss,
                                              use_only_first_fold = TRUE,
                                              is_classification = TRUE,
                                              my_threshold = MY_THRESHOLD)

ridge_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST,
                                                       my_metric_names = METRICS_NAMES,
                                                       my_x = X_mm_yes_interaction_sss,
                                                       my_y = sss$y,
                                                       my_alpha = 0,
                                                       my_lambda_vals = lambda_vals,
                                                       my_weights = MY_WEIGHTS_sss,
                                                       my_metrics_functions = MY_USED_METRICS,
                                                       my_ncores = N_CORES,
                                                       use_only_first_fold = TRUE,
                                                       is_classification = TRUE,
                                                       my_threshold = MY_THRESHOLD)

ridge_yes_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                          my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                                          my_one_se_best = TRUE,
                                          my_higher_more_complex = FALSE,
                                          my_se_matrix = ridge_yes_interaction_metrics[["se"]],
                                          my_metric_names = METRICS_NAMES,
                                          indexes_metric_max = 4)



PlotAndSave(function()(
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                my_se_matrix = ridge_yes_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(ridge_yes_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge yes interaction metrics",
                my_xlab = " log lambda")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "ridge_yes_int_metrics_plot.jpeg",
                       collapse = ""))

print("ridge_yes_int_best_summary")
ridge_yes_int_best_summary

ridge_yes_interaction = glmnet(x = X_mm_yes_interaction_sss,
                               y = sss$y,
                               alpha = 0,
                               lambda = ridge_yes_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])



# Plan B using default R library 
# ridge_cv_yes_interaction = cv.glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                     alpha = 0, nfols = K_FOLDS,
#                                     lambda.min.ratio = 1e-07)
# 
# plot(ridge_cv_yes_interaction)
# 
# # define plotting function
# temp_plotting_fun = function(){
#   plot(ridge_cv_yes_interaction, main = "ridge yes interaction")
# }
# 
# PlotAndSave(temp_plotting_fun,
#             my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
#                                  "ridge_yes_int_plot.jpeg",
#                                  collapse = ""))
# 
# 
# print(paste("ridge_cv_yes_interaction ", cv_criterion, collapse = ""))
# ridge_cv_yes_interaction[[cv_criterion]]
# 
# 
# ridge_yes_interaction = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                    alpha = 0,
#                                    lambda = ridge_cv_yes_interaction[[cv_criterion]])

# rm(ridge_cv_yes_interaction)

temp_pred = predict(ridge_yes_interaction, newx = X_mm_yes_interaction_vvv)
pred_list$ridge_yes_interaction = temp_pred

# previsione ed errore
df_metrics = Add_Test_Metric(df_metrics,
                             "ridge_yes_interaction",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")

file_name_ridge_yes_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                        "ridge_yes_interaction",
                                        ".Rdata", collapse = "", sep = "")

save(ridge_yes_interaction, file = file_name_ridge_yes_interaction)

rm(ridge_yes_interaction)
gc()



# Lasso ------
# NO Interaction -----------

lambda_vals = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

lasso_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction_sss,
                                              my_y = sss$y,
                                              my_alpha = 1,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS_sss,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                              is_classification = TRUE,
                                              my_threshold = MY_THRESHOLD)

# lasso_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST,
#                                                       my_metric_names = METRICS_NAMES,
#                                                       my_x = X_mm_no_interaction_sss,
#                                                       my_y = sss$y,
#                                                       my_alpha = 1,
#                                                       my_lambda_vals = lambda_vals,
#                                                       my_weights = MY_WEIGHTS_sss,
#                                                       my_metrics_functions = MY_USED_METRICS,
#                                                       my_ncores = N_CORES,
#                                                       use_only_first_fold = USE_ONLY_FIRST_FOLD)

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

lasso_no_interaction = glmnet(x = X_mm_no_interaction_sss,
                              y = sss$y,
                              alpha = 1,
                              lambda = lasso_no_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])



# Plan B using default R library 
# lasso_cv_no_interaction = cv.glmnet(x = X_mm_no_interaction_sss, y = sss$y,
#                                     alpha = 1, nfols = K_FOLDS,
#                                     lambda.min.ratio = 1e-07)
# 
# plot(lasso_cv_no_interaction)
# 
# # define plotting function
# temp_plotting_fun = function(){
#   plot(lasso_cv_no_interaction, main = "lasso no interaction")
# }
# 
# PlotAndSave(temp_plotting_fun,
#             my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
#                                  "lasso_no_int_plot.jpeg",
#                                  collapse = ""))
# 
# 
# print(paste("lasso_cv_no_interaction ", cv_criterion, collapse = ""))
# lasso_cv_no_interaction[[cv_criterion]]
# 
# 
# lasso_no_interaction = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
#                                    alpha = 1,
#                                    lambda = lasso_cv_no_interaction[[cv_criterion]])

# rm(lasso_cv_no_interaction)

temp_pred = predict(lasso_no_interaction, newx = X_mm_no_interaction_vvv)
pred_list$lasso_no_interaction = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_no_interaction",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")

file_name_lasso_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                       "lasso_no_interaction",
                                       ".Rdata", collapse = "", sep = "")

save(lasso_no_interaction, file = file_name_lasso_no_interaction)

rm(lasso_no_interaction)
gc()

# YES Interaction -----------
lambda_vals = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

lasso_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST,
                                               my_metric_names = METRICS_NAMES,
                                               my_x = X_mm_yes_interaction_sss,
                                               my_y = sss$y,
                                               my_alpha = 1,
                                               my_lambda_vals = lambda_vals,
                                               my_weights = MY_WEIGHTS_sss,
                                               use_only_first_fold = TRUE,
                                               is_classification = TRUE,
                                               my_threshold = MY_THRESHOLD)

# lasso_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST,
#                                                        my_metric_names = METRICS_NAMES,
#                                                        my_x = X_mm_yes_interaction_sss,
#                                                        my_y = sss$y,
#                                                        my_alpha = 1,
#                                                        my_lambda_vals = lambda_vals,
#                                                        my_weights = MY_WEIGHTS_sss,
#                                                        my_metrics_functions = MY_USED_METRICS,
#                                                        my_ncores = N_CORES,
#                                                        use_only_first_fold = TRUE,
#                                                        is_classification = TRUE,
#                                                        my_threshold = MY_THRESHOLD)

lasso_yes_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                          my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                                          my_one_se_best = TRUE,
                                          my_higher_more_complex = FALSE,
                                          my_se_matrix = lasso_yes_interaction_metrics[["se"]],
                                          my_metric_names = METRICS_NAMES,
                                          indexes_metric_max = 4)



PlotAndSave(function()(
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                my_se_matrix = lasso_yes_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(lasso_yes_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso yes interaction metrics",
                my_xlab = " log lambda")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "lasso_yes_int_metrics_plot.jpeg",
                       collapse = ""))

print("lasso_yes_int_best_summary")
lasso_yes_int_best_summary

lasso_yes_interaction = glmnet(x = X_mm_yes_interaction_sss,
                               y = sss$y,
                               alpha = 1,
                               lambda = lasso_yes_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])



# Plan B using default R library 
# lasso_cv_yes_interaction = cv.glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                     alpha = 1, nfols = K_FOLDS,
#                                     lambda.min.ratio = 1e-07)
# 
# plot(lasso_cv_yes_interaction)
# 
# # define plotting function
# temp_plotting_fun = function(){
#   plot(lasso_cv_yes_interaction, main = "lasso yes interaction")
# }
# 
# PlotAndSave(temp_plotting_fun,
#             my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
#                                  "lasso_yes_int_plot.jpeg",
#                                  collapse = ""))
# 
# 
# print(paste("lasso_cv_yes_interaction ", cv_criterion, collapse = ""))
# lasso_cv_yes_interaction[[cv_criterion]]
# 
# 
# lasso_yes_interaction = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                    alpha = 1,
#                                    lambda = lasso_cv_yes_interaction[[cv_criterion]])

# rm(lasso_cv_yes_interaction)

temp_pred = predict(lasso_yes_interaction, newx = X_mm_yes_interaction_vvv)
pred_list$lasso_yes_interaction = temp_pred

# previsione ed errore
df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_yes_interaction",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")

file_name_lasso_yes_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                        "lasso_yes_interaction",
                                        ".Rdata", collapse = "", sep = "")

save(lasso_yes_interaction, file = file_name_lasso_yes_interaction)

rm(lasso_yes_interaction)
gc()





rm(X_mm_yes_interaction_sss)
rm(X_mm_yes_interaction_vvv)
gc()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Tree -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

# Model selection

# 0) Full tree to be pruned ------

# default: overfit
tree_full = tree(factor(y) ~.,
                 data = sss[id_cb1,],
                 control = tree.control(nobs = length(id_cb1),
                                        mindev = 1e-04,
                                        minsize = 5))



# check overfitting
plot(tree_full)


# Selection of size parameter, we have two possible ways

# 1.a) Size: CV ----------
# Selection of size parameter

TREE_MAX_SIZE = 50


# tree_cv_metrics = ManualCvTree(my_id_list_cv_train = ID_CV_LIST,
#                                        my_metric_names = METRICS_NAMES,
#                                        my_data = sss,
#                                        my_max_size = TREE_MAX_SIZE,
#                                        my_weights = MY_WEIGHTS_sss,
#                                        my_mindev = 1e-04,
#                                        my_minsize = 2,
#                                        is_classification = TRUE,
#                                        my_threshold = MY_THRESHOLD,
#                                use_only_first_fold = TRUE)

# if parallel shows problems use the non parallel version
tree_cv_metrics = ManualCvTreeParallel(my_id_list_cv_train = ID_CV_LIST,
                                       my_metric_names = METRICS_NAMES,
                                       my_data = sss,
                                       my_max_size = TREE_MAX_SIZE,
                                       my_metrics_functions = MY_USED_METRICS,
                                       my_ncores = N_CORES,
                                       my_weights = MY_WEIGHTS_sss,
                                       my_mindev = 1e-04,
                                       my_minsize = 5,
                                       use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                       is_classification = TRUE,
                                       my_threshold = MY_THRESHOLD)

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
                my_xlab = "size")),
  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                       "tree_cv_metrics_plot.jpeg",
                       collapse = ""))


tree_best_size = tree_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]]

print("tree best size")
tree_best_size


# 1.b) Size: train - test -----------

# pruning
# tree_pruned = prune.tree(tree_full,
#                          newdata = sss[-id_cb1,])
# 
# tree_best_size = tree_pruned$size[which.min(tree_pruned$dev)]
# 
# print("tree best size")
# tree_best_size
# 
# temp_plot_function = function(){
#   plot(tree_pruned)
#   plot(tree_pruned, xlim = c(0, 40))
#   
#   abline(v = tree_best_size, col = "red")
# }
# 
# PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
#                                                      "tree_test_deviance_plot.jpeg",
#                                                      collapse = ""))



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


# check the index
temp_pred = predict(final_tree_pruned, newdata = vvv)[,2]
pred_list$tree_pruned = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "tree_pruned best",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))


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
save(df_metrics, pred_list, file = "df_metrics.Rdata")


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello Additivo ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(gam)

# step selection via GCV

# stepwise forward: AIC based on generalized df
gam0 = gam(y ~ 1, data = sss, family = "binomial")

# gam recognizes factor predictors
my_gam_scope = gam.scope(sss[,-y_index], arg = c("df=2", "df=3", "df=4", "df=5", "df=6"))

# try parallel (linux only)
# require(doMC)
# registerDoMC(cores= N_CORES)
# step.Gam(gam0, my_gam_scope, parallel=TRUE)

gam_step = step.Gam(gam0, scope = my_gam_scope)

temp_pred = predict(gam_step,
                    newdata = vvv,
                    type = "response")

pred_list$gam_step = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "gam_step",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics
rm(temp_pred)

file_name_gam_step = paste(MODELS_FOLDER_RELATIVE_PATH,
                           "gam_step",
                           ".Rdata", collapse = "", sep = "")

save(gam_step, file = file_name_gam_step)



rm(gam_step)
rm(gam0)
gc()

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")

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
num_index = which(colnames(sss[,-y_index]) %in% var_num_names)
factor_index = setdiff(1:NCOL(sss[,-y_index]), num_index)

library(polspline)

# step selection via GCV
# only interaction of two terms are admitted 
# (computational and time constraint)

# if problems: -> but usually give problems
# weights = MY_WEIGHTS_sss

mars_step = polymars(responses = sss$y[id_cb1],
                     predictors = sss[id_cb1,-y_index],
                     gcv = 1,
                     factors = factor_index,
                     maxsize = 50,
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

# pay attention to the column
temp_pred = predict(mars_step, x = vvv[,-y_index])[,2]

pred_list$mars_step = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "MARS",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")

mars_names = colnames(sss[,-y_index])

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

ppr_metrics = PPRRegulationCVParallel(my_data = sss,
                                      my_id_list_cv_train = ID_CV_LIST,
                                      my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                      my_spline_df = PPR_DF_SM,
                                      my_metrics_names = METRICS_NAMES,
                                      my_weights = MY_WEIGHTS_sss,
                                      my_metrics_functions = MY_USED_METRICS,
                                      my_ncores = N_CORES,
                                      use_only_first_fold = TRUE,
                                      is_classification = TRUE,
                                      my_threshold = MY_THRESHOLD)



# 2) final model -------

ppr_best_params = PPRExtractBestParams(ppr_metrics)

print("ppr best params")
ppr_best_params

ppr_model = ppr(y ~ .,
                data = sss,
                nterms = ppr_best_params[[METRIC_CHOSEN_NAME]][["n_ridge_functions"]],
                sm.method = "spline",
                df = ppr_best_params[[METRIC_CHOSEN_NAME]][["spline_df"]]) 


temp_pred = predict(ppr_model, vvv)

pred_list$ppr_model = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "PPR",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")

file_name_ppr_model = paste(MODELS_FOLDER_RELATIVE_PATH,
                            "ppr_model",
                            ".Rdata", collapse = "", sep = "")

save(ppr_model, file = file_name_ppr_model)

rm(ppr_model)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Nota: se manca il tempo eseguo prima la RandomForest del Bagging
# visto che quest'ultimo è un sotto caso particolare 
# della RandomForest (selezione di tutte le variabili per ogni split)


# Implementazione in parallelo
library(ranger)

library(snowfall)



sfInit(cpus = N_CORES, parallel = T)
sfLibrary(ranger)
sfExport(list = c("sss"))


# esportiamo tutti gli oggetti necessari

# scelta del numero di esplicative a ogni split
# adattiamo 50 alberi per core e di ciascun albero ritorniamo l'errore out of bag
# successivamente sommiamo gli errori out of bag per ogni mtry

# massimo numero di esplicative presenti
RF_MAX_VARIABLES = NCOL(sss) - 1 # sottraggo 1 per la variabile risposta
# ridurlo per considerazioni computazionali

# regolazione
# procedura sub-ottimale, ma la impiego per ragioni computazionali
# prima scelgo il numero di esplicative a ogni split,
# una volta scelto controllo la convergenza dell'errore basata sul numero di alberi
err = rep(NA, RF_MAX_VARIABLES)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°

for(i in seq(2, RF_MAX_VARIABLES)){
  sfExport(list = c("i"))
  
  err[i] = sum(sfSapply(rep(1:8),
                        function(x) ranger(factor(y) ~., data = sss,
                                           mtry = i,
                                           num.trees = 50,
                                           probability = TRUE,
                                           oob.error = TRUE)$prediction.error))
  print(paste("mtry: ", i, collapse = ""))
  gc()
}

print("Random forest error for each mtry")
err 

best_mtry = which.min(err)

print("best mtry random forest")
best_mtry

sfExport(list = c("best_mtry"))

# uso il valore trovato e controllo la convergenza rispetto al numero di alberi

err_rf_trees = rep(NA, 90)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
for(j in 10:100){
  sfExport(list = c("j"))
  err_rf_trees[j] = sum(sfSapply(rep(1:4),
                                 function(x) ranger(factor(y) ~., sss,
                                                    mtry = best_mtry,
                                                    num.trees = j,
                                                    probability = TRUE,
                                                    oob.error = TRUE)$prediction.error))
  print(paste("number of trees: ", j*4, collapse = ""))
  gc()
}

sfStop()

PlotAndSave(my_plotting_function =  function()plot((1:length(err_rf_trees)) * 4, err_rf_trees,
                                                   xlab = "Bootstrap trees number",
                                                   ylab = "Out of bag Error",
                                                   pch = 16,
                                                   main = "Random Forest"),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "random_forest_convergence_plot.jpeg",
                                 collapse = ""))




# modello finale e previsioni
random_forest_model = ranger(factor(y) ~., sss,
                             mtry = best_mtry,
                             num.trees = 400,
                             oob.error = TRUE,
                             probability = TRUE,
                             importance = "permutation")

# Warning check index
temp_pred = predict(random_forest_model, data = vvv,
                    type = "response")$predictions[,2]

pred_list$random_forest = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "Random Forest",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics
rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")


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


library(ipred)
sfInit(cpus = N_CORES, parallel = T)
sfExport(list = c("sss"))

sfLibrary(ipred)

err_bg_trees = rep(NA, 90)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°

# controllo la convergenza dell'errore rispetto al numero di alberi
# parto da 40 alberi bootstrap
for(j in 10:100){
  sfExport(list = c("j"))
  err_bg_trees[j] = sum(sfSapply(rep(1:4),
                                 function(x) bagging(factor(y) ~., sss,
                                                     nbag = j,
                                                     coob = TRUE)$err))
  print(j*4)
  gc()
}

sfStop()

PlotAndSave(my_plotting_function = function() plot((1:length(err_bg_trees)) * 4, err_bg_trees,
                                                   xlab = "numero di alberi bootstrap",
                                                   ylab = "errore out of bag",
                                                   pch = 16,
                                                   main = "Bagging"),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "bagging_convergence_plot.jpeg",
                                 collapse = "")
)


# se il numero di replicazioni bootstrap arriva a convergenza allora

bagging_model = bagging(y ~., sss, nbag = 400, coob = FALSE)

temp_pred = predict(bagging_model, data = vvv,
                    type = "response")$predictions[,2]

pred_list$bagging = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "Bagging",
                             USED.Metrics(temp_pred > MY_THRESHOLD,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))


df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")


# save metrics and model
file_name_bagging = paste(MODELS_FOLDER_RELATIVE_PATH,
                          "bagging",
                          ".Rdata", collapse = "", sep = "")

save(bagging_model, file = file_name_bagging)


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

# per questo problema usiamo l'errore di classificazione

# massima profondità = 10
err_boost = rep(NA, 20)

iter_boost = 200

# Stump 
m_boost_stump = ada(x = sss[id_cb1, -y_index],
                    y = sss$y[id_cb1],
                    test.x = sss[-id_cb1, -y_index],
                    test.y = sss$y[-id_cb1],
                    iter = iter_boost,
                    control = rpart.control(maxdepth=1,
                                            cp=-1,
                                            minsplit=0,xval=0))

plot(m_boost_stump, test = T)

par(mfrow = c(1,1))

# update
m_boost_stump = update(m_boost_stump,
                       x = sss[id_cb1, -y_index],
                       y = sss$y[id_cb1],
                       test.x = sss[-id_cb1, -y_index],
                       test.y = sss$y[-id_cb1],
                       n.iter = 400)

PlotAndSave(my_plotting_function = function() plot(m_boost_stump,
                                                   test = T),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "boosting_convergence.jpeg",
                                 collapse = ""))


pred_boost_stump = predict(m_boost_stump, vvv, type = "prob")[,2]
pred_list$pred_boost_stump = as.vector(pred_boost_stump)

df_err_qual = Add_Test_Metric(df_err_qual,
                              "Boosting Stump",
                              USED.Metrics(pred_boost_stump > MY_THRESHOLD,
                                        vvv$y,
                                        weights = MY_WEIGHTS_vvv))

rm(m_boost_stump)
rm(pred_boost_stump)
gc()

# 3 split
# guardiamo quando si stabilizza l'errore

m_boost_2 = ada(x = sss[id_cb1, -y_index],
                y = sss$y[id_cb1],
                test.x = sss[-id_cb1, -y_index],
                test.y = sss$y[-id_cb1],
                iter = iter_boost,
                control = rpart.control(maxdepth = 6))

plot(m_boost_2, test = T)

par(mfrow = c(1,1))

# update
m_boost_2 = update(m_boost_2,
                   x = sss[id_cb1, -y_index],
                   y = sss$y[id_cb1],
                   test.x = sss[-id_cb1, -y_index],
                   test.y = sss$y[-id_cb1],
                   n.iter = 700)



PlotAndSave(my_plotting_function = function() plot(m_boost_2, test = T),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "boosting_3_split_convergence.jpeg",
                                 collapse = ""))


pred_boost_2 = predict(m_boost_2, vvv, type = "prob")[,2]
pred_list$boosting_3_splits = pred_boost_2

df_err_qual = Add_Test_Metric(df_err_qual,
                              "Boosting 3 Split",
                              USED.Metrics(pred_boost_2 > MY_THRESHOLD,
                                        vvv$y,
                                        weights = MY_WEIGHTS_vvv))

df_err_qual

rm(m_boost_2)
rm(pred_boost_2)

gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Support Vector Machine ---------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(e1071)


# Stima Convalida
# Kernel Radiale

t = ManualCvSVM(my_id_list_cv_train = ID_CV_LIST,
                my_metric_names = METRICS_NAMES,
                my_data = sss,
                my_params_vector = seq(2,3,by=0.5),
                my_kernel_name = "polynomial",
                my_weights = MY_WEIGHTS_sss,
                use_only_first_fold = TRUE,
                my_id_list_cv_test = NULL)

ranges = seq(2,20,by=0.5)
err_svm <- matrix(NA,length(ranges),2)
for (i in 1:length(ranges)){
  s1<- svm(factor(y)~., data= sss[id_cb1,], cost=ranges[i], kernel = "radial")
  pr1 <- predict(s1, newdata= sss[-id_cb1,]) #, decision.values=TRUE)
  uso <- USED.Metrics(pr1, sss$y[-id_cb1], weights = MY_WEIGHTS_sss[-id_cb1])
  # eventualmente cambia il tipo di errore
  err_svm[i,]<-c(ranges[i], uso[1])
  print(i)
}
plot(err_svm, type="b", pch = 16,
     xlab = "costo", ylab = "errore",
     main = "SVM radiale")

ranges[which.min(err_svm[,2])]
# ATTENZIONE: modificare 
# 15 

m_svm =  svm( factor(y)~., data= sss, cost= ranges[which.min(err_svm[,2])])
pred_svm_radial = predict(m_svm, newdata = vvv)

df_err_qual = Add_Test_Metric(df_err_qual,
                              "SVM radial",
                              USED.Loss(pred_svm_radial,
                                        vvv$y))

df_err_qual

pred_list$pred_svm_radial = pred_svm_radial



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rete neurale -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Solo in parallelo altrimenti ci mette troppo tempo


# decay = 10^seq(-3, -1, length=10)
# nodi = 1:10
# 
# hyp_grid = expand.grid(decay,nodi)

# # Costruiamo una funzione che prenda come input una matrice parametri,
# # stimi la rete per ogni valore, e restiuisca una matrice con valori dei parametri + errori su convalida
# regola_nn = function(pars, sss, id_cb1){
#   err = data.frame(pars, err = NA)
#   for(i in 1:NROW(pars)){
#     n1 = nnet(y ~ . , data=sss[id_cb1,], 
#               size=pars[i,2], decay=pars[i,1],
#               MaxNWts = 1500, maxit = 500, 
#               trace = T)
#     err$err[i] = MSE.Loss(predict(n1, sss[-id_cb1,], type = 'raw'), sss$y[-id_cb1])
#   }
#   return(err)
# }
# 
# # proviamo
# regola_nn(hyp_grid[21:23,], sss, id_cb1)


# Parallelo
# Per mitigare il load balance possiamo assegnare a caso ai vari processori
# In questo modo ogni processore avra' sia valori di parametri "semplici" (pochi nodi)
# Che complessi (tanti nodi)

# Conviene creare una lista in cui ogni elemento sia la matrice di parametri provati
# da quel processore

# pp = sample(rep(1:4, each = NROW(hyp_grid)/4))
# pars_list = lapply(1:4, function(l) hyp_grid[pp == l,])
# 
# 
# library(snowfall)
# sfInit(cpus = 4, parallel = T)
# sfLibrary(nnet) # carichiamo la libreria
# sfExport(list = c("sss", "id_cb1", "regola_nn", "MSE.Loss")) # esportiamo tutte le quantita' necessarie
# 
# # Non restituisce messaggi, possiamo solo aspettare
# nn_error = sfLapply(pars_list, function(x) regola_nn(x, sss, id_cb1))
# sfStop()
# 
# err_nn = do.call(rbind, nn_error)
# 
# par(mfrow = c(1,2))
# plot(err_nn$Var1, err_nn$err, xlab = "Weight decay", ylab = "Errore di convalida", pch = 16)
# plot(err_nn$Var2, err_nn$err, xlab = "Numero di nodi", ylab = "Errore di convalida", pch = 16)
# 
# err_nn[which.min(err_nn$err),]
# 
# # 0.03593814    4 0.5818981 (ovviamente potrebbe variare a seconda di: punti iniziali, stima/convalida, etc
# 
# set.seed(123)
# mod_nn = nnet(diagnosi ~ . , data=sss[,], size = 4, decay = 0.03593,
#               MaxNWts = 1500, maxit = 2000, trace = T)
# 
# pr_nn = predict(mod_nn, vvv, type = "class")

# /////////////////////////////////////////////////////////////////
#------------------------ Sintesi Finale -------------------------
# /////////////////////////////////////////////////////////////////

rounded_df = cbind(df_metrics[,1],
                   apply(df_metrics[,2:NCOL(df_metrics)], 2, function(col) round(as.numeric(col), 2)))

rounded_df %>% orde

# LIFT e ROC --------------
# eventualmente crea una nuova lista ridotta
# con solo i modelli con l'errore minimo
model_names_all = names(pred_list)

paste(model_names_all, collapse = "','")

# selection of "best" models so Lift and ROC curve are readable
model_names = c('ridge_yes_interaction_lmin',
                'lasso_yes_interaction_lmin',
                'gam_step',
                'mars_step',
                'pred_tree',
                'pred_bagging',
                'pred_boost_2',
                'pred_random_forest',
                'pred_svm_radial')

pred_list_ridotta = pred_list[which(model_names_all %in% model_names)]

pred_list_ridotta$pred_glm_no_interaction = NULL


curve = lapply(pred_list_ridotta, function(x) lift.roc(x, vvv$y, plot.it=F, type = "bin"))

model_printed_names = sapply(model_names, FUN = function(x) gsub("pred_", "", x))


library(RColorBrewer)
colori = c("black", brewer.pal(12, "Set3"))
# @@@@@@@@@@@@@@@@@@@@@
# LIFT ----------------
# @@@@@@@@@@@@@@@@@@@@@

plot(curve[[1]][[1]], curve[[1]][[2]], type = "b", ylim = c(0,2.5), 
     xlab = "Frazione di predetti positivi", ylab = "Lift",
     pch = 1, main = "LIFT", lwd = 2)
# e disegnamo le altre sovrapponendole
for(j in 2:length(curve)) {
  lines(curve[[j]][[1]], curve[[j]][[2]],type = "b", col = colori[j], lwd = 2, pch = j)
}

legend("bottomright",col = colori,
       legend = model_printed_names, pch = 1:length(model_printed_names), 
       ncol = 2, lwd = 2,cex = .5,
       y.intersp = 1.5,
       text.width = 0.3,
       
)

# @@@@@@@@@@@@@@@@@@@@@
# ROC ----------------
# @@@@@@@@@@@@@@@@@@@@@
plot(curve[[1]][[3]], curve[[1]][[4]], type = "b", ylim = c(0,1), 
     xlab = "1-specificità", ylab = "sensibilità",
     pch = 1, main = "ROC", lwd = 2)
# e disegnamo le altre sovrapponendole
for(j in 2:length(curve)) {
  lines(curve[[j]][[3]], curve[[j]][[4]],type = "b", col = colori[j], lwd = 2, pch = j)
}

legend("bottomright",col = colori,
       legend = model_printed_names, pch = 1:length(model_printed_names), 
       ncol = 2, lwd = 2,cex = .5,
       y.intersp = 1.5,
       text.width = 0.3,
       
)


# Comparazione modelli
# selezione modelli migliori e commenti su questi:
# es -> vanno meglio modelli con interazioni oppure modelli additivi

# select the first k numbers with greater absolute value

coef(lm_step_no_interaction) %>% sort(decreasing = T)

coef(lm_step_no_interaction) %>% boxplot()

# Linear Step ---------------

load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "lm_step_no_interaction",
           ".Rdata", collapse = "", sep = ""))

temp_coef = coef(lm_step_no_interaction)
temp_main = "(abs) greatest linear model coefficients no interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -1) | (temp_coef > 1)) ] %>% sort()

PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_lm_no_int_plot.jpeg",
                                 collapse = ""))




load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "lm_step_yes_interaction",
           ".Rdata", collapse = "", sep = ""))

temp_coef = coef(lm_step_yes_interaction)
temp_main = "(abs) greatest linear model coefficients yes interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -1) | (temp_coef > 1)) ] %>% sort()
PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_lm_yes_int_plot.jpeg",
                                 collapse = ""))



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


# Gam ------------------
load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "gam_step",
           ".Rdata", collapse = "", sep = ""))

summary(gam_step)

PlotAndSave(my_plotting_function = function() plot(gam_step, terms = c("s(x8, df = 4)"), se = T),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "gam_1_plot.jpeg",
                                 collapse = ""))



# MARS -----------------
load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "mars_step",
           ".Rdata", collapse = "", sep = ""))

print("mars step model")
mars_step$model

mars_names = colnames(sss[,-y_index])

# get the index by variable name
temp_index = which(mars_names == "x8")


# plots
PlotAndSave(my_plotting_function = function() plot(mars_step, predictor1 = temp_index),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "mars_1_plot.jpeg",
                                 collapse = ""))




# plot(mars_step, predictor1 = 7, predictor2 = 30)



