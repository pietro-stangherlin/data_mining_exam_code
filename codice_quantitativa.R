# data wrangling
library(dplyr)

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
# change functions for specific problems

# in generale uso sia MAE che MSE
USED.Metrics = function(y.pred, y.test, weights = 1){
  return(c(MAE.Loss(y.pred, y.test, weights), MSE.Loss(y.pred, y.test, weights)))
}


df_metrics = data.frame(name = NA, MAE = NA, MSE = NA)

METRICS_NAMES = colnames(df_metrics[,-1])

N_METRICS = length(METRICS_NAMES)

# names used to extract the metric added to df_metrics
# change based on the spefific problem
METRIC_VALUES_NAME = "metric_values"
METRIC_CHOSEN_NAME = "MSE"

# names used for accessing list CV matrix (actual metrics and metrics se)
LIST_METRICS_ACCESS_NAME = "metrics"
LIST_SD_ACCESS_NAME = "se"

# metrics names + USED.Loss
# WARNING: the order should be same as in df_metrics
MY_USED_METRICS = c("USED.Metrics", "MAE.Loss", "MSE.Loss")

# /////////////////////////////////////////////////////////////////
#------------------------ Train & Test ------------------------
# /////////////////////////////////////////////////////////////////


# eventually change the proportion
id_stima = sample(1:NROW(dati), 0.75 * NROW(dati))

sss = dati[id_stima,]
vvv = dati[-id_stima,]


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
#------------------------ Explorative Data Analysis ---------------
# /////////////////////////////////////////////////////////////////


# (on train set)

# check distribution of marginal response

hist(sss$y,nclass = 100)
summary(sss$y)

# check logaritm, a transformation (traslation) is maybe needed before
hist(log(sss$y), nclass = 100)

# NOTE: if logarithm is considered as response the difference of log
# is the log of the ratio

# /////////////////////////////////////////////////////////////////
#------------------------ MODELS ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Mean and Median --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Not considering predictors

df_metrics = Add_Test_Metric(df_metrics,
                              "sss mean",
                              USED.Metrics(mean(sss$y),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))

df_metrics = Add_Test_Metric(df_metrics,
                              "sss median",
                              USED.Metrics(median(sss$y),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))

df_metrics = na.omit(df_metrics)

df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step linear model --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# AIC criterion is used for model selection

lm0 = lm(y ~ 1, data = sss)

# NO Interaction -----------
lm_step_no_interaction = step(lm0, scope = formula_no_interaction_yes_intercept,
                 direction = "forward")

formula(lm_step_no_interaction)

# load(file_name_lm_step_no_interaction)

df_metrics = Add_Test_Metric(df_metrics,
                              "lm_step_no_interaction",
                              USED.Metrics(predict(lm_step_no_interaction, newdata = vvv),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))
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

formula(lm_step_yes_interaction)


df_metrics = Add_Test_Metric(df_metrics,
                              "lm_step_yes_interaction",
                              USED.Metrics(predict(lm_step_yes_interaction, newdata = vvv),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))

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


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge & Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# sparse is preferred is there are many categorical predictors (sparse matrix)
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

ridge_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction_sss,
                                              my_y = sss$y,
                                              my_alpha = 0,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS_sss,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD)

# ridge_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
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
                                         my_metric_names = METRICS_NAMES)


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

df_metrics = Add_Test_Metric(df_metrics,
                              "ridge_no_interaction",
                              USED.Metrics(predict(ridge_no_interaction, newx = X_mm_no_interaction_vvv),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

file_name_ridge_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                          "ridge_no_interaction",
                                          ".Rdata", collapse = "", sep = "")

save(ridge_no_interaction, file = file_name_ridge_no_interaction)

rm(ridge_no_interaction)
gc()

# YES Interaction -----------
lambda_vals = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

# ridge_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv = ID_CV_LIST,
#                                               my_metric_names = METRICS_NAMES,
#                                               my_x = X_mm_yes_interaction_sss,
#                                               my_y = sss$y,
#                                               my_alpha = 0,
#                                               my_lambda_vals = lambda_vals,
#                                               my_weights = MY_WEIGHTS_sss,
#                                               use_only_first_fold = USE_ONLY_FIRST_FOLD)

ridge_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
                                                      my_metric_names = METRICS_NAMES,
                                                      my_x = X_mm_yes_interaction_sss,
                                                      my_y = sss$y,
                                                      my_alpha = 0,
                                                      my_lambda_vals = lambda_vals,
                                                      my_weights = MY_WEIGHTS_sss,
                                                      my_metrics_functions = MY_USED_METRICS,
                                                      my_ncores = N_CORES,
                                                      use_only_first_fold = TRUE)

ridge_yes_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                                         my_one_se_best = FALSE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = ridge_yes_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES)



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

# previsione ed errore
df_metrics = Add_Test_Metric(df_metrics,
                             "ridge_yes_interaction",
                             USED.Metrics(predict(ridge_yes_interaction, newx = X_mm_yes_interaction_vvv),
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

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

lasso_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction_sss,
                                              my_y = sss$y,
                                              my_alpha = 1,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS_sss,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD)

# lasso_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
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
                                         my_metric_names = METRICS_NAMES)



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

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_no_interaction",
                             USED.Metrics(predict(lasso_no_interaction, newx = X_mm_no_interaction_vvv),
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

file_name_lasso_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                       "lasso_no_interaction",
                                       ".Rdata", collapse = "", sep = "")

save(lasso_no_interaction, file = file_name_lasso_no_interaction)


rm(lasso_no_interaction)
gc()

# YES Interaction -----------
lambda_vals = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

# lasso_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv = ID_CV_LIST,
#                                                my_metric_names = METRICS_NAMES,
#                                                my_x = X_mm_yes_interaction_sss,
#                                                my_y = sss$y,
#                                                my_alpha = 1,
#                                                my_lambda_vals = lambda_vals,
#                                                my_weights = MY_WEIGHTS_sss,
#                                                use_only_first_fold = USE_ONLY_FIRST_FOLD)

lasso_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
                                                      my_metric_names = METRICS_NAMES,
                                                      my_x = X_mm_yes_interaction_sss,
                                                      my_y = sss$y,
                                                      my_alpha = 1,
                                                      my_lambda_vals = lambda_vals,
                                                      my_weights = MY_WEIGHTS_sss,
                                                      my_metrics_functions = MY_USED_METRICS,
                                                      my_ncores = N_CORES,
                                                      use_only_first_fold = TRUE)

lasso_yes_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                          my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                                          my_one_se_best = FALSE,
                                          my_higher_more_complex = FALSE,
                                          my_se_matrix = lasso_yes_interaction_metrics[["se"]],
                                          my_metric_names = METRICS_NAMES)

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

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_yes_interaction",
                             USED.Metrics(predict(lasso_yes_interaction, newx = X_mm_yes_interaction_vvv),
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

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

# 0) Full tree which to be pruned ------

# default: overfit
tree_full = tree(y ~.,
                 data = sss[id_cb1,],
                 control = tree.control(nobs = length(id_cb1),
                                        mindev = 1e-04,
                                        minsize = 5))



# check overfitting
plot(tree_full)


# Selection of size parameter, we have two possible ways

# 1.a) Size: CV ----------
# Selection of size parameter

TREE_MAX_SIZE = 100


# if parallel shows problems use the non parallel version
tree_cv_metrics = ManualCvTreeParallel(my_id_list_cv = ID_CV_LIST,
                                       my_metric_names = METRICS_NAMES,
                                       my_data = sss,
                                       my_max_size = TREE_MAX_SIZE,
                                       my_metrics_functions = MY_USED_METRICS,
                                       my_ncores = N_CORES,
                                       my_weights = MY_WEIGHTS_sss,
                                       my_mindev = 1e-04,
                                       my_minsize = 5,
                                       use_only_first_fold = USE_ONLY_FIRST_FOLD)

tree_best_summary = CvMetricBest(my_param_values = 2:TREE_MAX_SIZE,
                                 my_metric_matrix = tree_cv_metrics[["metrics"]],
                                 my_one_se_best = TRUE,
                                 my_higher_more_complex = TRUE,
                                 my_se_matrix = tree_cv_metrics[["se"]],
                                 my_metric_names = METRICS_NAMES)


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

plot(final_tree_pruned)
text(final_tree_pruned, cex = 0.7)

df_metrics = Add_Test_Metric(df_metrics,
                             "tree_pruned best",
                             USED.Metrics(predict(final_tree_pruned, newdata = vvv),
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))


df_metrics

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
gam0 = gam(y ~ 1, data = sss)

# gam recognizes factor predictors
my_gam_scope = gam.scope(sss[,-y_index], arg = c("df=2", "df=3", "df=4", "df=5", "df=6"))

# try parallel (linux only)
# require(doMC)
# registerDoMC(cores= N_CORES)
# step.Gam(gam0, my_gam_scope, parallel=TRUE)

gam_step = step.Gam(gam0, scope = my_gam_scope)

df_metrics = Add_Test_Metric(df_metrics,
                              "gam_step",
                              USED.Metrics(predict(gam_step, newdata = vvv),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))

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

num_index = which(colnames(X_mm_no_interaction_sss) %in% var_num_names)
factor_index = setdiff(1:NCOL(X_mm_no_interaction_sss), num_index)


library(polspline)

# step selection via GCV
# only interaction of two terms are admitted 
# (computational and time constraint)

mars_step = polymars(responses = sss$y,
                 predictors = X_mm_no_interaction_sss,
                 gcv = 1,
                 factors = factor_index,
                 maxsize = 60)


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



df_metrics = Add_Test_Metric(df_metrics,
                              "MARS",
                              USED.Metrics(predict(mars_step, x = X_mm_no_interaction_vvv),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

file_name_mars_step = paste(MODELS_FOLDER_RELATIVE_PATH,
                           "mars_step",
                           ".Rdata", collapse = "", sep = "")

save(mars_step, file = file_name_mars_step)



rm(mars_step)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PPR ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# max number of ridge functions
PPR_MAX_RIDGE_FUNCTIONS = 4

# possible spline degrees of freedom
PPR_DF_SM = 2:6


# given a list with elements parameters
# number of  ridge_functions
# spline degrees of freedom

# 1.a) Regulation: train - test ---------

#' @param my_data (data.frame)
#' @param my_id_train (vector of ints)
#' @param my_max_ridge_functions (vector of ints)
#' @param my_spline_df (vector in mums): values of possibile smoothing splines degrees of freedom
#' @param my_metrics_names (vector of chars)
#' @param my_weights (vector of nums):
#'  same length as the difference: NROW(my_data) - length(my_id_train)
#' 
#' @return (array):
#' first dimension (with names): 1:my_max_ridge_functions
#' second dimension (with names): my_spline_df
#' third dimension (with names): my_metrics_names
#' 
#' each cell contains the metric value of the model fitted on my_data[my_id_train,]
#' and tested on my_data[-my_id_train,] for each metric value used
PPRRegulationTrainTest = function(my_data = sss,
                                  my_id_train = id_cb1,
                                  my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                  my_spline_df = PPR_DF_SM,
                                  my_metrics_names = METRICS_NAMES,
                                  my_weights = MY_WEIGHTS_sss){
  metrics_array = array(NA,
                        dim = c(my_max_ridge_functions,
                                length(my_spline_df),
                                length(my_metrics_names)),
                        
                        dimnames = list(1:my_max_ridge_functions,
                                        my_spline_df,
                                        my_metrics_names))
  
  for(r in 1:my_max_ridge_functions){
    for(df in 1: length(my_spline_df)){
      mod = ppr(y ~ .,
                data = my_data[my_id_train,],
                nterms = r,
                sm.method = "spline",
                df = my_spline_df[df])
      
      metrics_array[r, df, ] = USED.Metrics(predict(mod, my_data[-my_id_train,]),
                                                my_data$y[-my_id_train],
                                                weights = my_weights)
    }
    print(paste0("n ridge functions: ", r, collapse = ""))
  }
  
  rm(mod)
  gc()
  
  
  return(metrics_array)
}



#' @param my_data (data.frame)
#' @param my_id_train (vector of ints)
#' @param my_max_ridge_functions (vector of ints)
#' @param my_spline_df (vector in mums): values of possibile smoothing splines degrees of freedom
#' @param my_metrics_names (vector of chars)
#' @param my_weights (vector of nums):
#'  same length as the difference: NROW(my_data) - length(my_id_train)
#'  
#'  
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#' @param my_ncores
#' 
#' @return (array):
#' first dimension (with names): 1:my_max_ridge_functions
#' second dimension (with names): my_spline_df
#' third dimension (with names): my_metrics_names
#' 
#' each cell contains the metric value of the model fitted on my_data[my_id_train,]
#' and tested on my_data[-my_id_train,] for each metric value used
PPRRegulationTrainTestParallel = function(my_data = sss,
                                          my_id_train = id_cb1,
                                          my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                          my_spline_df = PPR_DF_SM,
                                          my_metrics_names = METRICS_NAMES,
                                          my_weights = MY_WEIGHTS_sss,
                                          my_metrics_functions = MY_USED_METRICS,
                                          my_ncores = N_CORES){
  
  metrics_array = array(NA,
                        dim = c(my_max_ridge_functions,
                                length(my_spline_df),
                                length(my_metrics_names)),
                        
                        dimnames = list(1:my_max_ridge_functions,
                                        my_spline_df,
                                        my_metrics_names))
  
  my_n_metrics = length(my_metrics_names)
  
  
  # needed to do parallel
  # each list element contains a vector of length 2
  # first element is the number of ridge functions
  # second element are the spline degrees of freedom
  params_list = list()
  
  counter = 1
  
  for (r in 1:my_max_ridge_functions){
    for(df in my_spline_df){
      params_list[[counter]] = c(r, df)
      
      counter = counter + 1
    }
  }
  
  
  # init parallel
  sfInit(cpus = my_ncores, parallel = T)
  
  sfExport(list = c("my_data", my_metrics_functions,
                    "my_id_train", "my_max_ridge_functions", "my_spline_df", "params_list",
                    "my_weights"))
  
  temp_metric = sfLapply(params_list,
                         fun = function(el) 
                           USED.Metrics(predict(ppr(y ~ .,
                                                    data = my_data[my_id_train,],
                                                    nterms = el[1],
                                                    sm.method = "spline",
                                                    df = el[2]),
                                                my_data[-my_id_train,]), my_data$y[-my_id_train],
                                        weights = my_weights))
  
  # stop cluster
  sfStop()
  

  counter = 1
  
  for (r in 1:my_max_ridge_functions){
    for(df in 1:length(my_spline_df)){
      metrics_array[r, df, ] = temp_metric[[counter]]
      
      counter = counter + 1
    }
  }
  
  rm(temp_metric)
  gc()
  
  return(metrics_array)
  }


# ppr_metrics = PPRRegulationTrainTestParallel(my_data = sss,
#                                              my_id_train = id_cb1,
#                                              my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
#                                              my_spline_df = PPR_DF_SM,
#                                              my_metrics_names = METRICS_NAMES,
#                                              my_weights = MY_WEIGHTS_sss,
#                                              my_metrics_functions = MY_USED_METRICS,
#                                              my_ncores = N_CORES)



# 1.b) Regulation: CV -------

ppr_metrics = PPRRegulationCVParallel(my_data = sss,
                                      my_id_list_cv = ID_CV_LIST,
                                      my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                      my_spline_df = PPR_DF_SM,
                                      my_metrics_names = METRICS_NAMES,
                                      my_weights = MY_WEIGHTS_sss,
                                      my_metrics_functions = MY_USED_METRICS,
                                      my_ncores = N_CORES,
                                      use_only_first_fold = TRUE)



# 2) final model -------

ppr_best_params = PPRExtractBestParams(ppr_metrics)

print("ppr best params")
ppr_best_params

ppr_model = ppr(y ~ .,
               data = sss,
               nterms = ppr_best_params[[METRIC_CHOSEN_NAME]][["n_ridge_functions"]],
               sm.method = "spline",
               df = ppr_best_params[[METRIC_CHOSEN_NAME]][["spline_df"]]) 

df_metrics = Add_Test_Metric(df_metrics,
                              "PPR",
                              USED.Metrics(predict(ppr_model, vvv),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))

df_metrics

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
m_max = NCOL(sss) - 1 # sottraggo 1 per la variabile risposta

# se m_max è grande eventualmente ridurlo per considerazioni computazionali


# regolazione
# procedura sub-ottimale, ma la impiego per ragioni computazionali
# prima scelgo il numero di esplicative a ogni split,
# una volta scelto controllo la convergenza dell'errore basata sul numero di alberi
err = rep(NA, m_max - 1)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°

for(i in seq(2, m_max)){
  sfExport(list = c("i"))
  
  err[i] = sum(sfSapply(rep(1:8),
                        function(x) ranger(y ~., data = sss,
                                           mtry = i,
                                           num.trees = 50,
                                           oob.error = TRUE)$prediction.error))
  print(i)
  gc()
}

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
                                 function(x) ranger(y ~., sss,
                                                    mtry = best_mtry,
                                                    num.trees = j,
                                                    oob.error = TRUE)$prediction.error))
  print(j)
  gc()
}

sfStop()

plot((1:length(err_rf_trees)) * 4, err_rf_trees,
     xlab = "numero di alberi bootstrap",
     ylab = "errore out of bag",
     pch = 16,
     main = "Random Forest")

# ATTENZIONE: cambiare 
# best_mtry = 2

# modello finale e previsioni
random_forest_model = ranger(y ~., sss,
                             mtry = best_mtry,
                             num.trees = 400,
                             oob.error = TRUE,
                             importance = "permutation")

df_metrics = Add_Test_Metric(df_metrics,
                              "Random Forest",
                              USED.Metrics(predict(random_forest_model, data = vvv,
                                                type = "response")$predictions,
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))

df_metrics

# Importanza delle variabili
vimp = importance(random_forest_model)

PlotAndSave(my_plotting_function =  function() dotchart(vimp[order(vimp)]),
            my_path_plot = my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
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
# Il bagging tiene conto del compromesso 
# varianza distorsione tramite errore out of bag (bootstrap)

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
                                 function(x) bagging(y ~., sss,
                                                    nbag = j,
                                                    coob = TRUE)$err))
  print(j)
  gc()
}

sfStop()

plot((1:length(err_bg_trees)) * 4, err_bg_trees,
     xlab = "numero di alberi bootstrap",
     ylab = "errore out of bag",
     pch = 16,
     main = "Bagging")

# se il numero di replicazioni bootstrap arriva a convergenza allora

bagging_model = bagging(y ~., sss, nbag = 400, coob = FALSE)

df_metrics = Add_Test_Metric(df_metrics,
                              "Bagging",
                              USED.Metrics(predict(bagging_model, newdata = vvv),
                                           vvv$y,
                                           weights = MY_WEIGHTS_vvv))


df_metrics

# save metrics and model
file_name_bagging = paste(MODELS_FOLDER_RELATIVE_PATH,
                                 "bagging",
                                 ".Rdata", collapse = "", sep = "")

save(bagging_model, file = file_name_bagging)


rm(bagging_model)
gc()

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

cbind(df_metrics[,1],
      apply(df_metrics[,2:NCOL(df_metrics)], 2, function(col) round(as.numeric(col), 2)))
# Comparazione modelli
# selezione modelli migliori e commenti su questi:
# es -> vanno meglio modelli con interazioni oppure modelli additivi

# Linear Step ---------------

load(MODELS_FOLDER_RELATIVE_PATH,
      "lm_step_no_interaction",
      ".Rdata", collapse = "", sep = "")

load(MODELS_FOLDER_RELATIVE_PATH,
      "lm_step_yes_interaction",
      ".Rdata", collapse = "", sep = "")

# Ridge - Lasso ----------------

load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "ridge_no_interaction",
           ".Rdata", collapse = "", sep = ""))

load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "ridge_yes_interaction",
           ".Rdata", collapse = "", sep = ""))

load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "lasso_no_interaction",
           ".Rdata", collapse = "", sep = ""))

load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "lasso_yes_interaction",
           ".Rdata", collapse = "", sep = ""))

# Tree -----------------
load(MODELS_FOLDER_RELATIVE_PATH,
      "final_tree_pruned",
      ".Rdata", collapse = "", sep = "")

# Gam ------------------
load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "gam_step",
           ".Rdata", collapse = "", sep = ""))

# MARS -----------------
load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "mars_step",
           ".Rdata", collapse = "", sep = ""))



# Ridge 
# Lasso

# guarda coefficienti
# predict.glmnet(oggetto, type = "coefficients")


# Modello additivo : lo devo ri-stimare nel caso
# plot(gam_step, terms = c("s(Durata, df = 4)"), se = T)

# summary(gam_step)

# MARS:
# mars_step$model

# mars_step_pred_names_matrix

# plot(mars_step, predictor1 = 40, predictor2 = 30)





