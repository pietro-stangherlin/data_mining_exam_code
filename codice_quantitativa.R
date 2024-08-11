# data wrangling
library(dplyr)

# parallel computing
library(snowfall)

# number of cores
N_CORES = parallel::detectCores()


#////////////////////////////////////////////////////////////////////////////
# Costruzione metrica di valutazione e relativo dataframe -------------------
#////////////////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Quantitativa -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

source("loss_functions.R")

# °°°°°°°°°°°°°°°°°°°°°°° Warning: °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# cambia la funzione di errore per il problema specifico

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
#------------------------ Stima e Verifica ------------------------
# /////////////////////////////////////////////////////////////////


# Eventualmente modificare la proporzione
id_stima = sample(1:NROW(dati), 0.75 * NROW(dati))

sss = dati[id_stima,]
vvv = dati[-id_stima,]


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Selezione parametri - Convalida sulla stima  -------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# In caso di convalida nell'insieme di stima
id_cb1 = sample(1:NROW(sss), 0.8 * NROW(sss))

# rimozione dati originali
rm(dati)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Selezione parametri Costruzione ID Fold convalida incrociata  -------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# selezione parametri tramite cv

# numero fold
K_FOLDS = 10

NROW_sss = NROW(sss)

# matrice degli id dei fold della convalida incrociata
# NOTA: data la non garantita perfetta divisibilità del numero di osservazioni
# per il numero di fold è possibile che un fold abbia meno osservazioni degli altri

# ordine casuale degli id
SHUFFLED_ID = sample(1:NROW_sss, NROW_sss)

id_matrix_cv = matrix(SHUFFLED_ID, ncol = K_FOLDS)

# converto la matrice in lista per poter avere degli elementi
# (vettori) con un diverso numero di osservazioni
# ogni colonna diventa un elemento della lista

ID_CV_LIST = list()

for(j in 1:ncol(id_matrix_cv)){
  ID_CV_LIST[[j]] = id_matrix_cv[,j]
}

rm(id_matrix_cv)
gc()

# se ottengo Warning: non divisibilità perfetta
# significa che l'ultimo elemento lista contiene 
# degli id che sono presenti anche nel primo elemento
# sistemo eliminando dall'ultimo elemento della lista gli id presenti anche nel primo elemento

# controllo il resto della divisione
integer_division_cv = NROW_sss %/% K_FOLDS
modulo_cv = NROW_sss %% K_FOLDS

if(modulo_cv != 0){
  ID_CV_LIST[[K_FOLDS]] = ID_CV_LIST[[K_FOLDS]][1:integer_division_cv]
}

source("cv_functions.R")


# /////////////////////////////////////////////////////////////////
#------------------------ Analisi esplorative ---------------------
# /////////////////////////////////////////////////////////////////

# Analisi esplorativa sulla stima 
# eventuali inflazioni di zeri

hist(sss$y,nclass = 100)
summary(sss$y)

# possiamo provare a trasformare la risposta
# ATTENZIONE se y è <= 0 -> trasforma in modo adeguato
hist(log(sss$y), nclass = 100)

# anche se le distribuzioni marginali non 
# forniscono informazioni riguardo alle condizionate
# se per il problema in questione è sensato possiamo impiegare 
# come nuova rispota il logaritmo della precedente y

# nota: in in questo modo la differenza dei logaritmi corrisponde
# al logaritmo del rapporto



# /////////////////////////////////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Media e Mediana --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# aggiunta media e mediana della risposta sull'insieme di stima come possibili modelli
# (per valutare se modelli più complessi hanno senso)

df_metrics = Add_Test_Metric(df_metrics,
                              "sss mean",
                              USED.Metrics(mean(sss$y), vvv$y))

df_metrics = Add_Test_Metric(df_metrics,
                              "sss median",
                              USED.Metrics(median(sss$y), vvv$y))

df_metrics = na.omit(df_metrics)

df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello lineare Forward --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# AIC criterion is used for model selection

lm0 = lm(y ~ 1, data = sss)

# NO Interaction -----------
# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
lm_step_no_interaction = step(lm0, scope = formula_no_interaction_yes_intercept,
                 direction = "forward")

formula(lm_step_no_interaction)

# load(file_name_lm_step_no_interaction)

df_metrics = Add_Test_Metric(df_metrics,
                              "lm_step_no_interaction",
                              USED.Metrics(predict(lm_step_no_interaction, newdata = vvv), vvv$y))
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

# computazionalmente costoso (probabilmente)
lm_step_yes_interaction = step(lm0, scope = formula_yes_interaction_yes_intercept,
                               direction = "forward")

formula(lm_step_yes_interaction)


df_metrics = Add_Test_Metric(df_metrics,
                              "lm_step_yes_interaction",
                              USED.Metrics(predict(lm_step_yes_interaction, newdata = vvv), vvv$y))

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
# Ridge e Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Compromesso varianza - distorsione: convalida incrociata sull'insieme di stima

# valuta: se ci sono molte esplicative qualitative -> model.matrix con molti zeri
library(Matrix)
X_mm_no_interaction_sss =  sparse.model.matrix(formula_no_interaction_no_intercept, data = sss)
X_mm_no_interaction_vvv =  sparse.model.matrix(formula_no_interaction_no_intercept, data = vvv)

# # oneroso
X_mm_yes_interaction_sss =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = sss)
X_mm_yes_interaction_vvv =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = vvv)

# default
# stima 
# X_mm_no_interaction_sss = model.matrix(formula_no_interaction_no_intercept, data = sss)
# X_mm_no_interaction_vvv = model.matrix(formula_no_interaction_no_intercept, data = vvv)

# Interazioni: stima 
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
                                              my_weights = MY_WEIGHTS)

# ridge_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
#                                                       my_metric_names = METRICS_NAMES,
#                                                       my_x = X_mm_no_interaction_sss,
#                                                       my_y = sss$y,
#                                                       my_alpha = 0,
#                                                       my_lambda_vals = lambda_vals,
#                                                       my_weights = MY_WEIGHTS,
#                                                       my_metrics_functions = MY_USED_METRICS,
#                                                       my_ncores = N_CORES)

ridge_no_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = ridge_no_interaction_metrics[["metrics"]],
                                         my_one_se_best = TRUE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = ridge_no_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES)

temp_plot_function = function(){
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = ridge_no_interaction_metrics[["metrics"]],
                my_se_matrix = ridge_no_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(ridge_no_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge no interaction CV metrics",
                my_xlab = " log lambda")
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
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

# previsione ed errore
df_metrics = Add_Test_Metric(df_metrics,
                              "ridge_no_interaction",
                              USED.Metrics(predict(ridge_no_interaction, newx = X_mm_no_interaction_vvv),vvv$y))

df_metrics

file_name_ridge_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                          "ridge_no_interaction",
                                          ".Rdata", collapse = "", sep = "")

save(ridge_no_interaction, file = file_name_ridge_no_interaction)

# elimino dalla memoria l'oggetto ridge_cv
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
#                                               my_weights = MY_WEIGHTS)

ridge_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
                                                      my_metric_names = METRICS_NAMES,
                                                      my_x = X_mm_yes_interaction_sss,
                                                      my_y = sss$y,
                                                      my_alpha = 0,
                                                      my_lambda_vals = lambda_vals,
                                                      my_weights = MY_WEIGHTS,
                                                      my_metrics_functions = MY_USED_METRICS,
                                                      my_ncores = N_CORES)

ridge_yes_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                                         my_one_se_best = TRUE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = ridge_yes_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES)

temp_plot_function = function(){
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                my_se_matrix = ridge_yes_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(ridge_yes_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge yes interaction CV metrics",
                my_xlab = " log lambda")
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
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
                             USED.Metrics(predict(ridge_yes_interaction, newx = X_mm_yes_interaction_vvv),vvv$y))

df_metrics

file_name_ridge_yes_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                       "ridge_yes_interaction",
                                       ".Rdata", collapse = "", sep = "")

save(ridge_yes_interaction, file = file_name_ridge_yes_interaction)

# elimino dalla memoria l'oggetto ridge_cv
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
                                              my_weights = MY_WEIGHTS)

# lasso_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
#                                                       my_metric_names = METRICS_NAMES,
#                                                       my_x = X_mm_no_interaction_sss,
#                                                       my_y = sss$y,
#                                                       my_alpha = 1,
#                                                       my_lambda_vals = lambda_vals,
#                                                       my_weights = MY_WEIGHTS,
#                                                       my_metrics_functions = MY_USED_METRICS,
#                                                       my_ncores = N_CORES)

lasso_no_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                                         my_one_se_best = TRUE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = lasso_no_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES)

temp_plot_function = function(){
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                my_se_matrix = lasso_no_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(lasso_no_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso no interaction CV metrics",
                my_xlab = " log lambda")
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
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

# previsione ed errore
df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_no_interaction",
                             USED.Metrics(predict(lasso_no_interaction, newx = X_mm_no_interaction_vvv),vvv$y))

df_metrics

file_name_lasso_no_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                       "lasso_no_interaction",
                                       ".Rdata", collapse = "", sep = "")

save(lasso_no_interaction, file = file_name_lasso_no_interaction)

# elimino dalla memoria l'oggetto lasso_cv
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
#                                                my_weights = MY_WEIGHTS)

lasso_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
                                                      my_metric_names = METRICS_NAMES,
                                                      my_x = X_mm_yes_interaction_sss,
                                                      my_y = sss$y,
                                                      my_alpha = 1,
                                                      my_lambda_vals = lambda_vals,
                                                      my_weights = MY_WEIGHTS,
                                                      my_metrics_functions = MY_USED_METRICS,
                                                      my_ncores = N_CORES)

lasso_yes_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                          my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                                          my_one_se_best = TRUE,
                                          my_higher_more_complex = FALSE,
                                          my_se_matrix = lasso_yes_interaction_metrics[["se"]],
                                          my_metric_names = METRICS_NAMES)

temp_plot_function = function(){
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                my_se_matrix = lasso_yes_interaction_metrics[["se"]],
                my_best_param_values =log(ExtractBestParams(lasso_yes_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso yes interaction CV metrics",
                my_xlab = " log lambda")
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
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

# previsione ed errore
df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_yes_interaction",
                             USED.Metrics(predict(lasso_yes_interaction, newx = X_mm_yes_interaction_vvv),vvv$y))

df_metrics

file_name_lasso_yes_interaction = paste(MODELS_FOLDER_RELATIVE_PATH,
                                        "lasso_yes_interaction",
                                        ".Rdata", collapse = "", sep = "")

save(lasso_yes_interaction, file = file_name_lasso_yes_interaction)

# elimiyes dalla memoria l'oggetto lasso_cv
rm(lasso_yes_interaction)
gc()



rm(X_mm_yes_interaction_sss)
rm(X_mm_yes_interaction_vvv)
gc()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Albero -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

# Selezione del modello

# 0) Full tree which will be pruned ------

# default: molto fitto
tree_full = tree(y ~.,
                 data = sss[id_cb1,],
                 control = tree.control(nobs = length(id_cb1),
                                        mindev = 1e-04,
                                        minsize = 5))



# controllo che sia sovraadattato
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
                                       my_weights = MY_WEIGHTS,
                                       my_mindev = 1e-04,
                                       my_minsize = 5)

tree_best_summary = CvMetricBest(my_param_values = 2:TREE_MAX_SIZE,
                                 my_metric_matrix = tree_cv_metrics[["metrics"]],
                                 my_one_se_best = TRUE,
                                 my_higher_more_complex = TRUE,
                                 my_se_matrix = tree_cv_metrics[["se"]],
                                 my_metric_names = METRICS_NAMES)


temp_plot_function = function(){
  PlotCvMetrics(my_param_values = 2:TREE_MAX_SIZE,
                my_metric_matrix = tree_cv_metrics[["metrics"]],
                my_se_matrix = tree_cv_metrics[["se"]],
                my_best_param_values = ExtractBestParams(tree_best_summary),
                my_metric_names = METRICS_NAMES,
                my_main = "Tree CV metrics",
                my_xlab = "size")
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                     "tree_cv_metrics_plot.jpeg",
                                                     collapse = ""))


tree_best_size = tree_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]]

print("tree best size")
tree_best_size


# 1.b) Size: train - test -----------

# potatura
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
                             USED.Metrics(predict(final_tree_pruned, newdata = vvv), vvv$y))


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

# Controllo del compromesso varianza distorsione: 
# selezione step tramite gradi di libertà equivalenti

# stepwise forward: AIC based on generalized df
gam0 = gam(y ~ 1, data = sss)

# riconosce le qualitative se sono fattori
my_gam_scope = gam.scope(sss[,-y_index], arg = c("df=2", "df=3", "df=4", "df=5", "df=6"))

# prova anche parallelo
# require(doMC)
# registerDoMC(cores= N_CORES)
# step.Gam(gam0, my_gam_scope, parallel=TRUE)

gam_step = step.Gam(gam0, scope = my_gam_scope)

# salvo il modello finale
# y ~ x2 + x3 + x7 + s(x8, df = 2)

# gam_step = gam(y ~ Sottocategoria + s(Obiettivo, df = 4) + s(Durata, df = 4) + Anno,
#                 data = sss)

df_metrics = Add_Test_Metric(df_metrics,
                              "gam_step",
                              USED.Metrics(predict(gam_step, newdata = vvv), vvv$y))

df_metrics

file_name_gam_step = paste(MODELS_FOLDER_RELATIVE_PATH,
                                    "gam_step",
                                    ".Rdata", collapse = "", sep = "")

save(gam_step, file = file_name_gam_step)



rm(gam_step)
gc()

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MARS ---------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# devo ottenere gli indici delle colonne
# delle variabili qualitative della matrice del disegno (senza intercetta)
# poichè trasformando il data.frame in matrice del modello 
# i nomi delle variabili quantitative rimangono invariati
# selezioniamo prima quest'ultime 
num_index = which(colnames(X_mm_no_interaction_sss) %in% var_num_names)
factor_index = setdiff(1:NCOL(X_mm_no_interaction_sss), num_index)


library(polspline)

# Controllo del compromesso varianza distorsione: 
# criterio della convalida incrociata generalizzata

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
         legend = c("crescita", "potatura"),
         col = c("black","red"),
         pch = 16)
  
  
  abline(v = min_size_mars)
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                     "mars_gcv_plot.jpeg",
                                                     collapse = ""))



df_metrics = Add_Test_Metric(df_metrics,
                              "MARS",
                              USED.Metrics(predict(mars_step, x = X_mm_no_interaction_vvv),vvv$y))

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

# Scelgo il parametro di regolazione: numero di funzioni dorsali tramite
# stima convalida sul sottoinsieme di stima

# numero di possibili funzioni dorsali
PPR_MAX_RIDGE_FUNCTIONS = 4

# numero di possibili gradi di libertà (equivalenti) delle smoothing splines
PPR_E_DF_SM = 2:6

# 1.a) Regulation: train - test ---------

metrics_ppr_array = array(NA,
                          dim = c(PPR_MAX_RIDGE_FUNCTIONS, length(PPR_E_DF_SM), N_METRICS),
                          dimnames = list(1:PPR_MAX_RIDGE_FUNCTIONS,
                                          PPR_E_DF_SM,
                                          METRICS_NAMES))

for(r in 1:PPR_MAX_RIDGE_FUNCTIONS){
  for(df in 1: length(PPR_E_DF_SM)){
    mod = ppr(y ~ .,
              data = sss[id_cb1,],
              nterms = r,
              sm.method = "spline",
              df = PPR_E_DF_SM[df])
    
    metrics_ppr_array[r, df, ] = USED.Metrics(predict(mod, sss[-id_cb1,]),
                                           sss$y[-id_cb1],
                                           weights = MY_WEIGHTS)
  }
  print(r)
}

rm(mod)
gc()

# 1.b) Regulation: CV -------


# 2) final model -------

mod_ppr1 = ppr(y ~ .,
               data = sss,
               nterms = ppr_best_n_ridges,
               sm.method = "spline",
               df = 2) 

df_metrics = Add_Test_Metric(df_metrics,
                              "PPR",
                              USED.Metrics(predict(mod_ppr1, vvv), vvv$y))

df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Nota: se manca il tempo eseguo prima la RandomForest del Bagging
# visto che quest'ultimo è un sotto caso particolare 
# della RandomForest (selezione di tutte le variabili per ogni split)


# Implementazione in parallelo
library(ranger)

library(snowfall)

parallel::detectCores() # quanti core a disposizione?

sfLibrary(ranger)


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
  
  err[i] = sum(sfSapply(rep(1:4),
                        function(x) ranger(y ~., data = sss,
                                           mtry = i,
                                           num.trees = 50,
                                           oob.error = TRUE)$prediction.error))
  print(i)
  gc()
}

err 

best_mtry = which.min(err)
best_mtry
# 2

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
                                                type = "response")$predictions, vvv$y))

df_metrics

# Importanza delle variabili
vimp = importance(random_forest_model)

dotchart(vimp[order(vimp)])

rm(random_forest_model)
gc()



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Bagging ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Il bagging tiene conto del compromesso 
# varianza distorsione tramite errore out of bag (bootstrap)

library(ipred)
sfInit(cpus = 4, parallel = T)
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
                              USED.Metrics(predict(bagging_model, newdata = vvv), vvv$y))


df_metrics

rm(bagging_model)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rete neurale -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Solo in parallelo altrimenti ci mette troppo tempo


decay = 10^seq(-3, -1, length=10)
nodi = 1:10

hyp_grid = expand.grid(decay,nodi)

# Costruiamo una funzione che prenda come input una matrice parametri,
# stimi la rete per ogni valore, e restiuisca una matrice con valori dei parametri + errori su convalida
regola_nn = function(pars, sss, id_cb1){
  err = data.frame(pars, err = NA)
  for(i in 1:NROW(pars)){
    n1 = nnet(y ~ . , data=sss[id_cb1,], 
              size=pars[i,2], decay=pars[i,1],
              MaxNWts = 1500, maxit = 500, 
              trace = T)
    err$err[i] = MSE.Loss(predict(n1, sss[-id_cb1,], type = 'raw'), sss$y[-id_cb1])
  }
  return(err)
}

# proviamo
regola_nn(hyp_grid[21:23,], sss, id_cb1)


# Parallelo
# Per mitigare il load balance possiamo assegnare a caso ai vari processori
# In questo modo ogni processore avra' sia valori di parametri "semplici" (pochi nodi)
# Che complessi (tanti nodi)

# Conviene creare una lista in cui ogni elemento sia la matrice di parametri provati
# da quel processore

pp = sample(rep(1:4, each = NROW(hyp_grid)/4))
pars_list = lapply(1:4, function(l) hyp_grid[pp == l,])


library(snowfall)
sfInit(cpus = 4, parallel = T)
sfLibrary(nnet) # carichiamo la libreria
sfExport(list = c("sss", "id_cb1", "regola_nn", "MSE.Loss")) # esportiamo tutte le quantita' necessarie

# Non restituisce messaggi, possiamo solo aspettare
nn_error = sfLapply(pars_list, function(x) regola_nn(x, sss, id_cb1))
sfStop()

err_nn = do.call(rbind, nn_error)

par(mfrow = c(1,2))
plot(err_nn$Var1, err_nn$err, xlab = "Weight decay", ylab = "Errore di convalida", pch = 16)
plot(err_nn$Var2, err_nn$err, xlab = "Numero di nodi", ylab = "Errore di convalida", pch = 16)

err_nn[which.min(err_nn$err),]

# 0.03593814    4 0.5818981 (ovviamente potrebbe variare a seconda di: punti iniziali, stima/convalida, etc

set.seed(123)
mod_nn = nnet(diagnosi ~ . , data=sss[,], size = 4, decay = 0.03593,
              MaxNWts = 1500, maxit = 2000, trace = T)

pr_nn = predict(mod_nn, vvv, type = "class")

# /////////////////////////////////////////////////////////////////
#------------------------ Sintesi Finale -------------------------
# /////////////////////////////////////////////////////////////////


cbind(df_metrics[,1],
      apply(df_metrics[,2:NCOL(df_metrics)], 2, function(col) round(as.numeric(col), 2)))
# Comparazione modelli
# selezione modelli migliori e commenti su questi:
# es -> vanno meglio modelli con interazioni oppure modelli additivi



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

# per il grafico lo devo ristimare
# plot(mars_step, predictor1 = 40, predictor2 = 30)

# Random Forest: guarda grafico importanza variabili



