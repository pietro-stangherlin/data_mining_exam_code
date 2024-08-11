library(dplyr)

# parallel
library(snowfall)


# Descrizione -----------------------
# pochi dati: usiamo convalida incrociata come criterio per confrontare i modelli finali
# partizioniamo i dati in K insiemi (FOLD) e usiamo la procedura di convalida incrociata
# sia per identificare i valori ottimali dei parametri (rispetto all'errore di previsione) 
# che per confrontare i migliori modelli scelti (per tutti i modelli i FOLD di convalida sono gli stessi).

# NOTA: questa procedura può risultare troppo ottimista in quanto il confronto tra i modelli migliori
# avviene rispetto all'errore in corrispondenza del parametro selezionato:
# 1) Tramite CV seleziono il parametro ottimale: quello che minimizza l'errore medio di convalida
# 2) Per ogni modello finale così selezionato confronto tali errori medi convalida
# ma è un compromesso data la scarsità dei dati

#////////////////////////////////////////////////////////////////////////////
# Costruzione metrica di valutazione e relativo dataframe -------------------
#////////////////////////////////////////////////////////////////////////////

source("loss_functions.R")

# in generale uso sia MAE che MSE
USED.Metrics = function(y.pred, y.test, weights = 1){
  return(c(MAE.Loss(y.pred, y.test, weights), MSE.Loss(y.pred, y.test, weights)))
}


# anche qua
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


#////////////////////////////////////////////////////////////////////////////
# Costruzione ID Fold convalida incrociata  -------------------
#////////////////////////////////////////////////////////////////////////////

# numero fold
K_FOLDS = 10

NROW_DF = NROW(dati)

# matrice degli id dei fold della convalida incrociata
# NOTA: data la non garantita perfetta divisibilità del numero di osservazioni
# per il numero di fold è possibile che un fold abbia meno osservazioni degli altri

# ordine casuale degli id
SHUFFLED_ID = sample(1:NROW_DF, NROW_DF)

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
integer_division_cv = NROW_DF %/% K_FOLDS
modulo_cv = NROW_DF %% K_FOLDS

if(modulo_cv != 0){
  ID_CV_LIST[[K_FOLDS]] = ID_CV_LIST[[K_FOLDS]][1:integer_division_cv]
}

# /////////////////////////////////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Media e Mediana --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# aggiunta media e mediana della risposta sull'insieme di stima come possibili modelli
# (per valutare se modelli più complessi hanno senso)

# in questo caso non ci sono parametri: solo fold (righe) e metriche (2: MSE e MAE)

# media 
temp_err_matrix_cv = matrix(NA, nrow = K_FOLDS, ncol = N_METRICS)
colnames(temp_err_matrix_cv) = colnames(df_metrics[,-1])


for (i in 1:K_FOLDS){
  temp_err_matrix_cv[i,] = USED.Metrics(mean(dati$y[unlist(ID_CV_LIST[-i])]),
                                     dati$y[ID_CV_LIST[[i]]])
}



df_metrics = Add_Test_Metric(df_metrics,
                              "cv mean",
                              colMeans(temp_err_matrix_cv))


# mediana
temp_err_matrix_cv = matrix(NA, nrow = K_FOLDS, ncol = N_METRICS)


for (i in 1:K_FOLDS){
  temp_err_matrix_cv[i,] = USED.Metrics(median(dati$y[unlist(ID_CV_LIST[-i])]),
                                     dati$y[ID_CV_LIST[[i]]])
}



df_metrics = Add_Test_Metric(df_metrics,
                              "cv median",
                              colMeans(temp_err_matrix_cv))

df_metrics = na.omit(df_metrics)

df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge e Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(glmnet)
library(Matrix)

source("cv_functions.R")

# Compromesso varianza - distorsione: convalida incrociata sia per la scelta del parametro 
# di regolazione che per il confronto finale
# NOTA: questa procedura è sub-ottimale, poichè non ci stiamo totalmente tutelando 
# contro il sovraadattamento, tuttavia, data la scarsa numerosità campionaria 
# non sono presenti valide alternative


# in questo caso dobbiamo creare una griglia di valori di lambda 
# creiamo la griglia adattando il modello su tutti i dati

# NO interazione 

# X_mm_no_interaction = model.matrix(formula_no_interaction_no_intercept, data = dati)
# sparsa
X_mm_no_interaction =  sparse.model.matrix(formula_no_interaction_no_intercept, data = dati)

# SI interazione

# X_mm_yes_interaction = model.matrix(formula_yes_interaction_no_intercept, data = dati)
#sparsa
X_mm_yes_interaction =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = dati)


# eventualmente basato sui risultati successivi
# lambda_vals = seq(1e-07,1e-03, by = 1e-05)

# Ridge ----------------

# NO interaction 
lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

ridge_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction,
                                              my_y = dati$y,
                                              my_alpha = 0,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS)

# ridge_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
#                                               my_metric_names = METRICS_NAMES,
#                                               my_x = X_mm_no_interaction,
#                                               my_y = dati$y,
#                                               my_alpha = 0,
#                                               my_lambda_vals = lambda_vals,
#                                               my_weights = MY_WEIGHTS,
#                                               my_metrics_functions = MY_USED_METRICS,
#                                               my_ncores = N_CORES)

ridge_no_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                            my_metric_matrix = ridge_no_interaction_metrics[["metrics"]],
                                            my_one_se_best = TRUE,
                                            my_higher_more_complex = FALSE,
                                            my_se_matrix = ridge_no_interaction_metrics[["se"]],
                                            my_metric_names = METRICS_NAMES)

temp_plot_function = function(){
  PlotCvMetrics(my_param_values = lambda_vals,
                my_metric_matrix = ridge_no_interaction_metrics[["metrics"]],
                my_se_matrix = ridge_no_interaction_metrics[["se"]],
                my_best_param_values = ExtractBestParams(ridge_no_int_best_summary),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge no interaction CV metrics",
                my_xlab = "lambda",
                my_xlim = c(0, 0.1))
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                     "ridge_no_int_metrics_plot.jpeg",
                                                     collapse = ""))

print("ridge_no_int_best_summary")
ridge_no_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                              "ridge_no_int",
                              ridge_no_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# YES interaction 
lambda_vals = glmnet(x = X_mm_yes_interaction, y = dati$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

# ridge_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv = ID_CV_LIST,
#                                               my_metric_names = METRICS_NAMES,
#                                               my_x = X_mm_yes_interaction,
#                                               my_y = dati$y,
#                                               my_alpha = 0,
#                                               my_lambda_vals = lambda_vals,
#                                               my_weights = MY_WEIGHTS)

ridge_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
                                                      my_metric_names = METRICS_NAMES,
                                                      my_x = X_mm_yes_interaction,
                                                      my_y = dati$y,
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
  PlotCvMetrics(my_param_values = lambda_vals,
                my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                my_se_matrix = ridge_yes_interaction_metrics[["se"]],
                my_best_param_values = ExtractBestParams(ridge_yes_int_best_summary),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge yes interaction CV metrics",
                my_xlab = "lambda",
                my_xlim = c(0, 0.1))
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                     "ridge_yes_int_metrics_plot.jpeg",
                                                     collapse = ""))

print("ridge_yes_int_best_summary")
ridge_yes_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "ridge_yes_int",
                             ridge_yes_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics




# Lasso ---------------
# NO interaction 
lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

# lasso_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv = ID_CV_LIST,
#                                               my_metric_names = METRICS_NAMES,
#                                               my_x = X_mm_no_interaction,
#                                               my_y = dati$y,
#                                               my_alpha = 1,
#                                               my_lambda_vals = lambda_vals,
#                                               my_weights = MY_WEIGHTS)

lasso_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
                                                      my_metric_names = METRICS_NAMES,
                                                      my_x = X_mm_no_interaction,
                                                      my_y = dati$y,
                                                      my_alpha = 1,
                                                      my_lambda_vals = lambda_vals,
                                                      my_weights = MY_WEIGHTS,
                                                      my_metrics_functions = MY_USED_METRICS,
                                                      my_ncores = N_CORES)

lasso_no_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                                         my_one_se_best = TRUE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = lasso_no_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES)

temp_plot_function = function(){
  PlotCvMetrics(my_param_values = lambda_vals,
                my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                my_se_matrix = lasso_no_interaction_metrics[["se"]],
                my_best_param_values = ExtractBestParams(lasso_no_int_best_summary),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso no interaction CV metrics",
                my_xlab = "lambda")
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                     "lasso_no_int_metrics_plot.jpeg",
                                                     collapse = ""))

print("lasso_no_int_best_summary")
lasso_no_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_no_int",
                             lasso_no_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# YES interaction 
lambda_vals = glmnet(x = X_mm_yes_interaction, y = dati$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

# lasso_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv = ID_CV_LIST,
#                                               my_metric_names = METRICS_NAMES,
#                                               my_x = X_mm_yes_interaction,
#                                               my_y = dati$y,
#                                               my_alpha = 1,
#                                               my_lambda_vals = lambda_vals,
#                                               my_weights = MY_WEIGHTS)

lasso_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv = ID_CV_LIST,
                                                       my_metric_names = METRICS_NAMES,
                                                       my_x = X_mm_yes_interaction,
                                                       my_y = dati$y,
                                                       my_alpha = 1,
                                                       my_lambda_vals = lambda_vals,
                                                       my_weights = MY_WEIGHTS,
                                                       my_metrics_functions = MY_USED_METRICS,
                                                       my_ncores = N_CORES)
temp_plot_function = function(){
  PlotCvMetrics(my_param_values = lambda_vals,
                my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                my_se_matrix = lasso_yes_interaction_metrics[["se"]],
                my_best_param_values = ExtractBestParams(lasso_no_int_best_summary),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso yes interaction CV metrics",
                my_xlab = "lambda")
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                     "lasso_yes_int_metrics_plot.jpeg",
                                                     collapse = ""))

print("lasso_yes_int_best_summary")
lasso_no_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_yes_int",
                             lasso_yes_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Tree -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

TREE_MAX_SIZE = 50


# if parallel shows problems use the non parallel version
tree_cv_metrics = ManualCvTreeParallel(my_id_list_cv = ID_CV_LIST,
                                       my_metric_names = METRICS_NAMES,
                                       my_data = dati,
                                       my_max_size = TREE_MAX_SIZE,
                                       my_metrics_functions = MY_USED_METRICS,
                                       my_ncores = N_CORES,
                                       my_weights = MY_WEIGHTS,
                                       my_mindev = 1e-05,
                                       my_minsize = 2)

tree_best_summary = CvMetricBest(my_param_values = 2:TREE_MAX_SIZE,
                                    my_metric_matrix = tree_cv_metrics[["metrics"]],
                                    my_se_matrix = tree_cv_metrics[["se"]],
                                    my_metric_names = METRICS_NAMES,
                                    my_main = "Tree CV metrics",
                                    my_xlab = "Size",
                                    my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                         "tree_metrics_plot.jpeg"))

tree_best_summary


df_metrics = Add_Test_Metric(df_metrics,
                              "tree",
                             tree_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello Additivo ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(gam)

# TO DO -----------

# Here the model selection is harder vs other models, except for MARS.
# We give a brief description:
# In additive models, under the non - interaction constraint 
# (i.e. we do not consider variables which are a functions of two or more distinct variables),
# we have two possible regulation parameters
# 1) how many predictors (of course we also need to know what specific predictors)
# 2) for each quantitative predictor in the model what is its smoothing parameter?

# Since an exaustive search over all possibilities would require near infinite time and resources
# we adopt this sub-optimal procedure:
# for each training fold a model is selected based on a stepwise AIC selection 
# (based on generalized degrees of freedom), its error on its cv test fold is computed (as usual);
# The procedure is then repetead for all the folds and a 

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MARS ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# TO DO -----------


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PPR ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# K: numero di possibili funzioni dorsali
PPR_MAX_RIDGE_FUNCTIONS = 4

ppr_cv_metrics = ManualCvPPR(n_k_fold = K_FOLDS,
                                     my_id_list_cv = ID_CV_LIST,
                                     my_n_metrics = N_METRICS,
                                     my_metric_names = METRICS_NAMES,
                                     my_data = dati,
                                     my_max_ridges = PPR_MAX_RIDGE_FUNCTIONS)

ppr_best_summary = CvMetricBest(my_param_values = 1:PPR_MAX_RIDGE_FUNCTIONS,
                                    my_metric_matrix = ppr_cv_metrics[["metrics"]],
                                    my_se_matrix =  ppr_cv_metrics[["se"]],
                                    my_metric_names = METRICS_NAMES,
                                    my_main = "PPR CV metrics",
                                    my_xlab = "Number of ridge functions",
                                    my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                         "ppr_metrics_plot.jpeg"))

ppr_best_summary


df_metrics = Add_Test_Metric(df_metrics,
                             "PPR",
                             ppr_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# sub - optimal procedure: 
# 1) fix the number of bootstrap trees
# 2) using CV find the optimal number of variables chosen at each split
# 3) using the optimal number check if the CV error stabilizes with respect to
# the number of bootstrap trees
# eventually repeat until stabilization

# NOTE: here we're NOT using OOB errors because the results need to be confronted
# with other models fitted and tested on the same folds.

# max variables at each split (can be changed)
RF_MAX_VARIABLES = 30

# number of bootstrap trees (can be changed)
RF_N_BS_TREES = 200

library(randomForest)

# number of split variable selection

rf_cv_metrics = ManualCvRFParallel(n_k_fold = K_FOLDS,
                                 my_id_list_cv = ID_CV_LIST,
                                 my_n_metrics = N_METRICS,
                                 my_metric_names = METRICS_NAMES,
                                 my_data = dati,
                                 my_n_variables = 1:RF_MAX_VARIABLES,
                                 my_n_bs_trees = 400,
                                 fix_trees_bool = TRUE,
                                 my_ncores = N_CORES)

rf_best_summary = CvMetricBest(my_param_values = 1:RF_MAX_VARIABLES,
                                   my_metric_matrix = rf_cv_metrics[["metrics"]],
                                   my_se_matrix = rf_cv_metrics[["se"]],
                                   my_metric_names = METRICS_NAMES,
                                   my_main = "RF CV metrics",
                                   my_xlab = "Number of variables at each split",
                                   my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                        "rf_metrics_plot.jpeg"))

rf_best_summary

# check convergence with respect to number of bootstrap trees

# sequence of bootstrap trees
BTS_TREES_N_SEQ = seq(30, 400, 10)

rf_cv_metrics_bts_trees = ManualCvRFParallel(n_k_fold = K_FOLDS,
                                 my_id_list_cv = ID_CV_LIST,
                                 my_n_metrics = N_METRICS,
                                 my_metric_names = METRICS_NAMES,
                                 my_data = dati,
                                 my_n_variables = rf_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]],
                                 my_n_bs_trees = BTS_TREES_N_SEQ,
                                 fix_trees_bool = FALSE,
                                 my_ncores = N_CORES)

rf_best_summary_bts_trees = CvMetricBest(my_param_values = BTS_TREES_N_SEQ,
                                  my_metric_matrix = rf_cv_metrics_bts_trees[["metrics"]],
                                  my_se_matrix = rf_cv_metrics_bts_trees[["se"]],
                                  my_metric_names = METRICS_NAMES,
                                  my_main = "RF CV metrics",
                                  my_xlab = "Number of bootstrap trees",
                                  my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                       "rf_metrics_plot_bts.jpeg"))

# if there's convergence add the the RF CV metrics to df_metrics
# otherwise, do again the variable number selection fixing a new number of bootstrap trees


df_metrics = Add_Test_Metric(df_metrics,
                             "Random Forest",
                             rf_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics



# consider now the full data and the two parameters selected
# we can evaluate variable importance on the complete dataset by OOB error

rf_model = randomForest(y ~., data = dati,
                        mtry = rf_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]],
                        ntree = rf_best_summary_bts_trees[[METRIC_CHOSEN_NAME]][["best_param_value"]],
                        importance = TRUE)


vimp = importance(rf_model)[,1]

dotchart(vimp[order(vimp)], pch = 16,
         main = "RF Increase MSE % variable importance",
         xlab = "% MSE Increase")

# save it 

jpeg(paste(FIGURES_FOLDER_RELATIVE_PATH,
           "rf_var_imp_plot.jpeg",
           collapse = ""),
     width = FIGURE_WIDTH, height = FIGURE_HEIGHT,
     pointsize = FIGURE_POINT_SIZE, quality = FIGURE_QUALITY)

dotchart(vimp[order(vimp)], pch = 16,
         main = "RF Increase MSE % variable importance",
         xlab = "% MSE Increase")


dev.off()

rm(rf_model)
gc()



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Bagging ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Because bagging is just a Random Forest when all the variables are selected at each split
# we can use the Random Forest CV function with fixed number of variables and check the convergence
# with respect to the bootstrap trees number


# sequence of bootstrap trees
BTS_TREES_N_SEQ = seq(30, 400, 10)

bagging_cv_metrics_bts_trees = ManualCvRFParallel(n_k_fold = K_FOLDS,
                                           my_id_list_cv = ID_CV_LIST,
                                           my_n_metrics = N_METRICS,
                                           my_metric_names = METRICS_NAMES,
                                           my_data = dati,
                                           my_n_variables = NCOL(dati) - 1,
                                           my_n_bs_trees = BTS_TREES_N_SEQ,
                                           fix_trees_bool = FALSE)

bagging_best_summary_bts_trees = CvMetricBest(my_param_values = BTS_TREES_N_SEQ,
                                            my_metric_matrix = bagging_cv_metrics_bts_trees[["metrics"]],
                                            my_se_matrix = bagging_cv_metrics_bts_trees[["se"]],
                                            my_metric_names = METRICS_NAMES,
                                            my_main = "Bagging CV metrics",
                                            my_xlab = "Number of bootstrap trees",
                                            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                                 "bagging_metrics_plot_bts.jpeg"))

df_metrics = Add_Test_Metric(df_metrics,
                             "Bagging",
                             bagging_best_summary_bts_trees[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rete Neurale ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#////////////////////////////////////////////////////////////////////////////
# Conclusioni -------------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////
# TO FIX

df_metrics = na.omit(df_metrics)
df_metrics[,-1] = as.numeric(df_metrics[,-1])

df_metrics[,-1] = apply(df_metrics[,-1], 2, function(col) round(col, 3))

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modelli migliori ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


