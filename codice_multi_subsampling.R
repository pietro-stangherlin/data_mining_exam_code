# data wrangling
library(dplyr)

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
USED.Metrics = function(y.pred, y.test, weights = 1){
  return(c(MissErr(y.pred, y.test, weights), 0))
}


# anche qua
df_metrics = data.frame(name = NA,
                        missclass = NA,
                        filler = NA)

METRICS_NAMES = colnames(df_metrics[,-1])

N_METRICS = length(METRICS_NAMES)

# names used to extract the metric added to df_metrics
# change based on the specific problem
METRIC_VALUES_NAME = "metric_values"
METRIC_CHOSEN_NAME = "missclass"

# names used for accessing list CV matrix (actual metrics and metrics se)
LIST_METRICS_ACCESS_NAME = "metrics"
LIST_SD_ACCESS_NAME = "se"

# metrics names + USED.Loss
# WARNING: the order should be same as in df_metrics
MY_USED_METRICS = c("USED.Metrics", "MissErr")

MY_WEIGHTS = rep(1, nrow(dati))

# /////////////////////////////////////////////////////////////////
#------------------------ Sottocampionamento ----------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter tuning: cross validation on train: building cv folds  -------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

K_FOLDS = 4

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

ID_CV_LIST_BALANCED = ID_CV_LIST_UNBALANCED

# Balanced ID CV creation --------------

# CV_PROP = 0.5
# 
# 
# ID_CV_LIST_BALANCED = list()
# 
# for(i in 1:length(ID_CV_LIST_UNBALANCED)){
#   # cambiare valori all'occorrenza
#   ids_few =ID_CV_LIST_UNBALANCED[[i]][which(dati$y[ID_CV_LIST_UNBALANCED[[i]]] == 1)]
#   ids_lot =ID_CV_LIST_UNBALANCED[[i]][which(dati$y[ID_CV_LIST_UNBALANCED[[i]]] == 0)]
#   
#   tot = round(length(ids_few)/CV_PROP)
#   
#   ID_CV_LIST_BALANCED[[i]] = c(ids_few,
#                                sample(ids_lot, size = tot - length(ids_few), replace = FALSE))
# }
# 
# 
# 
# 
# 
# source("cv_functions.R")
# 
# BALANCED_ID_vector = unlist(ID_CV_LIST_BALANCED)
# UNBALANCED_ID_vector = unlist(ID_CV_LIST_UNBALANCED)

MY_WEIGHTS = rep(1, NROW_dati)

USE_ONLY_FIRST_FOLD = FALSE

library(nnet)
Y_dati = class.ind(dati$y)

Y_LEVELS_SORTED = colnames(Y_dati)

# //////////////////////////# ////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge & Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# sparse is preferred if there are many categorical predictors (sparse matrix)
library(Matrix)
# X_mm_no_interaction =  sparse.model.matrix(formula_no_interaction_no_intercept, data = dati)
# 
# # computational heavy
# X_mm_yes_interaction =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = dati)

# default
X_mm_no_interaction = model.matrix(formula_no_interaction_no_intercept, data = dati)

X_mm_yes_interaction_dati = model.matrix(formula_yes_interaction_no_intercept, data = dati)

library(glmnet)
lambda_vals = glmnet(x = X_mm_no_interaction, y = Y_dati[,1],
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

lasso_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                                              my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction,
                                              my_y = Y_dati,
                                              my_alpha = 1,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS,
                                              use_only_first_fold = USE_ONLY_FIRST_FOLD,
                                              is_classification = FALSE,
                                              is_multiclass = TRUE)

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

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_no_interaction",
                             lasso_no_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# save the df_metrics as .Rdata
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

TREE_MAX_SIZE = 20


tree_cv_metrics = ManualCvTree(my_id_list_cv_train = ID_CV_LIST_BALANCED,
                               my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
                               my_metric_names = METRICS_NAMES,
                               my_data = dati,
                               my_max_size = TREE_MAX_SIZE,
                               my_weights = MY_WEIGHTS,
                               my_mindev = 1e-05,
                               my_minsize = 5,
                               is_classification = TRUE,
                               use_only_first_fold = USE_ONLY_FIRST_FOLD,
                               is_multiclass = TRUE)

# if parallel shows problems use the non parallel version
# tree_cv_metrics = ManualCvTreeParallel(my_id_list_cv_train = ID_CV_LIST_BALANCED,
#                                        my_id_list_cv_test = ID_CV_LIST_UNBALANCED,
#                                        my_metric_names = METRICS_NAMES,
#                                        my_data = dati,
#                                        my_max_size = TREE_MAX_SIZE,
#                                        my_metrics_functions = MY_USED_METRICS,
#                                        my_ncores = N_CORES,
#                                        my_weights = MY_WEIGHTS,
#                                        my_mindev = 1e-05,
#                                        my_minsize = 5,
#                                        use_only_first_fold = USE_ONLY_FIRST_FOLD,
#                                        is_classification = TRUE,
#                                        is_multiclass = TRUE)

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
                           is_multiclass = TRUE)

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
#                            is_multiclass = TRUE)


rf_cv_metrics_best = CvMetricBest(my_param_values = 2:RF_MAX_VARIABLES,
                                  my_metric_matrix = rf_cv_metrics[["metrics"]],
                                  my_one_se_best = TRUE,
                                  my_higher_more_complex = TRUE,
                                  my_se_matrix = rf_cv_metrics[["se"]],
                                  my_metric_names = METRICS_NAMES)

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
                                            is_multiclass = TRUE)

bagging_best_summary = CvMetricBest(my_param_values = RF_TREE_NUMBER_SEQ,
                                    my_metric_matrix = bagging_n_tree_metrics[["metrics"]],
                                    my_one_se_best = TRUE,
                                    my_higher_more_complex = TRUE,
                                    my_se_matrix = bagging_n_tree_metrics[["se"]],
                                    my_metric_names = METRICS_NAMES)

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
