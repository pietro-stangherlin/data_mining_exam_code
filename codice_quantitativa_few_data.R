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

MY_WEIGHTS = rep(1, nrow(dati))
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

source("cv_functions.R")
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

# eventualmente basato sui risultati successivi
# lambda_vals = seq(1e-07,1e-03, by = 1e-05)

# Ridge ----------------

# NO interaction ------------------
lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

ridge_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction,
                                              my_y = dati$y,
                                              my_alpha = 0,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS)

# ridge_no_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST,
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
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = ridge_no_interaction_metrics[["metrics"]],
                my_se_matrix = ridge_no_interaction_metrics[["se"]],
                my_best_param_values = log(ExtractBestParams(ridge_no_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge no interaction CV metrics",
                my_xlab = " log lambda")
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

# YES interaction -------------------
lambda_vals = glmnet(x = X_mm_yes_interaction, y = dati$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

ridge_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_yes_interaction,
                                              my_y = dati$y,
                                              my_alpha = 0,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS)

# ridge_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST,
#                                               my_metric_names = METRICS_NAMES,
#                                               my_x = X_mm_yes_interaction,
#                                               my_y = dati$y,
#                                               my_alpha = 0,
#                                               my_lambda_vals = lambda_vals,
#                                               my_weights = MY_WEIGHTS,
#                                               my_metrics_functions = MY_USED_METRICS,
#                                               my_ncores = N_CORES)

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
                my_best_param_values = log(ExtractBestParams(ridge_yes_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "Ridge Yes interaction CV metrics",
                my_xlab = " log lambda")
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
save(df_metrics, file = "df_metrics.Rdata")

# Lasso ---------------
# NO interaction 
lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

lasso_no_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST,
                                              my_metric_names = METRICS_NAMES,
                                              my_x = X_mm_no_interaction,
                                              my_y = dati$y,
                                              my_alpha = 1,
                                              my_lambda_vals = lambda_vals,
                                              my_weights = MY_WEIGHTS)


lasso_no_int_best_summary = CvMetricBest(my_param_values = lambda_vals,
                                         my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                                         my_one_se_best = FALSE,
                                         my_higher_more_complex = FALSE,
                                         my_se_matrix = lasso_no_interaction_metrics[["se"]],
                                         my_metric_names = METRICS_NAMES)

temp_plot_function = function(){
  PlotCvMetrics(my_param_values = log(lambda_vals),
                my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                my_se_matrix = lasso_no_interaction_metrics[["se"]],
                my_best_param_values = log(ExtractBestParams(lasso_no_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso no interaction CV metrics",
                my_xlab = " log lambda")
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

save(df_metrics, file = "df_metrics.Rdata")

# YES interaction -------------------
lambda_vals = glmnet(x = X_mm_yes_interaction, y = dati$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

lasso_yes_interaction_metrics = ManualCvGlmnet(my_id_list_cv_train = ID_CV_LIST,
                                               my_metric_names = METRICS_NAMES,
                                               my_x = X_mm_yes_interaction,
                                               my_y = dati$y,
                                               my_alpha = 1,
                                               my_lambda_vals = lambda_vals,
                                               my_weights = MY_WEIGHTS)

# lasso_yes_interaction_metrics = ManualCvGlmnetParallel(my_id_list_cv_train = ID_CV_LIST,
#                                               my_metric_names = METRICS_NAMES,
#                                               my_x = X_mm_yes_interaction,
#                                               my_y = dati$y,
#                                               my_alpha = 1,
#                                               my_lambda_vals = lambda_vals,
#                                               my_weights = MY_WEIGHTS,
#                                               my_metrics_functions = MY_USED_METRICS,
#                                               my_ncores = N_CORES)

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
                my_best_param_values = log(ExtractBestParams(lasso_yes_int_best_summary)),
                my_metric_names = METRICS_NAMES,
                my_main = "lasso Yes interaction CV metrics",
                my_xlab = " log lambda")
}

PlotAndSave(temp_plot_function, my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                     "lasso_yes_int_metrics_plot.jpeg",
                                                     collapse = ""))

print("lasso_yes_int_best_summary")
lasso_yes_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_yes_int",
                             lasso_yes_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

save(df_metrics, file = "df_metrics.Rdata")


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Tree -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

TREE_MAX_SIZE = 40


# if parallel shows problems use the non parallel version
tree_cv_metrics = ManualCvTreeParallel(my_id_list_cv_train = ID_CV_LIST,
                                       my_metric_names = METRICS_NAMES,
                                       my_data = dati,
                                       my_max_size = TREE_MAX_SIZE,
                                       my_metrics_functions = MY_USED_METRICS,
                                       my_ncores = N_CORES,
                                       my_weights = MY_WEIGHTS,
                                       my_mindev = 1e-05,
                                       my_minsize = 5)

tree_best_summary = CvMetricBest(my_param_values = 2:TREE_MAX_SIZE,
                                 my_metric_matrix = tree_cv_metrics[["metrics"]],
                                 my_one_se_best = FALSE,
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

tree_best_summary


df_metrics = Add_Test_Metric(df_metrics,
                              "tree",
                             tree_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

save(df_metrics, file = "df_metrics.Rdata")


tree_full = tree(y ~.,
                 data = dati,
                 control = tree.control(nobs = NROW(dati),
                                        mindev = 1e-04,
                                        minsize = 5))


# check overfitting
plot(tree_full)

final_tree_pruned = prune.tree(tree_full,
                               best = tree_best_size)

temp_plot_function = function(){
  plot(final_tree_pruned)
  text(final_tree_pruned, cex = 0.7)}


PlotAndSave(temp_plot_function,
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "tree_pruned_plot.jpeg",
                                 collapse = ""))

file_name_final_tree_pruned = paste(MODELS_FOLDER_RELATIVE_PATH,
                                    "final_tree_pruned",
                                    ".Rdata", collapse = "", sep = "")

save(final_tree_pruned, file = file_name_final_tree_pruned)


rm(final_tree_pruned)
rm(tree_full)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PPR ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# max number of ridge functions
PPR_MAX_RIDGE_FUNCTIONS = 4

# possible spline degrees of freedom
PPR_DF_SM = 2:6

ppr_metrics = PPRRegulationCV(my_data = dati,
                              my_id_list_cv_train = ID_CV_LIST,
                              my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                              my_spline_df = PPR_DF_SM,
                              my_metrics_names = METRICS_NAMES,
                              my_weights = MY_WEIGHTS,
                              is_classification = FALSE)




# 1.b) Regulation: CV -------

ppr_metrics = PPRRegulationCVParallel(my_data = dati,
                                      my_id_list_cv_train = ID_CV_LIST,
                                      my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                      my_spline_df = PPR_DF_SM,
                                      my_metrics_names = METRICS_NAMES,
                                      my_weights = MY_WEIGHTS,
                                      my_metrics_functions = MY_USED_METRICS,
                                      my_ncores = N_CORES,
                                      is_classification = FALSE)


# 2) final model -------

ppr_best_params = PPRExtractBestParams(ppr_metrics)

ppr_n_ridges_best = ppr_best_params[[METRIC_CHOSEN_NAME]][[1]]
ppr_df_best = ppr_best_params[[METRIC_CHOSEN_NAME]][[2]]


print("ppr best params")
ppr_best_params


df_metrics = Add_Test_Metric(df_metrics,
                             "PPR",
                             ppr_metrics[ppr_n_ridges_best,ppr_df_best,])

df_metrics

rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

# file_name_ppr_model = paste(MODELS_FOLDER_RELATIVE_PATH,
#                             "ppr_model",
#                             ".Rdata", collapse = "", sep = "")
# 
# ppr_model = ppr(y ~ .,
#                 data = dati[BALANCED_ID,],
#                 nterms = ppr_best_params[[METRIC_CHOSEN_NAME]][["n_ridge_functions"]],
#                 sm.method = "spline",
#                 df = ppr_best_params[[METRIC_CHOSEN_NAME]][["spline_df"]]) 
# 
# save(ppr_model, file = file_name_ppr_model)
# 
# rm(ppr_model)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(ranger)

# Nota: se manca il tempo eseguo prima la RandomForest del Bagging
# visto che quest'ultimo è un sotto caso particolare 
# della RandomForest (selezione di tutte le variabili per ogni split)


# massimo numero di esplicative presenti
RF_MAX_VARIABLES = 20 # sottraggo 1 per la variabile risposta
# ridurlo per considerazioni computazionali

RF_ITER = 400

RF_TREE_NUMBER_SEQ = seq(10, 400, 10)

rf_cv_metrics = ManualCvRF(my_id_list_cv_train = ID_CV_LIST,
                           my_metric_names = METRICS_NAMES,
                           my_data = dati,
                           my_n_variables = 2:RF_MAX_VARIABLES,
                           my_n_bs_trees = RF_ITER,
                           fix_trees_bool = TRUE,
                           my_weights = MY_WEIGHTS,
                           use_only_first_fold = FALSE,
                           is_classification = FALSE,
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
#                            is_classification = FALSE,
#                            is_multiclass = FALSE)


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

print("rf best mtry")
best_mtry
# check convergence

rf_n_tree_metrics = ManualCvRFParallel(my_id_list_cv_train = ID_CV_LIST,
                                       my_metric_names = METRICS_NAMES,
                                       my_data = dati,
                                       my_n_variables = best_mtry,
                                       my_n_bs_trees = RF_TREE_NUMBER_SEQ,
                                       my_ncores = N_CORES,
                                       my_metrics_functions = MY_USED_METRICS,
                                       fix_trees_bool = FALSE,
                                       my_weights = MY_WEIGHTS,
                                       is_classification = FALSE,
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
                             importance = "permutation")


df_metrics = Add_Test_Metric(df_metrics,
                             "Random Forest",
                             rf_cv_metrics_best[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")


# Importanza delle variabili
vimp = importance(random_forest_model)

PlotAndSave(my_plotting_function =  function() dotchart(vimp[order(vimp, decreasing = T)[1:20]],
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

bagging_n_tree_metrics = ManualCvRFParallel(my_id_list_cv_train = ID_CV_LIST,
                                            my_metric_names = METRICS_NAMES,
                                            my_data = dati,
                                            my_n_variables = NCOL(dati) - 1,
                                            my_n_bs_trees = RF_TREE_NUMBER_SEQ,
                                            my_ncores = N_CORES,
                                            my_metrics_functions = MY_USED_METRICS,
                                            fix_trees_bool = FALSE,
                                            my_weights = MY_WEIGHTS,
                                            is_classification = FALSE,
                                            is_multiclass = FALSE)

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


# save the df_metrics as .Rdata
save(df_metrics, file = "df_metrics.Rdata")

rm(bagging_model)
gc()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rete Neurale ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#////////////////////////////////////////////////////////////////////////////
# Conclusioni -------------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////
# TO FIX


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modelli migliori ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

rounded_df = cbind(df_metrics[,1],
                   apply(df_metrics[,2:NCOL(df_metrics)], 2, function(col) round(as.numeric(col), 2)))

rounded_df
# RIDGE and Lasso ----------------

ridge_no_interaction = glmnet(x = X_mm_no_interaction,
                              y = dati$y,
                              alpha = 0,
                              lambda = ridge_no_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])

temp_glmnet_object = predict(ridge_no_interaction, type = "coef") %>% as.matrix()
temp_coef = temp_glmnet_object[,1]

temp_main = "(abs) greatest ridge coefficients no interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -2) | (temp_coef > 10)) ] %>% sort()

PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_ridge_no_int_plot.jpeg",
                                 collapse = ""))

ridge_yes_interaction = glmnet(x = X_mm_yes_interaction,
                               y = dati$y,
                               alpha = 0,
                               lambda = ridge_yes_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])

temp_glmnet_object = predict(ridge_yes_interaction, type = "coef") %>% as.matrix()
temp_coef = temp_glmnet_object[,1]

temp_main = "(abs) greatest ridge coefficients yes interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -0.8) | (temp_coef > 0.5)) ] %>% sort()

PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_ridge_yes_int_plot.jpeg",
                                 collapse = ""))

lasso_no_interaction = glmnet(x = X_mm_no_interaction,
                              y = dati$y,
                              alpha = 1,
                              lambda = lasso_no_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])

temp_glmnet_object = predict(lasso_no_interaction, type = "coef") %>% as.matrix()
temp_coef = temp_glmnet_object[,1]

temp_main = "(abs) greatest lasso coefficients no interaction"
summary(temp_coef)

sorted_temp_coef = temp_coef[which((temp_coef < -7) | (temp_coef > 7)) ] %>% sort()

PlotAndSave(my_plotting_function = function() sorted_temp_coef %>% dotchart(pch = 16, main = temp_main),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "coef_lasso_no_int_plot.jpeg",
                                 collapse = ""))

lasso_yes_interaction = glmnet(x = X_mm_yes_interaction,
                               y = dati$y,
                               alpha = 1,
                               lambda = lasso_yes_int_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]])

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


