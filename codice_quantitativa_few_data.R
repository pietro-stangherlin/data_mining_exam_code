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
USED.Loss = function(y.pred, y.test, weights = 1){
  return(c(MAE.Loss(y.pred, y.test, weights), MSE.Loss(y.pred, y.test, weights)))
}


# anche qua
df_metrics = data.frame(name = NA, MAE = NA, MSE = NA)

N_METRICS_df_metrics = NCOL(df_metrics) - 1

METRICS_NAMES = colnames(df_metrics[,-1])

# names used to extract the metric added to df_metrics
# change based on the spefific problem
METRIC_VALUES_NAME = "metric_values"
METRIC_CHOSEN_NAME = "MSE"

# names used for accessing list CV matrix (actual metrics and metrics sd)
LIST_METRICS_ACCESS_NAME = "metrics"
LIST_SD_ACCESS_NAME = "sd"


#////////////////////////////////////////////////////////////////////////////
# Costruzione ID Fold convalida incrociata  -------------------
#////////////////////////////////////////////////////////////////////////////

# numero fold
K_FOLD = 4

NROW_DF = NROW(dati)

# matrice degli id dei fold della convalida incrociata
# NOTA: data la non garantita perfetta divisibilità del numero di osservazioni
# per il numero di fold è possibile che un fold abbia meno osservazioni degli altri

# ordine casuale degli id
SHUFFLED_ID = sample(1:NROW_DF, NROW_DF)

id_matrix_cv = matrix(SHUFFLED_ID, ncol = K_FOLD)

# converto la matrice in lista per poter avere degli elementi
# (vettori) con un diverso numero di osservazioni
# ogni colonna diventa un elemento della lista

id_list_cv = list()

for(j in 1:ncol(id_matrix_cv)){
  id_list_cv[[j]] = id_matrix_cv[,j]
}

rm(id_matrix_cv)
gc()

# se ottengo Warning: non divisibilità perfetta
# significa che l'ultimo elemento lista contiene 
# degli id che sono presenti anche nel primo elemento
# sistemo eliminando dall'ultimo elemento della lista gli id presenti anche nel primo elemento

# controllo il resto della divisione
integer_division_cv = NROW_DF %/% K_FOLD
modulo_cv = NROW_DF %% K_FOLD

if(modulo_cv != 0){
  id_list_cv[[K_FOLD]] = id_list_cv[[K_FOLD]][1:integer_division_cv]
}



#' Plot Cross-Validation metrics function
#' 
#' @description 
#' 
#' 
#' @param my_param_values (vector): vector of parameters
#' @param my_metric_matrix (matrix): rows -> parameters in the same order of my_param_values,
#'                                 columns -> metrics, 
#'   each cell contains the cv metric already averaged
#' @param my_sd_matrix (matrix): matrix of standard errors of cv metrics
#' @param my_metric_names (vector of strings): names of the errors,
#'
#' WARNING, pay attention to this parameter:
#' @param indexes_metric_max (vector of ints): indexes for which high metric values is best (ex f1 score)
#' (default NULL)
#'
#' @param plot_bool (bool): if TRUE plot the result, else don't
#' @param: other parameters are the usual plot parameters
#' 
#' @param my_path_plot (char): path of the where the plot will be saved on disk
#' @param my_width (int): pixel width of saved plot
#' @param my_height (int): pixel height of saved plot
#' @param my_point_size (int): point size of saved plot
#' @param my_quality (int): quality of saved plot
#' 
#' @return: best metric list (list):
#' the list has nested elements and is of the type
#' list[[metric_name]][[x]]
#' where x are: "best_param_index" (int), "best_param_value" (num), "metric_values" (vector of num)

CvMetricPlotMin = function(my_param_values, my_metric_matrix,
                           my_sd_matrix = 0,
                           my_metric_names,
                           indexes_metric_max = NULL,
                           my_main = "Model metrics", my_xlab = "parameter", my_legend_coords = "topright",
                           my_xlim = NULL, my_ylim = NULL,
                           my_path_plot,
                           my_width = FIGURE_WIDTH,
                           my_height = FIGURE_HEIGHT,
                           my_point_size = FIGURE_POINT_SIZE,
                           my_quality = FIGURE_QUALITY){
  
  # PLOT section
  
  # compute standard deviations matrix lower and upper
  sd_lower = my_metric_matrix - my_sd_matrix
  sd_upper = my_metric_matrix + my_sd_matrix
  
  # if xlim null use entire x axis (default)
  if (is.null(my_xlim)){
    my_xlim = c(min(my_param_values), max(my_param_values))
  }
  
  # if ylim is null use min and max over the entire matrix
  if (is.null(my_ylim)){
    my_ylim = c(min(sd_lower), max(sd_upper))
  }
  
  
  # BEST PARAM VALUES section
  
  n_col_metric_matrix = NCOL(my_metric_matrix)
  
  best_params = rep(NA, n_col_metric_matrix)
  names(best_params) = my_metric_names
  
  
  # Check metrics min and max best
  
  if(is.null(indexes_metric_max)){
    indexes_best_params = apply(my_metric_matrix, 2, which.min)
  }
  
  else{
    indexes_best_params = c(apply(my_metric_matrix[,-indexes_metric_max], 2, which.min),
                            apply(my_metric_matrix[,indexes_metric_max], 2, which.max))
  }
  
  
  best_params = my_param_values[indexes_best_params]
  
  # return indexes and best param values
  
  returned_list = list()
  
  # cycle over all metrics
  for (i in 1:n_col_metric_matrix){
    # add index
    returned_list[[my_metric_names[i]]][["best_param_index"]] = indexes_best_params[i]
    # add param value
    returned_list[[my_metric_names[i]]][["best_param_value"]] = best_params[i]
    # add all metrics relative to that row (best param index row)
    returned_list[[my_metric_names[i]]][["metric_values"]] = my_metric_matrix[indexes_best_params[i],]
  }
  
  # plot showed
  
  plot(my_param_values, my_metric_matrix[,1],
       xlab = my_xlab, ylab = "metric",
       main = my_main, pch = 16,
       xlim = my_xlim,
       ylim = my_ylim)
  
  arrows(my_param_values, sd_lower[,1],
         my_param_values, sd_upper[,1],
         length = 0.03, angle = 90)
  
  for (i in 2:NCOL(my_metric_matrix)){
    points(my_param_values, my_metric_matrix[,i],
           pch = 15 + i, col = i)
    arrows(my_param_values, sd_lower[,i],
           my_param_values, sd_upper[,i],
           length = 0.03, angle = 90, col = i)
  }
  
  legend(my_legend_coords,
         legend = my_metric_names,
         col = 1:NCOL(my_metric_matrix),
         pch = 15 + (1:NCOL(my_metric_matrix)))
  
  # plot them
  for (i in 1:length(best_params))
    abline(v = best_params[i], col = i)
  
  
  
  # plot saved on disk
  jpeg(my_path_plot,
       width = my_width, height = my_height,
       pointsize = my_point_size, quality = my_quality)
  
  plot(my_param_values, my_metric_matrix[,1],
       xlab = my_xlab, ylab = "metric",
       main = my_main, pch = 16,
       xlim = my_xlim,
       ylim = my_ylim)
  
  arrows(my_param_values, sd_lower[,1],
         my_param_values, sd_upper[,1],
         length = 0.03, angle = 90)
  
  for (i in 2:NCOL(my_metric_matrix)){
    points(my_param_values, my_metric_matrix[,i],
           pch = 15 + i, col = i)
    
    arrows(my_param_values, sd_lower[,i],
           my_param_values, sd_upper[,i],
           length = 0.03, angle = 90, col = i)
  }
  
  legend(my_legend_coords,
         legend = my_metric_names,
         col = 1:NCOL(my_metric_matrix),
         pch = 15 + (1:NCOL(my_metric_matrix)))
  
  # plot them
  for (i in 1:length(best_params))
    abline(v = best_params[i], col = i)
  
  dev.off()
  
  
  return(returned_list)
  
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
temp_err_matrix_cv = matrix(NA, nrow = K_FOLD, ncol = N_METRICS_df_metrics)
colnames(temp_err_matrix_cv) = colnames(df_metrics[,-1])


for (i in 1:K_FOLD){
  temp_err_matrix_cv[i,] = USED.Loss(mean(dati$y[unlist(id_list_cv[-i])]),
                                     dati$y[id_list_cv[[i]]])
}



df_metrics = Add_Test_Metric(df_metrics,
                              "cv mean",
                              colMeans(temp_err_matrix_cv))


# mediana
temp_err_matrix_cv = matrix(NA, nrow = K_FOLD, ncol = N_METRICS_df_metrics)


for (i in 1:K_FOLD){
  temp_err_matrix_cv[i,] = USED.Loss(median(dati$y[unlist(id_list_cv[-i])]),
                                     dati$y[id_list_cv[[i]]])
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

# Compromesso varianza - distorsione: convalida incrociata sia per la scelta del parametro 
# di regolazione che per il confronto finale
# NOTA: questa procedura è sub-ottimale, poichè non ci stiamo totalmente tutelando 
# contro il sovraadattamento, tuttavia, data la scarsa numerosità campionaria 
# non sono presenti valide alternative

# Funzioni generali per ridge e lasso

#' GLMNET CV cycles in case of few data sample
#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of metrics functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#'
#' @param my_x (matrix): complete model matrix passed to glmnet
#' @param my_y (vector): y glmnet argument
#' @param my_alpha (int): alpha passed to glmnet (0 -> ridge, 1 -> lasso)
#' @param my_lambda_vals (vector): vector of lambda used
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computet standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "sd"
FewDataCVCycleGLMNET = function(n_k_fold, my_id_list_cv,
                                my_n_metrics, my_metric_names,
                                my_x, my_y, my_alpha, my_lambda_vals){
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, length(my_lambda_vals), my_n_metrics))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    
    
    temp_glmnet = glmnet(x = my_x[id_train,], 
                         y = my_y[id_train], alpha = my_alpha,
                         lambda = my_lambda_vals)
    
    temp_predictions = predict(temp_glmnet, my_x[id_test,])
    
    for (j in 1:length(my_lambda_vals)){
      temp_metrics_array_cv[k,j,] = USED.Loss(temp_predictions[,j], my_y[id_test])
    }
    
    rm(temp_glmnet)
    rm(temp_predictions)
    gc()
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = length(my_lambda_vals), ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_sd = matrix(NA, nrow = length(my_lambda_vals), ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_sd) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_sd[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "sd" = cv_metrics_sd))
  
}

library(Matrix)

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

ridge_no_interaction_metrics = FewDataCVCycleGLMNET(n_k_fold = K_FOLD,
                                                    my_id_list_cv = id_list_cv,
                                                    my_n_metrics = N_METRICS_df_metrics,
                                                    my_metric_names = METRICS_NAMES,
                                                    my_x = X_mm_no_interaction,
                                                    my_y = dati$y,
                                                    my_alpha = 0,
                                                    my_lambda_vals = lambda_vals)

ridge_no_int_best_summary = CvMetricPlotMin(my_param_values = lambda_vals,
                                            my_metric_matrix = ridge_no_interaction_metrics[["metrics"]],
                                            my_sd_matrix = ridge_no_interaction_metrics[["sd"]],
                                            my_metric_names = METRICS_NAMES,
                                            my_main = "Ridge no interaction CV metrics",
                                            my_xlab = "lambda",
                                            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                                 "ridge_no_int_metrics_plot.jpeg",
                                                                 collapse = ""))

ridge_no_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                              "ridge_no_int",
                              ridge_no_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# YES interaction 
lambda_vals = glmnet(x = X_mm_yes_interaction, y = dati$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda


ridge_yes_interaction_metrics = FewDataCVCycleGLMNET(n_k_fold = K_FOLD,
                                                  my_id_list_cv = id_list_cv,
                                                  my_n_metrics = N_METRICS_df_metrics,
                                                  my_metric_names = METRICS_NAMES,
                                                  my_x = X_mm_yes_interaction,
                                                  my_y = dati$y,
                                                  my_alpha = 0,
                                                  my_lambda_vals = lambda_vals)

ridge_yes_int_best_summary = CvMetricPlotMin(my_param_values = lambda_vals,
                                            my_metric_matrix = ridge_yes_interaction_metrics[["metrics"]],
                                            my_sd_matrix = ridge_yes_interaction_metrics[["sd"]],
                                            my_metric_names = METRICS_NAMES,
                                            my_main = "Ridge yes interaction CV metrics",
                                            my_xlab = "lambda",
                                            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                                 "ridge_yes_int_metrics_plot.jpeg",
                                                                 collapse = ""))

ridge_yes_int_best_summary


df_metrics = Add_Test_Metric(df_metrics,
                             "ridge_yes_int",
                             ridge_yes_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# Lasso ---------------

# NO interaction
lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda

lasso_no_interaction_metrics = FewDataCVCycleGLMNET(n_k_fold = K_FOLD,
                                                    my_id_list_cv = id_list_cv,
                                                    my_n_metrics = N_METRICS_df_metrics,
                                                    my_metric_names = METRICS_NAMES,
                                                    my_x = X_mm_no_interaction,
                                                    my_y = dati$y,
                                                    my_alpha = 1,
                                                    my_lambda_vals = lambda_vals)

lasso_no_int_best_summary = CvMetricPlotMin(my_param_values = lambda_vals,
                                            my_metric_matrix = lasso_no_interaction_metrics[["metrics"]],
                                            my_sd_matrix = lasso_no_interaction_metrics[["sd"]],
                                            my_metric_names = METRICS_NAMES,
                                            my_main = "lasso no interaction CV metrics",
                                            my_xlab = "lambda",
                                            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                                 "lasso_no_int_metrics_plot.jpeg",
                                                                 collapse = ""))
lasso_no_int_best_summary

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_no_int",
                             lasso_no_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# YES interaction

lambda_vals = glmnet(x = X_mm_yes_interaction, y = dati$y,
                     alpha = 1, lambda.min.ratio = 1e-07)$lambda


lasso_yes_interaction_metrics = FewDataCVCycleGLMNET(n_k_fold = K_FOLD,
                                                     my_id_list_cv = id_list_cv,
                                                     my_n_metrics = N_METRICS_df_metrics,
                                                     my_metric_names = METRICS_NAMES,
                                                     my_x = X_mm_yes_interaction,
                                                     my_y = dati$y,
                                                     my_alpha = 1,
                                                     my_lambda_vals = lambda_vals)

lasso_yes_int_best_summary = CvMetricPlotMin(my_param_values = lambda_vals,
                                             my_metric_matrix = lasso_yes_interaction_metrics[["metrics"]],
                                             my_sd_matrix = lasso_yes_interaction_metrics[["sd"]],
                                             my_metric_names = METRICS_NAMES,
                                             my_main = "lasso yes interaction CV metrics",
                                             my_xlab = "lambda",
                                             my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                                  "lasso_yes_int_metrics_plot.jpeg",
                                                                  collapse = ""))

lasso_yes_int_best_summary


df_metrics = Add_Test_Metric(df_metrics,
                             "lasso_yes_int",
                             lasso_yes_int_best_summary[[METRIC_CHOSEN_NAME]][[METRIC_VALUES_NAME]])

df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Tree -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

TREE_MAX_SIZE = 50

#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of loss functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_max_size (int): max size of the pruned tree
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
FewDataCVCycleTree = function(n_k_fold, my_id_list_cv,my_n_metrics,
                              my_metric_names, my_data,
                              my_max_size = TREE_MAX_SIZE){
  
  # we use my_max_size - 1 because we start with size = 2
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, my_max_size - 1, my_n_metrics))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    
    # full grown tree
    temp_tree_full = tree(y ~.,
                          data = my_data[id_train,],
                          control = tree.control(nobs = length(id_train),
                                                 mindev = 1e-05,
                                                 minsize = 2))
    # if maximum tree depth error
    # change minsize = 2 to higher values and so do it with 
    # mindev
    
    # pruned tree: problem -> each fold can have different pruning inducing
    # split sizes whose CV error cannot be averaged
    # so I need to do it manually: select a set of size values
    # for each value prune the full tree on the id_train (sub-optimal and too optimistic)
    # (but given the scarsity of data we need a compromise)
    # and keep track of the reduced deviance on the id_test
    
    for (s in 2:my_max_size){
      temp_tree_pruned = prune.tree(temp_tree_full, best = s)
      # prediction error
      # s-1 because we start by size = 2
      temp_metrics_array_cv[k,s-1,] = USED.Loss(predict(temp_tree_pruned, my_data[id_test,]),
                                            my_data$y[id_test])
      print(paste("tree size: ", s, collapse = ""))
    }
    
    
    rm(temp_tree_full)
    rm(temp_tree_pruned)
    gc()
    
    print(paste("fold ", k))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = my_max_size - 1, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_sd = matrix(NA, nrow = my_max_size - 1, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_sd) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_sd[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "sd" = cv_metrics_sd))
  
}

# parallelize inner cycle

#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of loss functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_max_size (int): max size of the pruned tree
#' 
#' @param my_loss_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_loss_functions = c("USED.Loss", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Loss contains some other functions they must be present as well, like the example
#' which is also the default
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
FewDataCVCycleTreeParallel = function(n_k_fold, my_id_list_cv,my_n_metrics,
                                           my_metric_names, my_data,
                                           my_max_size = TREE_MAX_SIZE,
                                           my_loss_functions = c("USED.Loss", "MAE.Loss", "MSE.Loss")){
  # init parallel
  sfInit(cpus = parallel::detectCores(), parallel = T)
  
  sfLibrary(tree)
  sfExport(list = c("my_data", my_loss_functions))
  
  
  # we use my_max_size - 1 because we start with size = 2
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, my_max_size - 1, my_n_metrics))
  
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    
    # full grown tree
    temp_tree_full = tree(y ~.,
                          data = my_data[id_train,],
                          control = tree.control(nobs = length(id_train),
                                                 mindev = 1e-05,
                                                 minsize = 2))
    
    sfExport(list = c("temp_tree_full", "id_train", "id_test", "my_max_size"))
    # if maximum tree depth error
    # change minsize = 2 to higher values and so do it with 
    # mindev
    
    # pruned tree: problem -> each fold can have different pruning inducing
    # split sizes whose CV error cannot be averaged
    # so I need to do it manually: select a set of size values
    # for each value prune the full tree on the id_train (sub-optimal and too optimistic)
    # (but given the scarsity of data we need a compromise)
    # and keep track of the reduced deviance on the id_test
    
    # for better readability
    temp_metric = sfLapply(2:my_max_size,
                           fun = function(s) 
                             USED.Loss(predict(prune.tree(temp_tree_full, best = s),
                                               my_data[id_test,]), my_data$y[id_test]))
    
    # unlist to the right dimensions matrix
    temp_metrics_array_cv[k,(2:my_max_size)-1,] = matrix(unlist(temp_metric), ncol = my_n_metrics, byrow = T)
    
    
    rm(temp_tree_full)
    rm(temp_metric)
    gc()
    
    print(paste("fold ", k))
  }
  
  # stop parallel cluster
  sfStop()
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = my_max_size - 1, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_sd = matrix(NA, nrow = my_max_size - 1, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_sd) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_sd[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "sd" = cv_metrics_sd))
  
  return(cv_metrics)
  
  
}

# if parallel shows problems use the non parallel version
tree_cv_metrics = FewDataCVCycleTreeParallel(n_k_fold = K_FOLD,
                                                  my_id_list_cv = id_list_cv,
                                                  my_n_metrics = N_METRICS_df_metrics,
                                                  my_metric_names = METRICS_NAMES,
                                                  my_data = dati,
                                                  my_max_size = TREE_MAX_SIZE)

tree_best_summary = CvMetricPlotMin(my_param_values = 2:TREE_MAX_SIZE,
                                    my_metric_matrix = tree_cv_metrics[["metrics"]],
                                    my_sd_matrix = tree_cv_metrics[["sd"]],
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

# PPR CV function
#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of loss functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_max_ridges (int): max number of ridge functions
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
FewDataCVCyclePPR = function(n_k_fold, my_id_list_cv,my_n_metrics,
                              my_metric_names, my_data,
                              my_max_ridges = PPR_MAX_RIDGE_FUNCTIONS){
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, my_max_ridges, my_n_metrics))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    # cycle through different numbers of ridge functions
    for (r in 1:my_max_ridges){
      temp_ppr = ppr(y ~ .,
                     data = my_data[id_train,],
                     nterms = r)
      # prediction error
      temp_metrics_array_cv[k,r,] = USED.Loss(predict(temp_ppr, my_data[id_test,]),
                                            my_data$y[id_test])
    }
    
    
    rm(temp_ppr)
    gc()
    
    print(paste("fold ", k))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = my_max_ridges, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_sd = matrix(NA, nrow = my_max_ridges, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_sd) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_sd[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "sd" = cv_metrics_sd))
  
}


ppr_cv_metrics = FewDataCVCyclePPR(n_k_fold = K_FOLD,
                                     my_id_list_cv = id_list_cv,
                                     my_n_metrics = N_METRICS_df_metrics,
                                     my_metric_names = METRICS_NAMES,
                                     my_data = dati,
                                     my_max_ridges = PPR_MAX_RIDGE_FUNCTIONS)

ppr_best_summary = CvMetricPlotMin(my_param_values = 1:PPR_MAX_RIDGE_FUNCTIONS,
                                    my_metric_matrix = ppr_cv_metrics[["metrics"]],
                                    my_sd_matrix =  ppr_cv_metrics[["sd"]],
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

# function to choose the optimal number of variables at each split
# or alternatively (based on fix_tress_bool parameter) check convergence of validation error
# with respect to number of tress for a fixed my_max_variables

#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of loss functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_n_variables: (int) or (vector of int) number of variables chosen at each split
#' @param my_n_bs_trees: (int) or (vector of int) number of bootstrap trees 
#' @param fix_tress_bool (bool): TRUE if fixed number of bootstrap trees, number of variables changes,
#' else FALSE
#' 
#' @description this function can be used (separately, not simultaneuosly) for two parameters check
#' 1) if fix_tress_bool == TRUE -> my_n_bs_trees is fixed at its maximum if not already an integer
#' and the procedure compute the CV error for varying number of variables at each split
#' according to the the vector (supposed to be a sequence)
#' 2) if fix_tress_bool == FALSE -> my_n_variables is fixed at its maximum if not already an integer,
#' but a warning is given, because it should be just an integer, not a vector.
#' the procedure compute the CV error for varying number of bootstrap trees
#' according to the the vector (supposed to be a sequence)
#'  
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
FewDataCVCycleRF = function(n_k_fold, my_id_list_cv,my_n_metrics,
                              my_metric_names, my_data,
                            my_n_variables = 1:RF_MAX_VARIABLES,
                            my_n_bs_trees = 10:RF_N_BS_TREES,
                            fix_trees_bool = TRUE){
  
  # fixed number of bootstrap trees, number of variables changes
  if(fix_trees_bool == TRUE){
    tuning_parameter_length = length(my_n_variables)
    
    # fix the number of trees to max if my_n_bs_trees is not already an int
    my_n_bs_trees = max(my_n_bs_trees)
  }
  # fixed number of number of variables, number of bootstrap tress changes
  else{
    tuning_parameter_length = length(my_n_bs_trees)
    
    # check warning
    if(length(my_n_variables) > 1){
      print("Warning: my_n_variables should be an integer, not a sequence of numbers, maximum is taken")
    }
    
    my_n_variables = max(my_n_variables)
    
  }
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, tuning_parameter_length, my_n_metrics))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    # it's ugly I know and a bad pratice but I'll do two separate loop based on if condition
    
    
    if(fix_trees_bool == TRUE){
      for (m in 1:tuning_parameter_length){
        temp_rf = randomForest(y ~., data = my_data[id_train,],
                               mtry = my_n_variables[m], ntree = my_n_bs_trees)
        # prediction error
        temp_metrics_array_cv[k,m,] = USED.Loss(predict(temp_rf, my_data[id_test,]),
                                            my_data$y[id_test])
        print(paste(c("n var = ", m), collapse = ""))
      }
    }
    
    else{
      for (t in 1:tuning_parameter_length){
        temp_rf = randomForest(y ~., data = my_data[id_train,],
                               mtry = my_n_variables, ntree = my_n_bs_trees[t])
        # prediction error
        temp_metrics_array_cv[k,t,] = USED.Loss(predict(temp_rf, my_data[id_test,]),
                                            my_data$y[id_test])
        print(paste(c("n trees = ", my_n_bs_trees[t]), collapse = ""))
      }
    }
    
    
    rm(temp_rf)
    gc()
    
    print(paste("fold ", k))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_sd = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_sd) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_sd[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "sd" = cv_metrics_sd))
  
}


# function to choose the optimal number of variables at each split
# or alternatively (based on fix_tress_bool parameter) check convergence of validation error
# with respect to number of tress for a fixed my_max_variables

#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of loss functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_n_variables: (int) or (vector of int) number of variables chosen at each split
#' @param my_n_bs_trees: (int) or (vector of int) number of bootstrap trees 
#' @param fix_tress_bool (bool): TRUE if fixed number of bootstrap trees, number of variables changes,
#' else FALSE
#' 
#' @param my_loss_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_loss_functions = c("USED.Loss", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Loss contains some other functions they must be present as well, like the example
#' which is also the default

#' 
#' @description this function can be used (separately, not simultaneuosly) for two parameters check
#' 1) if fix_tress_bool == TRUE -> my_n_bs_trees is fixed at its maximum if not already an integer
#' and the procedure compute the CV error for varying number of variables at each split
#' according to the the vector (supposed to be a sequence)
#' 2) if fix_tress_bool == FALSE -> my_n_variables is fixed at its maximum if not already an integer,
#' but a warning is given, because it should be just an integer, not a vector.
#' the procedure compute the CV error for varying number of bootstrap trees
#' according to the the vector (supposed to be a sequence)
#'  
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
FewDataCVCycleRFParallel = function(n_k_fold, my_id_list_cv,my_n_metrics,
                            my_metric_names, my_data,
                            my_n_variables = 1:RF_MAX_VARIABLES,
                            my_n_bs_trees = 10:RF_N_BS_TREES,
                            fix_trees_bool = TRUE,
                            my_loss_functions = c("USED.Loss", "MAE.Loss", "MSE.Loss")){
  
  # init parallel
  sfInit(cpus = parallel::detectCores(), parallel = T)
  
  sfLibrary(randomForest)
  
  # fixed number of bootstrap trees, number of variables changes
  if(fix_trees_bool == TRUE){
    tuning_parameter_length = length(my_n_variables)
    
    # fix the number of trees to max if my_n_bs_trees is not already an int
    my_n_bs_trees = max(my_n_bs_trees)
  }
  # fixed number of number of variables, number of bootstrap tress changes
  else{
    tuning_parameter_length = length(my_n_bs_trees)
    
    # check warning
    if(length(my_n_variables) > 1){
      print("Warning: my_n_variables should be an integer, not a sequence of numbers, maximum is taken")
    }
    
    my_n_variables = max(my_n_variables)
    
  }

  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, tuning_parameter_length, my_n_metrics))
  
  # allocate relevant variables in cluster
  sfExport(list = c("my_data", my_loss_functions,
                    "my_n_bs_trees", "tuning_parameter_length", "my_n_variables"))
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    sfExport(list = c("id_train", "id_test"))
    
    # it's ugly I know and a bad pratice but I'll do two separate loop based on if condition
    
    if(fix_trees_bool == TRUE){
      # for better readability
      temp_metric = sfLapply(1:tuning_parameter_length,
                             fun = function(m) 
                               USED.Loss(predict(randomForest(y ~., data = my_data[id_train,],
                                                              mtry = my_n_variables[m], ntree = my_n_bs_trees),
                                                 my_data[id_test,]), my_data$y[id_test]))
      
      # unlist to the right dimensions matrix
      temp_metrics_array_cv[k,my_n_variables,] = matrix(unlist(temp_metric), ncol = my_n_metrics, byrow = T)
      
      rm(temp_metric)
      gc()

    }
    
    else{
      
      # for better readability
      temp_metric = sfLapply(1:tuning_parameter_length,
                             fun = function(t) 
                               USED.Loss(predict(randomForest(y ~., data = my_data[id_train,],
                                                              mtry = my_n_variables, ntree = my_n_bs_trees[t]),
                                                 my_data[id_test,]), my_data$y[id_test]))
      
      # unlist to the right dimensions matrix
      temp_metrics_array_cv[k,1:tuning_parameter_length,] = matrix(unlist(temp_metric), ncol = my_n_metrics, byrow = T)
      
      rm(temp_metric)
      gc()
    }
    
    gc()
    
    print(paste("fold ", k))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_sd = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_sd) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_sd[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "sd" = cv_metrics_sd))
  
}



# number of split variable selection

rf_cv_metrics = FewDataCVCycleRFParallel(n_k_fold = K_FOLD,
                                 my_id_list_cv = id_list_cv,
                                 my_n_metrics = N_METRICS_df_metrics,
                                 my_metric_names = METRICS_NAMES,
                                 my_data = dati,
                                 my_n_variables = 1:RF_MAX_VARIABLES,
                                 my_n_bs_trees = 400,
                                 fix_trees_bool = TRUE)

rf_best_summary = CvMetricPlotMin(my_param_values = 1:RF_MAX_VARIABLES,
                                   my_metric_matrix = rf_cv_metrics[["metrics"]],
                                   my_sd_matrix = rf_cv_metrics[["sd"]],
                                   my_metric_names = METRICS_NAMES,
                                   my_main = "RF CV metrics",
                                   my_xlab = "Number of variables at each split",
                                   my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                                        "rf_metrics_plot.jpeg"))

rf_best_summary

# check convergence with respect to number of bootstrap trees

# sequence of bootstrap trees
BTS_TREES_N_SEQ = seq(30, 400, 10)

rf_cv_metrics_bts_trees = FewDataCVCycleRFParallel(n_k_fold = K_FOLD,
                                 my_id_list_cv = id_list_cv,
                                 my_n_metrics = N_METRICS_df_metrics,
                                 my_metric_names = METRICS_NAMES,
                                 my_data = dati,
                                 my_n_variables = rf_best_summary[[METRIC_CHOSEN_NAME]][["best_param_value"]],
                                 my_n_bs_trees = BTS_TREES_N_SEQ,
                                 fix_trees_bool = FALSE)

rf_best_summary_bts_trees = CvMetricPlotMin(my_param_values = BTS_TREES_N_SEQ,
                                  my_metric_matrix = rf_cv_metrics_bts_trees[["metrics"]],
                                  my_sd_matrix = rf_cv_metrics_bts_trees[["sd"]],
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

bagging_cv_metrics_bts_trees = FewDataCVCycleRFParallel(n_k_fold = K_FOLD,
                                           my_id_list_cv = id_list_cv,
                                           my_n_metrics = N_METRICS_df_metrics,
                                           my_metric_names = METRICS_NAMES,
                                           my_data = dati,
                                           my_n_variables = NCOL(dati) - 1,
                                           my_n_bs_trees = BTS_TREES_N_SEQ,
                                           fix_trees_bool = FALSE)

bagging_best_summary_bts_trees = CvMetricPlotMin(my_param_values = BTS_TREES_N_SEQ,
                                            my_metric_matrix = bagging_cv_metrics_bts_trees[["metrics"]],
                                            my_sd_matrix = bagging_cv_metrics_bts_trees[["sd"]],
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


