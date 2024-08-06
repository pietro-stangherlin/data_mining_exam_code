library(dplyr)

# Descrizione -----------------------
# pochi dati: usiamo convalida incrociata come criterio per confrontare i modelli finali
# partizioniamo i dati in K insiemi (FOLD) e usiamo la procedura di convalida incrociata
# sia per identificare i valori ottimali dei parametri (rispetto all'errore di previsione) 
# che per confrontare i migliori modelli scelti (per tutti i modelli i FOLD di convalida sono gli stessi).

# NOTA: questa procedura può risultare troppo ottimista in quanto il confronto tra i modelli migliori
# avviene rispetto all'errore in corrispondenza del parametro selezionato:
# 1) Tramite CV seleziono il parametro ottimale: quello che minimizza l'errore medio di convalida
# 2) Per ogni modello finale così selezionato confronto tali errori medi convalida

#////////////////////////////////////////////////////////////////////////////
# Costruzione metrica di valutazione e relativo dataframe -------------------
#////////////////////////////////////////////////////////////////////////////

source("loss_functions.R")

# in generale uso sia MAE che MSE
USED.Loss = function(y.pred, y.test, weights = 1){
  return(c(MAE.Loss(y.pred, y.test, weights), MSE.Loss(y.pred, y.test, weights)))
}


# anche qua
df_err_quant = data.frame(name = NA, MAE = NA, MSE = NA)

NLoss_df_err = NCOL(df_err_quant) - 1


# Funzione per aggiornare il data.frame degli errori
# (inefficiente, ma amen, tanto le operazioni che deve eseguire sono sempre limitate)



#////////////////////////////////////////////////////////////////////////////
# Costruzione ID Fold convalida incrociata  -------------------
#////////////////////////////////////////////////////////////////////////////

# numero fold
K_FOLD = 4

NROW_DF = NROW(dati)

# matrice degli id dei fold della convalida incrociata
# NOTA: data la non garantita perfetta divisibilità del numero di osservazioni
# per il numero di fold è possibile che un fold abbia meno osservazioni degli altri

# ordine causale degli id
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
temp_err_matrix_cv = matrix(NA, nrow = K_FOLD, ncol = NLoss_df_err)
colnames(temp_err_matrix_cv) = colnames(df_err_quant[,-1])


for (i in 1:K_FOLD){
  temp_err_matrix_cv[i,] = USED.Loss(mean(dati$y[unlist(id_list_cv[-i])]),
                                     dati$y[id_list_cv[[i]]])
}



df_err_quant = Add_Test_Error(df_err_quant,
                              "cv mean",
                              colMeans(temp_err_matrix_cv))


# mediana
temp_err_matrix_cv = matrix(NA, nrow = K_FOLD, ncol = NLoss_df_err)


for (i in 1:K_FOLD){
  temp_err_matrix_cv[i,] = USED.Loss(median(dati$y[unlist(id_list_cv[-i])]),
                                     dati$y[id_list_cv[[i]]])
}



df_err_quant = Add_Test_Error(df_err_quant,
                              "cv median",
                              colMeans(temp_err_matrix_cv))

df_err_quant = na.omit(df_err_quant)

df_err_quant

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge e Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(glmnet)

# Compromesso varianza - distorsione: convalida incrociata sia per la scelta del parametro 
# di regolazione che per il confronto finale
# NOTA: questa procedura è sub-ottimale, poichè non ci stiamo totalmente tutelando 
# contro il sovradattamento, tuttavia, data la scarsa numerosità campioanaria 
# non sono presenti valide alternative

# Funzioni generali per ridge e lasso

# GLMNET CV cycles in case of few data sample
# @input n_k_fold (int): number of fold used, use the global variable
# @input my_id_list_cv (list):ids in each fold , use the global variable
# @input my_nloss_df_err (int): number of loss functions used, use global variable
#
# @input my_x (matrix): complete model matrix passed to glmnet
# @input my_y (vector): y glmnet argument
# @input my_alpha (int): alpha passed to glmnet (0 -> ridge, 1 -> lasso)
# @input my_lambda_vals (vector): vector of lambda used

# @return: matrix of CV folds averaged errors for each parameter value and each loss function 
FewDataCVCycleGLMNET = function(n_k_fold, my_id_list_cv,my_nloss_df_err,
                                my_x, my_y, my_alpha, my_lambda_vals){
  
  temp_err_array_cv = array(NA, dim = c(n_k_fold, length(my_lambda_vals), my_nloss_df_err))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    
    
    temp_glmnet = glmnet(x = my_x[id_train,], 
                         y = my_y[id_train], alpha = my_alpha,
                         lambda = my_lambda_vals)
    
    for (j in 1:length(my_lambda_vals)){
      temp_err_array_cv[k,j,] = USED.Loss(predict(temp_glmnet, my_x[id_test,])[,j],
                                          my_y[id_test])
    }
    
    rm(temp_glmnet)
    gc()
  }
  
  glmnet_cv_errs = matrix(NA, nrow = length(my_lambda_vals), ncol = my_nloss_df_err)
  
  for (i in 1:my_nloss_df_err){
    glmnet_cv_errs[,i] = apply(temp_err_array_cv[,,i], 2, mean)
  }
  
  return(glmnet_cv_errs)
  
}

# Plot CV errors
# @input my_param_values (vector): vector of parameters
# @input my_errs_matrix (matrix): rows -> parameters in the same order of my_param_values,
#                                 columns -> loss functions, 
#   each cell contains the cv error already averaged
# @input my_err_names (vector of strings): names of the errors,
# @input: other parameters are the usual plot parameters
# 
# @return: vector of param values minizing each loss

CvErrsPlotMin = function(my_param_values, my_errs_matrix, my_err_names,
                         my_main, my_xlab, my_legend_coords = "topright",
                         my_xlim = NULL, my_ylim = NULL){
  
  # PLOT section
  
  # if xlim null use entire x axis (default)
  if (is.null(my_xlim)){
    my_xlim = c(min(my_param_values), max(my_param_values))
  }
  
  # if ylim is null use min and max over the entire matrix
  if (is.null(my_ylim)){
    my_ylim = c(min(my_errs_matrix), max(my_errs_matrix))
  }
  
  
  
  plot(my_param_values, my_errs_matrix[,1],
       xlab = my_xlab, ylab = "error",
       main = my_main, pch = 16,
       xlim = my_xlim,
       ylim = my_ylim)
  
  for (i in 2:NCOL(my_errs_matrix)){
    points(my_param_values, my_errs_matrix[,i],
           pch = 15 + i, col = i)
  }
  
  legend(my_legend_coords,
         legend = my_err_names,
         col = 1:NCOL(my_errs_matrix),
         pch = 15 + (1:NCOL(my_errs_matrix)))
  
  
  # BEST PARAM VALUES section
  
  best_params = rep(NA, NCOL(my_errs_matrix))
  names(best_params) = my_err_names
  
  best_params = my_param_values[apply(my_errs_matrix, 2, which.min)]
  
  # plot them
  
  for (i in 1:length(best_params))
    abline(v = best_params[i], col = i)
  
  # return them
  return(best_params)
  
}


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


# Valori di lambda

lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
        alpha = 0, lambda.min.ratio = 1e-07)$lambda

# eventualmente basato sui risultati successivi
# lambda_vals = seq(1e-07,1e-03, by = 1e-05)

# Ridge ----------------
# Senza interazione 

ridge_no_interaction_errs = FewDataCVCycleGLMNET(K_FOLD, id_list_cv, NLoss_df_err,
                            X_mm_no_interaction, dati$y, 0,
                            lambda_vals)

ridge_no_int_mse_lambda = CvErrsPlotMin(lambda_vals, ridge_no_interaction_errs, colnames(df_err_quant[,-1]),
              "Ridge no interaction CV error", "lambda")

# lambda vals
# 0.02737854 0.02737854

# aggiungo l'errore: eventualmente scegliere anche il lambda basato su MAE

df_err_quant = Add_Test_Error(df_err_quant,
                              "ridge_no_int_mse_lambda",
                              c(ridge_no_interaction_errs[which.min(ridge_no_interaction_errs[,2]),1],
                                ridge_no_interaction_errs[which.min(ridge_no_interaction_errs[,2]),2]))



ridge_yes_interaction_errs = FewDataCVCycleGLMNET(K_FOLD, id_list_cv, NLoss_df_err,
                                                 X_mm_yes_interaction, dati$y, 0,
                                                 lambda_vals)

ridge_yes_int_mse_lambda = CvErrsPlotMin(lambda_vals, ridge_yes_interaction_errs, colnames(df_err_quant[,-1]),
                                        "Ridge yes interaction CV error", "lambda")

# lambda vals
#  1.030784 1.030784

df_err_quant = Add_Test_Error(df_err_quant,
                              "ridge_yes_int_mse_lambda",
                              c(ridge_yes_interaction_errs[which.min(ridge_yes_interaction_errs[,2]),1],
                                ridge_yes_interaction_errs[which.min(ridge_yes_interaction_errs[,2]),2]))

# Lasso ---------------
lasso_no_interaction_errs = FewDataCVCycleGLMNET(K_FOLD, id_list_cv, NLoss_df_err,
                                                 X_mm_no_interaction, dati$y, 1,
                                                 lambda_vals)

lasso_no_int_mse_lambda = CvErrsPlotMin(lambda_vals, lasso_no_interaction_errs, colnames(df_err_quant[,-1]),
                                        "lasso no interaction CV error", "lambda")

# lambda vals
# 0.003880837 0.003880837

# aggiungo l'errore: eventualmente scegliere anche il lambda basato su MAE

df_err_quant = Add_Test_Error(df_err_quant,
                              "lasso_no_int_mse_lambda",
                              c(lasso_no_interaction_errs[which.min(lasso_no_interaction_errs[,2]),1],
                                lasso_no_interaction_errs[which.min(lasso_no_interaction_errs[,2]),2]))



lasso_yes_interaction_errs = FewDataCVCycleGLMNET(K_FOLD, id_list_cv, NLoss_df_err,
                                                  X_mm_yes_interaction, dati$y, 1,
                                                  lambda_vals)

lasso_yes_int_mse_lambda = CvErrsPlotMin(lambda_vals, lasso_yes_interaction_errs, colnames(df_err_quant[,-1]),
                                        "lasso yes interaction CV error", "lambda")

# lambda vals
#  1.030784 1.030784

df_err_quant = Add_Test_Error(df_err_quant,
                              "lasso_yes_int_mse_lambda",
                              c(lasso_yes_interaction_errs[which.min(lasso_yes_interaction_errs[,2]),1],
                                lasso_yes_interaction_errs[which.min(lasso_yes_interaction_errs[,2]),2]))

df_err_quant

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Albero -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

# GLMNET CV cycles in case of few data sample
# @input n_k_fold (int): number of fold used, use the global variable
# @input my_id_list_cv (list):ids in each fold , use the global variable
# @input my_nloss_df_err (int): number of loss functions used, use global variable
#
# @input my_max_size (int): max size of the a pruned tree
#
# @return: matrix of CV folds averaged errors for each parameter value and each loss function 
FewDataCVCycleTree = function(n_k_fold, my_id_list_cv,my_nloss_df_err,
                              my_max_size = 30){
  
  # we use my_max_size - 1 because we start with size = 2
  temp_err_array_cv = array(NA, dim = c(n_k_fold, my_max_size - 1, my_nloss_df_err))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    
    # full grown tree
    temp_tree_full = tree(y ~.,
                          data = dati[id_train,],
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
      temp_err_array_cv[k,s-1,] = USED.Loss(predict(temp_tree_pruned,
                                                  dati[id_test,]),
                                          dati$y[id_test])
    }
    
    
    rm(temp_tree_full)
    rm(temp_tree_pruned)
    gc()
    
    print(paste("fold ", k))
  }
  
  # to fix it
  tree_cv_errs = matrix(NA, nrow = (my_max_size - 1), ncol = my_nloss_df_err)
  
  for (i in 1:my_nloss_df_err){
    tree_cv_errs[,i] = apply(temp_err_array_cv[,,i], 2, mean)
  }
  
  return(tree_cv_errs)
  
}

tree_cv_errs = FewDataCVCycleTree(K_FOLD, id_list_cv, NLoss_df_err)


tree_best_size = CvErrsPlotMin(lambda_vals, ridge_yes_interaction_errs, colnames(df_err_quant[,-1]),
                                         "Ridge yes interaction CV error", "lambda")

# tree_best_size
#  10

df_err_quant = Add_Test_Error(df_err_quant,
                              "ridge_yes_int_mse_lambda",
                              c(ridge_yes_interaction_errs[which.min(ridge_yes_interaction_errs[,2]),1],
                                ridge_yes_interaction_errs[which.min(ridge_yes_interaction_errs[,2]),2]))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello Additivo ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(gam)

# questo è più problematico...

















