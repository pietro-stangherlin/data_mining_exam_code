library(dplyr)

# Descrizione -----------------------
# pochi dati: usiamo convalida incrociata come criterio per confrontare i modelli finali


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
temp_err_matrix_cv = matrix(NA, nrow = K_FOLD, ncol = 2)


for (i in 1:K_FOLD){
  temp_err_matrix_cv[i,] = USED.Loss(mean(dati$y[unlist(id_list_cv[-i])]),
                                     dati$y[id_list_cv[[i]]])
}



df_err_quant = Add_Test_Error(df_err_quant,
                              "cv mean",
                              colMeans(temp_err_matrix_cv))


# mediana
temp_err_matrix_cv = matrix(NA, nrow = K_FOLD, ncol = 2)


for (i in 1:K_FOLD){
  temp_err_matrix_cv[i,] = USED.Loss(median(dati$y[unlist(id_list_cv[-i])]),
                                     dati$y[id_list_cv[[i]]])
}



df_err_quant = Add_Test_Error(df_err_quant,
                              "cv median",
                              colMeans(temp_err_matrix_cv))

# df_err_quant = df_err_quant[-which(is.na(df_err_quant)),]

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

# in questo caso dobbiamo creare una griglia di valori di lambda 
# creiamo la griglia adattando il modello su tutti i dati

X_mm_no_interaction_sss = model.matrix(formula_no_interaction_no_intercept,
                                       data = dati)

# sparsa
# X_mm_no_interaction_sss =  sparse.model.matrix(formula_no_interaction_no_intercept, data = dati)

lambda_vals = glmnet(x = X_mm_no_interaction_sss, y = dati$y,
        alpha = 0, lambda.min.ratio = 1e-07)$lambda

# eventualmente basato sui risultati successivi
# lambda_vals = seq(1e-07,1e-03, by = 1e-05)


# Ridge ----------------
# Senza interazione 

temp_err_array_cv = array(NA, dim = c(K_FOLD, length(lambda_vals), 2))


for (k in 1:K_FOLD){
  id_train = unlist(id_list_cv[-k])
  id_test = id_list_cv[[k]]
  
  
  
  temp_ridge = glmnet(x = X_mm_no_interaction_sss[id_train,], 
                      y = dati$y[id_train], alpha = 0,
                      lambda = lambda_vals)
  
  for (j in 1:length(lambda_vals)){
    temp_err_array_cv[k,j,] = USED.Loss(predict(temp_ridge, X_mm_no_interaction_sss[id_test,])[,j],
                                        dati$y[id_test])
  }
  
  rm(temp_ridge)
  gc()
}


ridge_no_interaction_errs = cbind(apply(temp_err_array_cv[,,1], 2, mean),
                                  apply(temp_err_array_cv[,,2], 2, mean))

names(ridge_no_interaction_errs) = c("MAE", "MSE")

plot(lambda_vals, ridge_no_interaction_errs[,2],
     xlab = "lambda_values", ylab = "MSE",
     main = "RIDGE no interaction CV error", pch = 16)

points(lambda_vals, ridge_no_interaction_errs[,1],
     xlab = "lambda_values", ylab = "MAE",
     main = "RIDGE no interaction CV error", pch = 16, col = "red")

legend("topright",
       legend = c("MAE", "MSE"),
       col = c("black", "red"),
       pch = 16)

# in questo caso riprovo con valori piccoli di lambda
best_lambda_ridge_no_interaction_mse = lambda_vals[which.min(ridge_no_interaction_errs[,2])]
best_lambda_ridge_no_interaction_mae = lambda_vals[which.min(ridge_no_interaction_errs[,1])]

abline(v = best_lambda_ridge_no_interaction_mse)
abline(v = best_lambda_ridge_no_interaction_mae, col = "red")

# aggiungo l'errore: eventualente scegliere anche il lambda basato su MAE

df_err_quant = Add_Test_Error(df_err_quant,
                              "ridge_no_int_mse_lambda",
                              c(ridge_no_interaction_errs[which.min(ridge_no_interaction_errs[,2]),1],
                                ridge_no_interaction_errs[which.min(ridge_no_interaction_errs[,2]),2]))


# Con interazione 

# X_mm_yes_interaction_sss = model.matrix(formula_yes_interaction_no_intercept,
#                                        data = dati)
#
# sparsa
# X_mm_yes_interaction_sss =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = dati)
#
#
# # lambda_vals = glmnet(x = X_mm_no_interaction_sss, y = dati$y,
# #                      alpha = 0, lambda.min.ratio = 1e-07)$lambda
# 
# # eventualmente basato sui risultati successivi
# # lambda_vals = seq(1e-07,1e-03, by = 1e-05)
# 
# 
# temp_err_array_cv = array(NA, dim = c(K_FOLD, length(lambda_vals), 2))
# 
# 
# for (k in 1:K_FOLD){
#   id_train = unlist(id_list_cv[-k])
#   id_test = id_list_cv[[k]]
#   
#   
#   
#   temp_ridge = glmnet(x = X_mm_yes_interaction_sss[id_train,], 
#                       y = dati$y[id_train], alpha = 0,
#                       lambda = lambda_vals)
#   
#   for (j in 1:length(lambda_vals)){
#     temp_err_array_cv[k,j,] = USED.Loss(predict(temp_ridge, X_mm_yes_interaction_sss[id_test,])[,j],
#                                         dati$y[id_test])
#   }
#   
#   rm(temp_ridge)
#   gc()
# }
# 
# 
# ridge_yes_interaction_errs = cbind(apply(temp_err_array_cv[,,1], 2, mean),
#                                   apply(temp_err_array_cv[,,2], 2, mean))
# 
# names(ridge_yes_interaction_errs) = c("MAE", "MSE")
# 
# plot(lambda_vals, ridge_yes_interaction_errs[,2],
#      xlab = "lambda_values", ylab = "MSE",
#      main = "RIDGE yes interaction CV error", pch = 16)
# 
# points(lambda_vals, ridge_yes_interaction_errs[,1],
#        xlab = "lambda_values", ylab = "MAE",
#        main = "RIDGE yes interaction CV error", pch = 16, col = "red")
# 
# legend("topright",
#        legend = c("MAE", "MSE"),
#        col = c("black", "red"),
#        pch = 16)
# 
# # in questo caso riprovo con valori piccoli di lambda
# best_lambda_ridge_yes_interaction_mse = lambda_vals[which.min(ridge_yes_interaction_errs[,2])]
# best_lambda_ridge_yes_interaction_mae = lambda_vals[which.min(ridge_yes_interaction_errs[,1])]
# 
# abline(v = best_lambda_ridge_yes_interaction_mse)
# abline(v = best_lambda_ridge_yes_interaction_mae, col = "red")
# 
# # aggiungo l'errore: eventualente scegliere anche il lambda basato su MAE
# 
# df_err_quant = Add_Test_Error(df_err_quant,
#                               "ridge_yes_int_mse_lambda",
#                               c(ridge_yes_interaction_errs[which.min(ridge_yes_interaction_errs[,2]),1],
#                                 ridge_yes_interaction_errs[which.min(ridge_yes_interaction_errs[,2]),2]))


# Lasso ---------------



temp_err_array_cv = array(NA, dim = c(K_FOLD, length(lambda_vals), 2))


for (k in 1:K_FOLD){
  id_train = unlist(id_list_cv[-k])
  id_test = id_list_cv[[k]]
  
  
  
  temp_lasso = glmnet(x = X_mm_no_interaction_sss[id_train,], 
                      y = dati$y[id_train], alpha = 1,
                      lambda = lambda_vals)
  
  for (j in 1:length(lambda_vals)){
    temp_err_array_cv[k,j,] = USED.Loss(predict(temp_lasso, X_mm_no_interaction_sss[id_test,])[,j],
                                        dati$y[id_test])
  }
  
  rm(temp_lasso)
  gc()
}


lasso_no_interaction_errs = cbind(apply(temp_err_array_cv[,,1], 2, mean),
                                  apply(temp_err_array_cv[,,2], 2, mean))

names(lasso_no_interaction_errs) = c("MAE", "MSE")

plot(lambda_vals, lasso_no_interaction_errs[,2],
     xlab = "lambda_values", ylab = "MSE",
     main = "lasso no interaction CV error", pch = 16)

points(lambda_vals, lasso_no_interaction_errs[,1],
       xlab = "lambda_values", ylab = "MAE",
       main = "lasso no interaction CV error", pch = 16, col = "red")

legend("topright",
       legend = c("MAE", "MSE"),
       col = c("black", "red"),
       pch = 16)

# in questo caso riprovo con valori piccoli di lambda
best_lambda_lasso_no_interaction_mse = lambda_vals[which.min(lasso_no_interaction_errs[,2])]
best_lambda_lasso_no_interaction_mae = lambda_vals[which.min(lasso_no_interaction_errs[,1])]

abline(v = best_lambda_lasso_no_interaction_mse)
abline(v = best_lambda_lasso_no_interaction_mae, col = "red")

# aggiungo l'errore: eventualente scegliere anche il lambda basato su MAE

df_err_quant = Add_Test_Error(df_err_quant,
                              "lasso_no_int_mse_lambda",
                              c(lasso_no_interaction_errs[which.min(lasso_no_interaction_errs[,2]),1],
                                lasso_no_interaction_errs[which.min(lasso_no_interaction_errs[,2]),2]))


# Con interazione 
# # lambda_vals = glmnet(x = X_mm_no_interaction_sss, y = dati$y,
# #                      alpha = 1, lambda.min.ratio = 1e-07)$lambda
# 
# # eventualmente basato sui risultati successivi
# # lambda_vals = seq(1e-07,1e-03, by = 1e-05)
# 
# 
# temp_err_array_cv = array(NA, dim = c(K_FOLD, length(lambda_vals), 2))
# 
# 
# for (k in 1:K_FOLD){
#   id_train = unlist(id_list_cv[-k])
#   id_test = id_list_cv[[k]]
#   
#   
#   
#   temp_lasso = glmnet(x = X_mm_yes_interaction_sss[id_train,], 
#                       y = dati$y[id_train], alpha = 1,
#                       lambda = lambda_vals)
#   
#   for (j in 1:length(lambda_vals)){
#     temp_err_array_cv[k,j,] = USED.Loss(predict(temp_lasso, X_mm_yes_interaction_sss[id_test,])[,j],
#                                         dati$y[id_test])
#   }
#   
#   rm(temp_lasso)
#   gc()
# }
# 
# 
# lasso_yes_interaction_errs = cbind(apply(temp_err_array_cv[,,1], 2, mean),
#                                    apply(temp_err_array_cv[,,2], 2, mean))
# 
# names(lasso_yes_interaction_errs) = c("MAE", "MSE")
# 
# plot(lambda_vals, lasso_yes_interaction_errs[,2],
#      xlab = "lambda_values", ylab = "MSE",
#      main = "lasso yes interaction CV error", pch = 16)
# 
# points(lambda_vals, lasso_yes_interaction_errs[,1],
#        xlab = "lambda_values", ylab = "MAE",
#        main = "lasso yes interaction CV error", pch = 16, col = "red")
# 
# legend("topright",
#        legend = c("MAE", "MSE"),
#        col = c("black", "red"),
#        pch = 16)
# 
# # in questo caso riprovo con valori piccoli di lambda
# best_lambda_lasso_yes_interaction_mse = lambda_vals[which.min(lasso_yes_interaction_errs[,2])]
# best_lambda_lasso_yes_interaction_mae = lambda_vals[which.min(lasso_yes_interaction_errs[,1])]
# 
# abline(v = best_lambda_lasso_yes_interaction_mse)
# abline(v = best_lambda_lasso_yes_interaction_mae, col = "red")
# 
# # aggiungo l'errore: eventualente scegliere anche il lambda basato su MAE
# 
# df_err_quant = Add_Test_Error(df_err_quant,
#                               "lasso_yes_int_mse_lambda",
#                               c(lasso_yes_interaction_errs[which.min(lasso_yes_interaction_errs[,2]),1],
#                                 lasso_yes_interaction_errs[which.min(lasso_yes_interaction_errs[,2]),2]))



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Albero -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Come per la ridge: anche in questo caso la procedura è sub-ottima


library(tree)


temp_err_array_cv = array(NA, dim = c(K_FOLD, length(lambda_vals), 2))


for (k in 1:K_FOLD){
  id_train = unlist(id_list_cv[-k])
  id_test = id_list_cv[[k]]
  
  
  
  temp_full_tree = glmnet(x = X_mm_no_interaction_sss[id_train,], 
                      y = dati$y[id_train], alpha = 0,
                      lambda = lambda_vals)
  
  for (j in 1:length(lambda_vals)){
    temp_err_array_cv[k,j,] = USED.Loss(predict(temp_ridge, X_mm_no_interaction_sss[id_test,])[,j],
                                        dati$y[id_test])
  }
  
  rm(temp_ridge)
  gc()
}


ridge_no_interaction_errs = cbind(apply(temp_err_array_cv[,,1], 2, mean),
                                  apply(temp_err_array_cv[,,2], 2, mean))

names(ridge_no_interaction_errs) = c("MAE", "MSE")

plot(lambda_vals, ridge_no_interaction_errs[,2],
     xlab = "lambda_values", ylab = "MSE",
     main = "RIDGE no interaction CV error", pch = 16)

points(lambda_vals, ridge_no_interaction_errs[,1],
       xlab = "lambda_values", ylab = "MAE",
       main = "RIDGE no interaction CV error", pch = 16, col = "red")

legend("topright",
       legend = c("MAE", "MSE"),
       col = c("black", "red"),
       pch = 16)

# in questo caso riprovo con valori piccoli di lambda
best_lambda_ridge_no_interaction_mse = lambda_vals[which.min(ridge_no_interaction_errs[,2])]
best_lambda_ridge_no_interaction_mae = lambda_vals[which.min(ridge_no_interaction_errs[,1])]

abline(v = best_lambda_ridge_no_interaction_mse)
abline(v = best_lambda_ridge_no_interaction_mae, col = "red")

# aggiungo l'errore: eventualente scegliere anche il lambda basato su MAE

df_err_quant = Add_Test_Error(df_err_quant,
                              "ridge_no_int_mse_lambda",
                              c(ridge_no_interaction_errs[which.min(ridge_no_interaction_errs[,2]),1],
                                ridge_no_interaction_errs[which.min(ridge_no_interaction_errs[,2]),2]))























