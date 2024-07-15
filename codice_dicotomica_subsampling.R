source("lift_roc.R")
library(dplyr)

# In caso di dati sbilanciati
# sottocampiono i negativi

#////////////////////////////////////////////////////////////////////////////
# Costruzione metrica di valutazione e relativo dataframe -------------------
#////////////////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Qualitativa -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

tabella.sommario = function(previsti, osservati,
                            print_bool = FALSE,
                            ready_table = NA){
  # inizializza: per evitare casi in cui la tabella non è 2x2
  n <-  matrix(0, nrow = 2, ncol = 2)
  
  for (i in 1:length(previsti)){
    if(previsti[i] == osservati[i]){
      # 0 == 0 case
      if (previsti[i] == 0){
        n[1,1] = n[1,1] + 1
      }
      # 1 == 1
      else{
        n[2,2] = n[2,2] + 1}
    }
    
    else{
      # 0 != 1
      if (previsti[i] == 0){
        n[1,2] = n[1,2] + 1
      }
      # 1 != 0
      else{
        n[2,1] = n[2,1] + 1
      }
      
    }
  }
  
  if(typeof(ready_table) != "logical"){
    n = ready_table
  }
  
  err.tot <- 1-sum(diag(n))/sum(n)
  zeros.observed = sum(n[1,1] + n[2,1])
  ones.observed = sum(n[1,2] + n[2,2])
  
  fn <- n[1,2]/ones.observed
  fp <- n[2,1]/zeros.observed
  
  tp = 1 - fn
  tn = 1 - fp
  
  f.score = 2*tp / (2*tp + fp + fn)
  
  if(print_bool == TRUE){
    print(n)
    print(c("err tot", "fp", "fn", "f.score"))
    print(c(err.tot, fp, fn, f.score))}
  
  return(round(c(err.tot, fp, fn, f.score), 4))
}


# funzione di convenienza
Null.Loss = function(y.pred, y.test, weights = 1){
  NULL
}


# °°°°°°°°°°°°°°°°°°°°°°° Warning: °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# cambia la funzione di errore per il problema specifico

USED.Loss = function(y.pred, y.test, ready_table_used = NA){
  return(tabella.sommario(y.pred, y.test, ready_table = ready_table_used))
}


# anche qua
df_err_qual = data.frame(name = NA,
                         misclassification = NA,
                         fp = NA,
                         fn = NA,
                         f_score = NA)


# Funzione per aggiornare il data.frame degli errori
# (inefficiente, ma amen, tanto le operazioni che deve eseguire sono sempre limitate)
# Add the error to the df_error data.frame:
# if the df_error already has a model name in name column with the same as input: update the error value
# otherwise append the new name and error
# arguments:
# @df_error (data.frame): data.frame with columns: [1]: name and [2]: error
# @model_name (char): character with the model name
# @loss_value (num): numeric with the error on the test set
# @return: df_error

Add_Test_Error = function(df_error, model_name, loss_value){
  # check if the model name is already in the data.frame
  is_name = model_name %in% df_error[,1]
  
  # if yes: get the index and subscribe
  if(is_name){
    df_error[which(df_error[,1] == model_name),2:ncol(df_error)] = loss_value
  }
  
  else{
    # get the last index
    df_error[NROW(df_error) + 1,] = c(model_name, loss_value)
  }
  
  return(df_error)
}


# /////////////////////////////////////////////////////////////////
#------------------------ Stima e Verifica ------------------------
# /////////////////////////////////////////////////////////////////


# Eventualmente modificare la proporzione
id_stima = sample(1:NROW(dati), 0.75 * NROW(dati))

sss = dati[id_stima,]
vvv = dati[-id_stima,]


# In caso di convalida nell'insieme di stima
id_cb1 = sample(1:NROW(sss), 0.8 * NROW(sss))
id_cb2 = setdiff(1:NROW(sss), id_cb1)


# /////////////////////////////////////////////////////////////////
#------------------------ Analisi esplorative ---------------------
# /////////////////////////////////////////////////////////////////

# Analisi esplorativa sulla stima 
# eventuali inflazioni di zeri

# valutiamo se è sbilanciata 
# ed eventualmente se è ragionevole cambiare la solita soglia a 0.5
table(sss$y)

# soglia di classificazione: cambia eventualmente con
table(sss$y)[2] / NROW(sss)

threshold = 0.3


# elimino i data.frame di stima e verifica visto che non servono
rm(sss)
rm(vvv)
rm(id_cb1)
rm(id_cb2)
# /////////////////////////////////////////////////////////////////
#------------------------ Sottocampionamento ----------------------
# /////////////////////////////////////////////////////////////////

K_FOLD = 4


# Stimiamo i modelli su dati bilanciati, sia in stima che dentro ogni fold di convalida 

# fold sbilanciati, usati per verifica
cv_id_unbal_matr = matrix(sample(1:NROW(dati)), ncol = K_FOLD)

# Fold bilanciati, utilizziamo per la stima

# proportion of "ones" in each balanced fold 
ONES_OBS_PROPORTION = 0.3

# per ogni fold controlla il numero massimo di osservazioni di "ones"

max_ones_for_fold = 0

# aggiorna il valore
for(k in 1:K_FOLD){
  id_tmp = cv_id_unbal_matr[,k]
  temp_max_length = length(which(dati$y[id_tmp] == 1)) # tutti gli 1 di quel fold
  if(temp_max_length > max_ones_for_fold){
    max_ones_for_fold = temp_max_length}}

max_ones_for_fold

BALANCED_OBS_NUMBER = max_ones_for_fold / ONES_OBS_PROPORTION

# matrice dei dati di stima bilanciati
cv_id_bal_matr = matrix(NA, BALANCED_OBS_NUMBER, K_FOLD)


for(k in 1:K_FOLD){
  id_tmp = cv_id_unbal_matr[,k]
  mm = id_tmp[which(dati$y[id_tmp] == 1)] # tutti gli 1 di quel fold
  vv = id_tmp[sample(which(dati$y[id_tmp] == 0),  size = (BALANCED_OBS_NUMBER - length(mm)))]
  cv_id_bal_matr[,k] = c(mm,vv)
}


# Possiamo vettorizzare cv_id per avere tutte le osservazioni che mi creano un dataset bilanciato
ids_bal = c(cv_id_bal_matr)
table(dati[ids_bal,]$y)

# id non bilanciati vettorizzati
ids_unbal = c(cv_id_unbal_matr)

# /////////////////////////////////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Classificazione Casuale --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# modello classificazione casuale sull'insieme di verifica

df_err_qual = Add_Test_Error(df_err_qual,
                             "random threshold",
                             USED.Loss(rbinom(length(ids_unbal), 1, threshold), dati$y[ids_unbal]))

df_err_qual = na.omit(df_err_qual)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello lineare Forward --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# se y non è già numerica
dati$y = as.numeric(dati$y) # controlla la codifica

# Tengo conto del compromesso varianza - distorsione tramite AIC
# per la selezione della dimensione del modello
lm0 = lm(y ~ 1, data = dati[ids_bal,])

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
lm_step_no_interaction = step(lm0, scope = formula_no_interaction_yes_intercept,
                              direction = "forward")

rm(lm0)


# salvo la formula: ATTENZIONE: cambiare
# y ~ x7 + x2 + x8 + x1 + anno

# Convalida per la valutazione
# il numero di colonne della matrice corrisponde con l'output di tabella sommario
err_cv_tmp_array = matrix(0, K_FOLD, 4)
for(j in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-j])
  id_verifica = cv_id_unbal_matr[,j] 
  
  temp_cv_model = lm(as.formula(lm_step_no_interaction), dati[id_stima,])
  pr_tmp = predict(temp_cv_model, dati[id_verifica, ], type = "response")
  err_cv_tmp_array[j,] = USED.Loss(pr_tmp > threshold, dati$y[id_verifica])
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "lm_step_no_interaction",
                             colMeans(err_cv_tmp_array))
df_err_qual


# salvo i coefficienti e rimuovo gli oggetti dalla memoria
lm_step_no_interaction_coef = coef(lm_step_no_interaction)

rm(lm_step_no_interaction)
gc()

# computazionalmente costoso (probabilmente)
lm_step_yes_interaction = step(lm0, scope = formula_yes_interaction_yes_intercept,
                              direction = "forward")

# y ~ x7 + x2 + x8 + anno + x2:x8 + x7:x8

err_cv_tmp_array = matrix(0, K_FOLD, 4)
for(j in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-j])
  id_verifica = cv_id_unbal_matr[,j] 
  
  temp_cv_model = lm(as.formula(lm_step_yes_interaction), dati[id_stima,])
  pr_tmp = predict(temp_cv_model, dati[id_verifica, ], type = "response")
  err_cv_tmp_array[j,] = USED.Loss(pr_tmp > threshold, dati$y[id_verifica])
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "lm_step_yes_interaction",
                             colMeans(err_cv_tmp_array))
df_err_qual


# salvo i coefficienti e rimuovo gli oggetti dalla memoria
lm_step_yes_interaction_coef = coef(lm_step_yes_interaction)

rm(lm_step_yes_interaction)
gc()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Logistica forward --------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

glm0 = glm(y ~ 1, data = dati[ids_bal,],
           family = "binomial")
# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
glm_step_no_interaction = step(glm0, scope = formula_no_interaction_yes_intercept,
                              direction = "forward")

rm(lm0)


# salvo la formula: ATTENZIONE: cambiare
# y ~ x7 + x2 + x8 + anno

# Convalida per la valutazione
# il numero di colonne della matrice corrisponde con l'output di tabella sommario
err_cv_tmp_array = matrix(0, K_FOLD, 4)
for(j in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-j])
  id_verifica = cv_id_unbal_matr[,j] 
  
  temp_cv_model = glm(as.formula(glm_step_no_interaction), dati[id_stima,], family = "binomial")
  pr_tmp = predict(temp_cv_model, dati[id_verifica, ], type = "response")
  err_cv_tmp_array[j,] = USED.Loss(pr_tmp > threshold, dati$y[id_verifica])
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "glm_step_no_interaction",
                             colMeans(err_cv_tmp_array))
df_err_qual


# salvo i coefficienti e rimuovo gli oggetti dalla memoria
glm_step_no_interaction_coef = coef(glm_step_no_interaction)

rm(glm_step_no_interaction)
gc()

# computazionalmente costoso (probabilmente)
glm_step_yes_interaction = step(glm0, scope = formula_yes_interaction_yes_intercept,
                               direction = "forward")

# y ~ x7 + x2 + x8 + anno + x2:x8 + x7:x8

err_cv_tmp_array = matrix(0, K_FOLD, 4)
for(j in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-j])
  id_verifica = cv_id_unbal_matr[,j] 
  
  temp_cv_model = glm(as.formula(glm_step_yes_interaction), dati[id_stima,], family = "binomial")
  pr_tmp = predict(temp_cv_model, dati[id_verifica, ], type = "response")
  err_cv_tmp_array[j,] = USED.Loss(pr_tmp > threshold, dati$y[id_verifica])
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "lm_step_yes_interaction",
                             colMeans(err_cv_tmp_array))
df_err_qual


# salvo i coefficienti e rimuovo gli oggetti dalla memoria
glm_step_yes_interaction_coef = coef(glm_step_yes_interaction)

rm(glm_step_yes_interaction)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge e Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# Sparse
# valuta: se ci sono molte esplicative qualitative -> model.matrix con molti zeri
library(Matrix)
X_mm_no_interaction =  sparse.model.matrix(formula_no_interaction_no_intercept, data = dati)

# # oneroso
X_mm_yes_interaction =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = dati)


# default
# X_mm_no_interaction = model.matrix(formula_no_interaction_no_intercept,
#                                              data = dati)

# Interazioni: stima 
# X_mm_yes_interaction = model.matrix(formula_yes_interaction_no_intercept,
#                                    dati)

library(glmnet)

lambda_vals = glmnet(x = X_mm_no_interaction, y = dati$y,
                     alpha = 0, lambda.min.ratio = 1e-07)$lambda

# eventualmente basato sui risultati successivi
# lambda_vals = seq(1e-07,1e-03, by = 1e-05)


# Ridge -----------------
# NO interazione 

# l'ultima dimnensione (4) è dovuta alle 4 metriche della tabella sommario
err_cv_tmp_array = array(NA, dim = c(K_FOLD, length(lambda_vals), 4))

for(k in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-k])
  id_verifica = cv_id_unbal_matr[,k] 
  
  # stima sul dataset bilanciato
  temp_cv_model = glmnet(x = X_mm_no_interaction[id_stima,], y = dati$y[id_stima],
                 alpha = 0, lambda = lambda_vals)
  
  tmp_pred = predict(temp_cv_model,
                     X_mm_no_interaction[id_verifica,],
                     type = "response")
  
  for (j in 1:length(lambda_vals)){
    err_cv_tmp_array[k,j,] = USED.Loss((tmp_pred[,j] > threshold) %>% as.numeric(),
                          dati$y[id_verifica])
  }
  
  
  rm(temp_cv_model)
  rm(tmp_pred)
  gc()
  
  print(k)
}



# controlla gli errori e seleziona il modello migliore 

cv_errs = cbind(apply(err_cv_tmp_array[,,1], 2, mean),
                apply(err_cv_tmp_array[,,2], 2, mean),
                apply(err_cv_tmp_array[,,3], 2, mean),
                apply(err_cv_tmp_array[,,4], 2, mean))



names(cv_errs) = c("miss", "fp", "fn", "f_score")

plot(lambda_vals, cv_errs[,1],
     xlab = "lambda_values", ylab = "Metrics",
     main = "RIDGE no interaction CV error", pch = 16,
     ylim = c(0,1))

points(lambda_vals, cv_errs[,2],
       main = "RIDGE no interaction CV error", pch = 16, col = "red")

points(lambda_vals, cv_errs[,3],
       main = "RIDGE no interaction CV error", pch = 16, col = "blue")

points(lambda_vals, cv_errs[,4],
       main = "RIDGE no interaction CV error", pch = 16, col = "violet")

legend("topright",
       legend = c("miss", "fp", "fn", "f_score"),
       col = c("black", "red", "blue", "violet"),
       pch = 16)


# di default scelgo  f_score come criterio
# in base al problema può cambiare 

ridge_no_inter_index_best = which.max(cv_errs[,4])
lambda_vals[ridge_no_inter_index_best]
#  0.0003413895

# potrebbe essere necessario ristimare il modello

# scegli il modello migliore rispetto al criterio
df_err_qual = Add_Test_Error(df_err_qual,
                             "ridge_no_interaction",
                             cv_errs[ridge_no_inter_index_best,])
df_err_qual

# SI interazione 

# l'ultima dimnensione (4) è dovuta alle 4 metriche della tabella sommario
err_cv_tmp_array = array(NA, dim = c(K_FOLD, length(lambda_vals), 4))

for(k in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-k])
  id_verifica = cv_id_unbal_matr[,k] 
  
  # stima sul dataset bilanciato
  temp_cv_model = glmnet(x = X_mm_yes_interaction[id_stima,], y = dati$y[id_stima],
                 alpha = 0, lambda = lambda_vals)
  
  tmp_pred = predict(temp_cv_model,
                     X_mm_yes_interaction[id_verifica,],
                     type = "response")
  
  for (j in 1:length(lambda_vals)){
    err_cv_tmp_array[k,j,] = USED.Loss((tmp_pred[,j] > threshold) %>% as.numeric(),
                          dati$y[id_verifica])
  }
  
  
  rm(temp_cv_model)
  rm(tmp_pred)
  gc()
  
  print(k)
}



# controlla gli errori e seleziona il modello migliore 

cv_errs = cbind(apply(err_cv_tmp_array[,,1], 2, mean),
                apply(err_cv_tmp_array[,,2], 2, mean),
                apply(err_cv_tmp_array[,,3], 2, mean),
                apply(err_cv_tmp_array[,,4], 2, mean))



names(cv_errs) = c("miss", "fp", "fn", "f_score")

plot(lambda_vals, cv_errs[,1],
     xlab = "lambda_values", ylab = "Metrics",
     main = "RIDGE yes interaction CV error", pch = 16,
     ylim = c(0,1))

points(lambda_vals, cv_errs[,2],pch = 16, col = "red")

points(lambda_vals, cv_errs[,3],pch = 16, col = "blue")

points(lambda_vals, cv_errs[,4],pch = 16, col = "violet")

legend("topright",
       legend = c("miss", "fp", "fn", "f_score"),
       col = c("black", "red", "blue", "violet"),
       pch = 16)


# di default scelgo  f_score come criterio
# in base al problema può cambiare 

ridge_yes_int_index_best = which.max(cv_errs[,4])
lambda_vals[ridge_yes_int_index_best]
#  0.0003413895

# potrebbe essere necessario ristimare il modello

# scegli il modello migliore rispetto al criterio
df_err_qual = Add_Test_Error(df_err_qual,
                             "ridge_yes_interaction",
                             cv_errs[ridge_yes_int_index_best,])
df_err_qual


# Lasso -----------------

# NO interazione 

# l'ultima dimnensione (4) è dovuta alle 4 metriche della tabella sommario
err_cv_tmp_array = array(NA, dim = c(K_FOLD, length(lambda_vals), 4))

for(k in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-k])
  id_verifica = cv_id_unbal_matr[,k] 
  
  # stima sul dataset bilanciato
  temp_cv_model = glmnet(x = X_mm_no_interaction[id_stima,], y = dati$y[id_stima],
                 alpha = 1, lambda = lambda_vals)
  
  tmp_pred = predict(temp_cv_model,
                     X_mm_no_interaction[id_verifica,],
                     type = "response")
  
  for (j in 1:length(lambda_vals)){
    err_cv_tmp_array[k,j,] = USED.Loss((tmp_pred[,j] > threshold) %>% as.numeric(),
                          dati$y[id_verifica])
  }
  
  
  rm(temp_cv_model)
  rm(tmp_pred)
  gc()
  
  print(k)
}



# controlla gli errori e seleziona il modello migliore 

cv_errs = cbind(apply(err_cv_tmp_array[,,1], 2, mean),
                apply(err_cv_tmp_array[,,2], 2, mean),
                apply(err_cv_tmp_array[,,3], 2, mean),
                apply(err_cv_tmp_array[,,4], 2, mean))



names(cv_errs) = c("miss", "fp", "fn", "f_score")

plot(lambda_vals, cv_errs[,1],
     xlab = "lambda_values", ylab = "Metrics",
     main = "lasso no interaction CV error", pch = 16,
     ylim = c(0,1))

points(lambda_vals, cv_errs[,2],
       main = "lasso no interaction CV error", pch = 16, col = "red")

points(lambda_vals, cv_errs[,3],
       main = "lasso no interaction CV error", pch = 16, col = "blue")

points(lambda_vals, cv_errs[,4],
       main = "lasso no interaction CV error", pch = 16, col = "violet")

legend("topright",
       legend = c("miss", "fp", "fn", "f_score"),
       col = c("black", "red", "blue", "violet"),
       pch = 16)


# di default scelgo  f_score come criterio
# in base al problema può cambiare 

lasso_no_inter_index_best = which.max(cv_errs[,4])
lambda_vals[lasso_no_inter_index_best]
#  0.0003413895

# potrebbe essere necessario ristimare il modello

# scegli il modello migliore rispetto al criterio
df_err_qual = Add_Test_Error(df_err_qual,
                             "lasso_no_interaction",
                             cv_errs[lasso_no_inter_index_best,])
df_err_qual

# SI interazione 

# l'ultima dimnensione (4) è dovuta alle 4 metriche della tabella sommario
err_cv_tmp_array = array(NA, dim = c(K_FOLD, length(lambda_vals), 4))

for(k in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-k])
  id_verifica = cv_id_unbal_matr[,k] 
  
  # stima sul dataset bilanciato
  temp_cv_model = glmnet(x = X_mm_yes_interaction[id_stima,], y = dati$y[id_stima],
                 alpha = 1, lambda = lambda_vals)
  
  tmp_pred = predict(temp_cv_model,
                     X_mm_yes_interaction[id_verifica,],
                     type = "response")
  
  for (j in 1:length(lambda_vals)){
    err_cv_tmp_array[k,j,] = USED.Loss((tmp_pred[,j] > threshold) %>% as.numeric(),
                          dati$y[id_verifica])
  }
  
  
  rm(temp_cv_model)
  rm(tmp_pred)
  gc()
  
  print(k)
}



# controlla gli errori e seleziona il modello migliore 

cv_errs = cbind(apply(err_cv_tmp_array[,,1], 2, mean),
                apply(err_cv_tmp_array[,,2], 2, mean),
                apply(err_cv_tmp_array[,,3], 2, mean),
                apply(err_cv_tmp_array[,,4], 2, mean))



names(cv_errs) = c("miss", "fp", "fn", "f_score")

plot(lambda_vals, cv_errs[,1],
     xlab = "lambda_values", ylab = "Metrics",
     main = "lasso yes interaction CV error", pch = 16,
     ylim = c(0,1))

points(lambda_vals, cv_errs[,2],pch = 16, col = "red")

points(lambda_vals, cv_errs[,3],pch = 16, col = "blue")

points(lambda_vals, cv_errs[,4],pch = 16, col = "violet")

legend("topright",
       legend = c("miss", "fp", "fn", "f_score"),
       col = c("black", "red", "blue", "violet"),
       pch = 16)


# di default scelgo  f_score come criterio
# in base al problema può cambiare 

lasso_yes_int_index_best = which.max(cv_errs[,4])
lambda_vals[lasso_yes_int_index_best]
#  0.0003413895

# potrebbe essere necessario ristimare il modello

# scegli il modello migliore rispetto al criterio
df_err_qual = Add_Test_Error(df_err_qual,
                             "lasso_yes_interaction",
                             cv_errs[lasso_yes_int_index_best,])
df_err_qual

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Albero -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

# dimensione massima - 1 (albero)
TREE_SIZES_MAX = 30


err_cv_tmp_array = array(NA, dim = c(K_FOLD, TREE_SIZES_MAX, 4))

for(k in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-k])
  id_verifica = cv_id_unbal_matr[,k] 
  
  # stima sul dataset bilanciato
  temp_cv_model = tree(factor(y) ~ . , data = dati[id_stima,],
               split = "deviance",
               control=tree.control(nobs=length(id_stima),
                                    minsize = 2,
                                    mindev = 0.001))
  
  # Lista di alberi candidati (tutti i livelli di potatura)
  tree_list = lapply(2:(TREE_SIZES_MAX + 1), function(l) prune.tree(temp_cv_model, best = l))
  pred_list = lapply(tree_list, function(x) predict(x, dati[id_verifica, ]))
  
  for(j in 1:TREE_SIZES_MAX){
    err_cv_tmp_array[k,j,] = USED.Loss(pred_list[[j]][,2] > threshold, dati$y[id_verifica])
  }
  
  print(k)
  
  rm(tree_list)
  rm(pred_list)
  gc()
  }



# controlla gli errori e seleziona il modello migliore 

cv_errs = cbind(apply(err_cv_tmp_array[,,1], 2, mean),
                apply(err_cv_tmp_array[,,2], 2, mean),
                apply(err_cv_tmp_array[,,3], 2, mean),
                apply(err_cv_tmp_array[,,4], 2, mean))



names(cv_errs) = c("miss", "fp", "fn", "f_score")

plot(2:(TREE_SIZES_MAX + 1), cv_errs[,1],
     xlab = "tree size", ylab = "metrics",
     main = "Tree", pch = 16,
     ylim = c(0,1))

points(2:(TREE_SIZES_MAX + 1), cv_errs[,2],pch = 16, col = "red")

points(2:(TREE_SIZES_MAX + 1), cv_errs[,3],pch = 16, col = "blue")

points(2:(TREE_SIZES_MAX + 1), cv_errs[,4],pch = 16, col = "violet")

legend("topright",
       legend = c("miss", "fp", "fn", "f_score"),
       col = c("black", "red", "blue", "violet"),
       pch = 16)


tree_size_best = which.max(cv_errs[,4]) + 1
#  3

# potrebbe essere necessario ristimare il modello

# scegli il modello migliore rispetto al criterio
df_err_qual = Add_Test_Error(df_err_qual,
                             "tree",
                             cv_errs[tree_size_best - 1,])
df_err_qual

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# GAM ---------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# solo se ha senso: non tutte le esplicative sono qualitative
library(gam)

# selezione delle variabili impiegando i gradi di libertà equivalenti
gam0 = gam(y~ 1, dati[ids_bal,],
                  family = "binomial")

my_gam_scope = gam.scope(dati[ids_bal,-y_index], arg = c("df=2", "df=3", "df=4", "df=5", "df=6"))

gam_step = step.Gam(gam0, scope = my_gam_scope)
#y ~ x2 + x7 + x8 + anno

err_cv_tmp_array = matrix(0, K_FOLD, 4)

for(j in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-j])
  id_verifica = cv_id_unbal_matr[,j] 
  
  # cambia la formula
  temp_cv_model = gam(formula(gam_step), data = dati[id_stima,], family = "binomial")
  pr_tmp = predict(temp_cv_model, dati[id_verifica, ], type = "response")
  err_cv_tmp_array[j,] = USED.Loss(pr_tmp > threshold, dati$y[id_verifica])
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "gam",
                             colMeans(err_cv_tmp_array))
df_err_qual




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MARS ---------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

factor_index = which(colnames(X_mm_no_interaction) != var_num_names)

library(polspline)

m_mars = polymars(dati$y[ids_bal], X_mm_no_interaction[ids_bal,],
                  factors = factor_index, gcv=1,
                  maxsize = 60)
m_mars$model

m_mars$fitting$GCV


plot(m_mars$fitting$size, m_mars$fitting$GCV, 
     col = (m_mars$fitting[,1] + 1), pch = 16, ylab = "GCV", xlab = "Numero di basi",
     main = "MARS")
legend("topright", col = c(1,2), c("Crescita", "Potatura"), pch = 16)
abline(v = m_mars$fitting$size[which.min(m_mars$fitting$GCV)], col = "gold", lty = 2, 
       lwd = 2) 

mars_best_size = m_mars$fitting$size[which.min(m_mars$fitting$GCV)]

X_mars = design.polymars(m_mars, X_mm_no_interaction)
Xm = data.frame(y = dati$y, X_mars)

err_cv_tmp_array = matrix(0, K_FOLD, 4)

for(j in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-j])
  id_verifica = cv_id_unbal_matr[,j] 
  
  temp_cv_model = lm(y~ -1 + ., data = Xm[id_stima,])
  pr_tmp = predict(temp_cv_model, Xm[id_verifica, ], type = "response")
  err_cv_tmp_array[j,] = USED.Loss(pr_tmp > threshold, dati$y[id_verifica])
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "mars",
                             colMeans(err_cv_tmp_array))
df_err_qual



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(randomForest)

sfInit(cpus = 4, parallel = T)

sfLibrary(ranger)
sfExport(list = c("dati", "ids_bal", "threshold", "RangerOOBConfusion"))

rf_miss_err = sfLapply(1:10, fun = function(l) RangerOOBConfusion(dati$y[ids_bal],
                                                            df[ids_bal,],
                                                            l,
                                                            300))



rf_metrics_m = lapply(rf_all, function(x) USED.Loss(1,1, x$confusion))

rf_metrics_m = matrix(unlist(rf_metrics_m), ncol = 4,
                      byrow = T)

plot(rf_metrics_m[,4], type = "l", xlab = "numero di variabili campionate",
     ylab = "f_score", main = "Random Forest")

m_best = which.max(rf_metrics_m[,4])

# valutazione errore tramite cv
err_cv_tmp_array = matrix(NA, K_FOLD, 4)

# qua non mettiamo cutoff
for(k in 1:K_FOLD){
  
  id_stima = c(cv_id_bal_matr[,-j])
  id_verifica = cv_id_unbal_matr[,j] 
  
  rf_tmp = randomForest(factor(y)~., data = dati[id_stima,], mtry = m_best)
  pr_tmp = predict(rf_tmp, newdata =  dati[id_verifica, ],
                   type = "prob")
  
  err_cv_tmp_array[k,] =  USED.Loss(pr_tmp[,2] > threshold, dati$y[id_verifica])
  
  print(k)
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "Random Forest",
                             colMeans(err_cv_tmp_array))
df_err_qual



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Boosting ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(ada)

# per ogni iterazione calcoliamo l'errore di classificazione 
# OOB sui dati bilanciati
# nota: questa è una procedura sub-ottimale dovuta a motivi di tempo
# sub - ottima: stabilizzazione sull'insieme di stima
iter_boost = 300

# Stump 
m_boost = ada(y = dati$y[ids_bal], x = dati[ids_bal, -y_index],
              iter = iter_boost,
              control = rpart.control(maxdepth=1,
                                      cp=-1,
                                      minsplit=0,xval=0))
plot(m_boost, test = T)

err_boost = matrix(NA, 700, 4)

for(i in seq(iter_boost, 700, by = 20)){
  pr_tmp = predict(m_boost, dati[id_bal,],
                   n.iter = i,
                   type = "prob")[,2]
  
  err_boost[i,] = USED.Loss(pr_tmp > threshold, dati$morto[id])
  cat(i, "\n")
}

plot(err_boost[,4], pch = 16)
# non si stabilizza, dovrei provare ad andare avanti

err = matrix(NA, K_FOLD, 4)


for(k in 1:K_FOLD){
  
  id_stima = c(cv_id_bal_matr[,-k])
  id_verifica = cv_id_unbal_matr[,k] 
  
  m_tmp = ada(factor(y)~., data = dati[id_stima,], iter = 500, control = rpart.control(maxdepth=1,
                                                                                       cp=-1,
                                                                                       minsplit=0,xval=0))
  pr_tmp = predict(m_tmp, newdata =  dati[id_verifica, ],
                   type = "prob")
  
  err[k,] =  USED.Loss(pr_tmp[,2] > threshold, dati$y[id_verifica])
  print(k)
  
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "Boosting Stump ",
                             colMeans(colMeans(err)))
df_err_qual


# 3 Split
m_boost = ada(y ~., dati[ids_bal, ], iter = 500,
              control = rpart.control(maxdepth = 3))
plot(m_boost)

err_boost = matrix(NA, 700, 4)

for(i in seq(1,700, by = 20)){
  pr_tmp = predict(m_boost, dati[id_bal,],
                   n.iter = i,
                   type = "prob")[,2]
  
  err_boost[i,] = fun.errori(pr_tmp > threshold, dati$morto[id])
  cat(i, "\n")
}

plot(err_boost[,4], pch = 16)
# non si stabilizza, dovrei provare ad andare avanti

err = matrix(NA, K_FOLD, 4)


for(k in 1:K_FOLD){
  
  id_stima = c(cv_id_bal_matr[,-k])
  id_verifica = cv_id_unbal_matr[,k] 
  
  m_tmp = ada(factor(y)~., data = dati[id_stima,], iter = 500, control = rpart.control(maxdepth = 3))
  pr_tmp = predict(m_tmp, newdata =  dati[id_verifica, ],
                   type = "prob")
  
  err[k,] =  USED.Loss(pr_tmp[,2] > threshold, dati$y[id_verifica])
  print(k)
  
}

df_err_qual = Add_Test_Error(df_err_qual,
                             "Boosting 3 split ",
                             colMeans(colMeans(err)))
df_err_qual


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Support Vector Machine ---------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(e1071)


# Stima Convalida
# Kernel Radiale

ranges = seq(2,10,by=0.5)

# l'ultima dimnensione (4) è dovuta alle 4 metriche della tabella sommario
err_cv_tmp_array = array(NA, dim = c(K_FOLD, length(ranges), 4))

for(k in 1:K_FOLD) {
  # Stimiamo su un modello bilanciato
  id_stima = c(cv_id_bal_matr[,-k])
  id_verifica = cv_id_unbal_matr[,k] 
  
  # stima sul dataset bilanciato
  for (i in ranges){
    temp_cv_model <- svm(factor(y)~., data= dati[id_stima,], cost = ranges[i], kernel = "radial",
                         probability = TRUE)
    tmp_pred <- predict(temp_cv_model, newdata= dati[id_verifica,], probability = TRUE) #, decision.values=TRUE)
    err_cv_tmp_array[k,i,] = USED.Loss((tmp_pred[,i] > threshold) %>% as.numeric(),
                                       dati$y[id_verifica])
    rm(temp_cv_model)
    rm(tmp_pred)
    gc()
    
    print("i")
    print(i)
    }
  
  print("k")
  print(k)
}



# controlla gli errori e seleziona il modello migliore 

cv_errs = cbind(apply(err_cv_tmp_array[,,1], 2, mean),
                apply(err_cv_tmp_array[,,2], 2, mean),
                apply(err_cv_tmp_array[,,3], 2, mean),
                apply(err_cv_tmp_array[,,4], 2, mean))



names(cv_errs) = c("miss", "fp", "fn", "f_score")

plot(lambda_vals, cv_errs[,1],
     xlab = "lambda_values", ylab = "Metrics",
     main = "SVM radial CV error", pch = 16,
     ylim = c(0,1))

points(lambda_vals, cv_errs[,2], pch = 16, col = "red")

points(lambda_vals, cv_errs[,3], pch = 16, col = "blue")

points(lambda_vals, cv_errs[,4],  pch = 16, col = "violet")

legend("topright",
       legend = c("miss", "fp", "fn", "f_score"),
       col = c("black", "red", "blue", "violet"),
       pch = 16)


# di default scelgo  f_score come criterio
# in base al problema può cambiare 

svm_range_best = which.max(cv_errs[,4])
ranges[svm_range_best]
#  0.0003413895

# potrebbe essere necessario ristimare il modello

# scegli il modello migliore rispetto al criterio
df_err_qual = Add_Test_Error(df_err_qual,
                             "svm radial",
                             cv_errs[svm_range_best,])
df_err_qual


# /////////////////////////////////////////////////////////////////
#------------------------ Sintesi Finale -------------------------
# /////////////////////////////////////////////////////////////////


df_err_qual



