# l'unica funzione di perdita è il tasso di errata classificazione
source("lift_roc.R")
library(dplyr)


#////////////////////////////////////////////////////////////////////////////
# Costruzione metrica di valutazione e relativo dataframe -------------------
#////////////////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Qualitativa -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

MissErr = function(previsti, osservati){
  return( 1- sum(previsti == osservati) / length(previsti))
}



# funzione di convenienza
Null.Loss = function(y.pred, y.test, weights = 1){
  NULL
}

# °°°°°°°°°°°°°°°°°°°°°°° Warning: °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# cambia la funzione di errore per il problema specifico

USED.Loss = function(y.pred, y.test, weights = 1){
  return(MissErr(y.pred, y.test))
}


# anche qua
df_err_qual = data.frame(name = NA,
                         misclassification = NA)


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

# rimozione dataset originale
rm(dati)

# /////////////////////////////////////////////////////////////////
#------------------------ Analisi esplorative ---------------------
# /////////////////////////////////////////////////////////////////

# Analisi esplorativa sulla stima 
# eventuali inflazioni di zeri

# valutiamo se è sbilanciata 
table(sss$y)

y_rel_freqs = table(sss$y) / NROW(sss)

y_uniques = unique(sss$y)


# /////////////////////////////////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Classificazione Casuale --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# modello classificazione casuale

df_err_qual = Add_Test_Error(df_err_qual,
                             "sss threshold",
                             USED.Loss(sample(y_uniques, nrow(vvv), replace = TRUE, prob = y_rel_freqs),
                                       vvv$y))

df_err_qual

df_err_qual = na.omit(df_err_qual)

df_err_qual


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Regressione multinomiale --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(nnet)
m_multi0 = multinom(y ~., data = sss)

m_multi0


# scelgo il parametro di regolazione tramite AIC
# m_multi = step(m_multi0)
# salvo la formula 
# y ~ x2 + x7 + x8

m_multi = multinom(multinom(formula = y ~ x2 + x7 + x8, data = sss))
m_multi

pred_multinom = predict(m_multi, newdata = vvv)

df_err_qual = Add_Test_Error(df_err_qual,
                             "multinom",
                             USED.Loss(pred_multinom, vvv$y))

df_err_qual

rm(m_multi)
rm(pred_multinom)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge e Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Sparse
# valuta: se ci sono molte esplicative qualitative -> model.matrix con molti zeri
library(Matrix)
X_mm_no_interaction_sss =  sparse.model.matrix(formula_no_interaction_no_intercept, data = sss)
X_mm_no_interaction_vvv =  sparse.model.matrix(formula_no_interaction_no_intercept, data = vvv)

# # oneroso
X_mm_yes_interaction_sss =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = sss)
X_mm_yes_interaction_vvv =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = vvv)

# default
# stima 
# X_mm_no_interaction_sss = model.matrix(formula_no_interaction_no_intercept,
#                                              data = sss)
# 
# # verifica
# X_mm_no_interaction_vvv = model.matrix(formula_no_interaction_no_intercept,
#                                        data = vvv)

# Interazioni: stima 
# X_mm_yes_interaction_sss = model.matrix(formula_yes_interaction_no_intercept,
#                                    data = sss)

# Interazioni verifica
#
# X_mm_yes_interaction_vvv = model.matrix(formula_yes_interaction_no_intercept,
#                                         data = vvv)


library(glmnet)

# Lasso --------------------

# matrice del disegno tranne l'intercetta
X = model.matrix(m0)[,-1]

# Lasso mgaussian multivar impone che i coef ci siano sempre o mai
# contemporaneamente per ogni modalita' della risposta
m_lasso = glmnet(x = X[cb1,],
                 y = Y[cb1,],
                 family = "mgaussian")
x11()
par(mfrow = c(2,3))
plot(m_lasso)

pr_lasso_all = predict(m_lasso, X[cb2,], type = "response")
pr_lasso = matrix(NA, dim(pr_lasso_all)[1], dim(pr_lasso_all)[3])
for(k in 1:NCOL(pr_lasso)){
  pr_lasso[,k] = apply(pr_lasso_all[,,k], 1, which.max)
}

str(pr_lasso)
err_l = apply(pr_lasso, 2, function(x) ce(x, sss$diagnosi[cb2]))

par(mfrow = c(1,1))
plot(log(m_lasso$lambda), err_l)

bb = which.min(err_l)
bb

coef(m_lasso, m_lasso$lambda[bb])
# problema: lambda e' molto piccola
# in questo caso e' quasi come senza penalizzare

XXv = model.matrix(diagnosi ~., data = vvv)[,-1]
pr_lasso = predict(m_lasso, newx = XXv, s = log(m_lasso$lambda)[bb])

str(pr_lasso)
pr_lasso_class = apply(drop(pr_lasso), 1, which.max)

str(drop(pr_lasso))
errori[NROW(errori) + 1, ] = c("lin multi lasso",
                               ce(pr_lasso_class, vvv$diagnosi))



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MARS ---------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Albero -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

# albero che sovraadatta

# default: molto fitto
# tree_full = tree(factor(y) ~.,
#                  data = sss[id_cb1,],
#                  control = tree.control(nobs = length(id_cb1),
#                                         mindev = 1e-05,
#                                         minsize = 2))

# se per motivi computazionali l'albero sopra non può essere stimato
# aumento il numero di elementi in ogni foglia (sub-ottimale,
# ma meglio di non stimare il modello).
tree_full = tree(factor(y) ~.,
                 data = sss[id_cb1,],
                 control = tree.control(nobs = length(id_cb1),
                                        mindev = 1e-05,
                                        mincut = 100))


# controllo che sia sovraadattato
plot(tree_full)

# potatura
tree_pruned = prune.tree(tree_full, newdata = sss[-id_cb1,])

plot(tree_pruned)
plot(tree_pruned, xlim = c(10, 90))

tree_best_size = tree_pruned$size[which.min(tree_pruned$dev)]
# 15

abline(v = tree_best_size, col = "red")

final_tree_pruned = prune.tree(tree_full, best = tree_best_size)

plot(final_tree_pruned)
text(final_tree_pruned, cex = 0.7)

pred_tree_pruned = predict(final_tree_pruned, newdata = vvv, type = "class")

df_err_qual = Add_Test_Error(df_err_qual,
                             "tree_pruned best",
                             USED.Loss(pred_tree_pruned, vvv$y))

df_err_qual


rm(tree_full)
gc()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(snowfall)

# Implementazione in parallelo
library(ranger)

sfInit(cpus = 4, parallel = T)

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
  
  err[i] = mean(sfSapply(rep(1:4),
                         function(x) ranger(factor(y) ~., data = sss,
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
                                 function(x) ranger(factor(y) ~., sss,
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

# best_mtry = 2

# modello finale e previsioni
random_forest_model = ranger(factor(y) ~., sss,
                             mtry = best_mtry,
                             num.trees = 400,
                             oob.error = TRUE,
                             importance = "permutation",
                             probability = T)

pred_random_forest = predict(random_forest_model, data = vvv, type = "class")

df_err_qual = Add_Test_Error(df_err_qual,
                             "Random Forest",
                             USED.Loss(pred_random_forest,vvv$y))

df_err_qual

# Importanza delle variabili
vimp = importance(random_forest_model)

dotchart(vimp[order(vimp)])

rm(random_forest_model)
gc()










