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


sorted_y_values = sort(unique(sss$y))


# /////////////////////////////////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Creazione indicatrice --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#carico la libreria nnet in cui c'e' il comando class.ind che crea le 
# variabili indicatrici per le modalità della risposta
library(nnet)
Y_sss = class.ind(sss$y)

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
m_multi0 = multinom(Y_sss ~., data = sss[,-y_index])

m_multi0


# scelgo il parametro di regolazione tramite AIC
# m_multi = step(m_multi0)
# salvo la formula 
# y ~ x2 + x7 + x8

m_multi = multinom(multinom(formula = Y_sss ~ x2 + x7 + x8, data = sss[,-y_index]))
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

# Lasso mgaussian multivar impone che i coef ci siano sempre o mai
# contemporaneamente per ogni modalita' della risposta
# selezione dei parametri di regolazione: stima - convalida

# No interazione 
lasso_no_int = glmnet(x = X_mm_no_interaction_sss[id_cb1,],
                 y = Y_sss[id_cb1,],
                 family = "mgaussian",
                 lambda.min.ratio = 1e-07)

lambda_vals = lasso_no_int$lambda

# previsione sulla convalida
pr_lasso_no_int_matr = predict(lasso_no_int, X_mm_no_interaction_sss[-id_cb1,], type = "response")

dim(pr_lasso_no_int_matr)

# ogni riga: valore diverso di lambda
# ogni colonna: valore della classe stimata
pr_lasso_no_int = matrix(NA, nrow = length(lambda_vals), ncol = length(sss$y[-id_cb1]))

dim(pr_lasso_no_int)

for(i in 1:NROW(pr_lasso_no_int)){
  pr_lasso_no_int[i,] = t(apply(pr_lasso_no_int_matr[,,i], 1, which.max))
}

dim(pr_lasso_no_int)

# conversione in char
pr_lasso_no_int = t(apply(pr_lasso_no_int, 1, function(row) sorted_y_values[row]))

dim(pr_lasso_no_int)

# controllo 
pr_lasso_no_int[1:10, 1:15]

# errore sull'insieme di convalida
err_lasso_no_int = apply(pr_lasso_no_int, 1, function(row) USED.Loss(row, sss$y[-id_cb1]))

length(err_lasso_no_int)
err_lasso_no_int

par(mfrow = c(1,1))
plot(log(lambda_vals), err_lasso_no_int,
     xlab = "log lambda", ylab = "err", main = "Lasso no interaction",
     pch = 16)

best_lambda_no_int = lambda_vals[which.min(err_lasso_no_int)]
best_lambda_no_int
abline(v = log(best_lambda_no_int))

coef(lasso_no_int, lasso_no_int$lambda[best_lambda_no_int])
# problema: lambda e' molto piccola
# in questo caso e' quasi come senza penalizzare


# errore sull'insieme di convalida

pred_final_lasso_no_int_matr = drop(predict(lasso_no_int,
                                  newx = X_mm_no_interaction_vvv,
                                  s = best_lambda_no_int))
dim(pred_final_lasso_no_int_matr)


pred_final_lasso_no_int_class = apply(pred_final_lasso_no_int_matr, 1, which.max)


pred_final_lasso_no_int_class = sorted_y_values[pred_final_lasso_no_int_class]

length(pred_final_lasso_no_int_class)

df_err_qual = Add_Test_Error(df_err_qual,
                             "lasso no int",
                             USED.Loss(pred_final_lasso_no_int_class, vvv$y))

df_err_qual

rm(lasso_no_int)
rm(pred_final_lasso_no_int)
rm(pred_final_lasso_no_int_class)
rm(pred_final_lasso_no_int_matr)

# SI interazione

lasso_yes_int = glmnet(x = X_mm_yes_interaction_sss[id_cb1,],
                      y = Y_sss[id_cb1,],
                      family = "mgaussian",
                      lambda.min.ratio = 1e-07)

lambda_vals = lasso_yes_int$lambda

# previsione sulla convalida
pr_lasso_yes_int_matr = predict(lasso_yes_int, X_mm_yes_interaction_sss[-id_cb1,], type = "response")

dim(pr_lasso_yes_int_matr)

# ogni riga: valore diverso di lambda
# ogni colonna: valore della classe stimata
pr_lasso_yes_int = matrix(NA, nrow = length(lambda_vals), ncol = length(sss$y[-id_cb1]))

dim(pr_lasso_yes_int)

for(i in 1:NROW(pr_lasso_yes_int)){
  pr_lasso_yes_int[i,] = t(apply(pr_lasso_yes_int_matr[,,i], 1, which.max))
}

dim(pr_lasso_yes_int)

# conversione in char
pr_lasso_yes_int = t(apply(pr_lasso_yes_int, 1, function(row) sorted_y_values[row]))

dim(pr_lasso_yes_int)

# controllo 
pr_lasso_yes_int[1:10, 1:15]

# errore sull'insieme di convalida
err_lasso_yes_int = apply(pr_lasso_yes_int, 1, function(row) USED.Loss(row, sss$y[-id_cb1]))

length(err_lasso_yes_int)
err_lasso_yes_int

par(mfrow = c(1,1))
plot(log(lambda_vals), err_lasso_yes_int,
     xlab = "log lambda", ylab = "err", main = "Lasso yes interaction",
     pch = 16)

best_lambda_yes_int = lambda_vals[which.min(err_lasso_yes_int)]
best_lambda_yes_int
abline(v = log(best_lambda_yes_int))

coef(lasso_yes_int, lasso_yes_int$lambda[best_lambda_yes_int])
# problema: lambda e' molto piccola
# in questo caso e' quasi come senza penalizzare


# errore sull'insieme di convalida

pred_final_lasso_yes_int_matr = drop(predict(lasso_yes_int,
                                            newx = X_mm_yes_interaction_vvv,
                                            s = best_lambda_yes_int))
dim(pred_final_lasso_yes_int_matr)


pred_final_lasso_yes_int_class = apply(pred_final_lasso_yes_int_matr, 1, which.max)


pred_final_lasso_yes_int_class = sorted_y_values[pred_final_lasso_yes_int_class]

length(pred_final_lasso_yes_int_class)

df_err_qual = Add_Test_Error(df_err_qual,
                             "lasso yes int",
                             USED.Loss(pred_final_lasso_yes_int_class, vvv$y))

df_err_qual

rm(lasso_yes_int)
rm(pred_final_lasso_yes_int)
rm(pred_final_lasso_yes_int_class)
rm(pred_final_lasso_yes_int_matr)



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MARS ---------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(earth)
# earth fa sempre regressione
# mentre polyspline puo fare multi con approssimazione della devianza

# attenzione ale interazioni: degree = 2
# interazioni fino al secondo ordine
m_mars = earth(Y_sss ~., sss[,-y_index], degree = 2)

# di fatto modello di regressione multivariato
# con criterio di penalizzazione gcv



# per ogni numero di basi incluse
m_mars$gcv.per.subset

plot(m_mars$gcv.per.subset, pch = 16,
     xlab = "Numero di basi",
     ylab = "GCV")
abline(v = which.min(m_mars$gcv.per.subset), col = "gold",
       main = "MARS")


summary(m_mars)

# # prima risposta
# plotmo(m_mars, nresponse = 1, ylim = NA)
# 
# plotmo(m_mars, nresponse = 2, ylim = NA)

pred_mars = apply(predict(m_mars, vvv), 1, which.max)
pred_mars_class = sorted_y_values[pred_mars]

df_err_qual = Add_Test_Error(df_err_qual,
                             "mars",
                             USED.Loss(pred_mars_class, vvv$y))

df_err_qual


rm(m_mars)
gc()

# potremmo anche provare con degree 1
# per interpretazione 

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
                                        mindev = 1e-04,
                                        mincut = 40))


# controllo che sia sovraadattato
plot(tree_full)

# potatura
tree_pruned = prune.tree(tree_full, newdata = sss[-id_cb1,])

plot(tree_pruned)
plot(tree_pruned, xlim = c(10, 90))

tree_best_size = tree_pruned$size[which.min(tree_pruned$dev)]
tree_best_size
# 11

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

pred_random_forest = predict(random_forest_model, data = vvv, type = "response")$predictions
dim(pred_random_forest)

pred_random_forest_class = apply(pred_random_forest, 1, which.max)
pred_random_forest_class = sorted_y_values[pred_random_forest_class]

df_err_qual = Add_Test_Error(df_err_qual,
                             "Random Forest",
                             USED.Loss(pred_random_forest_class,vvv$y))

df_err_qual

# Importanza delle variabili
vimp = importance(random_forest_model)

dotchart(vimp[order(vimp)])

rm(random_forest_model)
gc()

# /////////////////////////////////////////////////////////////////
#------------------------ Sintesi Finale -------------------------
# /////////////////////////////////////////////////////////////////

# Dei modelli migliori in base alle metriche traccio le curve lift e ROC
# altrimenti i grafici sono difficili da leggere

df_err_qual









