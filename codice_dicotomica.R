source("lift_roc.R")
library(dplyr)
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



#////////////////////////////////////////////////////////////////////////////
# Costruzione metrica di valutazione e relativo dataframe -------------------
#////////////////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Qualitativa -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

tabella.sommario = function(previsti, osservati){
  n <-  table(previsti,osservati)
  err.tot <- 1-sum(diag(n))/sum(n)
  zeros.observed = sum(n[1,1] + n[2,1])
  ones.observed = sum(n[1,2] + n[2,2])
  
  fn <- n[1,2]/ones.observed
  fp <- n[2,1]/zeros.observed
  
  tp = 1 - fn
  tn = 1 - fp
  
  f.score = 2*tp / (2*tp + fp + fn)
  
  print(n)
  print(c("err tot", "fp", "fn", "f.score"))
  print(c(err.tot, fp, fn, f.score))
  
  return(round(c(err.tot, fp, fn, f.score), 4))
}



# funzione di convenienza
Null.Loss = function(y.pred, y.test, weights = 1){
  NULL
}

# La funzione di perdita può cambiare in base al problema
# Cambia MAE eventualmente

# °°°°°°°°°°°°°°°°°°°°°°° Warning: °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# cambia la funzione di errore per il problema specifico

USED.Loss = function(y.pred, y.test, weights = 1){
  return(tabella.sommario(y.pred, y.test))
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


# lista dei predittori (se stima - verifica)
pred_list = list()  


# /////////////////////////////////////////////////////////////////
#------------------------ Analisi esplorative ---------------------
# /////////////////////////////////////////////////////////////////

# Analisi esplorativa sulla stima 
# eventuali inflazioni di zeri

# valutiamo se è sbilanciata 
# ed eventualmente se è ragionevole cambiare la solita soglia a 0.5
table(sss$y)

# soglia di classificazione: cambia eventualmente con
# table(sss$y)[2] / NROW(sss)

threshold = 0.2


# /////////////////////////////////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Classificazione Casuale --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# modello classificazione casuale

df_err_qual = Add_Test_Error(df_err_qual,
                              "sss threshold",
                              USED.Loss(rbinom(nrow(vvv), 1, threshold), vvv$y))

df_err_qual

# solo la prima volta per rimuovere NA
# df_err_qual = df_err_qual[-1,]

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello lineare Forward --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# se y non è già numerica
# sss$y = as.numeric(sss$y)
# vvv$y = as.numeric(vvv$y)

lm0 = lm(y ~ 1, data = sss)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
lm_step_no_interaction = step(lm0, scope = formula_no_interaction_yes_intercept,
                 direction = "forward")

rm(lm0)


# salvo la formula: ATTENZIONE: cambiare
# y ~ x7 + x2 + x8 + anno
lm_step_no_interaction = lm(y ~ x7 + x2 + x8 + anno, data = sss)

pred_lm_no_interaction = predict(lm_step_no_interaction, newdata = vvv)

df_err_qual = Add_Test_Error(df_err_qual,
                              "lm_step_no_interaction",
                              USED.Loss(pred_lm_no_interaction > threshold %>% as.numeric(),
                                        vvv$y))
df_err_qual

pred_list$pred_lm_no_interaction = pred_lm_no_interaction

rm(pred_lm_no_interaction)

# salvo i coefficienti e rimuovo gli oggetti dalla memoria
lm_step_no_interaction_coef = coef(lm_step_no_interaction)

rm(lm_step_no_interaction)
gc()

# computazionalmente costoso (probabilmente)
# lm_step_yes_interaction = step(lm0, scope = formula_yes_interaction_yes_intercept,
#                               direction = "forward")

# salvo la formula

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Logistica forward --------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# converto in factor 
glm0 = glm(factor(y) ~ 1, data = sss, family = "binomial")

glm_step_no_interaction = step(glm0, scope = formula_no_interaction_yes_intercept,
                              direction = "forward")

rm(glm0)


# salvo la formula: ATTENZIONE: cambiare
# factor(y) ~ x7 + x2 + x8 + anno

glm_step_no_interaction = glm(factor(y) ~ x7 + x2 + x8 + anno,
                              family = "binomial",
                             data = sss)

pred_glm_no_interaction = predict(glm_step_no_interaction, newdata = vvv, type = "response")

df_err_qual = Add_Test_Error(df_err_qual,
                             "glm_step_no_interaction",
                             USED.Loss(pred_glm_no_interaction > threshold,
                                       vvv$y))
df_err_qual

pred_list$pred_glm_no_interaction = pred_glm_no_interaction
rm(pred_glm_no_interaction)

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
# Ridge ------

# ATTENZIONE: accertati che y sia numerica

# No Interaction
# Selezione tramite cv
ridge_cv_no_interaction = cv.glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                    alpha = 0,
                                    lambda.min.ratio = 1e-07)

plot(ridge_cv_no_interaction)

# salvo i risultati
# > ridge_cv_no_interaction$lambda.min
# [1] 0.0002575906
# > ridge_cv_no_interaction$lambda.1se
# [1] 1145154


# ri-stimo i modelli con il lambda scelto

# ridge_no_interaction_l1se = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
#                                    alpha = 0,
#                                    lambda = ridge_cv_no_interaction$lambda.1se)


ridge_no_interaction_lmin = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                   alpha = 0,
                                   lambda = ridge_cv_no_interaction$lambda.min)

# elimino dalla memoria l'oggetto ridge_cv
rm(ridge_cv_no_interaction)
gc()


# pred_ridge_no_interaction_l1se =  predict(ridge_no_interaction_l1se, newx = X_mm_no_interaction_vvv)
# 
# # previsione ed errore
# df_err_qual = Add_Test_Error(df_err_qual,
#                               "ridge_no_interaction_l1se",
#                               USED.Loss(pred_ridge_no_interaction_l1se > threshold %>% as.numeric(),
#                                         vvv$y))
# 
# pred_list$pred_ridge_no_interaction_l1se = as.vector(pred_ridge_no_interaction_l1se)
# rm(pred_ridge_no_interaction_l1se)


pred_ridge_no_interaction_lmin =  predict(ridge_no_interaction_lmin, newx = X_mm_no_interaction_vvv)

df_err_qual = Add_Test_Error(df_err_qual,
                              "ridge_no_interaction_lmin",
                              USED.Loss(pred_ridge_no_interaction_lmin > threshold %>% as.numeric(),
                                        vvv$y))

pred_list$pred_ridge_no_interaction_lmin = as.vector(pred_ridge_no_interaction_lmin)
rm(pred_ridge_no_interaction_lmin)

df_err_qual
# YES Interaction: potrebbe essere troppo computazionalmente oneroso
# Selezione tramite cv
ridge_cv_yes_interaction = cv.glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
                                    alpha = 0,
                                    lambda.min.ratio = 1e-07)

plot(ridge_cv_yes_interaction)

# salvo i risultati
# > ridge_cv_yes_interaction$lambda.min
# [1] 1866.315
# > ridge_cv_yes_interaction$lambda.1se
# [1] 1145154


# ri-stimo i modelli con il lambda scelto

# ridge_yes_interaction_l1se = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                    alpha = 0,
#                                    lambda = ridge_cv_yes_interaction$lambda.1se)

ridge_yes_interaction_lmin = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
                                   alpha = 0,
                                   lambda = ridge_cv_yes_interaction$lambda.min)

# elimino dalla memoria l'oggetto ridge_cv
rm(ridge_cv_yes_interaction)
gc()

# pred_ridge_yes_interaction_l1se = predict(ridge_yes_interaction_l1se, newx = X_mm_yes_interaction_vvv)

# previsione ed errore
# df_err_qual = Add_Test_Error(df_err_qual,
#                               "ridge_yes_interaction_l1se",
#                               USED.Loss(pred_ridge_yes_interaction_l1se > threshold %>% as.numeric(),
#                                         vvv$y))
# 
# pred_list$pred_ridge_yes_interaction_l1se = as.vector(pred_ridge_yes_interaction_l1se)
# rm(pred_ridge_yes_interaction_l1se)

pred_ridge_yes_interaction_lmin = predict(ridge_yes_interaction_lmin, newx = X_mm_yes_interaction_vvv)



df_err_qual = Add_Test_Error(df_err_qual,
                              "ridge_yes_interaction_lmin",
                              USED.Loss(predict(ridge_yes_interaction_lmin, newx = X_mm_yes_interaction_vvv)> threshold %>% as.numeric(),
                                        vvv$y))

pred_list$pred_ridge_yes_interaction_lmin = as.vector(pred_ridge_yes_interaction_lmin)
rm(pred_ridge_yes_interaction_lmin)



# Lasso ------

# No Interaction
# Selezione tramite cv
lasso_cv_no_interaction = cv.glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                    alpha = 1,
                                    lambda.min.ratio = 1e-07)

plot(lasso_cv_no_interaction)

# salvo i risultati
# > lasso_cv_no_interaction$lambda.min
# [1] 1866.315
# > lasso_cv_no_interaction$lambda.1se
# [1] 1145154


# ri-stimo i modelli con il lambda scelto

# lasso_no_interaction_l1se = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
#                                    alpha = 1,
#                                    lambda = lasso_cv_no_interaction$lambda.1se)


lasso_no_interaction_lmin = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                   alpha = 1,
                                   lambda = lasso_cv_no_interaction$lambda.min)

# elimino dalla memoria l'oggetto lasso_cv
rm(lasso_cv_no_interaction)
gc()


# pred_lasso_no_interaction_l1se =  predict(lasso_no_interaction_l1se, newx = X_mm_no_interaction_vvv)
# 
# # previsione ed errore
# df_err_qual = Add_Test_Error(df_err_qual,
#                              "lasso_no_interaction_l1se",
#                              USED.Loss(pred_lasso_no_interaction_l1se > threshold %>% as.numeric(),
#                                        vvv$y))
# 
# pred_list$pred_lasso_no_interaction_l1se = as.vector(pred_lasso_no_interaction_l1se)
# rm(pred_lasso_no_interaction_l1se)


pred_lasso_no_interaction_lmin =  predict(lasso_no_interaction_lmin, newx = X_mm_no_interaction_vvv)

df_err_qual = Add_Test_Error(df_err_qual,
                             "lasso_no_interaction_lmin",
                             USED.Loss(pred_lasso_no_interaction_lmin > threshold %>% as.numeric(),
                                       vvv$y))

pred_list$pred_lasso_no_interaction_lmin = as.vector(pred_lasso_no_interaction_lmin)
rm(pred_lasso_no_interaction_lmin)

df_err_qual
# YES Interaction: potrebbe essere troppo computazionalmente oneroso
# Selezione tramite cv
lasso_cv_yes_interaction = cv.glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
                                     alpha = 1,
                                     lambda.min.ratio = 1e-07)

plot(lasso_cv_yes_interaction)

# salvo i risultati
# > lasso_cv_yes_interaction$lambda.min
# [1] 1866.315
# > lasso_cv_yes_interaction$lambda.1se
# [1] 1145154


# ri-stimo i modelli con il lambda scelto

# lasso_yes_interaction_l1se = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                     alpha = 1,
#                                     lambda = lasso_cv_yes_interaction$lambda.1se)

lasso_yes_interaction_lmin = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
                                    alpha = 1,
                                    lambda = lasso_cv_yes_interaction$lambda.min)

# elimino dalla memoria l'oggetto lasso_cv
rm(lasso_cv_yes_interaction)
gc()

# pred_lasso_yes_interaction_l1se = predict(lasso_yes_interaction_l1se, newx = X_mm_yes_interaction_vvv)
# 
# # previsione ed errore
# df_err_qual = Add_Test_Error(df_err_qual,
#                              "lasso_yes_interaction_l1se",
#                              USED.Loss(pred_lasso_yes_interaction_l1se > threshold %>% as.numeric(),
#                                        vvv$y))
# 
# pred_list$pred_lasso_yes_interaction_l1se = as.vector(pred_lasso_yes_interaction_l1se)
# rm(pred_lasso_yes_interaction_l1se)

pred_lasso_yes_interaction_lmin = predict(lasso_yes_interaction_lmin, newx = X_mm_yes_interaction_vvv)



df_err_qual = Add_Test_Error(df_err_qual,
                             "lasso_yes_interaction_lmin",
                             USED.Loss(predict(lasso_yes_interaction_lmin, newx = X_mm_yes_interaction_vvv)> threshold %>% as.numeric(),
                                       vvv$y))

pred_list$pred_lasso_yes_interaction_lmin = as.vector(pred_lasso_yes_interaction_lmin)
rm(pred_lasso_yes_interaction_lmin)

df_err_qual

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

pred_tree_pruned = predict(final_tree_pruned, newdata = vvv)[,2]

df_err_qual = Add_Test_Error(df_err_qual,
                              "tree_pruned best",
                              USED.Loss(pred_tree_pruned > threshold, vvv$y))

df_err_qual

pred_list$pred_tree = as.vector(pred_tree_pruned)

rm(tree_full)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello Additivo ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(gam)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# stepwise forward
gam0 = gam(y ~ 1, data = sss, family = "binomial")
# riconosce le qualitative se sono fattori
my_gam_scope = gam.scope(sss[,-y_index], arg = c("df=2", "df=3", "df=4", "df=5", "df=6"))

# prova anche parallelo
# require(doMC)
# registerDoMC(cores=4)
# step.Gam(Gam.object,scope ,parallel=TRUE)

gam_step = step.Gam(gam0, scope = my_gam_scope)

# salvo il modello finale
# y ~ y ~ x2 + x7 + x8 + anno
gam_step = gam(y ~ x2 + x7 + x8 + anno,
                 data = sss)

object.size(gam_step)

pred_gam = predict(gam_step, newdata = vvv, type = "response")

df_err_qual = Add_Test_Error(df_err_qual,
                              "additivo_step",
                              USED.Loss(pred_gam > threshold %>% as.numeric(),
                                        vvv$y))

df_err_qual

pred_list$pred_gam_step = as.vector(pred_gam)

rm(gam_step)


gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MARS ---------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# devo ottenere gli indici delle colonne
# delle variabili qualitative della matrice del disegno (senza intercetta)
factor_index = which(colnames(X_mm_no_interaction_sss) != var_num_names)
num_index = which(colnames(X_mm_no_interaction_sss) == var_num_names)

library(polspline)

# tramite gcv
# °°°°°°°°°°°°°°°°°°°°°° Warning °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# ATTENZIONE: se prima X_mm_no_interaction_sss era sparsa ora la devo
# definire di nuovo non sparsa
mars1 = polymars(responses = sss$y,
                 predictors = X_mm_no_interaction_sss,
                 gcv = 1,
                 factors = factor_index,
                 maxsize = 40)


mars1$fitting
plot(mars1$fitting$size, mars1$fitting$GCV,
     col = as.factor(mars1$fitting$`0/1`),
     pch = 16,
     xlab = "numero di basi",
     ylab = "GCV")
legend(c("topright"),
       legend = c("crescita", "potatura"),
       col = c("black","red"),
       pch = 16)

min_size_mars = mars1$fitting$size[which.min(mars1$fitting$GCV)]
abline(v = min_size_mars)

pred_mars = predict(mars1, x = X_mm_no_interaction_vvv)

df_err_qual = Add_Test_Error(df_err_qual,
                              "MARS",
                              USED.Loss(pred_mars > threshold %>% as.numeric(),
                                        vvv$y))

df_err_qual

pred_list$pred_mars = as.vector(pred_mars)

mars1$model # salva output come commento
# pred1 knot1 level1 pred2 knot2         coefs           SE
# 1      0    NA     NA     0    NA  1.345771e+05 1.982770e+04
# 2     65    NA     NA     0    NA -1.467800e+00 9.380822e-01
# 3     35    NA      0     0    NA -2.436611e+04 3.591602e+03
# 4     40    NA      0     0    NA -2.483568e+04 3.774879e+03
# 5     14    NA      0     0    NA -2.744385e+04 2.161495e+03
# 6     65 11063     NA     0    NA  3.186730e+00 2.282815e+00
# 7      7    NA      0     0    NA  7.112268e+03 1.782258e+03
# 8     42    NA      0     0    NA  2.602210e+04 4.650814e+03
# 9     65 61904     NA     0    NA -1.396349e-01 1.535435e+00
# 10    39    NA      0     0    NA -7.829038e+03 2.562383e+03
# 11    65 39206     NA     0    NA -2.196532e+00 1.277251e+00
# 12     5    NA      0     0    NA -1.341572e+04 3.179571e+03
# 13    64    NA      0     0    NA -9.647382e+03 1.938139e+03
# 14    73    NA      0     0    NA -6.400039e+03 1.348250e+03
# 15    66    NA     NA     0    NA -9.575513e+01 1.980574e+02
# 16    74    NA      0     0    NA -5.812757e+03 1.375098e+03
# 17     9    NA      0     0    NA -1.158179e+04 3.395830e+03
# 18    72    NA      1     0    NA  3.283980e+03 1.274277e+03
# 19     8    NA      0     0    NA  9.131712e+03 3.104115e+03
# 20    66    29     NA     0    NA  3.236912e+03 1.018062e+03
# 21    66    53     NA     0    NA  5.994898e+03 2.489864e+03
# 22    65    NA     NA    66    NA  1.398396e-01 3.619635e-02
# 23    65    NA     NA    66    53 -1.844708e+00 1.172169e-01
# 24    65 61904     NA    66    NA -2.224549e-01 5.899403e-02
# 25    65 61904     NA    66    53  1.448929e+00 8.412837e-02
# 26    23    NA      0     0    NA  8.270854e+03 2.662728e+03
# 27    44    NA      0     0    NA -4.294871e+04 1.558417e+04
# 28    66    25     NA     0    NA -8.179595e+02 6.522641e+02
# 29    65    NA     NA    66    25 -1.932911e-01 5.372105e-02
# 30    65 61904     NA    66    25  1.067143e+00 7.764876e-02
# 31    31    NA      1     0    NA -7.283399e+03 3.916180e+03
# 32     6    NA      1     0    NA  5.020683e+03 2.213931e+03
# 33    20    NA      1     0    NA  5.930547e+03 3.213112e+03
# 34    27    NA      0     0    NA -6.423668e+03 4.100316e+03
# 35    66    61     NA     0    NA -2.799976e+03 1.024265e+03
# 36    66    33     NA     0    NA -2.973695e+03 1.540092e+03
# 37    65    NA     NA    66    33  6.306275e-03 7.939390e-02
# 38    65 61904     NA    66    33 -5.146211e-01 8.624953e-02
# 39    65    NA     NA    66    61  8.007892e-01 6.819904e-02
# 40    66    49     NA     0    NA -2.504278e+03 1.886772e+03
# 41    65    NA     NA    66    49  9.222163e-01 7.744037e-02
# 42    65 11063     NA    66    NA -1.191347e-01 6.911702e-02
# 43    65 11063     NA    66    33  3.686043e-01 5.574049e-02
# 44    65 39206     NA    66    NA  1.036151e-01 3.571673e-02
# 45    65 39206     NA    66    61 -6.305747e-01 9.198519e-02
# 46    56    NA      0     0    NA -1.093920e+04 6.612370e+03
# 47    46    NA      1     0    NA  6.236679e+03 3.051664e+03
# 48    63    NA      0     0    NA -4.660421e+03 2.329214e+03
# 49    66    37     NA     0    NA  1.091341e+02 1.230665e+03
# 50    65    NA     NA    66    37  1.421784e-01 6.207068e-02
# 51    65 61904     NA    66    37 -1.095875e+00 7.846485e-02
# 52    65 15404     NA     0    NA  7.040805e+00 2.162681e+00
# 53    65 15404     NA    66    NA -1.916707e-01 6.081694e-02

# paste(mars1$model, collapse = ", ") >>
mars_model_str = "c(0, 65, 35, 40, 14, 7, 42, 39, 5, 64, 74, 73, 66, 65, 9, 23, 65, 65, 44, 8, 72, 65, 65, 31, 66, 66, 65, 65, 66, 6, 20, 27, 66, 65, 65, 66, 65, 65, 66, 65, 65, 66, 65, 65, 66, 65, 56, 46, 63, 66, 65, 65, 65, 65, 65, 65, 66, 65, 65, 65, 65), c(NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 61640, 61640, NA, NA, NA, 39106, 39106, NA, 29, 53, NA, 61640, 25, NA, NA, NA, 49, NA, 61640, 61, NA, 61640, 37, NA, 61640, 41, NA, 61640, 21, NA, NA, NA, NA, 45, NA, 39106, 22881, 22881, 22881, 8019, 33, NA, 61640, 8019, 8019), c(NA, NA, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, NA, NA, 1, 0, NA, NA, 1, 0, 0, NA, NA, 0, NA, NA, NA, NA, NA, 1, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 1, 1, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA), c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 66, 0, 0, 0, 0, 66, 0, 0, 0, 66, 66, 0, 0, 0, 0, 0, 66, 66, 0, 66, 66, 0, 66, 66, 0, 66, 66, 0, 66, 0, 0, 0, 0, 66, 66, 0, 66, 66, 0, 0, 66, 66, 66, 66), c(NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 53, 53, NA, NA, NA, NA, NA, 49, 49, NA, 61, 61, NA, 37, 37, NA, 41, 41, NA, 21, NA, NA, NA, NA, 45, 45, NA, NA, 37, NA, NA, 33, 33, NA, 45), c(56160.0631406908, 4.73277017883286, -23722.4847921431, -26499.997282827, -29572.9437134434, 6787.65765429342, 25603.830121484, -7391.67360114537, -13911.3615710762, 9471.22949762025, -6452.49995908382, -5331.89055605198, 343.657905419548, -0.147100207961118, 10043.494368162, 8165.46181493377, -15.9332665886454, 0.507565575879143, 42816.0258459062, 9106.50736492801, -2966.77040866398, -7.0907303398058, 0.250915140971562, 7635.25430281457, 2908.24916300038, 11198.3541000731, -2.62272663574175, 2.91293188853425, \n-2645.5380484128, 3962.88127773202, -5668.80475847789, -7710.38719582726, -11320.8205438702, 2.19156945453529, -2.54024674697911, -3098.0463083301, 0.814312749986175, -1.06222825868313, -1496.82362442338, 0.281646955935303, -4.61694441939485, 3962.50295425648, -0.665486853742289, 3.63617321587881, 774.646320168858, 0.115476314131583, 11959.3946078128, 5835.52653387644, -4582.0707975685, 1128.31027277246, 0.0782701604358963, -0.243041781133891, 21.1309073042109, -0.680179836618281, 0.904444424864142, \n-4.91996252551403, -1884.74035776914, 0.0231791121362013, 0.886180656064442, 0.18733573102221, -0.39901614518142), c(12241.3253497659, 0.944519224779928, 3511.9397804544, 3726.51534326978, 2121.84988891918, 1748.9048688927, 4620.38939405852, 2525.49057438518, 3101.93474739228, 1922.01143205711, 1347.91279549189, 1326.30219251576, 250.5787358557, 0.0385741581613282, 3357.95126834383, 2613.16776757678, 1.39172063796444, 0.0426892922924196, 14599.1760280404, 3050.15209063977, 1250.16712671446, 1.99340378208462, 0.0580307892633984, 3829.45212751458, 1086.45572368882, 3164.38159367924, 0.209859520473076, 0.372413991146252, \n1267.6955928324, 2179.83087062679, 3147.28072282541, 4032.88022219072, 3561.81633919439, 0.20264335437014, 0.285897820545355, 1092.02347467678, 0.0822105768383733, 0.205054028083917, 2248.99961200262, 0.119797592397846, 0.153240118574596, 2192.79615881063, 0.114441120910513, 0.148312324194143, 865.681280878214, 0.0410378235105948, 6556.28326720728, 3013.4038378269, 2301.34651364492, 2162.8267452113, 0.131197481699462, 0.105081736202582, 2.05041558877983, 0.0637745464799678, 0.0807479446101127, 1.42388811331726, \n1608.86597381876, 0.059983378802878, 0.0721447067054154, 0.0444338477480245, 0.0877355535122962)"
# incolla dentro ""

# variabili senza interazioni

mars1_pred2_names

mars1_pred2_names = rep(NA, length(mars1$model$pred2))
mars1_pred2_names[which(mars1$model$pred2 != 0)] = colnames(X_mm_no_interaction_sss[,mars1$model$pred2[which(mars1$model$pred2 != 0)]])
mars1_pred2_names[which(mars1$model$pred2 == 0)] = "None"

mars1_pred_names_matrix = cbind(c("Intercept", colnames(X_mm_no_interaction_sss[,mars1$model$pred1])),
                                mars1_pred2_names)

colnames(mars1_pred_names_matrix) = c("pred1", "pred2")

mars1_pred_names_matrix

rm(mars1)
gc()

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: solo se necessario°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# default commentato

# in caso di perdita dell'oggetto mars1
# uso un trucco (pigro) per ricavare le matrice del disegno

dim(mars1$ranges.and.medians)
# 3 74

# paste(mars1$ranges.and.medians %>% as.numeric, collapse = ",")
# c(0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,0,2015609,3840,1,92,30,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0)
# c()

# ricava di nuovo la matrice
mars_ranges_medians_matrix = matrix(c(0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,0,2015609,3840,1,92,30,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0),
                                    nrow = 3, ncol = 74) # ATTENZIONE da modificare con le dimensioni sopra


# salvo il numero di righe e colonne di mars1$model
# dim(mars1$model)
# 61  7
mars_model_size = dim(mars1$model)[1]

# rimozione \n e ()

# mars_model_str = gsub("\\n", "", mars_model_str)
# mars_model_str = gsub("\\(", "", mars_model_str) 
# mars_model_str = gsub("\\)", "", mars_model_str) 
# 
# # separa nei vari vettori
# 
# mars_model_list = strsplit(mars_model_str, "c")[[1]]
# 
# # elimino l'elemento vuoto
# mars_model_list_2 = mars_model_list[-1]
# mars_model_list_2 = strsplit(mars_model_list_2, ", ")
# 
# names(mars_model_list_2) = c("pred1",  "knot1",  "level1", "pred2",  "knot2",  "coefs",  "SE")
# 
# mars_model_list_2 = lapply(mars_model_list_2, as.numeric) %>% as.data.frame()
# 
# # controllo 
# mars_model_list_2
# 
# 
# # design.polymars() ritorna la matrice del disegno voluta, 
# # ma richiede come primo argomento un oggetto polspline
# # nessun problema: 
# 
# empty_mars = polymars(responses = runif((NCOL(X_mm_no_interaction_sss))),
#                       predictors = matrix(1:(NCOL(X_mm_no_interaction_sss)^2),
#                                           ncol = NCOL(X_mm_no_interaction_sss)),
#                       gcv = 1)
# 
# # sostituisco la matrice del modello con quella di mars1
# empty_mars$model.size = mars_model_size # numero righe matrice del modello precedente
# empty_mars$ranges.and.medians = mars_ranges_medians_matrix
# empty_mars$model = mars_model_list_2
# 
# X_mars_sss = design.polymars(empty_mars, X_mm_no_interaction_sss)
# X_mars_vvv = design.polymars(empty_mars, X_mm_no_interaction_vvv)
# 
# W = t(X_mars_sss) %*% X_mars_sss
# 
# # attenzione poiché potrebbe essere singolare
# # previsione con il modello MARS
# df_err_qual[8,] = c("mars",
#                      USED.Loss(X_mars_vvv %*%
#                                 solve(t(X_mars_sss) %*% X_mars_sss) %*%
#                                 t(X_mars_sss) %*% as.matrix(sss$y) %>% as.numeric(),
#                                       vvv$y))


# tramite stima e verifica (per risposta qualitativa)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ClassMARS ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Richiede insieme di verifica

mars_class = polymars(responses = sss$y[id_cb1],
                 predictors = X_mm_no_interaction_sss[id_cb1,],
                 gcv = 1,
                 factors = factor_index,
                 maxsize = 40,
                 classify = T,
                 ts.pred = X_mm_no_interaction_sss[-id_cb1,],
                 ts.resp = sss$y[-id_cb1])


mars_class$fitting
plot(mars_class$fitting$size, mars_class$fitting$T.S.M.C.,
     col = as.factor(mars_class$fitting$`0/1`),
     pch = 16,
     xlab = "numero di basi",
     ylab = "GCV")
legend(c("topright"),
       legend = c("crescita", "potatura"),
       col = c("black","red"),
       pch = 16)

min_size_mars = mars_class$fitting$size[which.min(mars_class$fitting$GCV)]
abline(v = min_size_mars)

pred_mars_class = predict(mars_class, x = X_mm_no_interaction_vvv)

df_err_qual = Add_Test_Error(df_err_qual,
                             "MARS Class",
                             USED.Loss(pred_mars > threshold %>% as.numeric(),
                                       vvv$y))

df_err_qual

pred_list$pred_mars_class = as.vector(pred_mars_class)


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

pred_random_forest = predict(random_forest_model, data = vvv, type = "response")$predictions[,2]

df_err_qual = Add_Test_Error(df_err_qual,
                             "Random Forest",
                             USED.Loss(pred_random_forest > threshold %>% as.numeric(),
                                       vvv$y))

df_err_qual

pred_list$pred_random_forest = as.vector(pred_random_forest)

# Importanza delle variabili
vimp = importance(random_forest_model)

dotchart(vimp[order(vimp)])

rm(random_forest_model)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Bagging ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(ipred)

parallel::detectCores() # quanti core a disposizione?
sfInit(cpus = 4, parallel = T)
sfExport(list = c("sss"))
sfLibrary(ipred)



err_bg_trees = rep(NA, 90)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# parto da 40 alberi bootstrap
for(j in 10:100){
  sfExport(list = c("j"))
  err_bg_trees[j] = mean(sfSapply(rep(1:4),
                                 function(x) bagging(factor(y) ~., sss,
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

bagging_model = bagging(factor(y) ~., sss, nbag = 400, coob = FALSE)


pred_bagging = predict(bagging_model, newdata = vvv, type = "prob")[,2]

df_err_qual = Add_Test_Error(df_err_qual,
                              "Bagging",
                              USED.Loss(pred_bagging > threshold %>% as.numeric(), vvv$y))


df_err_qual

pred_list$pred_bagging = as.vector(pred_bagging)


bg_plot = lift.roc(pred_bagging, vvv$y)

rm(bagging_model)
gc()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Boosting ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(ada)


# Procedura Sub-ottimale per scegliere il numero di split di ciascun albero
# Fisso il numero di iterazioni e scelgo il modello con l'errore minore sulla convalida
# Poi sul modello scelto controllo la convergenza rispetto al numero di iterazioni
# NOTA: fissiamo la complessità massima degli alberi bassa dato che
# l'algoritmo di boosting non funziona correttamente per alberi troppo complessi
# (pochi o nessun errore commesso)

# per questo problema usiamo l'errore di classificazione


# massima profondità = 10
err_boost = rep(NA, 20)

iter_boost = 400


# questo sembra appropriato per essere eseguito in parallelo


# Stump 
m_boost_stump = ada(x = sss[id_cb1, -y_index],
                    y = sss$y[id_cb1],
                    test.x = sss[-id_cb1, -y_index],
                    test.y = sss$y[-id_cb1],
                    iter = iter_boost,
                    control = rpart.control(maxdepth=1,
                                           cp=-1,
                                           minsplit=0,xval=0))

plot(m_boost_stump, test = T)

par(mfrow = c(1,1))

# update
m_boost_stump = update(m_boost_stump,
                       x = sss[id_cb1, -y_index],
                       y = sss$y[id_cb1],
                       test.x = sss[-id_cb1, -y_index],
                       test.y = sss$y[-id_cb1],
                       n.iter = 700)

plot(m_boost_stump, test = T)

pred_boost_stump = predict(m_boost_stump, vvv, type = "prob")[,2]
pred_list$pred_boost_stump = as.vector(pred_boost_stump)

df_err_qual = Add_Test_Error(df_err_qual,
                             "Boosting Stump",
                             USED.Loss(pred_boost_stump > threshold %>% as.numeric(),
                                       vvv$y))

rm(m_boost_stump)
gc()

# 6 split (default)
# guardiamo quando si stabilizza l'errore

m_boost_2 = ada(x = sss[id_cb1, -y_index],
              y = sss$y[id_cb1],
              test.x = sss[-id_cb1, -y_index],
              test.y = sss$y[-id_cb1],
              iter = iter_boost,
              control = rpart.control(maxdepth = 6))

plot(m_boost_2, test = T)

par(mfrow = c(1,1))

# update
m_boost_2 = update(m_boost_2,
                 x = sss[id_cb1, -y_index],
                 y = sss$y[id_cb1],
                 test.x = sss[-id_cb1, -y_index],
                 test.y = sss$y[-id_cb1],
                 n.iter = 700)

plot(m_boost_2, test = T)

pred_boost_2 = predict(m_boost_2, vvv, type = "prob")[,2]

df_err_qual = Add_Test_Error(df_err_qual,
                             "Boosting 6 Split",
                             USED.Loss(pred_boost_2 > threshold %>% as.numeric(),
                                       vvv$y))

pred_list$pred_boost_2 = as.vector(pred_boost_2)

df_err_qual

rm(m_boost_2)

gc()



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Support Vector Machine ---------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(e1071)


# Stima Convalida
# Kernel Radiale

ranges = seq(2,20,by=0.5)
err_svm <- matrix(NA,length(ranges),2)
for (i in 1:length(ranges)){
  s1<- svm(factor(y)~., data= sss[id_cb1,], cost=ranges[i], kernel = "radial")
  pr1 <- predict(s1, newdata= sss[-id_cb1,], probability = FALSE) #, decision.values=TRUE)
  uso <- tabella.sommario(pr1, osserv= sss$y[-id_cb1])
  # eventualmente cambia il tipo di errore
  err_svm[i,]<-c(ranges[i], uso[1])
}
plot(err_svm, type="b", pch = 16,
     xlab = "costo", ylab = "errore",
     main = "SVM radiale")

ranges[which.min(err_svm[,2])]
# ATTENZIONE: modificare 
# 15 

m_svm =  svm( factor(y)~., data= sss, cost= ranges[which.min(err_svm[,2])])
pred_svm_radial = attr(predict(m_svm, newdata = vvv, decision.values=TRUE), "decision.values")[,1]

df_err_qual = Add_Test_Error(df_err_qual,
                             "SVM radial",
                             USED.Loss(pred_svm_radial > 0,
                                       vvv$y))

df_err_qual

pred_list$pred_svm_radial = pred_svm_radial



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rete neurale -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# /////////////////////////////////////////////////////////////////
#------------------------ Sintesi Finale -------------------------
# /////////////////////////////////////////////////////////////////

# Dei modelli migliori in base alle metriche traccio le curve lift e ROC
# altrimenti i grafici sono difficili da leggere

df_err_qual

# LIFT e ROC --------------
# eventualmente crea una nuova lista ridotta
# con solo i modelli con l'errore minimo
model_names_all = names(pred_list)

paste(model_names_all, collapse = "','")


model_names = c('pred_ridge_yes_interaction_lmin','pred_lasso_yes_interaction_lmin','pred_gam_step','pred_mars','pred_tree','pred_bagging','pred_boost_2','pred_random_forest','pred_svm_radial')

pred_list_ridotta = pred_list[which(model_names_all %in% model_names)]

pred_list_ridotta$pred_glm_no_interaction = NULL


curve = lapply(pred_list_ridotta, function(x) lift.roc(x, vvv$y, plot.it=F, type = "bin"))

model_printed_names = sapply(model_names, FUN = function(x) gsub("pred_", "", x))


library(RColorBrewer)
colori = c("black", brewer.pal(12, "Set3"))
# @@@@@@@@@@@@@@@@@@@@@
# LIFT ----------------
# @@@@@@@@@@@@@@@@@@@@@

plot(curve[[1]][[1]], curve[[1]][[2]], type = "b", ylim = c(0,2.5), 
     xlab = "Frazione di predetti positivi", ylab = "Lift",
     pch = 1, main = "LIFT", lwd = 2)
# e disegnamo le altre sovrapponendole
for(j in 2:length(curve)) {
  lines(curve[[j]][[1]], curve[[j]][[2]],type = "b", col = colori[j], lwd = 2, pch = j)
}

legend("bottomright",col = colori,
       legend = model_printed_names, pch = 1:length(model_printed_names), 
       ncol = 2, lwd = 2,cex = .5,
       y.intersp = 1.5,
       text.width = 0.3,
       
)

# @@@@@@@@@@@@@@@@@@@@@
# ROC ----------------
# @@@@@@@@@@@@@@@@@@@@@
plot(curve[[1]][[3]], curve[[1]][[4]], type = "b", ylim = c(0,1), 
     xlab = "1-specificità", ylab = "sensibilità",
     pch = 1, main = "ROC", lwd = 2)
# e disegnamo le altre sovrapponendole
for(j in 2:length(curve)) {
  lines(curve[[j]][[3]], curve[[j]][[4]],type = "b", col = colori[j], lwd = 2, pch = j)
}

legend("bottomright",col = colori,
       legend = model_printed_names, pch = 1:length(model_printed_names), 
       ncol = 2, lwd = 2,cex = .5,
       y.intersp = 1.5,
       text.width = 0.3,
       
)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Analisi modelli migliori ---------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# Comparazione modelli
# selezione modelli migliori e commenti su questi:
# es -> vanno meglio modelli con interazioni oppure modelli additivi



# Ridge 
# Lasso

# guarda coefficienti
# predict.glmnet(oggetto, type = "coefficients")

lasso_int_coef = predict(lasso_yes_interaction_lmin, type = "coef")
lasso_int_coef[which(abs(lasso_int_coef)> 1)]

# Modello additivo : lo devo ri-stimare nel caso
# plot(gam_step, terms = c("s(Durata, df = 4)"), se = T)

# summary(gam_step)

# MARS:
mars1$model

# nomi
mars1_pred_names_matrix

plot(mars1, 6)



colnames(X_mm_no_interaction_sss[,c(mars1$model$pred1, mars1$model$pred2)])

# per il grafico lo devo ristimare
# plot(mars1, predictor1 = 40, predictor2 = 30)

# Random Forest: guarda grafico importanza variabili

# SVM
# plot(m_svm, sss, salt ~ wine)


