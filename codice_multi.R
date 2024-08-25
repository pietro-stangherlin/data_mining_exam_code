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
  return(MissErr(y.pred, y.test, weights))
}


# anche qua
df_metrics = data.frame(name = NA,
                        missclass = NA)

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

# /////////////////////////////////////////////////////////////////
#------------------------ Train & Test ------------------------
# /////////////////////////////////////////////////////////////////


# eventually change the proportion
id_stima = sample(1:NROW(dati), 0.75 * NROW(dati))

sss = dati[id_stima,]
vvv = dati[-id_stima,]


# lista dei predittori (se stima - verifica)
# per LIFT e ROC
pred_list = list() 

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter tuning: Train & Test on Train subset  --------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

id_cb1 = sample(1:NROW(sss), 0.8 * NROW(sss))

# delete original data.frame from main memory
rm(dati)
gc()

sss$y = factor(sss$y)
vvv$y = factor(vvv$y)

# ///////////////////////////////////
# Weights ---------------
# //////////////////////////////////

# weights used for each metric function
# default 1
MY_WEIGHTS_sss = rep(1, NROW(sss)) 
MY_WEIGHTS_vvv = rep(1, NROW(vvv))

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter tuning: cross validation on train: building cv folds  -------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

K_FOLDS = 10

NROW_sss = NROW(sss)

SHUFFLED_ID = sample(1:NROW_sss, NROW_sss)

# NOTE: if the row number of sss is not a multiple of K_FOLDS
# the last fold repeats some ids from the first
# this is fixed in the code below
id_matrix_cv = matrix(SHUFFLED_ID, ncol = K_FOLDS)


# conversion of matrix in list of elements: each element contains a subset of ids
ID_CV_LIST = list()

for(j in 1:ncol(id_matrix_cv)){
  ID_CV_LIST[[j]] = id_matrix_cv[,j]
}

rm(id_matrix_cv)
gc()


# repeated ids fixing
integer_division_cv = NROW_sss %/% K_FOLDS
modulo_cv = NROW_sss %% K_FOLDS

if(modulo_cv != 0){
  ID_CV_LIST[[K_FOLDS]] = ID_CV_LIST[[K_FOLDS]][1:integer_division_cv]
}

source("cv_functions.R")

# FALSE = traditional CV on all folds
# TRUE -> use only first fold to test and all other to fit
USE_ONLY_FIRST_FOLD = FALSE
# /////////////////////////////////////////////////////////////////
#------------------------ Analisi esplorative ---------------------
# /////////////////////////////////////////////////////////////////

# Analisi esplorativa sulla stima 
# eventuali inflazioni di zeri

# valutiamo se è sbilanciata 
# ed eventualmente se è ragionevole cambiare la solita soglia a 0.5
table(sss$y)



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

Y_LEVELS_SORTED = colnames(Y_sss)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Linear Model -----------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# No Interaction ----------------

# Separate models for each modality and then get the highest value

# models list: where each model is stored
lm_no_int_models_list = list()

for(col in 1:NCOL(Y_sss)){
  lm0 = lm(Y_sss[,col] ~ 1, data = sss[,-y_index])
  lm_no_int_models_list[[col]] = step(lm0,
                             scope = formula_no_interaction_yes_intercept,
                             direction = "forward")
}

# error evaluation

temp_pred_scores = lapply(lm_no_int_models_list, function(el) predict(el, newdata = vvv))
temp_pred_scores = matrix(unlist(temp_pred_scores), ncol = NCOL(Y_sss))

temp_pred = Y_LEVELS_SORTED[apply(temp_pred_scores, 1, which.max)]

file_name_lm_no_int_models_list = paste(MODELS_FOLDER_RELATIVE_PATH,
                                 "lm_no_int_models_list",
                                 ".Rdata", collapse = "", sep = "")

save(lm_no_int_models_list, file = file_name_lm_no_int_models_list)

df_metrics = Add_Test_Metric(df_metrics,
                             "lm_no_int_models_list",
                             USED.Metrics(temp_pred,
                                          vvv$y,
                                          MY_WEIGHTS_vvv))

df_metrics = na.omit(df_metrics)

df_metrics


# Yes Interaction ----------------
# models list: where each model is stored
lm_yes_int_models_list = list()

for(col in 1:NCOL(Y_sss)){
  lm0 = lm(Y_sss[,col] ~ 1, data = sss[,-y_index])
  lm_yes_int_models_list[[col]] = step(lm0,
                                      scope = formula_yes_interaction_yes_intercept,
                                      direction = "forward")
}

# error evaluation

temp_pred_scores = lapply(lm_yes_int_models_list, function(el) predict(el, newdata = vvv))
temp_pred_scores = matrix(unlist(temp_pred_scores), ncol = NCOL(Y_sss))

temp_pred = Y_LEVELS_SORTED[apply(temp_pred_scores, 1, which.max)]

file_name_lm_yes_int_models_list = paste(MODELS_FOLDER_RELATIVE_PATH,
                                        "lm_yes_int_models_list",
                                        ".Rdata", collapse = "", sep = "")

save(lm_yes_int_models_list, file = file_name_lm_yes_int_models_list)

df_metrics = Add_Test_Metric(df_metrics,
                             "lm_no_yes_models_list",
                             USED.Metrics(temp_pred,
                                          vvv$y,
                                          MY_WEIGHTS_vvv))


df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Log Linear Model --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
m_multi0 = multinom(Y_sss ~ 1,
                    data = sss[,-y_index],
                    maxit = 400)

# scelgo il parametro di regolazione tramite AIC

m_multi_no_int = step(m_multi0, scope = formula_no_interaction_yes_intercept)

file_name_m_multi_no_int = paste(MODELS_FOLDER_RELATIVE_PATH,
                                 "m_multi_no_int",
                                 ".Rdata", collapse = "", sep = "")

save(m_multi_no_int, file = file_name_m_multi_no_int)

df_metrics = Add_Test_Metric(df_metrics,
                             "multi_no_int",
                             USED.Metrics(predict(m_multi_no_int, newdata = vvv),
                                       vvv$y,
                                        MY_WEIGHTS_vvv))

df_metrics




# forse pesante
# m_multi_yes_int = step(m_multi1, scope = formula_yes_interaction_yes_intercept)

# file_name_m_multi_yes_int = paste(MODELS_FOLDER_RELATIVE_PATH,
#                                  "m_multi_yes_int",
#                                  ".Rdata", collapse = "", sep = "")
# 
# save(m_multi_yes_int, file = file_name_m_multi_yes_int)

# df_metrics = Add_Test_Metric(df_metrics,
#                              "multi_yes_int",
#                              USED..Metrics(predict(m_multi_yes_int, newdata = vvv),
#                                        vvv$y,
#                                        MY_WEIGHTS_vvv))
# 
# df_metrics


save(df_metrics, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Multilogit --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(VGAM)

vglm0 = vglm(factor(y) ~ 1,
             multinomial,
             data = sss)

vglm_step = step4vglm(vglm0,
          scope = formula_no_interaction_yes_intercept,
          direction = "forward")

file_name_vglm_step = paste(MODELS_FOLDER_RELATIVE_PATH,
                                 "vglm_step",
                                 ".Rdata", collapse = "", sep = "")

save(vglm_step, file = file_name_vglm_step)


temp_pred = Y_LEVELS_SORTED[apply(predict(vglm_step, type = "response", newdata = vvv), 1, which.max)]


df_metrics = Add_Test_Metric(df_metrics,
                             "multi_no_int",
                             USED.Metrics(temp_pred,
                                          vvv$y,
                                          MY_WEIGHTS_vvv))

df_metrics

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# LDA & QDA ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(MASS)

vglm_step_formula = formula(vglm_step)

m_lda_r = lda(vglm_step_formula, data=sss)

df_metrics = Add_Test_Metric(df_metrics,
                             "LDA",
                             USED.Metrics(predict(m_lda_r, newdata=vvv)$class,
                                          vvv$y,
                                          MY_WEIGHTS_vvv))

m_qda_r = qda(vglm_step_formula, data=sss)

df_metrics = Add_Test_Metric(df_metrics,
                             "QDA",
                             USED.Metrics( predict(m_qda_r, newdata=vvv)$class,
                                           vvv$y,
                                           MY_WEIGHTS_vvv))


df_metrics

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

# No interazione ---------------------
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
pr_lasso_no_int = t(apply(pr_lasso_no_int, 1, function(row) Y_LEVELS_SORTED[row]))

dim(pr_lasso_no_int)

# controllo 
pr_lasso_no_int[1:10, 1:15]

# errore sull'insieme di convalida
err_lasso_no_int = apply(pr_lasso_no_int, 1, function(row) USED.Metrics(row, sss$y[-id_cb1]))

length(err_lasso_no_int)
err_lasso_no_int

best_lambda_no_int = lambda_vals[which.min(err_lasso_no_int)]
best_lambda_no_int

temp_plot_fun = function(){
plot(log(lambda_vals), err_lasso_no_int,
     xlab = "log lambda", ylab = "err", main = "Lasso no interaction",
     pch = 16)
abline(v = log(best_lambda_no_int))}

PlotAndSave(my_plotting_function = temp_plot_fun,
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "lasso_no_int_tuning_plot.jpeg",
                                 collapse = ""))

coef(lasso_no_int, lasso_no_int$lambda[best_lambda_no_int])
# problema: lambda e' molto piccola
# in questo caso e' quasi come senza penalizzare


# errore sull'insieme di convalida

pred_final_lasso_no_int_matr = drop(predict(lasso_no_int,
                                            newx = X_mm_no_interaction_vvv,
                                            s = best_lambda_no_int))
dim(pred_final_lasso_no_int_matr)


pred_final_lasso_no_int_class = apply(pred_final_lasso_no_int_matr, 1, which.max)


pred_final_lasso_no_int_class = Y_LEVELS_SORTED[pred_final_lasso_no_int_class]

length(pred_final_lasso_no_int_class)

df_metrics = Add_Test_Metric(df_metrics,
                              "lasso no int",
                              USED.Metrics(pred_final_lasso_no_int_class, vvv$y))

df_metrics

save(df_metrics, file = "df_metrics.Rdata")

rm(lasso_no_int)
rm(pred_final_lasso_no_int)
rm(pred_final_lasso_no_int_class)
rm(pred_final_lasso_no_int_matr)

# SI interazione --------------------

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
pr_lasso_yes_int = t(apply(pr_lasso_yes_int, 1, function(row) Y_LEVELS_SORTED[row]))

dim(pr_lasso_yes_int)

# controllo 
pr_lasso_yes_int[1:10, 1:15]

# errore sull'insieme di convalida
err_lasso_yes_int = apply(pr_lasso_yes_int, 1, function(row) USED.Metrics(row, sss$y[-id_cb1]))

length(err_lasso_yes_int)
err_lasso_yes_int

best_lambda_yes_int = lambda_vals[which.min(err_lasso_yes_int)]
best_lambda_yes_int

temp_plot_fun = function(){
  plot(log(lambda_vals), err_lasso_yes_int,
       xlab = "log lambda", ylab = "err", main = "Lasso Yes interaction",
       pch = 16)
  abline(v = log(best_lambda_yes_int))}

PlotAndSave(my_plotting_function = temp_plot_fun,
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "lasso_yes_int_tuning_plot.jpeg",
                                 collapse = ""))


coef(lasso_yes_int, lasso_yes_int$lambda[best_lambda_yes_int])
# problema: lambda e' molto piccola
# in questo caso e' quasi come senza penalizzare


# errore sull'insieme di convalida

pred_final_lasso_yes_int_matr = drop(predict(lasso_yes_int,
                                             newx = X_mm_yes_interaction_vvv,
                                             s = best_lambda_yes_int))
dim(pred_final_lasso_yes_int_matr)


pred_final_lasso_yes_int_class = apply(pred_final_lasso_yes_int_matr, 1, which.max)


pred_final_lasso_yes_int_class = Y_LEVELS_SORTED[pred_final_lasso_yes_int_class]

length(pred_final_lasso_yes_int_class)

df_metrics = Add_Test_Metric(df_metrics,
                             "lasso yes int",
                             USED.Metrics(pred_final_lasso_yes_int_class, vvv$y))

df_metrics

save(df_metrics, file = "df_metrics.Rdata")

rm(lasso_yes_int)
rm(pred_final_lasso_yes_int)
rm(pred_final_lasso_yes_int_class)
rm(pred_final_lasso_yes_int_matr)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello Additivo -----------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

library(gam)

# Separate models for each modality and then get the highest value

# models list: where each model is stored
gam_models_list = list()

# gam recognizes factor predictors
my_gam_scope = gam.scope(sss[,-y_index], arg = c("df=2", "df=3", "df=4", "df=5", "df=6"))

for(col in 1:NCOL(Y_sss)){
  gam0 = gam(Y_sss[,col] ~ 1, data =sss[,-y_index])
  gam_models_list[[col]] = step.Gam(gam0, scope = my_gam_scope)
}

# error evaluation

temp_pred_scores = lapply(gam_models_list, function(el) predict(el, newdata = vvv))
temp_pred_scores = matrix(unlist(temp_pred_scores), ncol = NCOL(Y_sss))

temp_pred = Y_LEVELS_SORTED[apply(temp_pred_scores, 1, which.max)]

file_name_gam_models_list = paste(MODELS_FOLDER_RELATIVE_PATH,
                                        "gam_models_list",
                                        ".Rdata", collapse = "", sep = "")

save(gam_models_list, file = file_name_gam_models_list)

df_metrics = Add_Test_Metric(df_metrics,
                             "gam_models_list",
                             USED.Metrics(temp_pred,
                                          vvv$y,
                                          MY_WEIGHTS_vvv))


df_metrics




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

temp_plot_fun = function(){
plot(m_mars$gcv.per.subset, pch = 16,
     xlab = "Numero di basi",
     ylab = "GCV",
     main = "MARS GCV")
abline(v = which.min(m_mars$gcv.per.subset), col = "gold",
       main = "MARS")}

PlotAndSave(my_plotting_function = temp_plot_fun,
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "mars_gcv_plot.jpeg",
                                 collapse = ""))
summary(m_mars)

# # prima risposta
# plotmo(m_mars, nresponse = 1, ylim = NA)
# 
# plotmo(m_mars, nresponse = 2, ylim = NA)

pred_mars = apply(predict(m_mars, vvv), 1, which.max)
pred_mars_class = colnames(Y_sss)[pred_mars]

df_metrics = Add_Test_Metric(df_metrics,
                              "mars",
                              USED.Metrics(pred_mars_class,
                                           vvv$y,
                                           MY_WEIGHTS_vvv))
df_metrics

file_name_mars = paste(MODELS_FOLDER_RELATIVE_PATH,
                       "tree",
                       ".Rdata", collapse = "", sep = "")

save(m_mars, file = file_name_mars)


rm(m_mars)
gc()

save(df_metrics, file = "df_metrics.Rdata")


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Albero -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

# default: molto fitto
# eventualmente aumenta mindev e minsize
# tree_full = tree(factor(y) ~ .,
#                  data = sss[id_cb1,],
#                  control = tree.control(nobs = length(id_cb1),
#                                         mindev = 1e-03,
#                                         minsize = 10))



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

temp_plot = function(){
  plot(final_tree_pruned)
  text(final_tree_pruned, cex = 0.7)
}

PlotAndSave(my_plotting_function = temp_plot,
             my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                  "pruned_tree_plot.jpeg",
                                  collapse = ""))


df_metrics = Add_Test_Metric(df_metric,
                              "tree_pruned best",
                              USED.Metrics(predict(final_tree_pruned, newdata = vvv,
                                                   type = "class"),
                                           vvv$y,
                                           MY_WEIGHTS_vvv))

df_metrics

file_name_tree = paste(MODELS_FOLDER_RELATIVE_PATH,
                                 "tree",
                                 ".Rdata", collapse = "", sep = "")

save(final_tree_pruned, file = file_name_tree)


rm(tree_full)
gc()

save(df_metrics, file = "df_metrics.Rdata")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(ranger)

# Nota: se manca il tempo eseguo prima la RandomForest del Bagging
# visto che quest'ultimo è un sotto caso particolare 
# della RandomForest (selezione di tutte le variabili per ogni split)


# massimo numero di esplicative presenti
RF_MAX_VARIABLES = NCOL(sss) - 2 # sottraggo 1 per la variabile risposta
# ridurlo per considerazioni computazionali

RF_ITER = 300

RF_TREE_NUMBER_SEQ = seq(10, 400, 10)

err = rep(NA, RF_MAX_VARIABLES)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°

for(i in seq(2, RF_MAX_VARIABLES)){
  
  err[i] = ranger(factor(y) ~., data = sss,
                  mtry = i,
                  num.trees = RF_ITER,
                  probability = TRUE,
                  oob.error = TRUE)$prediction.error
  
  
  print(paste("mtry: ", i, collapse = ""))
  gc()
}

print("Random forest error for each mtry")
err 

best_mtry = which.min(err)

print("best mtry random forest")
best_mtry


# uso il valore trovato e controllo la convergenza rispetto al numero di alberi

err_rf_trees = rep(NA, length(RF_TREE_NUMBER_SEQ))

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
for(j in 1:length(RF_TREE_NUMBER_SEQ)){
  err_rf_trees[j] = ranger(factor(y) ~., data = sss,
                           mtry = best_mtry,
                           num.trees = RF_TREE_NUMBER_SEQ[j],
                           oob.error = TRUE)$prediction.error
  
  print(paste("number of trees: ", RF_TREE_NUMBER_SEQ[j], collapse = ""))
}


PlotAndSave(my_plotting_function =  function()plot(RF_TREE_NUMBER_SEQ, err_rf_trees,
                                                   xlab = "Bootstrap trees number",
                                                   ylab = "Out of bag Error",
                                                   pch = 16,
                                                   main = "Random Forest"),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "random_forest_convergence_plot.jpeg",
                                 collapse = ""))




# modello finale e previsioni
random_forest_model = ranger(factor(y) ~., sss,
                             mtry = best_mtry,
                             num.trees = 400,
                             oob.error = TRUE,
                             importance = "permutation")

# Warning check index
temp_pred = predict(random_forest_model, data = vvv,
                    type = "response")$predictions

pred_list$random_forest = temp_pred

df_metrics = Add_Test_Metric(df_metrics,
                             "Random Forest",
                             USED.Metrics(temp_pred,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))

df_metrics
rm(temp_pred)

# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")


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

save(df_metrics, file = "df_metrics.Rdata")
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Bagging ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


library(ipred)

BAGGING_TREE_NUMBER_SEQ = seq(10, 400, 10)

err_bg_trees = rep(NA, length(BAGGING_TREE_NUMBER_SEQ))

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°

# controllo la convergenza dell'errore rispetto al numero di alberi
# parto da 40 alberi bootstrap
for(j in 1:length(BAGGING_TREE_NUMBER_SEQ)){
  err_bg_trees[j] = bagging(factor(y) ~., sss,
                            nbag = BAGGING_TREE_NUMBER_SEQ[j],
                            coob = TRUE)$err
  print(BAGGING_TREE_NUMBER_SEQ[j])
}


PlotAndSave(my_plotting_function = function() plot(BAGGING_TREE_NUMBER_SEQ, err_bg_trees,
                                                   xlab = "numero di alberi bootstrap",
                                                   ylab = "errore out of bag",
                                                   pch = 16,
                                                   main = "Bagging"),
            my_path_plot = paste(FIGURES_FOLDER_RELATIVE_PATH,
                                 "bagging_convergence_plot.jpeg",
                                 collapse = "")
)


# se il numero di replicazioni bootstrap arriva a convergenza allora

bagging_model = bagging(factor(y) ~., sss, nbag = max(BAGGING_TREE_NUMBER_SEQ), coob = FALSE)

temp_pred = predict(bagging_model, newdata = vvv,
                    type = "class")

df_metrics = Add_Test_Metric(df_metrics,
                             "Bagging",
                             USED.Metrics(temp_pred,
                                          vvv$y,
                                          weights = MY_WEIGHTS_vvv))


df_metrics

rm(temp_pred)
cor
# save the df_metrics as .Rdata
save(df_metrics, pred_list, file = "df_metrics.Rdata")


# save metrics and model
file_name_bagging = paste(MODELS_FOLDER_RELATIVE_PATH,
                          "bagging",
                          ".Rdata", collapse = "", sep = "")

save(bagging_model, file = file_name_bagging)


rm(bagging_model)
gc()

# /////////////////////////////////////////////////////////////////
#------------------------ Sintesi Finale -------------------------
# /////////////////////////////////////////////////////////////////

df_metrics


load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "m_multi_no_int",
           ".Rdata", collapse = "", sep = ""))

summary(m_multi_no_int)

load(paste(MODELS_FOLDER_RELATIVE_PATH,
           "m_mars",
           ".Rdata", collapse = "", sep = ""))

summary(m_mars)

plotmo(m_mars)

