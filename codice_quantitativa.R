rm(list = ls())
library(dplyr)


# Per Lapsus
# CTRL + C per commentare più righe di codice

# Template per risposta quantitativa + preprocessing
# con n grande e p piccolo
# ma devo fare tutti gli eventuali casi 
# quindi sia stima + convalida
# convalida  incrociata per l'insieme di stima
# convalida incrociata come valutazione finale

# /////////////////////////////////////////////////////////////////////////////
#  ----------------------- Lettura e Preprocessing -----------------------------
# /////////////////////////////////////////////////////////////////////////////


# per avere un'idea del file da terminale: 
# head nome_file.formato

dati = read.csv("df_quant.csv", sep = ",", stringsAsFactors = F)

# NOTA: converto in fattori solo alla fine del preprocessing
# in modo da non dover riconvertire tutto ogni volta 

# controllo
str(dati)

# §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
# Generico -------------------------------------------------------
# §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Valori mancanti 1 --------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# per ogni variabile controlla i dati mancanti per colonna
# definisco la funzione per contare il numero di unità
# uguali a un certo valore in un vettore



# Funzioni utili
CountEqualTo = function(vector_object, value){
  length(which(vector_object == value))
}


# controlla se value è presente nel vettore
IsValueInVector = function(vector_object, value){
  # definita prima
  if(CountEqualTo(vector_object, value) == 0){
    return(FALSE)
  }
  return(TRUE)
}



# Mancanti NA (classici)
missing_freq_NA = apply(dati, 2, function(col) sum(is.na(col)))
missing_freq_NA

# Mancanti "" (caratteri vuoti: "")

# Eventualmente cambia il parametro "" con altri in base al propblema
missing_freq_empty = apply(dati, 2, function(col) CountEqualTo(col, ""))
missing_freq_empty

# Totale mancanti: NA + empty
missing_freq_NA + missing_freq_empty


# righe dati mancanti NA
row_missing_index_NA = which(is.na(dati))

# righe dati mancanti empty

row_missing_index_empty = which((apply(dati, 1, function(row) (IsValueInVector(row, "")))) == TRUE)

head(row_missing_index_empty)

# eventuale gestione caso per caso

# Possibilità

# 1) Rimozione osservazioni se sono poche rispetto alla numerosità totale
# e n è molto grande
# dati = dati[-row_missing_index_NA,]
# dati = dati[-row_missing_index_empty,]


# 2) Se qualitativa: creazione di una nuova modalità
# es dati$v2 =  ifelse(da$v2, is.na, "missing", da$v2)


# 3) Se quantitativa:
# 3.1) rimozione
# 3.2) suddivisione in classi con aggiunta della modalità mancante
# 3.3) imputazione e imputazione con es. media o mediana -> mai fatto
#     un'altro modo per fare imputazione è trattare il predittore come risposta
#     rispetto agli altri predittori

# Per motivi di tempo nell'esame effettuo o 3.1) o 3.2)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rimozioni variabili --------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Rimozione di variabili esplicative
dati$X = NULL

# Rimozione Leaker 


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rimozione Osservazioni -----------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# seleziono un sotto insieme dei dati
# controllo valori mancanti
# se in frequenza sono relativamente pochi

# selezione di un sottoinsieme di dati per il problema in questione

# dati = dati[dati$Stato == "Successful",]
# dati$Stato = NULL

# controllo 
str(dati)

# rinomimo la risposta in y: cambia il primo "y" in base al problema
names(dati)[which(names(dati) == "y")] = "y"

# controllo 
str(dati)


# §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
# Specifico -----------------------------------------------------
# §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Testo ----------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# generale



# 2) gsub: sostitusice la stringa
# pattern = "-[[:digit:]][[:digit:]]-[[:digit:]][[:digit:]]"
# head(gsub(pattern = pattern, replacement = "", x = dati$variabile)

# 3) substring
# primi quattro caratteri (start = 1, stop = 4)
# head(substr(as.character(dati$variabile), 1, 4))

# grepl: controllo logico se è presente un pattern
# + grepl (logical): controllo presenza o assenza di stringa
# es mese di marzo
# head(grepl("-03-", x = dati$Data.di.lancio, ignore.case = TRUE))

# altro esempio con grepl -> weekend o no
# grepl(pattern = "dom|sab",
#       c("fwf w Domenica", "Sabato esco", "Luned dwdw"),
#       ignore.case = T)

# rimozione variabili ridondanti

# dati$Data.di.lancio = NULL
# dati$Data.di.termine = NULL

str(dati)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Data -----------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# %Y-%m-%d
date_time_default_str = "[[:digit:]][[:digit:]][[:digit:]][[:digit:]]-[[:digit:]][[:digit:]]-[[:digit:]][[:digit:]]"
date_expr_str = "[0-9]{4}-[0-9]{2}-[0-9]{2}" # esempio 2022-07-28
date_expr_str_raw = "[0-9]{8}" # esempio 20220728
date_format = "%Y-%m-%d"

# ATTENZIONE: da cambiare di caso in caso
temp_var_name = "x6"

# Se necessario prima devo pulire i dati con la data
# seleziono le righe non NA non empty per una particolare variabile con la data

# Per una variabile: per altre variabili copia e incolla

# tutti i valori da NON modificare
index_na_empty = which(is.na(dati[, temp_var_name]) | dati[, temp_var_name] == '')

# pulisci questi valori (dipende dal problema in questione)
# metodo 1: esempio -> se le posizioni sono fisse 
# substring("2022-09-08fwfwfw", 1, 10)

# dati[-index_na_empty, temp_var_name] = substring(dati[-index_na_empty, temp_var_name], 1, 10)
# 
# head(dati[-index_na_empty, temp_var_name])


# in alternativa
# regmatches("2022-09-08fwfwfw", gregexpr(date_expr_str, "2022-09-08fwfwfw"))[[1]]

dati[-index_na_empty, temp_var_name] = unlist(regmatches(dati[-index_na_empty, temp_var_name],
                                                           gregexpr(date_expr_str,
                                                                    dati[-index_na_empty, temp_var_name])))
head(dati[-index_na_empty, temp_var_name])

# modifica della data
# se il formato è diverso invertire i caratteri in tryFormat
dati[, temp_var_name] = as.Date(dati[,temp_var_name], tryFormat = "%Y-%m-%d")

# estraggo l'anno e il mese
dati$anno = format(dati[, temp_var_name], "%Y") # lo teniamo con categoriale
str(dati$anno)
table(dati$anno)

dati$mese =  format(dati[, temp_var_name], "%m")
table(dati$mese)


# rimozione variabile 
dati[,temp_var_name] = NULL
rm(temp_var_name)
rm(index_na_empty)

# Eventualmente nuova variabile differenza (in giorni)
# se ad esempio c'è una data di inizio e una di fine
# differenza in giorni
# dati$Durata = as.numeric(dati$Data.di.termine - dati$Data.di.lancio)
# 
# str(dati$Durata)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Valori mancanti 2 ----------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# per ogni variabile controlla i dati mancanti per colonna

# Mancanti NA (classici)
missing_freq_NA = apply(dati, 2, function(col) sum(is.na(col)))
missing_freq_NA

# Mancanti "" (caratteri vuoti: "")

# Eventualmente cambia il parametro "" con altri in base al propblema
missing_freq_empty = apply(dati, 2, function(col) CountEqualTo(col, ""))
missing_freq_empty

# Totale mancanti: NA + empty
missing_freq_NA + missing_freq_empty


# righe dati mancanti NA
row_missing_index_NA = which(is.na(dati))

# righe dati mancanti empty

row_missing_index_empty = which((apply(dati, 1, function(row) (IsValueInVector(row, "")))) == TRUE)

head(row_missing_index_empty)




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Riduzione categorie qualitative -------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
table(dati$Categoria)



# Funzioni 
# ===================================================================
# ritorna la tabella con le modalità e le rispettive frequenze
# per il data.frame df e la variabile var_name
TableFreqFun = function(df, var_name){
  table_modalita_decr = table(df[,var_name]) %>% sort(decreasing = T)
  # quante modalità
  
  print("Numero di modalità uniche")
  print( df[,var_name] %>% unique() %>% length())
  
  return(table_modalita_decr)
  
}


# ritorna le modalità di var_name con frequenza minore di soglia
# con il "valore nuova"
# NOTA: il return DEVE essere ASSEGNATO
RaggruppaModalita = function(df, var_name, tabella_freq_modalita, soglia, valore_nuova){
  
  # modalità al di sotto di una certa frequenza
  modalita_sotto_soglia = names(which(tabella_freq_modalita < soglia))
  
  # raggruppo queste modalità
  return(ifelse(df[,var_name] %in% modalita_sotto_soglia, valore_nuova, df[,var_name]))
}
# ==========================================================================================

# Attenzione: l'implementazione di tree NON permette esplicative 
# categoriali con più di 30 modalità
# quindi eventualmente ridurre a max 30 modalità
# (compromesso di perdita informazione)

# Esempio di applicazione: cambiare il nome della variabile e la soglia
# ==============================================================================

# °°°°°°°°°°°°°°°°°°°° Warning: cambia nome variabile °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
temp_table_freq = TableFreqFun(dati, "anno")
temp_table_freq

dati[,"Sottocategoria"] = RaggruppaModalita(dati, "Sottocategoria", temp_table_freq, 1500, "Altro")

# check
temp_table_freq = TableFreqFun(dati, "Sottocategoria")
temp_table_freq

rm(temp_table_freq)
# ================================================================================


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Esplicative qualitative e quantitative -------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


var_names = colnames(dati)

y_index = which(var_names == "y")

# +++++++++++++++++++++++++++++++++++++++++++++++++
# Individuo qualitative codificate come quantitative
# ++++++++++++++++++++++++++++++++++++++++++++++++++

# ottieni l'indice delle colonne delle variabili con il numero di modalità
# da eventualmente convertire in fattori
# !!!!RICHIESTA ATTENZIONE!!!!!

unique_vals_df = data.frame(nome = rep("", NCOL(dati)),
                            indice = rep(0, NCOL(dati)),
                            uniques = rep(0, NCOL(dati)))


unique_vals_df$nome = colnames(dati)
unique_vals_df$indice = as.numeric(1:NCOL(dati))
unique_vals_df$uniques = as.numeric(apply(dati, 2, function(col) length(unique(col))))


unique_vals_df

# eslcusione della y
unique_vals_df_no_y = unique_vals_df[-which(unique_vals_df$nome == "y"),]

# indici delle esplicative con meno di min_modalità modalità
# da aumentare in base al problema
min_modalita = 2

index_min_modalita = unique_vals_df_no_y$indice[which(unique_vals_df_no_y$uniques <= min_modalita)]
index_min_modalita

# trasformo in fattore queste ultime
for(i in index_min_modalita){
  dati[,i] = as.factor(dati[,i])
}

str(dati)

# +++++++++++++++++++++++++++++++++++++++++++++++++
# Nomi e indici di colonna delle variabili
# ++++++++++++++++++++++++++++++++++++++++++++++++++

# nomi delle esplicative qualitative e quantitative
# potrei dover effettuare questa operazione più volte


var_factor_index = which(sapply(dati, is.factor))
# se comprende l'indice della y  lo rimuovo
# da sistemare
if (y_index %in% var_factor_index){
  var_factor_index = var_factor_index[-which(var_factor_index == y_index)]}

var_char_index = which(sapply(dati, is.character))
# se comprende l'indice della y  lo rimuovo
# da sistemare
if (y_index %in% var_char_index){
var_char_index = var_char_index[-which(var_char_index == y_index)]}


# comprende anche int
var_num_index = as.numeric(which(sapply(dati, is.numeric)))
# se comprende l'indice della y lo rimuovo
if (y_index %in% var_num_index){
var_num_index = var_num_index[-which(var_num_index == y_index)]}


# +++++++++++++++++++++++++++++++++++++++++++++++++
# Conversione character in factor
# ++++++++++++++++++++++++++++++++++++++++++++++++++

for(i in var_char_index){
  dati[,i] = as.factor(dati[,i])
}


str(dati)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Aggiorno indici qualitative e nomi qualitative e quantitative
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

var_qual_index =  as.numeric(c(var_char_index, var_factor_index))

var_qual_names = var_names[var_qual_index]

var_num_names = var_names[var_num_index]

# check 
var_qual_index
var_num_index


var_qual_names
var_num_names


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Scope ----------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# funzione per creare le stringhe di interazione
# tra variabili della stessa tipologia
# (quantitativa - quantitativa e fattore - fattore)

# '@ input: array of strings   
# '@ return string formula of interaction terms
# example :
# input = c("a", "b", "c")
# output = "a:b + a:c + b:c" 
MakeSameInteractionsString = function(input_var_type_names){
  # preliminary checks
  if(length(input_var_type_names) == 0){
    cat("Warning: input_var_type_names is of length 0, return empty string")
    return("")
  }
  
  type_type_interactions_string = ""
  for (i in 1:length(input_var_type_names)){
    for (j in (i+1):length(input_var_type_names)){
      if (!(is.na(input_var_type_names[i]) | is.na(input_var_type_names[j])) & (j != i))
        type_type_interactions_string = paste(type_type_interactions_string,
                                              " + ",
                                              input_var_type_names[i],
                                              ":",
                                              input_var_type_names[j])
    }
  }
  
  # Remove the first " + " from the string
  type_type_interactions_string = substring(type_type_interactions_string, 6)
  
  return(type_type_interactions_string)
  
}

# stringhe intermedie
no_interaction_string = paste(var_names[-y_index], collapse = " + ")


qual_num_interactions_string = paste(outer(var_num_names,
                                               var_qual_names,
                                               FUN = function(x, y) paste(x, y, sep = ":")), collapse = " + ")

qual_qual_interactions_string = MakeSameInteractionsString(var_qual_names)


num_num_interactions_string = MakeSameInteractionsString(var_num_names)

# variabili quantitative al quadrato
num_vars_square_string <- paste("I(",
                                  var_num_names,
                                  "^2)",
                                  sep = "", collapse = " + ")


# string terms vector: vector of string terms
# return formula object
MakeFormula = function(string_terms_vector, intercept_bool = TRUE){
  base_formula = "y ~ "
  
  if (intercept_bool == FALSE){
    base_formula = paste(base_formula, " - 1 + ")
  }
  
  added_terms = paste(string_terms_vector, collapse = " + ")
  return(as.formula(paste(base_formula, added_terms)))
  
}


# creazione delle formule

# per evitare errori dovuti a formule troppo lunghe
options(expressions = 50000)

formula_yes_interaction_yes_intercept <- MakeFormula(c(no_interaction_string,
                                         num_vars_square_string,
                                         qual_qual_interactions_string,
                                         qual_num_interactions_string))

formula_yes_interaction_no_intercept <- MakeFormula(c(no_interaction_string,
                                                      num_vars_square_string,
                                                      qual_qual_interactions_string,
                                                      qual_num_interactions_string),
                                                intercept_bool = FALSE)

# formula senza interazioni
formula_no_interaction_yes_intercept = MakeFormula(no_interaction_string)
formula_no_interaction_no_intercept = MakeFormula(no_interaction_string, intercept_bool = FALSE)







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
# Quantitativa -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# per variabile risposta quantitativa: MSE
MSE.Loss = function(y.pred, y.test, weights = 1){
  sqrt(mean((y.test - y.pred)^2*weights))
}

# per variabile risposta quantitativa: MAE
MAE.Loss = function(y.pred, y.test, weights = 1){
  mean(abs(y.test - y.pred)*weights)
}


# funzione di convenienza
Null.Loss = function(y.pred, y.test, weights = 1){
  NULL
}

# La funzione di perdita può cambiare in base al problema
# Cambia MAE eventualmente

# °°°°°°°°°°°°°°°°°°°°°°° Warning: °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# cambia la funzione di errore il problema specifico

USED.Loss = function(y.pred, y.test, weights = 1){
  return(MAE.Loss(y.pred, y.test, weights))
}


# anche qua
df_err_quant = data.frame(name = NA, MAE = NA)


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
    df_error[which(df_error[,1] == model_name),2] = loss_value
  }
  
  else{
    # get the last index
    df_error[NROW(df_error) + 1,] = c(model_name, loss_value)
  }
  
  return(df_error)
}


# /////////////////////////////////////////////////////////////////
#------------------------ Analisi esplorative ---------------------
# /////////////////////////////////////////////////////////////////

# Analisi esplorativa sulla stima 
# eventuali inflazioni di zeri

hist(sss$y,nclass = 100)
summary(sss$y)

# possiamo provare a trasformare la risposta
hist(log(sss$y), nclass = 100)

# anche se le distribuzioni marginali non 
# forniscono informazioni riguardo alle condizionate
# se per il problema in questione è sensato possiamo impiegare 
# come nuova rispota il logaritmo della precedente y

# nota: in in questo modo la differenza dei logaritmi corrisponde
# al logaritmo del rapporto



# /////////////////////////////////////////////////////////////////
#------------------------ Modelli ---------------------------------
# /////////////////////////////////////////////////////////////////

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Media e Mediana --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# aggiunta media e mediana della risposta sull'insieme di stima come possibili modelli
# (per valutare se modelli più complessi hanno senso)

df_err_quant = Add_Test_Error(df_err_quant,
                              "sss mean",
                              USED.Loss(mean(sss$y), vvv$y))

df_err_quant = Add_Test_Error(df_err_quant,
                              "sss median",
                              USED.Loss(median(sss$y), vvv$y))

# df_err_quant = df_err_quant[-which(is.na(df_err_quant)),]


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello lineare Forward --------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

lm0 = lm(y ~ 1, data = sss)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
lm_step_no_interaction = step(lm0, scope = formula_no_interaction_yes_intercept,
                 direction = "forward")

rm(lm0)


# salvo la formula: ATTENZIONE: cambiare
# y ~ Obiettivo + Sottocategoria + Anno + Categoria + Durata + 
#   Nazione
lm_step_no_interaction = lm(y ~ Obiettivo + Sottocategoria + Anno + Categoria + Durata + Nazione, data = sss)






df_err_quant = Add_Test_Error(df_err_quant,
                              "lm_step_no_interaction",
                              USED.Loss(predict(lm_step_no_interaction, newdata = vvv), vvv$y))
df_err_quant

# salvo i coefficienti e rimuovo gli oggetti dalla memoria
lm_step_no_interaction_coef = coef(lm_step_no_interaction)

rm(lm_step_no_interaction)
gc()

# computazionalmente costoso (probabilmente)
# lm_step_yes_interaction = step(lm0, scope = formula_yes_interaction_yes_intercept,
#                               direction = "forward")

# salvo la formula


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge e Lasso ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# valuta: se ci sono molte esplicative qualitative -> model.matrix con molti zeri
# library(Matrix)
# X_mm_no_interaction_sss =  sparse.model.matrix(formula_no_interaction_no_intercept, data = sss)
# X_mm_no_interaction_vvv =  sparse.model.matrix(formula_no_interaction_no_intercept, data = vvv)

# # oneroso
# X_mm_yes_interaction_sss =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = sss)
# X_mm_yes_interaction_vvv =  sparse.model.matrix(formula_yes_interaction_no_intercept, data = vvv)

# default
# stima 
X_mm_no_interaction_sss = model.matrix(formula_no_interaction_no_intercept,
                                             data = sss)

# verifica
X_mm_no_interaction_vvv = model.matrix(formula_no_interaction_no_intercept,
                                       data = vvv)

# Interazioni: stima 
# X_mm_yes_interaction_sss = model.matrix(formula_yes_interaction_no_intercept,
#                                    data = sss)

# Interazioni verifica
#
# X_mm_yes_interaction_vvv = model.matrix(formula_yes_interaction_no_intercept,
#                                         data = vvv)

library(glmnet)
# Ridge ------


# No Interaction
# Selezione tramite cv
ridge_cv_no_interaction = cv.glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                    alpha = 0,
                                    lambda.min.ratio = 1e-07)

plot(ridge_cv_no_interaction)

# salvo i risultati
# > ridge_cv_no_interaction$lambda.min
# [1] 1866.315
# > ridge_cv_no_interaction$lambda.1se
# [1] 1145154


# ri-stimo i modelli con il lambda scelto

ridge_no_interaction_l1se = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                   alpha = 0,
                                   lambda = ridge_cv_no_interaction$lambda.1se)

ridge_no_interaction_lmin = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                   alpha = 0,
                                   lambda = ridge_cv_no_interaction$lambda.min)

# elimino dalla memoria l'oggetto ridge_cv
rm(ridge_cv_no_interaction)
gc()

# previsione ed errore
df_err_quant = Add_Test_Error(df_err_quant,
                              "ridge_no_interaction_l1se",
                              USED.Loss(predict(ridge_no_interaction_l1se, newx = X_mm_no_interaction_vvv),vvv$y))


df_err_quant = Add_Test_Error(df_err_quant,
                              "ridge_no_interaction_lmin",
                              USED.Loss(predict(ridge_no_interaction_lmin, newx = X_mm_no_interaction_vvv),vvv$y))

df_err_quant
# # YES Interaction: potrebbe essere troppo computazionalmente oneroso
# # Selezione tramite cv
# ridge_cv_yes_interaction = cv.glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                     alpha = 0,
#                                     lambda.min.ratio = 1e-07)
# 
# plot(ridge_cv_yes_interaction)
# 
# # salvo i risultati
# # > ridge_cv_yes_interaction$lambda.min
# # [1] 1866.315
# # > ridge_cv_yes_interaction$lambda.1se
# # [1] 1145154
# 
# 
# # ri-stimo i modelli con il lambda scelto
# 
# ridge_yes_interaction_l1se = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                    alpha = 0,
#                                    lambda = ridge_cv_yes_interaction$lambda.1se)
# 
# ridge_yes_interaction_lmin = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                    alpha = 0,
#                                    lambda = ridge_cv_yes_interaction$lambda.min)
# 
# # elimino dalla memoria l'oggetto ridge_cv
# rm(ridge_cv_yes_interaction)
# gc()
# 
# # previsione ed errore
# df_err_quant = Add_Test_Error(df_err_quant,
#                               "ridge_yes_interaction_l1se",
#                               USED.Loss(predict(ridge_yes_interaction_l1se, newx = X_mm_yes_interaction_vvv),vvv$y))
# 
# 
# df_err_quant = Add_Test_Error(df_err_quant,
#                               "ridge_yes_interaction_lmin",
#                               USED.Loss(predict(ridge_yes_interaction_lmin, newx = X_mm_yes_interaction_vvv),vvv$y))



# Lasso ------

# NO Interaction
# Selezione tramite cv
lasso_cv_no_interaction = cv.glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                    alpha = 1,
                                    lambda.min.ratio = 1e-07)

plot(lasso_cv_no_interaction)

# salvo i risultati
# > lasso_cv_no_interaction$lambda.min
# [1] 141.1799
# > lasso_cv_no_interaction$lambda.1se
# [1] 56994.57


# ri-stimo i modelli con il lambda scelto

lasso_no_interaction_l1se = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                   alpha = 1,
                                   lambda = lasso_cv_no_interaction$lambda.1se)

lasso_no_interaction_lmin = glmnet(x = X_mm_no_interaction_sss, y = sss$y,
                                   alpha = 1,
                                   lambda = lasso_cv_no_interaction$lambda.min)

# elimino dalla memoria l'oggetto lasso_cv
rm(lasso_cv_no_interaction)
gc()

# previsione ed errore
df_err_quant = Add_Test_Error(df_err_quant,
                              "lasso_no_interaction_l1se",
                              USED.Loss(predict(lasso_no_interaction_l1se, newx = X_mm_no_interaction_vvv),vvv$y))


df_err_quant = Add_Test_Error(df_err_quant,
                              "lasso_no_interaction_lmin",
                              USED.Loss(predict(lasso_no_interaction_lmin, newx = X_mm_no_interaction_vvv),vvv$y))


df_err_quant


# # YES Interaction: potrebbe essere troppo computazionalmente oneroso
# 
# # Selezione tramite cv
# lasso_cv_yes_interaction = cv.glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                     alpha = 1,
#                                     lambda.min.ratio = 1e-07)
# 
# plot(lasso_cv_yes_interaction)
# 
# # salvo i risultati
# # > lasso_cv_yes_interaction$lambda.min
# # [1] 141.1799
# # > lasso_cv_yes_interaction$lambda.1se
# # [1] 56994.57
# 
# 
# # ri-stimo i modelli con il lambda scelto
# 
# lasso_yes_interaction_l1se = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                    alpha = 1,
#                                    lambda = lasso_cv_yes_interaction$lambda.1se)
# 
# lasso_yes_interaction_lmin = glmnet(x = X_mm_yes_interaction_sss, y = sss$y,
#                                    alpha = 1,
#                                    lambda = lasso_cv_yes_interaction$lambda.min)
# 
# # elimino dalla memoria l'oggetto lasso_cv
# rm(lasso_cv_yes_interaction)
# gc()
# 
# # previsione ed errore
# df_err_quant = Add_Test_Error(df_err_quant,
#                               "lasso_yes_interaction_l1se",
#                               USED.Loss(predict(lasso_yes_interaction_l1se, newx = X_mm_yes_interaction_vvv),vvv$y))
# 
# 
# df_err_quant = Add_Test_Error(df_err_quant,
#                               "lasso_yes_interaction_lmin",
#                               USED.Loss(predict(lasso_yes_interaction_lmin, newx = X_mm_yes_interaction_vvv),vvv$y))
# 
# df_err_quant


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Albero -------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(tree)

# albero che sovraadatta
tree_full = tree(y ~.,
                 data = sss[id_cb1,],
                 control = tree.control(nobs = length(id_cb1),
                                        minsize = 2,
                                        mindev = 1e-05))

# controllo che sia sovraadattato
plot(tree_full)

# potatura
tree_pruned = prune.tree(tree_full, newdata = sss[-id_cb1,])

plot(tree_pruned)
plot(tree_pruned, xlim = c(0, 20))

tree_best_size = tree_pruned$size[which.min(tree_pruned$dev)]
# 2

abline(v = tree_best_size, col = "red")

final_tree_pruned = prune.tree(tree_full, best = tree_best_size)

plot(final_tree_pruned)
text(final_tree_pruned, cex = 0.7)

df_err_quant = Add_Test_Error(df_err_quant,
                              "tree_pruned best",
                              USED.Loss(predict(final_tree_pruned, newdata = vvv), vvv$y))


df_err_quant = Add_Test_Error(df_err_quant,
                              "tree_pruned 10",
                              USED.Loss(predict(prune.tree(tree_full, best = 10), newdata = vvv), vvv$y))

df_err_quant

rm(tree_full)
gc()
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Modello Additivo ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(gam)

# °°°°°°°°°°°°°°°°°°°°°°°°°°°Warning: lento°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# stepwise forward
gam0 = gam(y ~ 1, data = sss)
# riconosce le qualitative se sono fattori
my_gam_scope = gam.scope(sss[,-y_index], arg = c("df=2", "df=3", "df=4", "df=5", "df=6"))

# prova anche parallelo
# require(doMC)
# registerDoMC(cores=4)
# step.Gam(Gam.object,scope ,parallel=TRUE)

gam_step = step.Gam(gam0, scope = my_gam_scope)

# salvo il modello finale
# y ~ Sottocategoria + s(Obiettivo, df = 4) + s(Durata, df = 4) + Anno

# gam_step = gam(y ~ Sottocategoria + s(Obiettivo, df = 4) + s(Durata, df = 4) + Anno,
#                 data = sss)

object.size(gam_step)

df_err_quant = Add_Test_Error(df_err_quant,
                              "additivo_step",
                              USED.Loss(predict(gam_step, newdata = vvv), vvv$y))

df_err_quant

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
# °°°°°°°°°°°°°°°°°°°°°° Warning °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°à
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

df_err_quant = Add_Test_Error(df_err_quant,
                              "MARS",
                              USED.Loss(predict(mars1, x = X_mm_no_interaction_vvv),vvv$y))

df_err_quant

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


mars1_pred2_names

mars1_pred2_names = rep(NA, length(mars1$model$pred2))
mars1_pred2_names[which(mars1$model$pred2 != 0)] = colnames(X_mm_no_interaction_sss[,mars1$model$pred2[which(mars1$model$pred2 != 0)]])
mars1_pred2_names[which(mars1$model$pred2 == 0)] = "None"

mars1_pred_names_matrix = cbind(c("Intercept", colnames(X_mm_no_interaction_sss[,mars1$model$pred1])),
                                mars1_pred2_names)

colnames(mars1_pred_names_matrix) = c("pred1", "pred2")

mars1_pred_names_matrix



# rm(mars1)
# gc()

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
# df_err_quant[8,] = c("mars",
#                      USED.Loss(X_mars_vvv %*%
#                                 solve(t(X_mars_sss) %*% X_mars_sss) %*%
#                                 t(X_mars_sss) %*% as.matrix(sss$y) %>% as.numeric(),
#                                       vvv$y))


# tramite stima e verifica (per risposta qualitativa)



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PPR ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# stima convalida
# k le possibili funzioni dorsali
K = 4

err_ppr_test_validation = rep(NA, K)

for(k in 1:K){
  mod = ppr(y ~ .,
            data = sss[id_cb1,],
            nterms = k)
  err_ppr_test_validation[k] = mean((predict(mod, sss[-id_cb1,]) - sss$y[-id_cb1])^2)
}

rm(mod)

err_ppr_test_validation

mod_ppr1 = ppr(y ~ .,
               data = sss,
               nterms = 2)

df_err_quant = Add_Test_Error(df_err_quant,
                              "PPR",
                              USED.Loss(predict(mod_ppr1, vvv), vvv$y))

df_err_quant

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Bagging ------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(snowfall)
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
  err_bg_trees[j] = sum(sfSapply(rep(1:4),
                                 function(x) bagging(y ~., sss,
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

bagging_model = bagging(y ~., sss, nbag = 400, coob = FALSE)

df_err_quant = Add_Test_Error(df_err_quant,
                              "Bagging",
                              USED.Loss(predict(bagging_model, newdata = vvv), vvv$y))


df_err_quant


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Implemenetazione in parallelo
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
  
  err[i] = sum(sfSapply(rep(1:4),
                        function(x) ranger(y ~., data = sss,
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
                                 function(x) ranger(y ~., sss,
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

best_mtry = 2

# modello finale e previsioni
random_forest_model = ranger(y ~., sss,
                             mtry = best_mtry,
                             num.trees = 200,
                             oob.error = TRUE,
                             importance = "permutation")

df_err_quant = Add_Test_Error(df_err_quant,
                              "Random Forest",
                              USED.Loss(predict(random_forest_model, data = vvv,
                                                type = "response")$predictions, vvv$y))

df_err_quant

# Importanza delle variabili
vimp = importance(random_forest_model)

dotchart(vimp[order(vimp)])



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rete neurale -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



# /////////////////////////////////////////////////////////////////
#------------------------ Sintesi Finale -------------------------
# /////////////////////////////////////////////////////////////////


cbind(df_err_quant[,1], round(df_err_quant[,2] %>% as.numeric(), 2))

# Comparazione modelli
# selezione modelli migliori e commenti su questi:
# es -> vanno meglio modelli con interazioni oppure modelli additivi



# Ridge 
# Lasso

# guarda coefficienti
# predict.glmnet(oggetto, type = "coefficients")


# Modello additivo : lo devo ri-stimare nel caso
# plot(gam_step, terms = c("s(Durata, df = 4)"), se = T)

# summary(gam_step)

# MARS:
# mars1$model

# mars1_pred_names_matrix

# per il grafico lo devo ristimare
# plot(mars1, predictor1 = 40, predictor2 = 30)

# Random Forest: guarda grafico importanza variabili



