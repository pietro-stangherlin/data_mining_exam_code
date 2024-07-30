rm(list = ls())
library(dplyr)


# Per Lapsus
# CTRL + C per commentare più righe di codice

# /////////////////////////////////////////////////////////////////////////////
#  ----------------------- Lettura e Preprocessing -----------------------------
# /////////////////////////////////////////////////////////////////////////////


# per avere un'idea del file da terminale: 
# head nome_file.formato

dati = read.csv("test_dataset/df_multi.csv", sep = ",", stringsAsFactors = F)

# NOTA: converto in fattori solo alla fine del preprocessing
# in modo da non dover riconvertire tutto ogni volta 

# controllo
str(dati)

# rinomimo la risposta in y: cambia il primo "y" in base al problema
names(dati)[which(names(dati) == "y")] = "y"

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

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Valori mancanti EMPTY ----------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Mancanti "" (caratteri vuoti: "") (più facili da gestire)

# Eventualmente cambia il parametro "" con altri in base al propblema
missing_freq_empty = apply(dati, 2, function(col) CountEqualTo(col, ""))
missing_freq_empty

# se sono character sono quasi sicuramente categoriali (controlla comunque)
str(dati)

# aggiungo la modalità "EMPTY" al posto di ""

# ottieni i nomi per cui ci sono dei ""
names_missing = names(missing_freq_empty[which(missing_freq_empty > 0)])

# sostituisco i "" con il character "EMPTY" (attenzione al problema specifico)
# (potrebbe essere necessario cambiare nome alla modalità)
dati[,names_missing] = apply(dati[,names_missing], 2,
                             function(col) ifelse(col == '', "EMPTY", col))

head(dati[,names_missing])

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Valori mancanti NA ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Mancanti NA (classici) (più difficili da gestire)
missing_freq_NA = apply(dati, 2, function(col) sum(is.na(col)))
missing_freq_NA


str(dati)
# se ci sono dei dati character che presentano dei NA: di fatto posso considerarli
# come "" e ripetere la procedura "EMPTY"

# per questo specifico problema questa casistica si presenta con
# le variabili "anno" e "mese"
# copia - incolla:
case_specific_missing = c("anno", "mese")

dati[,case_specific_missing] = apply(dati[,case_specific_missing], 2,
                             function(col) ifelse(is.na(col), "EMPTY", col))


# Controllo
# righe dati mancanti empty
# non dovrebbero esserci
row_missing_index_empty = which((apply(dati, 1, function(row) (IsValueInVector(row, "")))) == TRUE)
length(row_missing_index_empty)


# NA
missing_freq_NA = apply(dati, 2, function(col) sum(is.na(col)))
missing_freq_NA
str(dati)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rimozione tutte righe con NA -----------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# righe dati mancanti NA
row_missing_index_NA = which((apply(dati, 1, function(row)( NA %in% row))))
# se sono poche le elimino
length(row_missing_index_NA)

# elimino le righe con dati mancanti se sono poche
# rispetto alla numerosità campionaria
# soglia arbitraria -> da cambiare

length(row_missing_index_NA) / NROW(dati)

if((length(row_missing_index_NA) / NROW(dati)) < 0.05){
  dati = dati[-row_missing_index_NA,]
}


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Valori mancanti NA Quantitative --------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Visto il tempo limitato adotto una delle seguenti opzioni:
# 1) eliminazione mancanti se sono pochi (in base al problema)
# 2) converto in classi i valori quantitativi e aggiungo la modalità EMPTY
# in corrispondenza dei NA

# Per risparmiare tempo discrimino quali variabili sottoporre a 1) o 2)
# in base alla frequenza relativa di NA (può cambiare da problema a problema):
# sotto una certa soglia adotto 1), sopra adotto 2)

relative_missing_freqs = missing_freq_NA / NROW(dati)

# seleziono le variabili con frequenza relativa di valori NA inferiore 
# a una certa soglia prefissata
temp_threshold = 0.05

var_names_to_remove = colnames(dati[,which(relative_missing_freqs > 0 & relative_missing_freqs < temp_threshold)])
var_names_to_remove

if(length(var_names_to_remove != 0)){
  row_missing_index_NA_to_remove = which((apply(dati[,var_names_to_remove], 1, function(row)( NA %in% row))))
  dati = dati[-row_missing_index_NA_to_remove,]}
str(dati)


# 2)
# funzione per una di queste variabili
# @param var_vector: vettore di variabile quantitativa con eventuali NA
# @param my_breaks vettore di separazione: default NA
# @param my_probs vettore di quantili: default quartili
# @return: vettore di character della variabile categorizzata per quantili 
# + modalità EMPTY al posto dei NA

ToCategoricalIncludeNA = function(var_vector,
                                  my_breaks = NA,
                                  my_probs = seq(0, 1, 0.25)){
  NOT_NA_index = which(!is.na(var_vector))
  
  used_breaks = my_breaks
  
  # default
  if(is.na(my_breaks)){
    used_breaks = quantile(var_vector, probs = my_probs, na.rm = TRUE)
  }
  
  to_breaks = as.character(cut(var_vector[NOT_NA_index],
                               breaks = unique(used_breaks),
                               include.lowest = TRUE))
  
  returned_vector = rep("EMPTY", length(var_vector))
  
  returned_vector[NOT_NA_index] = to_breaks
  
  return(returned_vector)
  
}


# test 
# ToCategoricalIncludeNA(c(1,2,NA,4,5,NA))


# lista di variabili quantitative da trasformare in qualitative
var_names_to_qual = colnames(dati[,which(relative_missing_freqs >= temp_threshold)])
var_names_to_qual

if(length(var_names_to_qual) != 0){
  # attenzione agli eventuali valori di default in ToCategoricalIncludeNA
  dati[,var_names_to_qual] = apply(dati[,var_names_to_qual], 2, 
                                   function(col) ToCategoricalIncludeNA(col))

}

rm(temp_threshold)

str(dati)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Valori unici ---------------------------------
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
unique_vals_df_no_y

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Riduzione categorie qualitative -------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Funzioni 
# ===================================================================

# @input my_table(table)
# @input first (int): how many values frequencies one should print (default all)
# @input print_lenght (bool): print the number of total unique values
# @return values frequency in decreasing frequency order
ReturnFirstTable = function(my_table,
                           first = NULL,
                           print_length = FALSE){
  
  if (print_length == TRUE){
    print(paste("number of unique values: ", length(my_table)))
  }
  
  # print all
  if(is.null(first)){
    first = length(my_table)
    return(my_table  %>% sort(decreasing = T))
  }
  
  # print only first
  else{
    return((my_table  %>% sort(decreasing = T))[1:min(first,length(my_table))])
  }
  
}

# @input my_df (data.frame)
# @input var_index_subset (vector of int): indexes of variables subset
# @input first (int): how many values frequencies one should print (default all)
# @input print_lenght (bool): print the number of total unique values
# @print values frequency in decreasing frequency order
# going forward with the "enter" input and backward with the "b" input

PrintAllTables = function(my_df,
                          var_index_subset = NULL,
                          first = NULL,
                          print_length = FALSE){
  
  # all variables
  if(is.null(var_index_subset)){
    var_index_subset = 1:NCOL(my_df)}
  
  var_index_counter = 0
  var_names_temp = colnames(my_df)
  
  print("press (enter) to forward and 'b' to backward and q to quit")
  
  while(var_index_counter < length(var_index_subset)){
    input = readline("")
    
    if((input == "q")){
      var_index_counter = length(var_index_subset) - 1}
    
    if((input != "b")){
      var_index_counter = var_index_counter + 1}
    
    if(input == "b"){
      var_index_counter = var_index_counter - 1}
    
    if(var_index_counter <= 0){
      var_index_counter = 1}
    
    
    print("--------------------------------------------")
    print(var_names_temp[var_index_subset[var_index_counter]])
    print(ReturnFirstTable(table(my_df[,var_index_subset[var_index_counter]]),
                    first,
                    print_length))
    print("--------------------------------------------")}
    
}



# Analisi prime 40 frequenze delle modalità di tutte
# le variabili

PrintAllTables(dati, first = 40)




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


# ================================================================================

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Riduzione modalità qualitative per frequenza ----------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Esempio di applicazione: cambiare il nome della variabile e la soglia
# ==============================================================================


unique_vals_df_no_y
str(dati)

# Variabile singola ----------------
# °°°°°°°°°°°°°°°°°°°° Warning: cambia nome variabile °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# Per una specifica variabile 

# temp_table_freq = TableFreqFun(dati, "x5")
# temp_table_freq
# 
# dati[,"x5"] = RaggruppaModalita(dati, "x5", temp_table_freq, 400, "Altro")
# 
# # check
# temp_table_freq = TableFreqFun(dati, "x5")
# temp_table_freq
# 
# rm(temp_table_freq)

# Tutte le variabili character ------------------------
# per motivi computazionali, al costo di perdere informazioni
# riduco le modalità a 25 modalità

# funzione per una singola variabile
GroupValuesQual = function(df, qual_vector_var_name, new_name = "Altro"){
  temp_table_freq = table(df[,qual_vector_var_name]) %>% sort(decreasing = T)
  
  # meno di 30 modalità: non c'è bisogno di nessuna modifica
  if (length(temp_table_freq) <= 25){
    return(df[, qual_vector_var_name])
  }
  
  # altrimenti riduci le modalità
  # seleziona la frequenza soglia oltre cui aggregare
  # (temp_table_freq è già ordinata in ordine decrescente per frequenza)
  freq_threshold = temp_table_freq[24]
  
  return(RaggruppaModalita(df, qual_vector_var_name, temp_table_freq,
                           freq_threshold, new_name))
}

char_var_names = colnames(dati[,-y_index])[which(unlist(lapply(dati[,-y_index], typeof)) == "character")]

for(name in char_var_names){
  dati[,name] = GroupValuesQual(dati, name, "Altro")
}

str(dati)

# check 
unique_vals_df = data.frame(nome = rep("", NCOL(dati)),
                            indice = rep(0, NCOL(dati)),
                            uniques = rep(0, NCOL(dati)))


unique_vals_df$nome = colnames(dati)
unique_vals_df$indice = as.numeric(1:NCOL(dati))
unique_vals_df$uniques = as.numeric(apply(dati, 2, function(col) length(unique(col))))


unique_vals_df

# eslcusione della y
unique_vals_df_no_y = unique_vals_df[-which(unique_vals_df$nome == "y"),]
unique_vals_df_no_y


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Riduzione quantitative in qualitative per poche modalità --------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# + qualitative che sono codificate numericamente

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
# Analisi istrogramma quantitative -------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# per ogni variabile esplicativa quantitativa 
# disegna l'istrogramma della sua distribuzione empirica
# e quella del suo logaritmo (opportunamente traslata)

# @input: my_df (data.frame)
# @input var_index_subset (vector of int): indexes of quantitative variables subset
# output: plots of each quantitative variable histogram
DrawQuantHist = function(my_df,
                        var_index_subset = NULL,
                        my_breaks = 50){
  
  # all variables
  if(is.null(var_index_subset)){
    var_index_subset = 1:NCOL(my_df)}
  
  var_index_counter = 0
  var_names_temp = colnames(my_df)
  
  par(mfrow = c(1,2))
  
  print("press (enter) to forward and 'b' to backward and q to quit")
  
  while(var_index_counter < length(var_index_subset)){
    input = readline("")
    
    if((input == "q")){
      var_index_counter = length(var_index_subset) - 1}
    
    if((input != "b")){
      var_index_counter = var_index_counter + 1}
    
    if(input == "b"){
      var_index_counter = var_index_counter - 1}
    
    if(var_index_counter <= 0){
      var_index_counter = 1}
    
    
    # original scale
    hist(my_df[,var_index_subset[var_index_counter]],
         breaks = my_breaks,
         main = var_names_temp[var_index_subset[var_index_counter]],
         xlab = "values")
    
    # log translated scale
    temp_min = min(my_df[,var_index_subset[var_index_counter]])
    
    if(temp_min > 0){
      temp_min = 0}
    
    hist(log(my_df[,var_index_subset[var_index_counter]] - temp_min + 1e-05 ),
         breaks = my_breaks,
         main = paste("log", var_names_temp[var_index_subset[var_index_counter]]),
         xlab = "log values")
    }
  
  par(mfrow = c(1,1))
  
}


# Analisi istogrammi
DrawQuantHist(dati, var_num_index)


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

num_vars_square_string = ""

if(length(var_num_names) != 0){
  num_vars_square_string <- paste("I(",
                                var_num_names,
                                "^2)",
                                sep = "", collapse = " + ")}


# string terms vector: vector of string terms
# return formula object
MakeFormula = function(string_terms_vector, intercept_bool = TRUE){
  base_formula = "y ~ "
  
  # remove empty vector terms
  string_terms_vector = string_terms_vector[which(string_terms_vector != "")]
  
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

formula_yes_interaction_yes_intercept
formula_yes_interaction_no_intercept

# formula senza interazioni
formula_no_interaction_yes_intercept = MakeFormula(no_interaction_string)
formula_no_interaction_no_intercept = MakeFormula(no_interaction_string, intercept_bool = FALSE)

formula_no_interaction_yes_intercept
formula_no_interaction_no_intercept

