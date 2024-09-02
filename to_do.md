# To do File

Descrizione delle modifiche e aggiunte da fare al codice.

## generale

### TO - DO: Change location of code to produce train-test or CV (+subsampling) in one script
So it's changed only once

### TO - DO: rename loss_functions.R to metrics_functions.R across all files

### Subsampling rispetto alle modalità di una covariata qualitativa (sia stima-verifica che CV)
Prima valuta le proprietà statistiche, ma di fatto è concettualmente simile alla stratificazione.

### File deeplearning per classificare immagini

## data_preprocessing

### TO - DO: Aggiungi altre tipologie di imputazione per dati mancanti quantitative

### DONE --- controlla e sistema posizione per la definizione di var_names nel file (+y_index etc).
E' possibile che la definizione di var_names sia troppo "a monte", in questo modo sono inclusi alcuni nomi di variabili che in realtà sono state eliminate.

### DONE --- join / merge function to merge dataframes based on key value
Use merge or dplyr

### DONE --- Funzione: prime k modalità più frequenti
Funzione per vedere le prime 30 più frequenti modalità di ciascuna variabile 
Premendo un tasto per andare in avanti, indietro e per chiudere.
Così posso avere un'idea delle modalità da tenere e di quelle da aggreggare.


### DONE --- Funzione: analisi esplorative esplicative quantitative
Per ogni esplicativa quantitativa disegnare l'istogramma della distribuzione per valutare se trasformarla con logaritmi o 
effettuare altre trasformazioni.
Premendo un tasto per andare in avanti, indietro e q per chiudere.


## loss_functions

### DONE --- Pesi tabella errata classificazione: dicotomica
Aggiungi il parametro dei pesi per l'errore di errata classificazione nella funzione apposita.

### Pesi basati sulla risposta: dicotomica e multiclasse
probabilmente può essere troppo lungo da implementare (per la multiclasse), ma potrebbe essere opportuno costruire una metrica che pesi in base alla tipologia di classificazione effettuata: es falso negativo pesa il doppio di un falso positivo, oppure peso maggiormente le osservazioni per cui nell'insieme di verifica la risposta assume cuna determinata modalità.


## Regressione e classificazione

### TO - DO: aggiungi e DESCRIVI ridge e lasso con perdita logit (binary)

### TO - DO: rivedi lasso mgaussian multiclasse ed eventualmente descrivi algoritmo + aggiungi altre tipologie
E trova un modo di descrivere l'effetto delle variabili nel caso multiclasse

### DONE - Generale: subsampling
Aggiungere un'opzione nelle funzioni di convalida incrociata per il subsampling:
Assumere che vi possano essere due diverse partizioni di fold: una bilanciata per la stima 
e una sbilanciata per la verifica.
Con l'opzione di default in cui coincidono.

### TO - DO: Gestisci dati aggregati (classificazione)
Es. tabelle di frequenza.

### DONE - Regressione per poche osservazioni: convalida incrociata

### DONE - Dicotomica per poche osservazioni + subsampling

### DONE - Multiclasse per poche osservazioni + subsampling

### TO - DO: sistema funzioni CV per metriche uniche (non vettoriali)

### TO - DO: nel grafico dei coefficienti, aggiungi indici di variabili di interesse

### DONE - Salvataggio output su file
Ad esempio per recuperare formule di modelli se R crasha.
Problema: con sink() è possibile ma ciò implica che l'output sia visibile SOLO sul file,
io voglio che sia visibile sia da console che salvato su file.

### DONE - Tutti i modelli: salvataggio locale modelli
Trova un modo di salvare i modelli su disco in modo da poter liberare la memoria centrale 
e riprendere i modelli per le conclusioni.

### DONE - Salva alcuni output + grafici anche su file


### TO - DO: Regressione sulle componenti principali
Può essere impiegata anche per modelli additivi e MARS, ovvero per modelli che operano una selezione delle variabili.

### TO - DO: Aggiungi EARTH come ulteriore MARS 

### TO - DO: converti molte funzioni in C++ con Armadillo (se ha senso e se induce miglioramenti tangibili)

## File report

### Crea diversi file report .odt per ogni sottocaso
Aggiungi un caso in cui si impieghi classificazione + regressione (es. con molti zeri.)

