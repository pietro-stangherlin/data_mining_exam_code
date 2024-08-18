# To do File

Descrizione delle modifiche e aggiunte da fare al codice.

## generale
### File deeplearning per classificare immagini

## data_preprocessing

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

### Generale: subsampling
Aggiungere un'opzione nelle funzioni di convalida incrociata per il subsampling:
Assumere che vi possano essere due diverse partizioni di fold: una bilanciata per la stima 
e una sbilanciata per la verifica.
Con l'opzione di default in cui coincidono.

### Gestisci dati aggregati (classificazione)
Es. tabelle di frequenza.

### Regressione per poche osservazioni: convalida incrociata

### Dicotomica per poche osservazioni + subsampling

### Multiclasse per poche osservazioni +  subsampling

### DONE - Salvataggio output su file
Ad esempio per recuperare formule di modelli se R crasha.
Problema: con sink() è possibile ma ciò implica che l'output sia visibile SOLO sul file,
io voglio che sia visibile sia da console che salvato su file.

### DONE-  Tutti i modelli: salvataggio locale modelli
Trova un modo di salvare i modelli su disco in modo da poter liberare la memoria centrale 
e riprendere i modelli per le conclusioni.

### DONE-  In alternativa: salva alcuni output + grafici anche su file


### Regressione sulle componenti principali
Può essere impiegata anche per modelli additivi e MARS, ovvero per modelli che operano una selezione delle variabili.

### Aggiungi EARTH come ulteriore MARS 


## File report

### Crea diversi file report .odt per ogni sottocaso