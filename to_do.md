# To do File

Descrizione delle modifiche e aggiunte da fare al codice.

## data_preprocessing

### Funzione: prime 30 modalità più frequenti
Funzione per vedere le prime 30 più frequenti modalità di ciascuna variabile 
Premendo un tasto per andare in avanti, indietro e per chiudere.
Così posso avere un'idea delle modalità da tenere e di quelle da aggreggare.


## loss_functions

### Pesi tabella errata classificazione: dicotomica
Aggiungi il parametro dei pesi per l'errore di errata classificazione nella funzione apposita.

### Pesi basati sulla risposta: dicotomica e multiclasse
probabilmente può essere troppo lungo da implementare (per la multiclasse), ma potrebbe essere opportuno costruire una metrica che pesi in base alla tipologia di classificazione effettuata: es falso negativo pesa il doppio di un falso positivo, oppure peso maggiormente le osservazioni per cui nell'insieme di verifica la risposta assume cuna determinata modalità.


## Regressione e classificazione
### Regressione sulle componenti principali
Può essere impiegata anche per modelli additivi e MARS, ovvero per modelli che operano una selezione delle variabili.

### Aggiiungi earth come ulteriore MARS 

