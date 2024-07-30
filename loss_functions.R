# Codice con le varie funzioni di perdita,
# in modo che eventuali modifiche vengano eseguite una volta sola

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

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Dicotomica -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# weights: solo per tasso di errata classificazione 
tabella.sommario = function(previsti, osservati,
                            print_bool = FALSE,
                            weights = 1){
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
  
  
  err.tot <- sum((previsti != osservati) * weights) / sum(length(previsti) * weights)
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


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Multiclasse -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# input: 
# previsti: vector of character or factors
# osservati: vector of character or factors

MissErr = function(previsti, osservati, weights = 1){
  
  # check
  if(length(previsti) != length(osservati)){
    print(paste("Warning", "length(previsti) = ", length(previsti), 
                "; length(osservati) = ", length(osservati)))
    print("return NULL")
    return(NULL)
  }
  
  return( 1- sum((previsti == osservati)*weights) / length(previsti))
}

previsti_test = c("a", "b", "a", "c")
osservati_test = c("a", "c", "a", "c")

MissErr(previsti_test, osservati_test)

#weight more the first observation
MissErr(previsti_test, osservati_test, weights = c(0.4, 0.2, 0.2, 0.2))

# error expected
MissErr(c(1,1), c(0,0,0))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Generale -------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

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
    df_error[which(df_error[,1] == model_name),2:length(loss_value)] = loss_value
  }
  
  else{
    # get the last index
    df_error[NROW(df_error) + 1,] = c(model_name, loss_value)
  }
  
  return(df_error)
}


