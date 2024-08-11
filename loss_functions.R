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
# if the df_metric already has a model name in name column with the same as input: update the metric value
# otherwise append the new name and error
# arguments:
# @df_metric (data.frame): data.frame with columns: [1]: name and [2] and more: metrics
# @model_name (char): character with the model name
# @metric_value (num): numeric with the metric on the test set
# @return: df_metric

Add_Test_Metric = function(df_metric, model_name, metric_value){
  # check if the model name is already in the data.frame
  is_name = model_name %in% df_metric[,1]
  
  # if yes: get the index and subscribe
  if(is_name){
    df_metric[which(df_metric[,1] == model_name),] = c(model_name, metric_value)
  }
  
  else{
    # get the last index
    df_metric[NROW(df_metric) + 1,] = c(model_name, metric_value)
  }
  
  return(df_metric)
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter Selection ------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#' @param my_param_values (vector): vector of parameters, ascendent order is assumed
#' @param my_metric_matrix (matrix): rows -> parameters in the same order of my_param_values,
#'                                 columns -> metrics, 
#'   each cell contains the cv metric already averaged
#' @param my_se_matrix (matrix): matrix of standard errors of cv metrics
#' @param my_metric_names (vector of strings): names of the errors,
#' @param my_one_se_best (bool): choose the parameter whose error is less than 1se distant 
#' from the one associate with best parameter (in the direction of less a complex model)
#' @param my_higher_more_complex (bool): if TRUE -> the higher the parameter value the higher
#' the model complexity (ex tree size); if FALSE -> the opposite (es lambda in ridge)
#' only used if my_one_se_best is TRUE
#'
#' WARNING, pay attention to this parameter:
#' @param indexes_metric_max (vector of ints): indexes for which high metric values is best (ex f1 score)
#' (default NULL)
#' 
#' @return: best metric list (list):
#' the list has nested elements and is of the type
#' list[[metric_name]][[x]]
#' where x are: "best_param_index" (int), "best_param_value" (num), "metric_values" (vector of num)

CvMetricBest = function(my_param_values,
                        my_metric_matrix,
                        my_se_matrix = 0,
                        my_metric_names,
                        my_one_se_best = TRUE,
                        my_higher_more_complex = TRUE,
                        indexes_metric_max = NULL){
  
  # this code is sub-optimal, someday I'll change it
  
  # compute standard deviations matrix lower and upper
  se_lower = my_metric_matrix - my_se_matrix
  se_upper = my_metric_matrix + my_se_matrix
  
  # BEST PARAM VALUES section
  
  n_col_metric_matrix = NCOL(my_metric_matrix)
  
  best_params = rep(NA, n_col_metric_matrix)
  names(best_params) = my_metric_names
  
  
  # Check metrics min and max best
  
  if(is.null(indexes_metric_max)){
    indexes_best_params = apply(my_metric_matrix, 2, which.min)
  }
  
  else{
    indexes_best_params = c(apply(my_metric_matrix[,-indexes_metric_max], 2, which.min),
                            apply(my_metric_matrix[,indexes_metric_max], 2, which.max))
  }
  
  if(my_one_se_best == TRUE){
    
    # non efficient procedure, but we assume the parameter space is small
    # find all indexes for which the best error is inside the 1se band
    for(i in 1:length(indexes_best_params)){
      # more readable code
      temp_best_metric = my_metric_matrix[indexes_best_params[i], i]
      # parameter indexes for which the metric is inside the 1se band of best param metric
      temp_param_indexes = which(my_param_values %in% my_param_values[which(temp_best_metric > se_lower[,i] &
                                                                              temp_best_metric < se_upper[,i])])
      if(my_higher_more_complex == TRUE){
        indexes_best_params[i]= temp_param_indexes[which.min(my_param_values[temp_param_indexes])]
      }
      
      if(my_higher_more_complex == FALSE){
        indexes_best_params[i]= temp_param_indexes[which.max(my_param_values[temp_param_indexes])]
      }
    }
  }
  
  best_params = my_param_values[indexes_best_params]
  
  # return indexes and best param values
  
  returned_list = list()
  
  # cycle over all metrics
  for (i in 1:n_col_metric_matrix){
    # add index
    returned_list[[my_metric_names[i]]][["best_param_index"]] = indexes_best_params[i]
    # add param value
    returned_list[[my_metric_names[i]]][["best_param_value"]] = best_params[i]
    # add all metrics relative to that row (best param index row)
    returned_list[[my_metric_names[i]]][["metric_values"]] = my_metric_matrix[indexes_best_params[i],]
  }
  
  
  return(returned_list)
  
}

#' @description extract a the vector of best parameter for each metric
#' from the output of CvMetricBest (list)
#' @param my_cv_metric_best_object (list): each element corresponds
#' to the name of the metric (ex. MSE) and the best parameter is accessed
#' by my_cv_metric_best_object[[i]]$best_param_value, where i is a generic index
#' 
#' @return vector of best params
ExtractBestParams = function(my_cv_metric_best_object){
  
  best_params_vector = rep(NA, length(my_cv_metric_best_object))
  
  for (i in 1:length(my_cv_metric_best_object)){
    best_params_vector[i] = my_cv_metric_best_object[[i]]$best_param_value
  }
  
  return(best_params_vector)
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter Selection Plotting --------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#' @param my_param_values (vector): vector of parameters, ascendent order is assumed
#' @param my_metric_matrix (matrix): rows -> parameters in the same order of my_param_values,
#'                                 columns -> metrics, 
#'   each cell contains the cv metric already averaged
#' @param my_se_matrix (matrix): matrix of standard errors of cv metrics
#' @param my_metric_names (vector of strings): names of the errors,
#' @param my_best_param_values (vector of nums): param values which will be plotted as vertical lines

PlotCvMetrics = function(my_param_values, my_metric_matrix,
                         my_se_matrix = 0,
                         my_metric_names,
                         my_best_param_values = c(),
                         my_main = "Model metrics", my_xlab = "parameter", my_legend_coords = "topright",
                         my_xlim = NULL, my_ylim = NULL){
  
  
  # compute standard deviations matrix lower and upper
  se_lower = my_metric_matrix - my_se_matrix
  se_upper = my_metric_matrix + my_se_matrix
  
  # if xlim null use entire x axis (default)
  if (is.null(my_xlim)){
    my_xlim = c(min(my_param_values), max(my_param_values))
  }
  
  # if ylim is null use min and max over the entire matrix
  if (is.null(my_ylim)){
    my_ylim = c(min(se_lower), max(se_upper))
  }
  
  # plot showed
  
  plot(my_param_values, my_metric_matrix[,1],
       xlab = my_xlab, ylab = "metric",
       main = my_main, pch = 16,
       xlim = my_xlim,
       ylim = my_ylim)
  
  arrows(my_param_values, se_lower[,1],
         my_param_values, se_upper[,1],
         length = 0.03, angle = 90)
  
  for (i in 2:NCOL(my_metric_matrix)){
    points(my_param_values, my_metric_matrix[,i],
           pch = 15 + i, col = i)
    arrows(my_param_values, se_lower[,i],
           my_param_values, se_upper[,i],
           length = 0.03, angle = 90, col = i)
  }
  
  legend(my_legend_coords,
         legend = my_metric_names,
         col = 1:NCOL(my_metric_matrix),
         pch = 15 + (1:NCOL(my_metric_matrix)))
  
  # plot them
  for (i in 1:length(my_best_param_values))
    abline(v = my_best_param_values[i], col = i)
}


#' @param my_plotting_function (function): function with NO ARGUMENTS
#' outputting the desired plot
#' @param my_path_plot (char): path of the where the plot will be saved on disk
#' @param my_width (int): pixel width of saved plot
#' @param my_height (int): pixel height of saved plot
#' @param my_point_size (int): point size of saved plot
#' @param my_quality (int): quality of saved plot
#' 
#' @description show plot determined by my_plotting_function and save it on disk
#' 
#' @return None
PlotAndSave = function(my_plotting_function,
                       my_path_plot,
                       my_width = FIGURE_WIDTH,
                       my_height = FIGURE_HEIGHT,
                       my_point_size = FIGURE_POINT_SIZE,
                       my_quality = FIGURE_QUALITY){
  
  # call to shown plot
  my_plotting_function()
  
  
  # plot saved on disk
  jpeg(my_path_plot,
       width = my_width, height = my_height,
       pointsize = my_point_size, quality = my_quality)
  
  # call to saved plot
  my_plotting_function()
  
  dev.off()
}

