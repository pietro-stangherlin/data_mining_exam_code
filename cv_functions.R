# Cross validation functions
library(dplyr)

# parallel
library(snowfall)
# 
N_CORES = parallel::detectCores()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge and Lasso ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 library(glmnet)

#' GLMNET CV cycles in case of few data sample
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#'
#' @param my_x (matrix): complete model matrix passed to glmnet
#' @param my_y (vector): y glmnet argument
#' @param my_alpha (int): alpha passed to glmnet (0 -> ridge, 1 -> lasso)
#' @param my_lambda_vals (vector): vector of lambda used
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"
ManualCvGlmnet = function(my_id_list_cv,
                          my_metric_names,
                          my_x,
                          my_y,
                          my_alpha,
                          my_lambda_vals,
                          my_weights = 1){
  
  n_k_fold = length(my_id_list_cv)
  my_n_metrics = length(my_metric_names)
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, length(my_lambda_vals), my_n_metrics))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    
    
    temp_glmnet = glmnet(x = my_x[id_train,], 
                         y = my_y[id_train], alpha = my_alpha,
                         lambda = my_lambda_vals)
    
    temp_predictions = predict(temp_glmnet, my_x[id_test,])
    
    for (j in 1:length(my_lambda_vals)){
      temp_metrics_array_cv[k,j,] = USED.Metrics(temp_predictions[,j], my_y[id_test],
                                                 weights = my_weights)
    }
    
    rm(temp_glmnet)
    rm(temp_predictions)
    gc()
    
    print(paste("fold ", k, collapse = ""))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = length(my_lambda_vals), ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = length(my_lambda_vals), ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}



#' GLMNET CV cycles in case of few data sample
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#'
#' @param my_x (matrix): complete model matrix passed to glmnet
#' @param my_y (vector): y glmnet argument
#' @param my_alpha (int): alpha passed to glmnet (0 -> ridge, 1 -> lasso)
#' @param my_lambda_vals (vector): vector of lambda used
#' 
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#' 
#' @param my_ncores (int): number of cores used for parallel computing
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"
ManualCvGlmnetParallel = function(my_id_list_cv,
                                  my_metric_names,
                                  my_x,
                                  my_y,
                                  my_alpha,
                                  my_lambda_vals,
                                  my_weights = 1,
                                  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss"),
                                  my_ncores = N_CORES){
  
  n_k_fold = length(my_id_list_cv)
  my_n_metrics = length(my_metric_names)
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, length(my_lambda_vals), my_n_metrics))
  
  # init parallel cluster
  sfInit(cpus = my_ncores, parallel = T)
  
  sfLibrary(glmnet)
  sfExport(list = c("my_x", "my_y", "my_alpha", "my_lambda_vals", "my_weights", my_metrics_functions))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    sfExport(list = c("id_train", "id_test"))
    
    # for better readability
    temp_metric = sfLapply(my_lambda_vals,
                           fun = function(lambda) 
                             USED.Metrics(predict(glmnet(x = my_x[id_train,], 
                                                         y = my_y[id_train], alpha = my_alpha,
                                                         lambda = lambda),
                                                  my_x[id_test,]), my_y[id_test]))
    
    # unlist to the right dimensions matrix
    temp_metrics_array_cv[k,,] = matrix(unlist(temp_metric), ncol = my_n_metrics, byrow = T)
  
    rm(temp_metric)
    gc()
    
    print(paste("fold ", k, collapse = ""))
  }
  
  # stop parallel cluster
  sfStop()
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = length(my_lambda_vals), ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = length(my_lambda_vals), ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PPR ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# K: numero di possibili funzioni dorsali
PPR_MAX_RIDGE_FUNCTIONS = 4

#' @param my_data (data.frame)
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_max_ridge_functions (vector of ints)
#' @param my_spline_df (vector in mums): values of possibile smoothing splines degrees of freedom
#' @param my_metrics_names (vector of chars)
#' @param my_weights (vector of nums):
#'  same length as the difference: NROW(my_data) - length(my_id_train)
#' 
#' @return (array):
#' first dimension (with names): 1:my_max_ridge_functions
#' second dimension (with names): my_spline_df
#' third dimension (with names): my_metrics_names
#' 
#' each cell contains the metric value of the model fitted on my_data[my_id_train,]
#' and tested on my_data[-my_id_train,] for each metric value used
PPRRegulationCV = function(my_data = sss,
                           my_id_list_cv,
                           my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                           my_spline_df = PPR_DF_SM,
                           my_metrics_names = METRICS_NAMES,
                           my_weights = MY_WEIGHTS){
  
  n_k_fold = length(my_id_list_cv)
  my_n_metrics = length(my_metrics_names)
  
  temp_metrics_array_cv = array(NA,
                        dim = c(my_max_ridge_functions,
                                length(my_spline_df),
                                length(my_metrics_names),
                                n_k_fold),
                        
                        dimnames = list(1:my_max_ridge_functions,
                                        my_spline_df,
                                        my_metrics_names,
                                        rep(NA, n_k_fold)))
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    for(r in 1:my_max_ridge_functions){
      for(df in 1:length(my_spline_df)){
        mod = ppr(y ~ .,
                  data = my_data[id_train,],
                  nterms = r,
                  sm.method = "spline",
                  df = my_spline_df[df])
        
        temp_metrics_array_cv[r, df, ,k] = USED.Metrics(predict(mod, my_data[-id_train,]),
                                              my_data$y[id_test],
                                              weights = MY_WEIGHTS)
      }
      print(paste0("n ridge functions: ", r, collapse = ""))
    }
    print(paste("fold = ", k, collapse = ""))
  }
  
  metrics_array = array(0,
                        dim = c(my_max_ridge_functions,
                                length(my_spline_df),
                                length(my_metrics_names)),
                        dimnames = list(1:my_max_ridge_functions,
                                        my_spline_df,
                                        my_metrics_names))
  
  # average over ridge functions
  for(k in 1:n_k_fold){
    metrics_array = metrics_array + temp_metrics_array_cv[, , ,k]
  }
  
  metrics_array = metrics_array / n_k_fold

  rm(mod)
  gc()
  
  return(metrics_array)
}

#' @param my_data (data.frame)
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_max_ridge_functions (vector of ints)
#' @param my_spline_df (vector in mums): values of possibile smoothing splines degrees of freedom
#' @param my_metrics_names (vector of chars)
#' @param my_weights (vector of nums):
#'  same length as the difference: NROW(my_data) - length(my_id_train)
#'
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#'
#' 
#' @return (array):
#' first dimension (with names): 1:my_max_ridge_functions
#' second dimension (with names): my_spline_df
#' third dimension (with names): my_metrics_names
#' 
#' each cell contains the metric value of the model fitted on my_data[my_id_train,]
#' and tested on my_data[-my_id_train,] for each metric value used
PPRRegulationCVParallel = function(my_data = sss,
                                   my_id_list_cv,
                                   my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                   my_spline_df = PPR_DF_SM,
                                   my_metrics_names = METRICS_NAMES,
                                   my_weights = MY_WEIGHTS,
                                   my_metrics_functions = MY_USED_METRICS,
                                   my_ncores = N_CORES){
  
  n_k_fold = length(my_id_list_cv)
  my_n_metrics = length(my_metrics_names)
  
  # needed to do parallel
  # each list element contains a vector of length 2
  # first element is the number of ridge functions
  # second element are the spline degrees of freedom
  params_list = list()
  
  counter = 1
  
  for (r in 1:my_max_ridge_functions){
    for(df in my_spline_df){
      params_list[[counter]] = c(r, df)
      
      counter = counter + 1
    }
  }
  
  # init parallel
  sfInit(cpus = my_ncores, parallel = T)
  
  sfExport(list = c("my_data", my_metrics_functions,
                    "my_max_ridge_functions", "my_spline_df", "params_list",
                    "my_weights"))
  
  temp_metrics_array_cv = array(NA,
                                dim = c(my_max_ridge_functions,
                                        length(my_spline_df),
                                        length(my_metrics_names),
                                        n_k_fold),
                                
                                dimnames = list(1:my_max_ridge_functions,
                                                my_spline_df,
                                                my_metrics_names,
                                                rep(NA, n_k_fold)))
  
  print("lol")
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    sfExport(list = c("id_train", "id_test"))
    
    temp_metric = sfLapply(params_list,
                           fun = function(el) 
                             USED.Metrics(predict(ppr(y ~ .,
                                                      data = my_data[id_train,],
                                                      nterms = el[1],
                                                      sm.method = "spline",
                                                      df = el[2]),
                                                  my_data[id_test,]), my_data$y[id_test],
                                          weights = my_weights))
    
    
    counter = 1
    
    print("lul")
    for (r in 1:my_max_ridge_functions){
      for(df in 1:length(my_spline_df)){
        temp_metrics_array_cv[r, df, ,k] = temp_metric[[counter]]
        
        counter = counter + 1
      }
    }
    
    print(paste("fold = ", k, collapse = ""))
  }
  
  print("lel")
  metrics_array = array(0,
                        dim = c(my_max_ridge_functions,
                                length(my_spline_df),
                                length(my_metrics_names)),
                        dimnames = list(1:my_max_ridge_functions,
                                        my_spline_df,
                                        my_metrics_names))
  
  # average over ridge functions
  for(k in 1:n_k_fold){
    metrics_array = metrics_array + temp_metrics_array_cv[, , ,k]
  }
  
  metrics_array = metrics_array / n_k_fold
  
  rm(mod)
  gc()
  
  return(metrics_array)
}


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Tree ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# TO DO: fix the y classification case -------------------

library(tree)

#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of loss functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_max_size (int): max size of the pruned tree
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
ManualCvTree = function(n_k_fold,
                        my_id_list_cv,my_n_metrics,
                        my_metric_names,
                        my_data,
                        my_max_size = TREE_MAX_SIZE,
                        my_weights = 1,
                        my_mindev = 1e-05,
                        my_minsize = 2){
  
  
  n_k_fold = length(my_id_list_cv)
  my_n_metrics = length(my_metric_names)
  
  # we use my_max_size - 1 because we start with size = 2
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, my_max_size - 1, my_n_metrics))
  
  fold_indexes = 1:n_k_fold
  
  for (k in fold_indexes) {
    
    # k-th fold set is used as validation set
    id_test = my_id_list_cv[[k]]
    
    # one set among those remaining is chosen as the pruning set
    # specifically we choose the one corresponding to the one with the minimum index 
    # among those not in the validation set
    
    id_pruning = min(fold_indexes[-k])
    
    # the remaining fold sets are used as training set
    id_train = unlist(my_id_list_cv[-c(k, id_pruning)])
    
    
    # full grown tree
    temp_tree_full = tree(y ~.,
                          data = my_data[id_train,],
                          control = tree.control(nobs = length(id_train),
                                                 mindev = my_mindev,
                                                 minsize = my_minsize))
    
    # it has to overfit
    plot(temp_tree_full)
    
    
    # if maximum tree depth error
    # change minsize = 2 to higher values and so do it with 
    # mindev
    
    # pruned tree: problem -> each fold can have different pruning inducing
    # split sizes whose CV error cannot be averaged
    # so I need to do it manually: select a set of size values
    # for each value prune the full tree on the id_train (sub-optimal and too optimistic)
    # (but given the scarsity of data we need a compromise)
    # and keep track of the reduced deviance on the id_test
    
    for (s in 2:my_max_size){
      temp_tree_pruned = prune.tree(temp_tree_full, best = s, newdata = my_data[id_pruning,])
      # prediction error
      # s-1 because we start by size = 2
      temp_metrics_array_cv[k,s-1,] = USED.Metrics(predict(temp_tree_pruned, my_data[id_test,]),
                                                   my_data$y[id_test], weights = my_weights)
      print(paste("tree size: ", s, collapse = ""))
    }
    
    
    rm(temp_tree_full)
    rm(temp_tree_pruned)
    gc()
    
    print(paste("fold ", k))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = my_max_size - 1, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = my_max_size - 1, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}

# parallelize inner cycle

#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of loss functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_max_size (int): max size of the pruned tree
#' 
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
ManualCvTreeParallel = function(my_id_list_cv,
                                my_metric_names,
                                my_data,
                                my_max_size = TREE_MAX_SIZE,
                                my_metrics_functions = MY_USED_METRICS,
                                my_ncores = N_CORES,
                                my_weights = 1,
                                my_mindev = 1e-05,
                                my_minsize = 2){
  
  n_k_fold = length(my_id_list_cv)
  my_n_metrics = length(my_metric_names)
  
  
  # init parallel
  sfInit(cpus = my_ncores, parallel = T)
  
  sfLibrary(tree)
  sfExport(list = c("my_data", my_metrics_functions))
  
  
  # we use my_max_size - 1 because we start with size = 2
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, my_max_size - 1, my_n_metrics))
  
  fold_indexes = 1:n_k_fold
  
  for (k in fold_indexes){
    
    # k-th fold set is used as validation set
    id_test = my_id_list_cv[[k]]
    
    # one set among those remaining is chosen as the pruning set
    # specifically we choose the one corresponding to the one with the minimum index 
    # among those not in the validation set
    
    id_pruning = min(fold_indexes[-k])
    
    # the remaining fold sets are used as training set
    id_train = unlist(my_id_list_cv[-c(k, id_pruning)])
    
    # full grown tree
    temp_tree_full = tree(y ~.,
                          data = my_data[id_train,],
                          control = tree.control(nobs = length(id_train),
                                                 mindev = my_mindev,
                                                 minsize = my_minsize))
    
    # it has to overfit
    plot(temp_tree_full)
    
    sfExport(list = c("temp_tree_full", "id_train", "id_test", "id_pruning",
                      "my_max_size", "my_weights"))
    # if maximum tree depth error
    # change minsize = 2 to higher values and so do it with 
    # mindev
    
    # pruned tree: problem -> each fold can have different pruning inducing
    # split sizes whose CV error cannot be averaged
    # so I need to do it manually: select a set of size values
    # for each value prune the full tree on the id_train (sub-optimal and too optimistic)
    # (but given the scarsity of data we need a compromise)
    # and keep track of the reduced deviance on the id_test
    
    # for better readability
    temp_metric = sfLapply(2:my_max_size,
                           fun = function(s) 
                             USED.Metrics(predict(prune.tree(temp_tree_full, best = s,
                                                             newdata = my_data[id_pruning,]),
                                                  my_data[id_test,]), my_data$y[id_test],
                                          weights = my_weights))
    
    # unlist to the right dimensions matrix
    temp_metrics_array_cv[k,,] = matrix(unlist(temp_metric), ncol = my_n_metrics, byrow = T)
    
    
    rm(temp_tree_full)
    rm(temp_metric)
    gc()
    
    print(paste("fold ", k))
  }
  
  # stop parallel cluster
  sfStop()
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = my_max_size - 1, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = my_max_size - 1, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
  return(cv_metrics)
  
  
}


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# TO DO: fix the y classification case -------------------


# max variables at each split (can be changed)
RF_MAX_VARIABLES = 30

# number of bootstrap trees (can be changed)
RF_N_BS_TREES = 200

library(randomForest)

# function to choose the optimal number of variables at each split
# or alternatively (based on fix_tress_bool parameter) check convergence of validation error
# with respect to number of tress for a fixed my_max_variables


#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_n_variables: (int) or (vector of int) number of variables chosen at each split
#' @param my_n_bs_trees: (int) or (vector of int) number of bootstrap trees 
#' @param fix_tress_bool (bool): TRUE if fixed number of bootstrap trees, number of variables changes,
#' else FALSE
#' 
#' @description this function can be used (separately, not simultaneuosly) for two parameters check
#' 1) if fix_tress_bool == TRUE -> my_n_bs_trees is fixed at its maximum if not already an integer
#' and the procedure compute the CV error for varying number of variables at each split
#' according to the vector my_n_variables (supposed to be a sequence)
#' 2) if fix_tress_bool == FALSE -> my_n_variables is fixed at its maximum if not already an integer,
#' but a warning is given, because it should be just an integer, not a vector.
#' the procedure compute the CV error for varying number of bootstrap trees
#' according to the the vector my_n_bs_trees (supposed to be a sequence)
#'  
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
ManualCvRF = function(my_id_list_cv,
                      my_metric_names,
                      my_data,
                      my_n_variables = 1:RF_MAX_VARIABLES,
                      my_n_bs_trees = 10:RF_N_BS_TREES,
                      fix_trees_bool = TRUE,
                      my_weights = 1){
  
  n_k_fold = length(my_id_list_cv)
  my_n_metrics = length(my_metric_names)
  
  # fixed number of bootstrap trees, number of variables changes
  if(fix_trees_bool == TRUE){
    tuning_parameter_length = length(my_n_variables)
    
    # fix the number of trees to max if my_n_bs_trees is not already an int
    my_n_bs_trees = max(my_n_bs_trees)
  }
  # fixed number of number of variables, number of bootstrap tress changes
  else{
    tuning_parameter_length = length(my_n_bs_trees)
    
    # check warning
    if(length(my_n_variables) > 1){
      print("Warning: my_n_variables should be an integer, not a sequence of numbers, maximum is taken")
    }
    
    my_n_variables = max(my_n_variables)
    
  }
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, tuning_parameter_length, my_n_metrics))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    # it's ugly I know and a bad pratice but I'll do two separate loop based on if condition
    
    
    if(fix_trees_bool == TRUE){
      for (m in 1:tuning_parameter_length){
        temp_rf = randomForest(y ~., data = my_data[id_train,],
                               mtry = my_n_variables[m], ntree = my_n_bs_trees)
        # prediction error
        temp_metrics_array_cv[k,m,] = USED.Metrics(predict(temp_rf, my_data[id_test,]),
                                                   my_data$y[id_test], weights = my_weights)
        print(paste(c("n var = ", m), collapse = ""))
      }
    }
    
    else{
      for (t in 1:tuning_parameter_length){
        temp_rf = randomForest(y ~., data = my_data[id_train,],
                               mtry = my_n_variables, ntree = my_n_bs_trees[t])
        # prediction error
        temp_metrics_array_cv[k,t,] = USED.Metrics(predict(temp_rf, my_data[id_test,]),
                                                   my_data$y[id_test], weights = my_weights)
        print(paste(c("n trees = ", my_n_bs_trees[t]), collapse = ""))
      }
    }
    
    
    rm(temp_rf)
    gc()
    
    print(paste("fold ", k))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}


# function to choose the optimal number of variables at each split
# or alternatively (based on fix_tress_bool parameter) check convergence of validation error
# with respect to number of tress for a fixed my_max_variables


#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#'
#' @param my_n_variables: (int) or (vector of int) number of variables chosen at each split
#' @param my_n_bs_trees: (int) or (vector of int) number of bootstrap trees 
#' @param fix_tress_bool (bool): TRUE if fixed number of bootstrap trees, number of variables changes,
#' else FALSE
#' 
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default

#' 
#' @description this function can be used (separately, not simultaneuosly) for two parameters check
#' 1) if fix_tress_bool == TRUE -> my_n_bs_trees is fixed at its maximum if not already an integer
#' and the procedure compute the CV error for varying number of variables at each split
#' according to the vector my_n_variables (supposed to be a sequence)
#' 2) if fix_tress_bool == FALSE -> my_n_variables is fixed at its maximum if not already an integer,
#' but a warning is given, because it should be just an integer, not a vector.
#' the procedure compute the CV error for varying number of bootstrap trees
#' according to the the vector my_n_bs_trees (supposed to be a sequence)
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
ManualCvRFParallel = function(my_id_list_cv,
                              my_metric_names,
                              my_data,
                              my_n_variables = 1:RF_MAX_VARIABLES,
                              my_n_bs_trees = 10:RF_N_BS_TREES,
                              fix_trees_bool = TRUE,
                              my_metrics_functions = MY_USED_METRICS,
                              my_weights = MY_WEIGHTS,
                              my_ncores = N_CORES){
  
  n_k_fold = length(my_id_list_cv)
  my_n_metrics = length(my_metric_names)
  
  # init parallel
  sfInit(cpus = my_ncores, parallel = T)
  
  sfLibrary(randomForest)
  
  # fixed number of bootstrap trees, number of variables changes
  if(fix_trees_bool == TRUE){
    tuning_parameter_length = length(my_n_variables)
    
    # fix the number of trees to max if my_n_bs_trees is not already an int
    my_n_bs_trees = max(my_n_bs_trees)
  }
  # fixed number of number of variables, number of bootstrap tress changes
  else{
    tuning_parameter_length = length(my_n_bs_trees)
    
    # check warning
    if(length(my_n_variables) > 1){
      print("Warning: my_n_variables should be an integer, not a sequence of numbers, maximum is taken")
    }
    
    my_n_variables = max(my_n_variables)
    
  }
  
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, tuning_parameter_length, my_n_metrics))
  
  # allocate relevant variables in cluster
  sfExport(list = c("my_data", my_metrics_functions,
                    "my_n_bs_trees", "tuning_parameter_length", "my_n_variables", "my_weights"))
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv[-k])
    id_test = my_id_list_cv[[k]]
    
    sfExport(list = c("id_train", "id_test"))
    
    # it's ugly I know and a bad pratice but I'll do two separate loop based on if condition
    
    if(fix_trees_bool == TRUE){
      # for better readability
      temp_metric = sfLapply(1:tuning_parameter_length,
                             fun = function(m) 
                               USED.Metrics(predict(randomForest(y ~., data = my_data[id_train,],
                                                                 mtry = my_n_variables[m], ntree = my_n_bs_trees),
                                                    my_data[id_test,]), my_data$y[id_test],
                                            weights = my_weights))
      
      # unlist to the right dimensions matrix
      temp_metrics_array_cv[k,my_n_variables,] = matrix(unlist(temp_metric), ncol = my_n_metrics, byrow = T)
      
      rm(temp_metric)
      gc()
      
    }
    
    else{
      
      # for better readability
      temp_metric = sfLapply(1:tuning_parameter_length,
                             fun = function(t) 
                               USED.Metrics(predict(randomForest(y ~., data = my_data[id_train,],
                                                                 mtry = my_n_variables, ntree = my_n_bs_trees[t]),
                                                    my_data[id_test,]), my_data$y[id_test],
                                            weights = my_weights))
      
      # unlist to the right dimensions matrix
      temp_metrics_array_cv[k,1:tuning_parameter_length,] = matrix(unlist(temp_metric), ncol = my_n_metrics, byrow = T)
      
      rm(temp_metric)
      gc()
    }
    
    gc()
    
    print(paste("fold ", k))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
    cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}


