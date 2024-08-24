# Cross validation functions
library(dplyr)

# parallel
library(snowfall)
# 
N_CORES = parallel::detectCores()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Linear Model & GLM ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#' @param my_id_list_cv_train (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#'
#' @param my_data (data.frame)
#' @param my_formula (model formula)
#' @param my_model_type (char): lm, glm, gam, polymars
#' 
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_id_list_cv_test (list): indexes of fold used to test
#' # if NULL my_id_list_cv_test = my_id_list_cv, and classical CV is performed
#' # if is not null it has to be the same number of elements as my_id_list_cv
#' # Can be used for example for unbalanced datasets:
#' my_id_list_cv -> balanced folds use to estimation
#' my_id_list_cv_test -> unbalanced folds used for testing
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"


ManualCvLmGlmGam = function(my_id_list_cv_train,
                          my_metric_names,
                          my_data,
                          my_formula = NULL,
                          my_data_matrix = NULL,
                          my_model_type,
                          my_weights = 1,
                          use_only_first_fold = FALSE,
                          is_classification = FALSE,
                          my_threshold = 0.5,
                          my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  temp_metrics_array_cv = matrix(NA, nrow = n_k_fold, ncol = my_n_metrics)
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    if(my_model_type == "lm"){
      temp_model= lm(formula = my_formula, data = my_data[id_train,])
    
      temp_predictions = predict(temp_model, my_data[id_test,])
    
    }
    
    if(my_model_type == "glm"){
      temp_model= glm(formula = my_formula,
                      family = "binomial",
                      data = my_data[id_train,])
      
      temp_predictions = predict(temp_model, my_data[id_test,],
                                 type = "response")
    }
    
    if(my_model_type == "gam"){
      
      if(is_classification){
        temp_model= gam(formula = my_formula,
                        family = "binomial",
                        data = my_data[id_train,])
        
        temp_predictions = predict(temp_model, my_data[id_test,],
                                   type = "response")
      }
      
      else{
        temp_model= gam(formula = my_formula,
                        data = my_data[id_train,])
        
        temp_predictions = predict(temp_model, my_data[id_test,],
                                   type = "response")
      }
      
    }
    
    
    
    if(is_classification){
      temp_predictions = temp_predictions > my_threshold %>% as.numeric
    }
      
      temp_metrics_array_cv[k,] = USED.Metrics(temp_predictions,
                                               my_data$y[id_test],
                                               weights = my_weights[id_test])

    
    rm(temp_model)
    rm(temp_predictions)
    gc()
    
    print(paste("fold ", k, collapse = ""))
    
  }
  
  # averaged metrics matrix
  cv_metrics = apply(temp_metrics_array_cv, 2, mean)
  
  return(cv_metrics)
  
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# POLYMARS ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#' @param my_id_list_cv_train (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#'
#' @param my_y (data.frame)
#' @param my_design_matrix (model formula)
#' 
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_id_list_cv_test (list): indexes of fold used to test
#' # if NULL my_id_list_cv_test = my_id_list_cv, and classical CV is performed
#' # if is not null it has to be the same number of elements as my_id_list_cv
#' # Can be used for example for unbalanced datasets:
#' my_id_list_cv -> balanced folds use to estimation
#' my_id_list_cv_test -> unbalanced folds used for testing
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"


ManualCvMARS = function(my_id_list_cv_train,
                            my_metric_names,
                            my_y,
                            my_design_matrix,
                            my_weights = 1,
                            use_only_first_fold = FALSE,
                            is_classification = FALSE,
                            my_threshold = 0.5,
                            my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  temp_metrics_array_cv = matrix(NA, nrow = n_k_fold, ncol = my_n_metrics)
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    W = solve(t(my_design_matrix[id_train,]) %*% my_design_matrix[id_train,])
    
    temp_predictions = my_design_matrix[id_test,] %*%
      W %*%
      t(my_design_matrix[id_train,]) %*%
      as.matrix(my_y[id_train])

     
     
     
    if(is_classification == TRUE){
      temp_predictions = temp_predictions > my_threshold
      
    }
    
    temp_metrics_array_cv[k,] = USED.Metrics(temp_predictions,
                                             my_y[id_test],
                                             weights = my_weights[id_test])
    
    
    rm(temp_predictions)
    gc()
    
    print(paste("fold ", k, collapse = ""))
    
  }
  
  # averaged metrics matrix
  cv_metrics = apply(temp_metrics_array_cv, 2, mean)
  
  return(cv_metrics)
  
}


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Ridge and Lasso ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 library(glmnet)

#' GLMNET CV cycles in case of few data sample
#' @param my_id_list_cv_train (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#'
#' @param my_x (matrix): complete model matrix passed to glmnet
#' @param my_y (vector): y glmnet argument
#' @param my_alpha (int): alpha passed to glmnet (0 -> ridge, 1 -> lasso)
#' @param my_lambda_vals (vector): vector of lambda used
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_id_list_cv_test (list): indexes of fold used to test
#' # if NULL my_id_list_cv_test = my_id_list_cv, and classical CV is performed
#' # if is not null it has to be the same number of elements as my_id_list_cv
#' # Can be used for example for unbalanced datasets:
#' my_id_list_cv -> balanced folds use to estimation
#' my_id_list_cv_test -> unbalanced folds used for testing
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"
ManualCvGlmnet = function(my_id_list_cv_train,
                          my_metric_names,
                          my_x,
                          my_y,
                          my_alpha,
                          my_lambda_vals,
                          my_weights = 1,
                          use_only_first_fold = FALSE,
                          is_classification = FALSE,
                          my_threshold = 0.5,
                          my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, length(my_lambda_vals), my_n_metrics))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    temp_glmnet = glmnet(x = my_x[id_train,], 
                         y = my_y[id_train], alpha = my_alpha,
                         lambda = my_lambda_vals)
    
    temp_predictions = predict(temp_glmnet, my_x[id_test,])
    
    if(is_classification == TRUE){
      temp_predictions = temp_predictions > my_threshold %>% as.numeric
    }
    
    
    
    for (j in 1:length(my_lambda_vals)){
      
      temp_metrics_array_cv[k,j,] = USED.Metrics(temp_predictions[,j], my_y[id_test],
                                                 weights = my_weights[id_test])
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
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    else{
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
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
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_id_list_cv_test (list): indexes of fold used to test
#' # if NULL my_id_list_cv_test = my_id_list_cv, and classical CV is performed
#' # if is not null it has to be the same number of elements as my_id_list_cv
#' # Can be used for example for unbalanced datasets:
#' my_id_list_cv -> balanced folds use to estimation
#' my_id_list_cv_test -> unbalanced folds used for testing
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#' 
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#'
#' @param my_ncores (int): number of cores used for parallel computing
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"
ManualCvGlmnetParallel = function(my_id_list_cv_train,
                                  my_metric_names,
                                  my_x,
                                  my_y,
                                  my_alpha,
                                  my_lambda_vals,
                                  my_weights = 1,
                                  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss"),
                                  my_ncores = N_CORES,
                                  use_only_first_fold = FALSE,
                                  is_classification = FALSE,
                                  my_threshold = 0.5,
                                  my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  temp_metrics_array_cv = array(0, dim = c(n_k_fold, length(my_lambda_vals), my_n_metrics))
  
  # init parallel cluster
  sfInit(cpus = my_ncores, parallel = T)
  
  sfLibrary(glmnet)
  sfExport(list = c("my_x", "my_y", "my_alpha",
                    "my_lambda_vals", "my_weights", my_metrics_functions,
                    "my_threshold"))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    #' @param lambda (num): lambda value
    #' @return (vector of nums): metrics values
    ParallelFunction = function(lambda){
      
      temp_predictions = predict(glmnet(x = my_x[id_train,], 
                                        y = my_y[id_train], alpha = my_alpha,
                                        lambda = lambda),
                                 my_x[id_test,])
      
      if(is_classification == TRUE){
        temp_predictions = as.numeric(temp_predictions > my_threshold)
      }
      
      return(USED.Metrics(y.pred = temp_predictions,
                          y.test = my_y[id_test],
                          weights = my_weights[id_test]))
    }
    
    sfExport(list = c("id_train", "id_test", "ParallelFunction"))
    
    temp_metrics = sfLapply(my_lambda_vals,
                            fun = ParallelFunction)
    
    # unlist to the right dimensions matrix
    temp_metrics_array_cv[k,,] = matrix(unlist(temp_metrics),
                                        nrow = length(lambda_vals),
                                        ncol = my_n_metrics,
                                        byrow = T)
    
    rm(temp_metrics)
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
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    else{
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PPR ---------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# 1.a) Regulation: train - test ---------

#' @param my_data (data.frame)
#' @param my_id_train (vector of ints)
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
PPRRegulationTrainTest = function(my_data = sss,
                                  my_id_train = id_cb1,
                                  my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                  my_spline_df = PPR_DF_SM,
                                  my_metrics_names = METRICS_NAMES,
                                  my_weights = MY_WEIGHTS_sss){
  metrics_array = array(NA,
                        dim = c(my_max_ridge_functions,
                                length(my_spline_df),
                                length(my_metrics_names)),
                        
                        dimnames = list(1:my_max_ridge_functions,
                                        my_spline_df,
                                        my_metrics_names))
  
  for(r in 1:my_max_ridge_functions){
    for(df in 1: length(my_spline_df)){
      mod = ppr(y ~ .,
                data = my_data[my_id_train,],
                nterms = r,
                sm.method = "spline",
                df = my_spline_df[df])
      
      metrics_array[r, df, ] = USED.Metrics(predict(mod, my_data[-my_id_train,]),
                                            my_data$y[-my_id_train],
                                            weights = my_weights)
    }
    print(paste0("n ridge functions: ", r, collapse = ""))
  }
  
  rm(mod)
  gc()
  
  
  return(metrics_array)
}



#' @param my_data (data.frame)
#' @param my_id_train (vector of ints)
#' @param my_max_ridge_functions (vector of ints)
#' @param my_spline_df (vector in mums): values of possibile smoothing splines degrees of freedom
#' @param my_metrics_names (vector of chars)
#' @param my_weights (vector of nums):
#'  same length as the difference: NROW(my_data) - length(my_id_train)
#'  
#'  
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#' @param my_ncores
#' 
#' @return (array):
#' first dimension (with names): 1:my_max_ridge_functions
#' second dimension (with names): my_spline_df
#' third dimension (with names): my_metrics_names
#' 
#' each cell contains the metric value of the model fitted on my_data[my_id_train,]
#' and tested on my_data[-my_id_train,] for each metric value used
PPRRegulationTrainTestParallel = function(my_data = sss,
                                          my_id_train = id_cb1,
                                          my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                          my_spline_df = PPR_DF_SM,
                                          my_metrics_names = METRICS_NAMES,
                                          my_weights = MY_WEIGHTS_sss,
                                          my_metrics_functions = MY_USED_METRICS,
                                          my_ncores = N_CORES){
  
  metrics_array = array(NA,
                        dim = c(my_max_ridge_functions,
                                length(my_spline_df),
                                length(my_metrics_names)),
                        
                        dimnames = list(1:my_max_ridge_functions,
                                        my_spline_df,
                                        my_metrics_names))
  
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
                    "my_id_train", "my_max_ridge_functions", "my_spline_df", "params_list",
                    "my_weights"))
  
  temp_metric = sfLapply(params_list,
                         fun = function(el) 
                           USED.Metrics(predict(ppr(y ~ .,
                                                    data = my_data[my_id_train,],
                                                    nterms = el[1],
                                                    sm.method = "spline",
                                                    df = el[2]),
                                                my_data[-my_id_train,]), my_data$y[-my_id_train],
                                        weights = my_weights))
  
  # stop cluster
  sfStop()
  
  
  counter = 1
  
  for (r in 1:my_max_ridge_functions){
    for(df in 1:length(my_spline_df)){
      metrics_array[r, df, ] = temp_metric[[counter]]
      
      counter = counter + 1
    }
  }
  
  rm(temp_metric)
  gc()
  
  return(metrics_array)
}


# K: numero di possibili funzioni dorsali
PPR_MAX_RIDGE_FUNCTIONS = 4

#' @param my_data (data.frame)
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_max_ridge_functions (vector of ints)
#' @param my_spline_df (vector in mums): values of possibile smoothing splines degrees of freedom
#' @param my_metrics_names (vector of chars)
#' @param my_weights (vector of nums):
#'  same length as the difference: NROW(my_data) - length(my_id_train)
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#' 
#' @return (array):
#' first dimension (with names): 1:my_max_ridge_functions
#' second dimension (with names): my_spline_df
#' third dimension (with names): my_metrics_names
#' 
#' each cell contains the metric value of the model fitted on my_data[my_id_train,]
#' and tested on my_data[-my_id_train,] for each metric value used
PPRRegulationCV = function(my_data = sss,
                           my_id_list_cv_train,
                           my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                           my_spline_df = PPR_DF_SM,
                           my_metrics_names = METRICS_NAMES,
                           my_weights = MY_WEIGHTS,
                           use_only_first_fold = FALSE,
                           is_classification = FALSE,
                           my_threshold = 0.5,
                           my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metrics_names)
  
  
  # first create 
  temp_metrics_array_cv = array(0,
                        dim = c(my_max_ridge_functions,
                                length(my_spline_df),
                                length(my_metrics_names),
                                n_k_fold),
                        
                        dimnames = list(1:my_max_ridge_functions,
                                        my_spline_df,
                                        my_metrics_names,
                                        rep(NA, n_k_fold)))
  
  # do after array creation, inefficient but should work
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    for(r in 1:my_max_ridge_functions){
      for(df in 1:length(my_spline_df)){
        
        mod = ppr(y ~ .,
                  data = my_data[id_train,],
                  nterms = r,
                  sm.method = "spline",
                  df = my_spline_df[df])
        
        temp_predictions = predict(mod, my_data[-id_train,])
        
        if(is_classification == TRUE){
          temp_predictions = temp_predictions > my_threshold
        }
        
        rm(mod)
        gc()
        
        temp_metrics_array_cv[r, df, ,k] = USED.Metrics(temp_predictions,
                                              my_data$y[id_test],
                                              weights = my_weights[id_test])
        rm(temp_predictions)
        gc()
      }
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
  
  return(metrics_array)
}

#' @param my_data (data.frame)
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_max_ridge_functions (vector of ints)
#' @param my_spline_df (vector in mums): values of possibile smoothing splines degrees of freedom
#' @param my_metrics_names (vector of chars)
#' @param my_weights (vector of nums):
#'  same length as the difference: NROW(my_data) - length(my_id_train)
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#'
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#' 
#' @return (array):
#' first dimension (with names): 1:my_max_ridge_functions
#' second dimension (with names): my_spline_df
#' third dimension (with names): my_metrics_names
#' 
#' each cell contains the metric value of the model fitted on my_data[my_id_train,]
#' and tested on my_data[-my_id_train,] for each metric value used
PPRRegulationCVParallel = function(my_data = sss,
                                   my_id_list_cv_train,
                                   my_max_ridge_functions = PPR_MAX_RIDGE_FUNCTIONS,
                                   my_spline_df = PPR_DF_SM,
                                   my_metrics_names = METRICS_NAMES,
                                   my_weights = MY_WEIGHTS,
                                   my_metrics_functions = MY_USED_METRICS,
                                   my_ncores = N_CORES,
                                   use_only_first_fold = FALSE,
                                   is_classification = FALSE,
                                   my_threshold = 0.5,
                                   my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
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
                    "my_weights", "is_classification", "my_threshold"))
  
  temp_metrics_array_cv = array(0,
                                dim = c(my_max_ridge_functions,
                                        length(my_spline_df),
                                        length(my_metrics_names),
                                        n_k_fold),
                                dimnames = list(1:my_max_ridge_functions,
                                                my_spline_df,
                                                my_metrics_names,
                                                1:n_k_fold))
  
  # do after array creation: inefficient but should work
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }

  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    
    sfExport(list = c("id_train", "id_test"))
    
    #' @params (vector of nums): 
    #' first element: number of ridge functions
    #' second element number of spline degrees of freedom
    ParallelFunction = function(params){
      temp_predictions = predict(ppr(y ~ .,
                       data = my_data[id_train,],
                       nterms = params[1],
                       sm.method = "spline",
                       df = params[2]),
                       my_data[id_test,])
      
      if(is_classification == TRUE){
        temp_predictions = temp_predictions > my_threshold
      }
      
      return(temp_predictions)
    }
    
    sfExport(list = c("ParallelFunction"))
    
    temp_predictions = sfLapply(params_list,
                                fun = ParallelFunction)

    counter = 1
    
    for (r in 1:my_max_ridge_functions){
      for(df in 1:length(my_spline_df)){
        temp_metrics_array_cv[r, df, ,k] = USED.Metrics(temp_predictions[[counter]],
                                                        my_data$y[id_test],
                                                        weights = my_weights[id_test])
        
        counter = counter + 1
        
      }
    }
    
    print(paste("fold = ", k, collapse = ""))
  }
  
  # stop cluster
  sfStop()
  
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
  
  gc()
  
  return(metrics_array)
}



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Tree ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# TO DO: check the multiclass case  -------------------

library(tree)

#' @param n_k_fold (int): number of fold used, use the global variable
#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_n_metrics (int): number of loss functions used, use global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#' 
#' @param is_multiclass (bool): multiclass classification (to be checked)
#' 
#' @param my_max_size (int): max size of the pruned tree
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
ManualCvTree = function(my_id_list_cv_train,
                        my_metric_names,
                        my_data,
                        my_max_size = TREE_MAX_SIZE,
                        my_weights = 1,
                        my_mindev = 1e-05,
                        my_minsize = 2,
                        use_only_first_fold = FALSE,
                        is_classification = FALSE,
                        my_threshold = 0.5,
                        is_multiclass = FALSE,
                        my_id_list_cv_test = NULL){
  
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  
  all_fold_indexes = 1:n_k_fold
  
  # this are the indexed of the folds used as test set
  # default all are used once as test set 
  fold_used_as_test_indexes = 1:length(my_id_list_cv_train)
  
  # if only the first fold is used a test set
  if(use_only_first_fold == TRUE){
    fold_used_as_test_indexes = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  # we use my_max_size - 1 because we start with size = 2
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, my_max_size - 1, my_n_metrics))
  
  
  for (k in fold_used_as_test_indexes) {
    
    # k-th fold set is used as validation set
    id_test = my_id_list_cv_test[[k]]
    
    # one set among those remaining is chosen as the pruning set
    # specifically we choose the one corresponding to the one with the minimum index 
    # among those not in the validation set
    
    # if the test set is the first index choose the last set a pruning set
    # else choose the test index set - 1 as pruning set
    
    if(k == 1){
      index_id_pruning = n_k_fold
      
    }
    
    if(k > 1){
      index_id_pruning = k - 1
    }
    
    id_pruning = my_id_list_cv_train[[index_id_pruning]]
    
    # the remaining fold sets are used as training set
    id_train = unlist(my_id_list_cv_train[-c(k, index_id_pruning)])
    
    
    # full grown tree
    temp_tree_full = tree(y ~.,
                          data = my_data[id_train,],
                          control = tree.control(nobs = length(id_train),
                                                 mindev = my_mindev,
                                                 minsize = my_minsize))
    
    if((is_classification == TRUE) | (is_multiclass == TRUE)){
      temp_tree_full = tree(factor(y) ~.,
                            data = my_data[id_train,],
                            control = tree.control(nobs = length(id_train),
                                                   mindev = my_mindev,
                                                   minsize = my_minsize))
    }
    
    # it has to overfit
    plot(temp_tree_full)
    
    
    # if maximum tree depth error
    # change minsize = 2 to higher values and so do it with 
    # mindev
    
    for (s in 2:my_max_size){
      temp_tree_pruned = prune.tree(temp_tree_full, best = s, newdata = my_data[id_pruning,])
      
      if(is_multiclass == TRUE){
        temp_predictions = predict(temp_tree_pruned,
                                   newdata = my_data[id_test,],
                                   type = "class")
      }
      
      if((is_classification == TRUE) & (is_multiclass == FALSE)){
        # tree gives probabilities for each class
        temp_predictions = predict(temp_tree_pruned,
                                   newdata = my_data[id_test,],
                                   type = "vector")
        
        positive_index = which(colnames(temp_predictions) == 1)
        temp_predictions = temp_predictions[,positive_index] > my_threshold
      }
      
      # default
      if((is_classification == FALSE) & (is_multiclass == FALSE)){
        temp_predictions = predict(temp_tree_pruned,
                                   newdata = my_data[id_test,],
                                   type = "vector")
      }
  
      
      # s-1 because we start by size = 2
      temp_metrics_array_cv[k,s-1,] = USED.Metrics(y.pred = temp_predictions,
                                                   y.test = my_data$y[id_test],
                                                   weights = my_weights[id_test])
    }
    
    
    rm(temp_tree_full)
    rm(temp_tree_pruned)
    rm(temp_predictions)
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
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    if(use_only_first_fold == TRUE){
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}

# parallelize inner cycle




#' @param my_id_list_cv (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data.frame used
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#' 
#' @param is_multiclass (bool): multiclass classification (to be checked)
#' 
#' @param my_max_size (int): max size of the pruned tree

#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#'
#' @return matrix of CV folds averaged errors for each parameter value and each loss function 
ManualCvTreeParallel = function(my_id_list_cv_train,
                        my_metric_names,
                        my_data,
                        my_max_size = TREE_MAX_SIZE,
                        my_weights = 1,
                        my_metrics_functions = MY_USED_METRICS,
                        my_ncores = N_CORES,
                        my_mindev = 1e-05,
                        my_minsize = 2,
                        use_only_first_fold = FALSE,
                        is_classification = FALSE,
                        my_threshold = 0.5,
                        is_multiclass = FALSE,
                        my_id_list_cv_test = NULL){
  
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  
  all_fold_indexes = 1:n_k_fold
  
  # this are the indexed of the folds used as test set
  # default all are used once as test set 
  fold_used_as_test_indexes = 1:length(my_id_list_cv_train)
  
  # if only the first fold is used a test set
  if(use_only_first_fold == TRUE){
    fold_used_as_test_indexes = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  # init parallel
  sfInit(cpus = my_ncores, parallel = T)
  
  sfLibrary(tree)
  sfExport(list = c("my_data", my_metrics_functions,
                    "my_weights", "my_max_size",
                    "is_classification","my_threshold",
                    "is_multiclass", "my_mindev", "my_minsize"))
  
  # we use my_max_size - 1 because we start with size = 2
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, my_max_size - 1, my_n_metrics))
  
  
  for (k in fold_used_as_test_indexes) {
    
    # k-th fold set is used as validation set
    id_test = my_id_list_cv_test[[k]]
    
    # one set among those remaining is chosen as the pruning set
    # specifically we choose the one corresponding to the one with the minimum index 
    # among those not in the validation set
    
    # if the test set is the first index choose the last set a pruning set
    # else choose the test index set - 1 as pruning set
    
    if(k == 1){
      index_id_pruning = n_k_fold
      
    }
    
    if(k > 1){
      index_id_pruning = k - 1
    }
    
    id_pruning = my_id_list_cv_train[[index_id_pruning]]
    
    # the remaining fold sets are used as training set
    id_train = unlist(my_id_list_cv_train[-c(k, index_id_pruning)])
    
    
    # full grown tree
    temp_tree_full = tree(y ~.,
                          data = my_data[id_train,],
                          control = tree.control(nobs = length(id_train),
                                                 mindev = my_mindev,
                                                 minsize = my_minsize))
    
    if(is_classification == TRUE){
      temp_tree_full = tree(factor(y) ~.,
                            data = my_data[id_train,],
                            control = tree.control(nobs = length(id_train),
                                                   mindev = my_mindev,
                                                   minsize = my_minsize))
    }
    
    # it has to overfit
    plot(temp_tree_full)
    
    #' @param size (num): size values
    #' @return (vector of nums): metrics values
    ParallelFunction = function(size){
      
      temp_tree_pruned = prune.tree(temp_tree_full,
                                    best = size,
                                    newdata = my_data[id_pruning,])
      
      
      if(is_multiclass == TRUE){
        temp_predictions = predict(temp_tree_pruned,
                                   newdata = my_data[id_test,],
                                   type = "class")
      }
      
      if((is_classification == TRUE) & (is_multiclass == FALSE)){
        # tree gives probabilities for each class
        temp_predictions = predict(temp_tree_pruned,
                                   newdata = my_data[id_test,],
                                   type = "vector")
        
        #debug
        print(temp_predictions)
        
        positive_index = which(colnames(temp_predictions) == 1)
        temp_predictions = temp_predictions[,positive_index] > my_threshold
      }
      
      # default
      if((is_classification == FALSE) & (is_multiclass == FALSE)){
        temp_predictions = predict(temp_tree_pruned,
                                   newdata = my_data[id_test,],
                                   type = "vector")
      }
      
      
      
      
      return(USED.Metrics(y.pred = temp_predictions,
                          y.test = my_data$y[id_test],
                          weights = my_weights[id_test]))
    }
    
    sfExport(list = c("temp_tree_full",
                      "id_train", "id_test", "id_pruning",
                      "ParallelFunction"))
    
    temp_metrics = sfLapply(2:my_max_size,
                            fun = ParallelFunction)
    
    # unlist to the right dimensions matrix
    temp_metrics_array_cv[k,,] = matrix(unlist(temp_metrics),
                                        nrow = my_max_size - 1,
                                        ncol = my_n_metrics,
                                        byrow = T)
    
    rm(temp_metrics)
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
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    if(use_only_first_fold == TRUE){
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Random Forest ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# TO DO: check the multiclass case  -------------------


# max variables at each split (can be changed)
RF_MAX_VARIABLES = 30

# number of bootstrap trees (can be changed)
RF_N_BS_TREES = 200

library(ranger)

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
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
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
ManualCvRF = function(my_id_list_cv_train,
                      my_metric_names,
                      my_data,
                      my_n_variables = 1:RF_MAX_VARIABLES,
                      my_n_bs_trees = 10:RF_N_BS_TREES,
                      fix_trees_bool = TRUE,
                      my_weights = 1,
                      use_only_first_fold = FALSE,
                      is_classification = FALSE,
                      my_threshold = 0.5,
                      is_multiclass = FALSE,
                      my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
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
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    # it's ugly I know and a bad practice but I'll do two separate loop based on if condition
    
    PredictFunction = function(my_pred_mtry, my_pred_n_trees){
      
      # predict class (greatest prob.)
      if(is_multiclass == TRUE){
        model = ranger(factor(y) ~.,
                             data = my_data[id_train,],
                             mtry = my_pred_mtry,
                             num.trees = my_pred_n_trees)
        
        temp_predictions = predict(model,
                                   type = "response",
                                   data = my_data[id_test,])$predictions
      }
      
      
      # predict probability
      if((is_classification == TRUE) & (is_multiclass == FALSE)){
        
        model = ranger(factor(y) ~.,
                             data = my_data[id_train,],
                             mtry = my_pred_mtry,
                       probability = TRUE,
                             num.trees = my_pred_n_trees)
        
        temp_predictions = predict(model,
                                   type = "response",
                                   data = my_data[id_test,])$predictions[,2] > my_threshold
      }
      
      # regression
      if((is_classification == FALSE) & (is_multiclass == FALSE)){
        model = ranger(y ~.,
                             data = my_data[id_train,],
                             mtry = my_pred_mtry,
                             num.trees = my_pred_n_trees)
        
        temp_predictions = predict(model,
                                   data = my_data[id_test,])$predictions
      }
      
      return(temp_predictions)
    }
    
    if(fix_trees_bool == TRUE){
      for (m in 1:length(my_n_variables)){
        # prediction error
        temp_metrics_array_cv[k,m,] = USED.Metrics(PredictFunction(my_pred_mtry = my_n_variables[m],
                                                                   my_pred_n_trees = my_n_bs_trees),
                                                   my_data$y[id_test],
                                                   weights = my_weights)
        print(paste(c("n var = ", my_n_variables[m]), collapse = ""))
      }
    }
    
    else{
      for (t in 1:length(my_n_bs_trees)){
        temp_metrics_array_cv[k,t,] = USED.Metrics(PredictFunction(my_pred_mtry = my_n_variables,
                                                                   my_pred_n_trees = my_n_bs_trees[t]),
                                                   my_data$y[id_test],
                                                   weights = my_weights[id_test])
        print(paste(c("n trees = ", my_n_bs_trees[t]), collapse = ""))
      }
    }
    
    
    print(paste("fold ", k))
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = tuning_parameter_length, ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    else{
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
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
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' 
#' @param my_metrics_functions (vector of characters): vector of loss function names feed to snowfall (parallel)
#' example  my_metrics_functions = c("USED.Metrics", "MAE.Loss", "MSE.Loss").
#' NOTE: if USED.Metrics contains some other functions they must be present as well, like the example
#' which is also the default
#' 
#' @param is_classification (bool): if TRUE adapt the metrics to classification problem
#' using the threshold
#' @param my_threshold (num): classification threshold used
#' 
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
ManualCvRFParallel = function(my_id_list_cv_train,
                              my_metric_names,
                              my_data,
                              my_n_variables = 2:20,
                              my_n_bs_trees = 300,
                              fix_trees_bool = TRUE,
                              my_metrics_functions = MY_USED_METRICS,
                              my_weights = MY_WEIGHTS,
                              my_ncores = N_CORES,
                              use_only_first_fold = FALSE,
                              is_classification = FALSE,
                              my_threshold = 0.5,
                              is_multiclass = FALSE,
                              my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  
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
  
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold,
                                            tuning_parameter_length,
                                            my_n_metrics))
  
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    # it's ugly I know and a bad pratice but I'll do two separate loop based on if condition
    
    PredictFunction = function(my_pred_mtry, my_pred_n_trees){
      
      # predict class (greatest prob.)
      if(is_multiclass == TRUE){
        model = ranger(factor(y) ~.,
                       data = my_data[id_train,],
                       mtry = my_pred_mtry,
                       num.trees = my_pred_n_trees)
        
        temp_predictions = predict(model,
                                   type = "response",
                                   data = my_data[id_test,])$predictions
      }
      
      
      # predict probability
      if((is_classification == TRUE) & (is_multiclass == FALSE)){
        
        model = ranger(factor(y) ~.,
                       data = my_data[id_train,],
                       mtry = my_pred_mtry,
                       probability = TRUE,
                       num.trees = my_pred_n_trees)
        
        temp_predictions = predict(model,
                                   type = "response",
                                   data = my_data[id_test,])$predictions[,2] > my_threshold
      }
      
      # regression
      if((is_classification == FALSE) & (is_multiclass == FALSE)){
        model = ranger(y ~.,
                       data = my_data[id_train,],
                       mtry = my_pred_mtry,
                       num.trees = my_pred_n_trees)
        
        temp_predictions = predict(model,
                                   data = my_data[id_test,])$predictions
      }
      
      return(temp_predictions)
    }
    
    # init parallel
    sfInit(cpus = my_ncores, parallel = T)
    
    sfLibrary(ranger)
    
    # allocate relevant variables in cluster
    sfExport(list = c("my_data", my_metrics_functions,
                      "my_n_bs_trees", "tuning_parameter_length",
                      "my_n_variables", "my_weights",
                      "my_threshold", "is_classification"))
    
    sfExport(list = c("id_train", "id_test", "PredictFunction"))
    
    if(fix_trees_bool == TRUE){
      
    
      # for better readability
      temp_metric = sfLapply(1:length(my_n_variables),
                             fun = function(m) 
                               USED.Metrics(PredictFunction(my_pred_mtry = my_n_variables[m],
                                                            my_pred_n_trees = my_n_bs_trees),
                                            my_data$y[id_test],
                                            weights = my_weights[id_test]))
      
      
      # unlist to the right dimensions matrix
      temp_metrics_array_cv[k,,] = matrix(unlist(temp_metric),
                                                        ncol = my_n_metrics,
                                                        byrow = T)
      
      rm(temp_metric)
      gc()
      
    }
    
    else{
      
      # for better readability
      temp_metric = sfLapply(1:length(my_n_bs_trees),
                             fun = function(t) 
                               USED.Metrics(PredictFunction(my_pred_mtry = my_n_variables,
                                                            my_pred_n_trees = my_n_bs_trees[t]),
                                            my_data$y[id_test],
                                            weights = my_weights))
      
      # unlist to the right dimensions matrix
      temp_metrics_array_cv[k,,] = matrix(unlist(temp_metric),
                                                                   ncol = my_n_metrics,
                                                                   byrow = T)
      
      rm(temp_metric)
      gc()
      
      sfStop()
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
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    else{
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Boosting -----------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(ada)

#' AdaBoost by Cross validation
#' @param my_id_list_cv_train (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data frame used
#' @param my_tree_depths (vector of int): 1: stump, do not go over 6
#' @param my_n_iterations (int): boosting iterations
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_id_list_cv_test (list): indexes of fold used to test
#' # if NULL my_id_list_cv_test = my_id_list_cv, and classical CV is performed
#' # if is not null it has to be the same number of elements as my_id_list_cv
#' # Can be used for example for unbalanced datasets:
#' my_id_list_cv -> balanced folds use to estimation
#' my_id_list_cv_test -> unbalanced folds used for testing
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"
ManualCvADAFixedIterations = function(my_id_list_cv_train,
                                      my_metric_names,
                                      my_data,
                                      my_tree_depths,
                                      my_n_iterations = 200,
                                      my_weights = 1,
                                      use_only_first_fold = FALSE,
                                      my_id_list_cv_test = NULL,
                                      my_threshold = 0.5){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, length(my_tree_depths), my_n_metrics))
  
  y_index = which(colnames(my_data) == "y")
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    
    for (j in 1:length(my_tree_depths)){
      
      model = ada(x = my_data[id_train, -y_index],
                  y = my_data$y[id_train],
                  iter = my_n_iterations,
                  control = rpart.control(maxdepth = my_tree_depths[j]))
      
      temp_predictions = predict(model,
                                 my_data[id_test,-y_index],
                                 type = "prob")[,2]
      
      temp_metrics_array_cv[k,j,] = USED.Metrics(temp_predictions > my_threshold,
                                                 my_data$y[id_test],
                                                 weights = my_weights[id_test])
      print(paste("tree depth: ", j, collapse = ""))
    }
    
    rm(temp_predictions)
    rm(model)
    gc()
    
    print(paste("fold ", k, collapse = ""))
    
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = length(my_tree_depths), ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = length(my_tree_depths), ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    else{
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}

#' AdaBoost by Cross validation
#' @param my_id_list_cv_train (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data frame used
#' @param my_tree__max_depth (int): 1: stump, do not go over 6
#' @param my_n_iterations (int): boosting iterations
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_id_list_cv_test (list): indexes of fold used to test
#' # if NULL my_id_list_cv_test = my_id_list_cv, and classical CV is performed
#' # if is not null it has to be the same number of elements as my_id_list_cv
#' # Can be used for example for unbalanced datasets:
#' my_id_list_cv -> balanced folds use to estimation
#' my_id_list_cv_test -> unbalanced folds used for testing
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"
ManualCvADAFixedDepth = function(my_id_list_cv_train,
                                      my_metric_names,
                                      my_data,
                                      my_tree_depth,
                                      my_n_iterations = 200,
                                      my_weights = 1,
                                      use_only_first_fold = FALSE,
                                      my_id_list_cv_test = NULL,
                                 my_threshold = 0.5){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, length(my_n_iterations), my_n_metrics))
  
  y_index = which(colnames(my_data) == "y")
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    
    for (j in 1:length(my_n_iterations)){
      
      model = ada(x = my_data[id_train, -y_index],
                  y = my_data$y[id_train],
                  iter = my_n_iterations[j],
                  control = rpart.control(maxdepth = my_tree_depth))
      
      temp_predictions = predict(model, my_data[id_test,-y_index], type = "prob")[,2]
                                 
      
      temp_metrics_array_cv[k,j,] = USED.Metrics(temp_predictions > my_threshold,
                                                 my_data$y[id_test],
                                                 weights = my_weights[id_test])
      print(paste("iteration: ", j, collapse = ""))
    }
    
    rm(temp_predictions)
    rm(model)
    gc()
    
    print(paste("fold ", k, collapse = ""))
    
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = length(my_n_iterations), ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = length(my_n_iterations), ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    else{
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SVM ----------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(e1071)

#' SVM by Cross validation
#' @param my_id_list_cv_train (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data frame used
#' @param my_params_vector (int): vector of penalty parameter to be optimized
#' @param my_kernel_name (char): name of kernel used (radial or polinomial)
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_id_list_cv_test (list): indexes of fold used to test
#' # if NULL my_id_list_cv_test = my_id_list_cv, and classical CV is performed
#' # if is not null it has to be the same number of elements as my_id_list_cv
#' # Can be used for example for unbalanced datasets:
#' my_id_list_cv -> balanced folds use to estimation
#' my_id_list_cv_test -> unbalanced folds used for testing
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"
ManualCvSVM = function(my_id_list_cv_train,
                          my_metric_names,
                          my_data,
                          my_params_vector,
                          my_kernel_name = "radial",
                          my_weights = 1,
                          use_only_first_fold = FALSE,
                          my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, length(my_params_vector), my_n_metrics))
  
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
   
    
    for (j in 1:length(my_params_vector)){
      
      temp_predictions = predict(svm(factor(y) ~.,
                                     data = my_data[id_train,],
                                     kernel = my_kernel_name,
                                     degree = 2,
                                     cost = my_params_vector[j],
                                     coef = 1),
                                 my_data[id_test,])
      
      temp_metrics_array_cv[k,j,] = USED.Metrics(temp_predictions, my_data$y[id_test],
                                                 weights = my_weights[id_test])
    }
  
    rm(temp_predictions)
    gc()
    
    print(paste("fold ", k, collapse = ""))
    
  }
  
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = length(my_params_vector), ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = length(my_params_vector), ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    else{
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}

#' SVM by Cross validation
#' @param my_id_list_cv_train (list):ids in each fold , use the global variable
#' @param my_metric_names (vector of string): ordered names of loss functions, use global variables
#' @param my_data (data.frame): data frame used
#' @param my_params_vector (int): vector of penalty parameter to be optimized (depending on kernel)
#' @param my_kernel_name (char): name of kernel used (radial or polinomial)
#' @param use_only_first_fold (bool): if yes fit the model on all except the first fold
#' and compute the metrics on that
#' @param my_id_list_cv_test (list): indexes of fold used to test
#' # if NULL my_id_list_cv_test = my_id_list_cv, and classical CV is performed
#' # if is not null it has to be the same number of elements as my_id_list_cv
#' # Can be used for example for unbalanced datasets:
#' my_id_list_cv -> balanced folds use to estimation
#' my_id_list_cv_test -> unbalanced folds used for testing
#'
#'
#' @return (list): list of two matrix 
#' the first contains the CV folds averaged metrics for each parameter value and each metric 
#' the second the CV computed standard errors of those metrics
#' first matrix is accessed by "metrics"
#' second matrix is accessed by "se"
ManualCvSVMParallel = function(my_id_list_cv_train,
                       my_metric_names,
                       my_data,
                       my_params_vector,
                       my_kernel_name = "radial",
                       my_weights = 1,
                       my_metrics_functions = MY_USED_METRICS,
                       my_ncores = N_CORES,
                       use_only_first_fold = FALSE,
                       my_id_list_cv_test = NULL){
  
  n_k_fold = length(my_id_list_cv_train)
  my_n_metrics = length(my_metric_names)
  
  if(use_only_first_fold == TRUE){
    n_k_fold = 1
  }
  
  if(is.null(my_id_list_cv_test)){
    my_id_list_cv_test = my_id_list_cv_train
  }
  
  temp_metrics_array_cv = array(NA, dim = c(n_k_fold, length(my_params_vector), my_n_metrics))
  
  # init parallel cluster
  sfInit(cpus = my_ncores, parallel = T)
  
  sfLibrary(e1071)
  sfExport(list = c("my_data",
                    "my_params_vector", "my_weights", "my_kernel_name",
                    my_metrics_functions))
  
  
  for (k in 1:n_k_fold){
    id_train = unlist(my_id_list_cv_train[-k])
    id_test = my_id_list_cv_test[[k]]
    
    #' @param cost (num): cost or degree value
    #' @param kernel_name (char)
    #' @return (vector of nums): metrics values
    ParallelFunction = function(cost){
      
      temp_predictions = predict(svm(factor(y) ~.,
                                     data = my_data[id_train,],
                                     kernel = my_kernel_name,
                                     degree = 2,
                                     cost = cost,
                                     coef = 1),
                                 my_data[id_test,])
      
      return(USED.Metrics(y.pred = temp_predictions,
                          y.test = my_data$y[id_test],
                          weights = my_weights[id_test]))
    }
    
    sfExport(list = c("id_train", "id_test", "ParallelFunction"))
    
    temp_metrics = sfLapply(my_params_vector,
                            fun = ParallelFunction)
    
    # unlist to the right dimensions matrix
    temp_metrics_array_cv[k,,] = matrix(unlist(temp_metrics),
                                        nrow = length(my_params_vector),
                                        ncol = my_n_metrics,
                                        byrow = T)
    
    rm(temp_metrics)
    gc()
    
    print(paste("fold ", k, collapse = ""))
  }
  sfStop()
    
  # averaged metrics matrix
  cv_metrics = matrix(NA, nrow = length(my_params_vector), ncol = my_n_metrics)
  
  # metrics standard deviations matrix
  cv_metrics_se = matrix(NA, nrow = length(my_params_vector), ncol = my_n_metrics)
  colnames(cv_metrics) = my_metric_names
  colnames(cv_metrics_se) = my_metric_names
  
  for (i in 1:my_n_metrics){
    if(use_only_first_fold == FALSE){
      cv_metrics[,i] = apply(temp_metrics_array_cv[,,i], 2, mean)
      cv_metrics_se[,i] = apply(temp_metrics_array_cv[,,i], 2, sd)}
    
    else{
      cv_metrics[,i] = temp_metrics_array_cv[1,,i]
      cv_metrics_se[,i] = 0
    }
  }
  
  return(list("metrics" = cv_metrics,
              "se" = cv_metrics_se))
  
}



