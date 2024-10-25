#### AMAZON EMPLOYEE ACCESS #####
library(tidyverse)
library(patchwork)
library(tidymodels)
library(embed)
library(vroom)
library(discrim)

## READ IN THE DATA ####
amazon_test = vroom("./test.csv")
amazon_train = vroom("./train.csv")

## CLEAN THE DATA
amazon_train$ACTION = factor(amazon_train$ACTION) #factor response variable
#

### PRINCIPLE COMPONENT -------
## PCA RECIPE ####
amazon_recipe <- recipe(ACTION ~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% #target encoding
  step_pca(all_predictors(), threshold=.9) #Threshold is between 0 and 1

amazon_prepped = prep(amazon_recipe)
baked <- bake(amazon_prepped, new_data = amazon_train)

### LOGISTIC REGRESSION -----------
## MODEL
log_reg_model <- logistic_reg() %>% 
  set_engine("glm")

## WORKFLOW
log_reg_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(log_reg_model) %>% 
  fit(data=amazon_train)

## PREDICTIONS
amazon_preds <- predict(log_reg_workflow,
                        new_data = amazon_test,
                        type="prob")

## Format predictions for kaggle upload
recipe_kaggle_submission <- amazon_preds %>% 
  bind_cols(., amazon_test) %>% 
  rename(ACTION=.pred_1) %>% 
  select(id, ACTION)

## Write out file
vroom_write(x=recipe_kaggle_submission, file="../../LogRegPreds.csv", delim=",")



### PENALIZED REGRESSION ---------------
pen_reg_model <- logistic_reg(penalty=tune(),
                              mixture=tune()) %>% 
  set_engine("glmnet")

## Put into a workflow
pen_reg_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(pen_reg_model)
## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels=5)
#### CV 
## Split data for CV
folds <- vfold_cv(amazon_train, v=5, repeats=1)
## Run the CV
CV_results <- pen_reg_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))
# (f_meas, sens, recall, spec, precision, accuracy))

## Find best tuning parameters
best_tune <- CV_results %>% 
  select_best(metric="roc_auc")
#### Finalize the workflow and fit it 
final_wf <- pen_reg_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data=amazon_train)

#### Make predictions 
amazon_preds <- final_wf %>% 
  predict(new_data = amazon_test, type="prob")

## Format predictions for kaggle upload
recipe_kaggle_submission <- amazon_preds %>% 
  bind_cols(., amazon_test) %>% 
  rename(ACTION=.pred_1) %>% 
  select(id, ACTION)

## Write out file
vroom_write(x=recipe_kaggle_submission, file="../../PRegPreds.csv", delim=",")



### K-NEAREST NEIGHBORS --------------
knn_model <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

## Put into a workflow
knn_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(knn_model)

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels=5)

#### CV 
## Split data for CV
folds <- vfold_cv(amazon_train, v=5, repeats=1)

## Run the CV
CV_results <- knn_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))
# metric_set(roc_auc, f_meas, sens, recall, spec, precision, accuracy)

## Find best tuning parameters
best_tune <- CV_results %>% 
  select_best(metric="roc_auc")

#### Finalize the workflow and fit it 
final_wf <- knn_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data=amazon_train)

#### Make predictions 
amazon_preds <- final_wf %>% 
  predict(new_data = amazon_test, type="prob")

## Format predictions for kaggle upload
recipe_kaggle_submission <- amazon_preds %>% 
  bind_cols(., amazon_test) %>% 
  rename(ACTION=.pred_1) %>% 
  select(id, ACTION)

## Write out file
vroom_write(x=recipe_kaggle_submission, file="../../KNNPreds.csv", delim=",")





### RANDOM FOREST ---------------
forest_model <- rand_forest(mtry = tune(),
                            min_n=tune(),
                            trees=500) %>%  # or 1000
  set_engine("ranger") %>%
  set_mode("classification")

## Put into a workflow
forest_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(forest_model)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(),
                            levels=5)

#### CV 
## Split data for CV
folds <- vfold_cv(amazon_train, v=5, repeats=1)

## Run the CV
CV_results <- forest_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, precision, accuracy))
# metric_set(roc_auc, f_meas, sens, recall, spec, precision, accuracy)

## Find best tuning parameters
best_tune <- CV_results %>% 
  select_best(metric="roc_auc")

#### Finalize the workflow and fit it 
final_wf <- forest_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data=amazon_train)

#### Make predictions 
amazon_preds <- final_wf %>% 
  predict(new_data = amazon_test, type="prob")

## Format predictions for kaggle upload
recipe_kaggle_submission <- amazon_preds %>% 
  bind_cols(., amazon_test) %>% 
  rename(ACTION=.pred_1) %>% 
  select(id, ACTION)

## Write out file
vroom_write(x=recipe_kaggle_submission, file="../../ForestPreds.csv", delim=",")





### NAIVE BAYES --------------
nb_model <- naive_Bayes(Laplace=tune(),
                      smoothness=tune()) %>%  
  set_engine("naivebayes") %>%
  set_mode("classification")

## Put into a workflow
nb_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(nb_model)

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels=5)

#### CV 
## Split data for CV
folds <- vfold_cv(amazon_train, v=5, repeats=1)

## Run the CV
CV_results <- nb_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, precision, accuracy))
# metric_set(roc_auc, f_meas, sens, recall, spec, precision, accuracy)

## Find best tuning parameters
best_tune <- CV_results %>% 
  select_best(metric="roc_auc")

#### Finalize the workflow and fit it 
final_wf <- nb_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data=amazon_train)

#### Make predictions 
amazon_preds <- final_wf %>% 
  predict(new_data = amazon_test, type="prob")

## Format predictions for kaggle upload
recipe_kaggle_submission <- amazon_preds %>% 
  bind_cols(., amazon_test) %>% 
  rename(ACTION=.pred_1) %>% 
  select(id, ACTION)

## Write out file
vroom_write(x=recipe_kaggle_submission, file="../../NBPreds.csv", delim=",")





