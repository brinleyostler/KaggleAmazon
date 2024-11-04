#### AMAZON EMPLOYEE ACCESS #####
library(tidyverse)
library(patchwork)
library(tidymodels)
library(embed)
library(vroom)
library(discrim)
library(kernlab)
library(themis)

## READ IN THE DATA ####
amazon_test = vroom("./test.csv")
amazon_train = vroom("./train.csv")

## CLEAN THE DATA
amazon_train$ACTION = factor(amazon_train$ACTION) #factor response variable
#

### SMOTE -------
## SMOTE RECIPE ####
amazon_recipe <- recipe(ACTION ~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) #target encoding
  #step_pca(all_predictors(), threshold=.9) #Threshold is between 0 and 1
  #step_smote(all_outcomes(), neighbors=5)
# also step_upsample() and step_downsample()

amazon_prepped = prep(amazon_recipe)
baked <- bake(amazon_prepped, new_data = amazon_train)


#### RANDOM FOREST ####
forest_model <- rand_forest(mtry = tune(),
                            min_n=tune(),
                            trees=1000) %>%  # or 500
  set_engine("ranger") %>%
  set_mode("classification")

## Put into a workflow
forest_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(forest_model)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,9)),
                            min_n(),
                            levels=5)

#### CV ####
## Split data for CV
folds <- vfold_cv(amazon_train, v=10, repeats=1)

## Run the CV
CV_results <- forest_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, precision, accuracy))
# metric_set(roc_auc, f_meas, sens, recall, spec, precision, accuracy)

## Find best tuning parameters
best_tune <- CV_results %>% 
  select_best(metric="roc_auc")

#### Finalize the workflow and fit it ####
final_wf <- forest_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data=amazon_train)

#### Make predictions ####
amazon_preds <- final_wf %>% 
  predict(new_data = amazon_test, type="prob")

## Format predictions for kaggle upload
recipe_kaggle_submission <- amazon_preds %>% 
  bind_cols(., amazon_test) %>% 
  rename(ACTION=.pred_1) %>% 
  select(id, ACTION)

## Write out file
vroom_write(x=recipe_kaggle_submission, file="./ForestPreds.csv", delim=",")



