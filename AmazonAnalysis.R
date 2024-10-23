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
#### NAIVE BAYES ####
#### FEATURE ENGINEERING ####
amazon_recipe <- recipe(ACTION ~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_nominal_predictors()) #target encoding (must
# also step_lencode_glm() and step_lencode_bayes()

amazon_prepped = prep(amazon_recipe)
baked <- bake(amazon_prepped, new_data = amazon_train)

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

#### CV ####
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
vroom_write(x=recipe_kaggle_submission, file="../../NBPreds.csv", delim=",")


