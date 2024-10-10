#### AMAZON EMPLOYEE ACCESS #####
library(tidyverse)
library(patchwork)
library(tidymodels)
library(embed)
library(vroom)

## READ IN THE DATA ####
amazon_test = read_csv("test.csv")
amazon_train = read_csv("train.csv")

## CLEAN THE DATA
amazon_train$ACTION = factor(amazon_train$ACTION) #factor response variable
#
#### EXPLORATORY DATA ANALYSIS ####
glimpse(amazon_train)

# Action Bar Plot
action_bar = ggplot(data = amazon_train, mapping = aes(x = ACTION)) +
  geom_bar()

# Role Title Bar Plot
title_hist = ggplot(data = amazon_train, mapping = aes(x = ROLE_TITLE)) +
  geom_histogram()

# create a 2 panel ggplot (patchwork) showing 4 key features of the dataset
action_bar + title_hist
#

#### FEATURE ENGINEERING ####
amazon_recipe <- recipe(ACTION ~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .1) %>% # combines categorical values that occur
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_normalize(all_nominal_predictors()) #target encoding (must
# also step_lencode_glm() and step_lencode_bayes()

amazon_prepped = prep(amazon_recipe)
baked <- bake(amazon_prepped, new_data = amazon_train)
#
#### LOGISTIC REGRESSION ####
log_reg_model <- logistic_reg() %>% 
  set_engine("glm")

## Put into a workflow
log_reg_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(log_reg_model) %>% 
  fit(data=amazon_train)

## Make predictions
amazon_preds <- predict(log_reg_workflow,
                              new_data = amazon_test,
                              type="prob")

## Format predictions for kaggle upload
recipe_kaggle_submission <- amazon_preds %>% 
  bind_cols(., amazon_test) %>% 
  rename(ACTION=.pred_1) %>% 
  select(id, ACTION)

## Write out file
vroom_write(x=recipe_kaggle_submission, file="./LogRegPreds.csv", delim=",")


