#### AMAZON EMPLOYEE ACCESS #####
library(tidyverse)
library(patchwork)
library(tidymodels)
library(embed)

## READ IN THE DATA ####
amazon_test = read_csv("test.csv")
amazon_train = read_csv("train.csv")
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
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_normalize(all_nominal_predictors()) #target encoding (must
# also step_lencode_glm() and step_lencode_bayes()

amazon_prepped = prep(amazon_recipe)
baked <- bake(amazon_prepped, new_data = amazon_train)
