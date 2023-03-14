# MUSHROOMS CLASSIFICATION
# https://www.kaggle.com/datasets/ulrikthygepedersen/mushroom-attributes

library(h2o)
library(ggplot2)
library(dplyr)
library(knitr)
library(wesanderson)


h2o.init(nthreads = -1,
         max_mem_size = '4G')

mushrooms <- read.csv('https://tinyurl.com/hmkhs9au')

# Adapt data set for further analysis
mushrooms <- mushrooms %>%
  # Remove not needed characters
  mutate(across(1:23, ~ substr(.x, 3,3))) %>%
  # Change columns to factor (initially set as strings)
  mutate(across(everything(), as.factor))

head(mushrooms)

mushrooms_hex <- as.h2o(mushrooms, destination_frame = 'mushrooms_hex')

# Split data set to train & test
mushrooms_split <- h2o.splitFrame(data = mushrooms_hex, ratios = 0.75)

mushrooms_train <- mushrooms_split[[1]]
mushrooms_test <- mushrooms_split[[2]]

mushrooms_model <- h2o.deeplearning(
                          y = 23,
                          x = 1:22,
                          training_frame = mushrooms_train,
                          validation_frame = mushrooms_test,
                          distribution = 'multinomial',
                          activation = 'RectifierWithDropout',
                          hidden = c(300, 300, 300),
                          input_dropout_ratio = 0.2,
                          l1 = 1e-5,
                          epochs = 40,
                          variable_importances = TRUE)

# Mushrooms predictions
mushroom_pred <- h2o.predict(object = mushrooms_model,
                             newdata = mushrooms_test)

# Confusion matrix from h2o functions
h2o.performance(mushrooms_model, valid = TRUE)

# Confusion matrix as simple table
table(as.data.frame(mushrooms_test[,23])[,1],
      as.data.frame(mushroom_pred[,1])[,1])


# Model evaluation
expl_mushroom <- h2o.explain(mushrooms_model, mushrooms_test)
print(expl_mushroom)

# Variables importance
h2o.varimp_plot(mushrooms_model)

# Odor vs class
ggplot(mushrooms, aes(odor)) +
  geom_bar(position="dodge", aes(fill = class)) +
  scale_fill_manual(name = 'Edible/Poisonous', values = wes_palette('Moonrise2', type = 'discrete')) +
  labs(x = 'Odor', y = 'Count')+
  scale_x_discrete(labels=c('almond', 'creosote', 'foul', 'anise', 'musty', 'none',
                            'pungent', 'spicy', 'fishy'))


# Odor vs class
ggplot(mushrooms, aes(spore.print.color)) +
  geom_bar(position="dodge", aes(fill = class)) +
  scale_fill_manual(name = 'Edible/Poisonous', values = wes_palette('Moonrise2', type = 'discrete')) +
  labs(x = 'Odor', y = 'Count')+
  scale_x_discrete(labels=c('buff', 'chocolate', 'black', 'brown', 'orange', 'green',
                            'purple', 'white', 'yellow'))


mushrooms_auoml <- h2o.automl(y = 23,
                              x = 1:22,
                              training_frame = mushrooms_train,
                              max_runtime_secs = 30,
                              max_models = 20)


h2o.pd_multi_plot(mushrooms_auoml, mushrooms_test, "odor")

df <- h2o.get_leaderboard(object = mushrooms_auoml, extra_columns = "ALL")
df <- as.data.frame(df)
df$model_id <- substr(df$model_id,1,5)

h2o.shutdown(prompt = TRUE)
