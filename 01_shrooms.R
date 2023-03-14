# MUSHROOMS CLASSIFICATION
# https://www.kaggle.com/datasets/ulrikthygepedersen/mushroom-attributes

library(h2o)
library(ggplot2)
library(dplyr)
library(wesanderson)

h2o.init(nthreads = -1,
         max_mem_size = '4G')

mushrooms <- read.csv('https://raw.githubusercontent.com/aradzikbys/UAM_Deep_Learning/master/mushroom.csv')

# Adapt data set for further analysis

mushrooms <- mushrooms %>%
  # Remove not needed characters
  mutate(across(1:23, ~ substr(.x, 3,3))) %>%
  # Change columns to factor (initially set as strings)
  mutate(across(everything(), as.factor))

skimr::skim(mushrooms)

mushrooms_hex <- as.h2o(mushrooms, destination_frame = 'mushrooms_hex')

ggplot(mushrooms, aes(cap.shape)) +
  geom_bar(position="dodge", aes(fill = class)) +
  scale_fill_manual(name = 'Edible/Poisonous', values = wes_palette('Moonrise2', type = 'discrete')) +
  labs(x = 'Cap shape', y = 'Count')
  

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
                          epochs = 60,
                          variable_importances = TRUE)

mushrooms_model

# Mushrooms predictions
mushroom_pred <- h2o.predict(object = mushrooms_model,
                            newdata = mushrooms_test)

# Confusion matrix from h2o functions
h2o.performance(mushrooms_model, valid = TRUE)

# Confusion matrix as simple table
table(as.data.frame(mushrooms_test[,23])[,1],
      as.data.frame(mushroom_pred[,1])[,1])


h2o.shutdown(prompt = TRUE)
