# MUSHROOMS CLASSIFICATION
# https://www.kaggle.com/datasets/ulrikthygepedersen/mushroom-attributes

library(h2o)
library(dplyr)

h2o.init(nthreads = -1,
         max_mem_size = '4G')

# setwd('C:/Users/User/OneDrive/Edu/Deep Learning')

mushrooms <- read.csv(file = 'mushroom.csv')

mushrooms <- mushrooms %>%
  mutate(across(everything(), as.factor))

skimr::skim(mushrooms)

mushrooms.hex <- as.h2o(mushrooms, destination_frame = 'mushrooms.hex')

mushrooms.split <- h2o.splitFrame(data = mushrooms.hex, ratios = 0.75)

mushrooms_test <- mushrooms.split[[1]]
mushrooms_train <- mushrooms.split[[2]]

mushroom_model <- h2o.deeplearning(
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

# Mushrooms predictions
mushroom_pred <- h2o.predict(object = mushroom_model,
                            newdata = mushrooms_test)

# Confusion matrix
h2o.performance(mushroom_model, valid = TRUE)


mushrooms_train[,23]
mushrooms_test[,23]
mushroom_pred[,1]

as.data.frame(mushrooms_test[,23])

# Table
table(as.data.frame(mushrooms_test[,23])[,1],
      as.data.frame(mushroom_pred[,1])[,1])

h2o.shutdown(prompt = TRUE)
