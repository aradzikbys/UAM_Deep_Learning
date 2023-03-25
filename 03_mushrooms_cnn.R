# Tensorflow
tensorflow::install_tensorflow()
library(keras)
library(tensorflow)

# Use python in your anaconda3 environment folder
reticulate::use_python("C:/Users/User/anaconda3/envs/tf_image", required = T)

# Data wrangling
library(dplyr)
library(ggplot2)
library(purrr)

# Image manipulation
library(imager)

# Deep learning
library(keras)

# Model Evaluation
library(caret)


# Prepare data set:
# Data set downloaded from: https://www.kaggle.com/datasets/derekkunowilliams/mushrooms
# 4 initial groups divided to 2 classes:
# Edible & conditionally edible >> EDIBLE
# Poisonous & deadly >> POISONOUS

# set directory paths
train_dir <- "C:/Users/User/OneDrive/Edu/Deep Learning/mushroom_dataset/"
train_edible_dir <- "C:/Users/User/OneDrive/Edu/Deep Learning/mushroom_dataset/edible/"
train_poisonous_dir <- "C:/Users/User/OneDrive/Edu/Deep Learning/mushroom_dataset/poisonous/"

folder_path <-c(train_edible_dir,train_poisonous_dir)

# Desired height and width of images
target_size <- c(128,128)

# Batch size for training the model
batch_size <- 32


# Image Generator
train_data_gen <- image_data_generator(rescale = 1/255,
                                      horizontal_flip = T,
                                      vertical_flip = T,
                                      fill_mode = 'nearest',
                                      validation_split = 0.2)

# Training Dataset
train_img_array <- flow_images_from_directory(directory = train_dir,
                                        generator = train_data_gen,
                                        classes = c("edible", "poisonous"),
                                        target_size = target_size,
                                        subset = 'training',
                                        batch_size = batch_size,
                                        color_mode = "rgb")

# Training Dataset
valid_img_array <- flow_images_from_directory(directory = train_dir,
                                        generator = train_data_gen,
                                        classes = c("edible", "poisonous"),
                                        target_size = target_size,
                                        batch_size = batch_size,
                                        subset = 'validation',
                                        color_mode = "rgb")

# Number of classes to predict:
(output_n <- n_distinct(train_img_array$classes))

# Define model architecture
model <- keras_model_sequential() %>%
  
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(target_size, 3)) %>%
  
  layer_batch_normalization() %>% 
  
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 64, activation = "relu") %>%
  
  layer_dense(units = output_n, activation = "softmax")

# Model summary
model

# Compile model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics ='accuracy')


# Train model
history <- model %>% fit_generator(train_generator,
                                   steps_per_epoch = length(train_img_array),
                                   epochs = 15,
                                   validation_data = valid_img_array,
                                   validation_steps = length(valid_img_array))


# Define model architecture
model <- keras_model_sequential() %>%
  
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(target_size, 3)) %>%
  
  layer_batch_normalization() %>% 
  
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>%
  
  layer_batch_normalization() %>% 
  
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 128, activation = "relu") %>%
  
  layer_dense(units = 64, activation = "relu") %>%
  
  layer_dense(units = output_n, activation = "softmax")

# Compile model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics ='accuracy')

# Train model
history <- model %>% fit_generator(train_img_array,
                                   steps_per_epoch = length(train_img_array),
                                   epochs = 15,
                                   validation_data = valid_img_array,
                                   validation_steps = length(valid_img_array))

plot(history)
