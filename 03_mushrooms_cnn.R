# Tensorflow
tensorflow::install_tensorflow()

# Deep learning
library(keras)
library(tensorflow)

# Data wrangling
library(tidyverse)

# Image manipulation
library(imager)

# Model Evaluation
library(caret)


# Prepare data set:
# Data set downloaded from: https://www.kaggle.com/datasets/derekkunowilliams/mushrooms
# 2 classes (from intial 4 classes):
#   Edible: edible + conditionally edible
#   Poisonous: poisonous + deadly

# set directory paths
train_dir <- "C:/Users/User/OneDrive/Edu/Deep Learning/mushroom_dataset/"

folders <- list.files(train_dir)

folders_paths <- paste0(train_dir,folders,'/') 

train_edible_dir <- folders_paths[1]
train_poisonous_dir <- folders_paths[2]

# Desired height and width of images
target_size <- c(128,128)

# Batch size for training the model
batch_size <- 32

# Image generator
train_data_gen <- image_data_generator(rescale = 1/255,
                                       horizontal_flip = T,
                                       vertical_flip = T,
                                       fill_mode = 'nearest',
                                       validation_split = 0.2)

# Training dataset
train_img_array <- flow_images_from_directory(directory = train_dir,
                                              generator = train_data_gen,
                                              classes = folders,
                                              target_size = target_size,
                                              subset = 'training',
                                              batch_size = batch_size,
                                              color_mode = "rgb")

# Training dataset
valid_img_array <- flow_images_from_directory(directory = train_dir,
                                              generator = train_data_gen,
                                              classes = folders,
                                              target_size = target_size,
                                              batch_size = batch_size,
                                              subset = 'validation',
                                              color_mode = "rgb")

# Number of classes to predict:
(output_n <- n_distinct(train_img_array$classes))

# Define model architecture
simple_model <- keras_model_sequential() %>%
  
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(target_size, 3)) %>%
  
  layer_batch_normalization() %>% 
  
  layer_average_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 64, activation = "relu") %>%
  
  layer_dense(units = output_n, activation = "softmax")

# Model summary
simple_model

# Compile model
simple_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics ='accuracy')

# Train model
simple_model %>% fit_generator(train_img_array,
                               steps_per_epoch = length(train_img_array),
                               epochs = 15,
                               validation_data = valid_img_array,
                               validation_steps = length(valid_img_array)) -> simple_model


plot(simple_model)