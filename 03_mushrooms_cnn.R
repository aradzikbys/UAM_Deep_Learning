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
# 2 classes (from initial 4 classes):
#   Edible: edible + conditionally edible
#   Poisonous: poisonous + deadly

# set directory paths
train_dir <- "C:/Users/User/OneDrive/Edu/Deep Learning/mushroom_dataset/"

folders <- list.files(train_dir)

folders_paths <- paste0(train_dir,folders,'/') 

train_edible_dir <- folders_paths[1]
train_poisonous_dir <- folders_paths[2]


# Get files names
file_name <- map(folders_paths, 
                 function(x) paste0(x, list.files(x))) %>% 
  unlist()

# first 6 file names
head(file_name)


# Randomly select image
set.seed(123)
sample_image <- sample(file_name, 6)

# Load image into R
img <- map(sample_image, load.image)

# Plot image
par(mfrow = c(2, 3)) # Create 2 x 3 image grid
map(img, plot)



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
  
  layer_dense(units = 128, activation = "relu") %>%
  
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
                               validation_steps = length(valid_img_array)) -> simple_model_hist



plot(simple_model_hist)


# Validation data set
val_data <- data.frame(file_name = paste0('C:/Users/User/OneDrive/Edu/Deep Learning/mushroom_dataset/',
                                          valid_img_array$filenames)) %>% 
  mutate(class = str_extract(file_name, 'edible|poisonous'))

head(val_data, 10)
tail(val_data,10)


# Function to convert image to array
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = target_size, 
                      grayscale = F # Set FALSE if image is RGB
    )
    
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x/255 # rescale image pixel
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

test_x <- image_prep(val_data$file_name)


# Check dimension of testing data set
dim(test_x)


pred_test <- simple_model %>%
  predict(test_x) %>%
  k_argmax()

head(pred_test, 10)

# Convert encoding to label
decode <- function(x){
  case_when(x == 0 ~ 'edible',
            x == 1 ~ 'poisonous')
  }

pred_test <- sapply(pred_test, decode) 

head(pred_test, 10)

confusionMatrix(as.factor(pred_test), 
                as.factor(val_data$class))


# Improve model
model_big <- keras_model_sequential() %>% 
  
  # First convolutional layer
  layer_conv_2d(filters = 32, kernel_size = c(5,5),
                padding = "same", activation = "relu",
                input_shape = c(target_size, 3)) %>% 
  
  # Second convolutional layer
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                padding = "same", activation = "relu") %>% 
  
  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Third convolutional layer
  layer_conv_2d(filters = 64, kernel_size = c(3,3),
                padding = "same", activation = "relu") %>% 
  
  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Fourth convolutional layer
  layer_conv_2d(filters = 128, kernel_size = c(3,3),
                padding = "same", activation = "relu") %>% 
  
  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Fifth convolutional layer
  layer_conv_2d(filters = 256, kernel_size = c(3,3),
                padding = "same", activation = "relu") %>% 
  
  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Flattening layer
  layer_flatten() %>% 
  
  # Dense layer
  layer_dense(units = 64, activation = "relu") %>% 
  
  # Output layer
  layer_dense(name = "Output", units = 2, activation = "softmax")

model_big

model_big %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = "accuracy")

model_big %>% fit_generator(train_img_array,
                            steps_per_epoch = length(train_img_array),
                            epochs = 50,
                            validation_data = valid_img_array,
                            validation_steps = length(valid_img_array)) -> model_big_history


plot(model_big_history)

pred_test <-  model_big %>% predict(test_x) %>% k_argmax()

head(pred_test, 10)

pred_test <- sapply(pred_test, decode) 

head(pred_test, 10)

confusionMatrix(as.factor(pred_test), 
                as.factor(val_data$class))
