# Source: https://rpubs.com/Argaadya/image_conv

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



getwd()

folder_class <- list.files('C:/Users/User/OneDrive/Edu/Deep Learning/c_d_p/') 
folder_path <- paste0('C:/Users/User/OneDrive/Edu/Deep Learning/c_d_p/', folder_class, '/')

folder_path


# Get file name
file_name <- map(folder_path, 
                 function(x) paste0(x, list.files(x))) %>% 
            unlist()

# first 6 file name
head(file_name)

# Desired height and width of images
target_size <- c(64, 64)

# Batch size for training the model
batch_size <- 32


# Image Generator
train_data_gen <- image_data_generator(rescale = 1/255, # Scaling pixel value
                                       horizontal_flip = T, # Flip image horizontally
                                       vertical_flip = T, # Flip image vertically 
                                       rotation_range = 45, # Rotate image from 0 to 45 degrees
                                       zoom_range = 0.25, # Zoom in or zoom out range
                                       validation_split = 0.2) # 20% data as validation data
# Training Dataset
train_image_array_gen <- flow_images_from_directory(directory = 'C:/Users/User/OneDrive/Edu/Deep Learning/c_d_p/', # Folder of the data
                                                    target_size = target_size, # target of the image dimension (64 x 64)  
                                                    color_mode = 'rgb', # use RGB color
                                                    batch_size = batch_size , 
                                                    seed = 123,  # set random seed
                                                    subset = 'training', # declare that this is for training data
                                                    generator = train_data_gen)

# Validation Dataset
val_image_array_gen <- flow_images_from_directory(directory = 'C:/Users/User/OneDrive/Edu/Deep Learning/c_d_p/',
                                                  target_size = target_size, 
                                                  color_mode = 'rgb', 
                                                  batch_size = batch_size ,
                                                  seed = 123,
                                                  subset = 'validation', # declare that this is the validation data
                                                  generator = train_data_gen)                                       

# Number of training samples
train_samples <- train_image_array_gen$n

# Number of validation samples
valid_samples <- val_image_array_gen$n

# Number of target classes/categories
output_n <- n_distinct(train_image_array_gen$classes)

# Get the class proportion
table('\nFrequency' = factor(train_image_array_gen$classes)
) %>% 
  prop.table()



model <- keras_model_sequential(name = 'simple_model') %>% 
  
  # Convolution Layer
  layer_conv_2d(filters = 16, kernel_size = c(3,3),
                padding = 'same', activation = 'relu',
                input_shape = c(target_size, 3)) %>% 
  
  # Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Flattening Layer
  layer_flatten() %>% 
  
  # Dense Layer
  layer_dense(units = 16,
              activation = 'relu') %>% 
  
  # Output Layer
  layer_dense(units = output_n,
              activation = 'softmax',
              name = 'Output')

model


model %>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics = 'accuracy')

# Fit data into model
history <- model %>% 
  fit(train_image_array_gen,
    steps_per_epoch = as.integer(train_samples / batch_size),
    epochs = 30,
    validation_data = val_image_array_gen,
    validation_steps = as.integer(valid_samples / batch_size))


plot(history)


val_data <- data.frame(file_name = paste0('C:/Users/User/OneDrive/Edu/Deep Learning/c_d_p/',
                                          val_image_array_gen$filenames)) %>% 
  mutate(class = str_extract(file_name, 'cat|dog|panda'))

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

# not valid anymore:
# pred_test <- predict_classes(model, test_x) 

pred_test <- model %>% predict(test_x) %>% k_argmax()

head(pred_test, 10)

# Convert encoding to label
decode <- function(x){
  case_when(x == 0 ~ "cat",
            x == 1 ~ "dog",
            x == 2 ~ "panda")
            }

pred_test <- sapply(pred_test, decode) 

head(pred_test, 10)

confusionMatrix(as.factor(pred_test), 
                as.factor(val_data$class))




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
  layer_dense(name = "Output", units = 3, activation = "softmax")

model_big

model_big %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = "accuracy")

model_big %>% fit_generator( train_image_array_gen,
                             steps_per_epoch = as.integer(train_samples / batch_size),
                             epochs = 50,
                             validation_data = val_image_array_gen,
                             validation_steps = as.integer(valid_samples / batch_size)) -> history

plot(history)


# not valid anymore:
# pred_test <- predict_classes(model_big, test_x) 

pred_test <-  model_big %>% predict(test_x) %>% k_argmax()

head(pred_test, 10)

# Convert encoding to label
decode <- function(x){
  case_when(x == 0 ~ "cat",
            x == 1 ~ "dog",
            x == 2 ~ "panda")
}

pred_test <- sapply(pred_test, decode) 

head(pred_test, 10)

confusionMatrix(as.factor(pred_test), 
                as.factor(val_data$class))
