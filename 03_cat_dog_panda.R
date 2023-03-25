getwd()

folder_class <- list.files("C:/Users/User/OneDrive/Edu/Deep Learning/cat_dog_panda/") 
folder_path <- paste0("C:/Users/User/OneDrive/Edu/Deep Learning/cat_dog_panda/", folder_class, "/")

folder_path


# Get file name
file_name <- map(folder_path, 
                 function(x) paste0(x, list.files(x))) %>% 
            unlist()

# first 6 file name
head(file_name)

length(file_name)


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
train_image_array_gen <- flow_images_from_directory(directory = "C:/Users/User/OneDrive/Edu/Deep Learning/cat_dog_panda/", # Folder of the data
                                                    target_size = target_size, # target of the image dimension (64 x 64)  
                                                    color_mode = "rgb", # use RGB color
                                                    batch_size = batch_size , 
                                                    seed = 123,  # set random seed
                                                    subset = "training", # declare that this is for training data
                                                    generator = train_data_gen)

# Validation Dataset
val_image_array_gen <- flow_images_from_directory(directory = "C:/Users/User/OneDrive/Edu/Deep Learning/cat_dog_panda/",
                                                  target_size = target_size, 
                                                  color_mode = "rgb", 
                                                  batch_size = batch_size ,
                                                  seed = 123,
                                                  subset = "validation", # declare that this is the validation data
                                                  generator = train_data_gen)                                       

# Number of training samples
train_samples <- train_image_array_gen$n

# Number of validation samples
valid_samples <- val_image_array_gen$n

# Number of target classes/categories
output_n <- n_distinct(train_image_array_gen$classes)

# Get the class proportion
table("\nFrequency" = factor(train_image_array_gen$classes)
) %>% 
  prop.table()



model <- keras_model_sequential(name = "simple_model") %>% 
  
  # Convolution Layer
  layer_conv_2d(filters = 16,
                kernel_size = c(3,3),
                padding = "same",
                activation = "relu",
                input_shape = c(target_size, 3) 
  ) %>% 
  
  # Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Flattening Layer
  layer_flatten() %>% 
  
  # Dense Layer
  layer_dense(units = 16,
              activation = "relu") %>% 
  
  # Output Layer
  layer_dense(units = output_n,
              activation = "softmax",
              name = "Output")

model


model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = "accuracy"
  )

(as.integer(train_samples / batch_size))
(as.integer(valid_samples / batch_size))

# Fit data into model
history <- model %>% 
  fit(
    # Training data
    # 2400 images belonging to 3 classes.
    train_image_array_gen,
    
    # 75 stops per epoch
    steps_per_epoch = as.integer(train_samples / batch_size), 
    # Training epochs
    epochs = 30, 
    
    # validation data
    # 600 images belonging to 3 classes
    validation_data = val_image_array_gen,
    # 18 validation steps
    validation_steps = as.integer(valid_samples / batch_size))
