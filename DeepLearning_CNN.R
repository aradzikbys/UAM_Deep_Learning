## Libraries
tensorflow::install_tensorflow()
library(keras)
library(tensorflow)
library(dplyr)
library(ggplot2)

## MNIST data set example

### Loading data set
mnist <- dataset_mnist()
c(x_train, y_train) %<-% mnist$train # Train set features, train set labels
c(x_test, y_test) %<-% mnist$test # Test set features, test set labels

dim(x_train) # The dimensions of the training feature set (the images)
dim(x_test) # No label


# The x data is a 3-d array (images, width, height) of grayscale values.
# The y data is an integer vector with values ranging from 0 to 9. 

### Data prepare
# These images are not in the the correct shape as tensors,
# as the number of channels is missing.


# Reshape
img_rows <- 28
img_cols <- 28

x_train <- array_reshape(x_train,
                         c(nrow(x_train),
                           img_rows,
                           # 1 channel >> since pictures are black and white
                           # For color pictures >> 3 canals
                           img_cols, 1))

x_test <- array_reshape(x_test,
                        c(nrow(x_test),
                          img_rows,
                          img_cols, 1))

# Input: 28x28x1 (for colored 28x28x3) instead of 28x28
input_shape <- c(img_rows,
                 img_cols, 1)


# The data must be normalized. Since the pixel values represent brigness
# on a scale from 0 (black) to 255 (white), they can all be rescaled by
# dividing each by the maximum value of 255.

# Rescale
x_train <- x_train / 255
x_test <- x_test / 255


# To prepare this data for training we one-hot encode the vectors into
# binary class matrices using the Keras to_categorical function.
# One hot encoding is a vector representation where all elements of the vector
# are 0 except one, which has 1 as its value (assigning 1 to working feature
# and 0’s to other idle features).
num_classes <- 10
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)


# 9 first encoded digits from trainig set are below
# (the first column corresponds to zero, second to one, etc.):
head(y_train, 9)


### Defining the model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = 'relu',
                # 28 x 28 x 1
                input_shape = input_shape) %>%
  # How many parameters? (3 * 3 * 1 + 1) * 32 = 320
  # (filter_rows * filter_cols * filters_previous_layer + 1) * filters
  # 2 * 32 + 2 * 32 = 128 (2 learnable & 2 non-learnable)
  layer_batch_normalization() %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, 
                kernel_size = c(3, 3), 
                activation = 'relu') %>% 
  # (3 * 3 * 32 + 1) * 64 = 18 496
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  # 5 * 5 * 64 neurons
  layer_flatten() %>% 
  # 1600 * 64 + 64 = 102 464
  layer_dense(units = 64,
              activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  # 64 * 10 + 10 = 650
  layer_dense(units = num_classes,
              activation = 'softmax') 


### Defining the model
model <- keras_model_sequential() %>% 
  # Convolution Layer
  layer_conv_2d(filters = 16,
                kernel_size = c(3,3),
                padding = "same",
                activation = "relu",
                input_shape = input_shape) %>% 
  
  # Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Flattening Layer
  layer_flatten() %>% 
  
  # Dense Layer
  layer_dense(units = 16,
              activation = "relu") %>% 
  
  # Output Layer
  layer_dense(units = num_classes,
              activation = "softmax",
              name = "Output")

### Summary the model >> 122 058$ learnable parameters.
summary(model)


### Plot the model
devtools::install_github('andrie/deepviz')
deepviz::plot_model(model)


### Compile the model
# Loss function: this measures how accurate the model is during training.
# We want to minimize this function to 'steer' the model in the right direction.

# Optimizer: this is how the model is updated based on the data it sees
# and its loss function.

# Metrics: used to monitor the training and testing steps.
# The following example uses accuracy, the fraction of the digits that are
# correctly classified.


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = 'accuracy')


### Training the model
# A mini-batch size of 128 will allow the tensors to fit into the memory
# most of NVidia graphics processing unit.
# The model will run over 15 epochs, with a validation split set at 0.2.


# tensorboard('logs/run_a')
model %>% fit(x_train, 
              y_train, 
              epochs = 15, # Number of epochs
              batch_size = 128, # Size of batch in single step
              validation_split = 0.2 # Percent of data in validation sets
              # callbacks = callback_tensorboard('logs/run_a')
) -> model_cnn 
# 375 = 0.8 * 60000 / 128
plot(model_cnn)


### Evaluate the model
model %>% evaluate(x_train, y_train, verbose = 0) # Evaluate the model’s performance on the train data
model %>% evaluate(x_test, y_test, verbose = 0) # Evaluate the model’s performance on the test data


### Predictions
model %>% predict(x_test) -> predictions # Predicted probabilities on test data
model %>% 
  predict(x_test) %>% 
  k_argmax() %>% 
  as.numeric() -> predicted_digits # Predicted digits on test data


# A prediction is an array of 10 numbers.
# These describe the 'confidence' of the model that the image corresponds
# to each of the 10 different digits.
# Let's plot several images with their predictions.
# Correct prediction labels are green and incorrect prediction labels are red.


par(mfcol = c(5, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = 'i', yaxs = 'i')
for (i in 1:25) { 
  img <- mnist$test$x[i, , ]
  img <- t(apply(img, 2, rev))
  if (predicted_digits[i] == mnist$test$y[i]) {
    color <- '#008800' 
} else {
  color <- '#bb0000'
}
image(1:28, 1:28, img, col = gray((255:0) / 255), xaxt = 'n', yaxt = 'n',
      main = paste0(predicted_digits[i], ' (',
                    mnist$test$y[i], ')'),
      col.main = color)
}

### Confusion matrix
data.frame(table(predicted_digits, mnist$test$y)) %>% 
  setNames(c('Prediction', 'Reference', 'Freq')) %>% 
  mutate(GoodBad = ifelse(Prediction == Reference, 'Correct', 'Incorrect')) -> conf_table

conf_table %>% 
  ggplot(aes(y = Reference, x = Prediction, fill = GoodBad, alpha = Freq)) + 
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 0.5, fontface  = 'bold', alpha = 1) + 
  scale_fill_manual(values = c(Correct = 'green', Incorrect = 'red')) +
  guides(alpha = 'none') + 
  theme_bw() + 
  ylim(rev(levels(conf_table$Reference)))