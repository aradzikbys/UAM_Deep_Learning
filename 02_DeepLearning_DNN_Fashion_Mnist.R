library(keras)
library(ggplot2)
library(dplyr)

fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

dim(train_images)
dim(train_labels)

dim(test_images)
dim(test_labels)

# Rescale from 0-255 to 0-1:
train_images <- train_images / 255
test_images <- test_images / 255

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')
par(mfcol = c(3, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = 'i', yaxs = 'i')
for (i in 1:15) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}

# Reshape
train_images <- array_reshape(train_images, c(nrow(train_images), 28 * 28))
test_images <- array_reshape(test_images, c(nrow(test_images), 28 * 28))

# Change to categorical
train_labels <- to_categorical(train_labels, 10)
test_labels <- to_categorical(test_labels, 10)


# Model 1: 3 dense layers with dropout, big batch size (480)
model_1 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = 28 * 28) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model_1)

model_1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = 'accuracy')

model_1 %>% fit(train_images,
              train_labels,
              # Number of epochs
              # (number of times the algorithm sees training data set)
              epochs = 50,
              # Percent of data in validation set
              # 0.8 x 60000 = 48000 images used for training
              # 0.2 x 60000 = 12000 used for validation
              validation_split = 0.2,
              # Size of batch in single step
              # 48000 / 480 = 100 passes in 1 epoch
              batch_size = 480, ) -> model_1_dnn


# Model 2: 2 dense layers w/o dropout, smaller batch size (128)
model_2 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = 'relu', input_shape = 28 * 28) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model_2)

model_2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = 'accuracy')

model_2 %>% fit(train_images,
                train_labels,
                epochs = 50,
                validation_split = 0.2,
                batch_size = 96, ) -> model_2_dnn


# Training and validation performance of 2 models
plot(model_1_dnn)
plot(model_2_dnn)

# Evaluate models
model_1 %>% evaluate(train_images, train_labels)
model_1 %>% evaluate(test_images, test_labels) 

model_2 %>% evaluate(train_images, train_labels)
model_2 %>% evaluate(test_images, test_labels) 

# Accuracy and loss are better in model_1

# Predicted probabilities on test data
model_1 %>% predict(test_images) -> predictions 

# Predicted classes on test data
model_1 %>% predict(test_images) %>%
  k_argmax() %>%
  as.numeric() -> predicted_clothes 


# See prediction results: prediction (real)
par(mfcol = c(5, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = 'i', yaxs = 'i')
for (i in 1:25) { 
  img <- fashion_mnist$test$x[i, , ]
  img <- t(apply(img, 2, rev))
  if (predicted_clothes[i] == fashion_mnist$test$y[i]) {
    color <- '#507B58' 
  } else {
    color <- '#902711'
  }
  image(1:28, 1:28, img, col = gray((255:0) / 255), xaxt = 'n', yaxt = 'n',
        # prediction 
        main = paste0(class_names[predicted_clothes[i]+1], ' (',
        # (real)
        class_names[fashion_mnist$test$y[i]+1], ')'),
        col.main = color)
        }

data.frame(table(predicted_clothes, fashion_mnist$test$y)) %>%
  setNames(c('Prediction', 'Reference', 'Freq')) %>%
  mutate(Results = ifelse(Prediction == Reference, 'Correct', 'Incorrect')) -> conf_table

conf_table %>%
  ggplot(aes(y = Reference, x = Prediction, fill = Results, alpha = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 0.5, fontface = 'bold', alpha = 1) +
  scale_fill_manual(values = c(Correct = '#507B58', Incorrect = '#902711')) +
  guides(alpha = FALSE) +
  ylim(rev(levels(conf_table$Reference))) +
  scale_x_discrete(labels=class_names) +
  scale_y_discrete(labels=class_names)

















