#### Installation ####
install.packages('h2o')
# The version of H2O in CRAN is often one release behind the current version.
# For the latest recommended version, download the latest stable H2O-3 build from http://h2o.ai/download.
library(h2o)
h2o.init(nthreads = -1,
         max_mem_size = '4G')
# Start H2O on your local machine using all available cores
# For best performance, the allocated memory should be 4x the size of your data,
# but never more than the total amount of memory on your computer. 
# On 64-bit Java, the heap size is 1/4 of the total memory available on the machine.


h2o.clusterInfo() # Check the status and health of the H2O cluster


# visual interface in the browser: look for "http://localhost:54321"

#### Data Preparation and simple statistics ####
# Although it may seem like you are manipulating the data in R, once the
# data has been passed to H2O, all data munging occurs in the H2O
# instance. You are limited by the total amount of memory allocated to the H2O
# instance, not by R’s ability to handle data.

# File path
airlinesURL <- 'https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv' 
# The U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics (BTS) 
# tracks the on-time performance of domestic flights operated by large air carriers. 


# Import data
# H20 frame - data is not in R, loaded to cluster
# you can check in browser (http://localhost:54321 >> GET FRAMES)
airlines.hex <- h2o.importFile(path = airlinesURL,
                               destination_frame = 'airlines.hex') 

# summary statistics - from H20 (calculations were performed in cluster
# created in H20, then downloaed to R)
summary(airlines.hex) 

# calc done in R
skimr::skim(airlines.hex)

# Quantiles
quantile(x = airlines.hex$ArrDelay, na.rm = TRUE) 

# Histogram
h2o.hist(airlines.hex$ArrDelay)

# If factor
h2o.anyFactor(airlines.hex) 

# Number of flights by airport
originFlights <- h2o.group_by(data = airlines.hex, 
                              #group by
                              by = 'Origin', 
                              # Calculate nrow()
                              nrow('Origin'), 
                              # Remove NA's
                              gb.control = list(na.methods = 'rm')) 
# Convert to data frame
as.data.frame(originFlights)


# Number of flights per month
flightsByMonth <- h2o.group_by(data = airlines.hex, 
                               by = 'Month', 
                               nrow('Month'),
                               gb.control = list(na.methods = 'rm'))
as.data.frame(flightsByMonth)



# Months with the highest cancellation ratio
cancellationsByMonth <- h2o.group_by(data = airlines.hex, 
                                     by = 'Month', 
                                     sum('Cancelled'), 
                                     gb.control = list(na.methods = 'rm'))
cancellation_rate <- cancellationsByMonth$sum_Cancelled / flightsByMonth$nrow
rates_table <- h2o.cbind(flightsByMonth$Month,
                         cancellation_rate)
as.data.frame(rates_table)

# Table
h2o.table(airlines.hex$Cancelled)
# Because H2O can handle larger datasets, it is possible to generate tables that 
# are larger than R‘s capacity, so use caution when executing this command.

#### Data Manipulation ####
# Import the iris data into H2O
# iris.hex - object in R, make it H20 as "destination_frame"
iris.hex <- as.h2o(iris, destination_frame = 'iris.hex')

h2o.ls() # List of objects
# http://localhost:54321/flow/index.html

# Splitting Frames
# h2o.splitFrame() does not give an exact split. H2O is designed
# to be efficient on big data using a probabilistic splitting method rather than an
# exact split. On small datasets, the sizes of the resulting splits will deviate from
# the expected value more than on big data, where they will be very close to
# exact.

# split data frame
iris.split <- h2o.splitFrame(data = iris.hex,
                             ratios = 0.75)
# Creates training set from 1st data set in split
iris.train <- iris.split[[1]]

# Creates testing set from 2st data set in split
iris.test <- iris.split[[2]]

# there is no iris.test >> we didn't specify "destination_frame"
h2o.ls()


# Adding Functions >> H20 can make some operations on data
simpleFun <- function(x){
  2 * x + 5
}
calculated <- simpleFun(iris.hex[, 'Sepal.Length'])
h2o.cbind(iris.hex[, 'Sepal.Length'], calculated)




#### Running Models ####
# Classification
# GBM model - gradient boosting machine
                    # where is y (which column)
iris.gbm <- h2o.gbm(y = 5, 
                    # which columns are parameters
                    x = 1:4, 
                    training_frame = iris.train) 

# Training history
iris.gbm@model$scoring_history

# Prediction
h2o.predict(object = iris.gbm,
            newdata = iris.test)

h2o.confusionMatrix(iris.gbm)



# NEW DATA
prostate.hex <- h2o.importFile(path = 'https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv', 
                               destination_frame = 'prostate.hex')

# Logistic classification
                        # name of feature
prostate.glm <- h2o.glm(y = 'CAPSULE', 
                        # names of parameters
                        x = c('AGE', 'RACE', 'PSA', 'DCAPS'), 
                        training_frame = prostate.hex, 
                        # CV
                        nfolds = 10,
                        # in logistic regression
                        family = 'binomial') 

prostate.glm@model$cross_validation_metrics

h2o.predict(object = prostate.glm,
            newdata = prostate.hex)

# AUC
h2o.auc(prostate.glm) 

# Confusion matrix
h2o.confusionMatrix(prostate.glm)

h2o.performance(prostate.glm)




# Pine age based on its height
# Regression

Loblolly.data <- Loblolly[, 1:2]

# Convert to h20 format (visible as Loblolly.hex in browser/h20)
Loblolly.hex <- as.h2o(Loblolly.data, destination_frame = 'Loblolly.hex')

# Regression
Loblolly.glm <- h2o.glm(y = 1, 
                        x = 2, 
                        training_frame = Loblolly.hex, 
                        nfolds = 10) 

lm.predict <- h2o.predict(object = Loblolly.glm,
                          newdata = Loblolly.hex)



Loblolly.automl <- h2o.automl(y = 1, 
                              x = 2, 
                              training_frame = Loblolly.hex,
                              max_runtime_secs = 60)
Loblolly.automl@leader # The leader model

automl.predict <- h2o.predict(object = Loblolly.automl,
                              newdata = Loblolly.hex)

library(ggplot2)
library(dplyr)
h2o.cbind(Loblolly.hex, lm.predict, automl.predict) %>% 
  as.data.frame() %>% 
  setNames(c('height', 'age', 'LM', 'AutoML')) %>% 
  tidyr::gather(Method, Prediction, -age) %>% 
  ggplot(aes(x = age, y = Prediction)) +
  geom_point(aes(color = Method)) + 
  geom_smooth(method = 'lm', se = FALSE, size = 0.5) + 
  theme_minimal()

# k-means
h2o.kmeans(training_frame = iris.hex, 
           k = 3, 
           x = 1:4)

# PCA
h2o.prcomp(training_frame = iris.hex[, -5], 
           transform = 'NORMALIZE', 
           k = 2)

#### CASE STUDY ####
library(h2o)
h2o.init(nthreads = -1)

train_file <- 'https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz'
test_file <- 'https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz'

train <- h2o.importFile(train_file) # 13MB
test <- h2o.importFile(test_file) # 2.1MB

# To see a brief summary of the data
summary(train)
summary(test)

# Specify the response and predictor columns
y <- 'C785'
x <- setdiff(names(train), y)

# Encode the response column as categorical for classification
train[, y] <- as.factor(train[, y])
test[, y] <- as.factor(test[, y])

# The example below illustrates the relative simplicity underlying most H2O Deep Learning model parameter
# configurations, as a result of the default settings. We use the first 282 = 784 values of each row to
# represent the full image and the final value to denote the digit class. Rectified linear activation is popular
# with image processing and has performed well on the MNIST database previously and dropout has been
# known to enhance performance on this dataset as well, so we train our model accordingly.

# Train a Deep Learning model and validate on a test set.
# Rectified linear activation is popular with image processing and has performed well on the MNIST
# database previously and dropout has been known to enhance performance on this dataset as well, 
# so we train our model accordingly.
model <- h2o.deeplearning(x = x,
                          y = y,
                          training_frame = train,
                          validation_frame = test,
                          distribution = 'multinomial',
                          activation = 'RectifierWithDropout',
                          hidden = c(200, 200, 200),
                          input_dropout_ratio = 0.2,
                          l1 = 1e-5,
                          epochs = 10,
                          variable_importances = TRUE)

model@parameters # View the specified parameters
model # Display all performance metrics
h2o.performance(model, train = TRUE) # Training set metrics
h2o.performance(model, valid = TRUE) # Validation set metrics
h2o.mse(model, valid = TRUE) # Get MSE only

h2o.varimp(model) # Variable importance

pred <- h2o.predict(model, newdata = test) # Predictions
head(pred)

# Grid search
hidden_opt <- list(c(200, 200), c(100, 300, 100), c(500, 500, 500))
l1_opt <- c(1e-5, 1e-7)
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt)
model_grid <- h2o.grid('deeplearning',
                       hyper_params = hyper_params,
                       x = x,
                       y = y,
                       distribution = 'multinomial',
                       training_frame = train,
                       epochs = 5,
                       validation_frame = test,
                       search_criteria = list(strategy = 'RandomDiscrete', 
                                              max_runtime_secs = 900))

summary(model_grid) # Print out all prediction errors of the models
# Print out the Test MSE for all of the models
lapply(model_grid@model_ids, 
       function(model_id) 
         paste("Test set MSE:", 
               round(h2o.mse(h2o.getModel(model_id), 
                             valid = TRUE), 6)))

h2o.shutdown(prompt = TRUE) # Shut down the specified instance. All data will be lost.
