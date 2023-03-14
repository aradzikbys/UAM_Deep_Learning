library(caret)
library(DALEX)
# Create model ####
model.ml <- train(mpg ~ ., 
                  data = mtcars, 
                  model = 'rf')

model.ml$finalModel

# Explainer ####
rf.explainer <- explain(model.ml$finalModel, 
                        data = mtcars, 
                        y = mtcars$mpg)

model_performance(rf.explainer)

# Shapley values ####
rf.sh <- predict_parts(rf.explainer, 
                       mtcars[1,], 
                       type = "shap")

plot(rf.sh, show_boxplots = FALSE)

rf.bd <- predict_parts(rf.explainer, 
                       mtcars[1,], 
                       type = "break_down_interactions")
plot(rf.bd, show_boxplots = FALSE)

rf.cp <- predict_profile(rf.explainer, 
                         mtcars[1,])
plot(rf.cp)

rf.mp <- model_profile(rf.explainer)
plot(rf.mp)
