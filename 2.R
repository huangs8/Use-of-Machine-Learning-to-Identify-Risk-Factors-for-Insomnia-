library(xgboost)

xgb_train <- function(train, target, params) {
  
  # Create DMatrix objects for training and testing data
  dtrain <- xgb.DMatrix(data.matrix(train), label = target)
  
  # Train the model using xgb.train()
  xgb_model <- xgb.train(
    params,
    data = dtrain,
    nrounds = 100,
    verbose = 0
  )
  
  # Return the trained model
  return(xgb_model)
}


# Set XGBoost parameters
params <- list(
  objective = "binary:logistic",
  learning_rate = 0.1,
  max_depth = 5
)

# This function takes in the training data (train), the target variable (target), and the XGBoost parameters (params) as inputs, and outputs a trained XGBoost model.



# Train the XGBoost model
xgb_model <- xgb_train(train = train_data, target = train_target, params = params)
# In this example, train_data and train_target are the training data and target variable, respectively. You can customize the XGBoost parameters to optimize the model performance for your specific problem.

