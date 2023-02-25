library(xgboost)
library(SHAPforxgboost)
library(ggplot2)

# Load data and create training and test sets
dtrain <- xgb.DMatrix(df.train$data, label = df.train$label)
dtest <- xgb.DMatrix(df.test$data)

# Train the XGBoost model
params <- list(objective = "binary:logistic", max_depth = 2, eta = 1, silent = 1)
xgb_model <- xgb.train(params, dtrain, nrounds=2)

# Generate SHAP values for test set
shap_values <- xgb.predict(xgb_model, dtest, predcontrib=TRUE, approxcontrib=FALSE, feature_contrib=TRUE)

# Create SHAP plot for a specific feature
feature <- "f2"
shap <- shap_values[[feature]]
df <- data.frame(shap=shap, feature_value=df.test$data[,feature])

# Plot the SHAP values
ggplot(df, aes(x=feature_value, y=shap)) +
  geom_point() +
  scale_x_continuous(limits=c(0,1), expand=c(0,0)) +
  ylab(paste0("SHAP value for ", feature)) +
  ggtitle(paste0("SHAP plot for ", feature)) +
  theme(plot.title = element_text(hjust = 0.5))