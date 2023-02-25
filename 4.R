library(xgboost)
library(SHAPforxgboost)

# Load data and train XGBoost model
data(df.train, package='xgboost')
dtrain <- xgb.DMatrix(df.train$data, label=df.train$label)
param <- list(max_depth = 2, eta = 1, silent = 1, objective = "binary:logistic")
xgb_model <- xgb.train(param, dtrain, nrounds = 2)

# Compute SHAP values
explainer <- shap.tree.explainer(model = xgb_model, data = df.train$data)
shap.values <- explainer$shap.values

# Compute feature importance
feature_importance <- shap.feature_importance(shap.values)

# Plot feature importance
plot(feature_importance, main = "SHAP Feature Importance")