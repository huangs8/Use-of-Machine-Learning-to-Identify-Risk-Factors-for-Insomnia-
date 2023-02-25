# Use-of-Machine-Learning-to-Identify-Risk-Factors-for-Insomnia-

README:

This project involves using machine learning techniques to identify lab, lifestyle, demographic, and nutritional risk factors for Insomnia. The dataset used in this project is the NHANES dataset, which contains information about the health and lifestyle of individuals in the United States.

The code is written in R and utilizes various packages such as foreign, readxl, MLDataR, cvms, tibble, farff, dplyr, xgboost, ggplot2, SHAPforxgboost, pROC, rms, and lspline. These packages are necessary for importing and manipulating data, building models, and visualizing results.

The main script is named "1.R" and is located in the code directory. The script imports data from various Excel and CSV files, merges them, and performs feature selection using linear regression models. The output of the feature selection process is saved in a CSV file named "sig_var.csv" in the code directory.

The selected features are then used to build an XGBoost model to predict the presence of sleep disorders. The XGBoost function is located in "2.R" in the code directory. The model is evaluated using various metrics such as AUC, accuracy, precision, and recall. The results are visualized using ROC curves, confusion matrices, and SHAP (SHapley Additive exPlanations) plots. The code for making SHAP plots is located in "3.R" and the code for feature importance is in "4.R".

The intermediate datasets used in the project are stored in the "data" directory. The NHANES dataset and other relevant files can be downloaded from the official NHANES website.

Note: Some of the paths in the code are specific to the author's machine and may need to be modified accordingly.
