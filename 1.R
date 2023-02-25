# Loads the foreign package for reading data from other software formats
library(foreign)

# Loads the readxl package for reading Excel files
library(readxl)

# Loads the MLDataR package for loading and manipulating machine learning datasets
library(MLDataR)

# Loads the cvms package for cross-validation and model selection
library(cvms)

# Loads the tibble package for creating data frames
library(tibble)

# Loads the farff package for reading and writing ARFF files used by Weka
library(farff)

# Loads the dplyr package for data manipulation and cleaning
library(dplyr)

# Loads the xgboost package for building and tuning gradient boosting models
library(xgboost)

# Loads the ggplot2 package for data visualization
library(ggplot2)

# Loads the SHAPforxgboost package for computing SHAP values for XGBoost models
library(SHAPforxgboost)

# Loads the pROC package for computing ROC curves and other metrics for classification models
library(pROC)

# Loads the rms package for regression modeling and validation
library(rms)

# Loads the lspline package for creating restricted cubic splines
library(lspline)

# Loads the caret package for machine learning model training and evaluation
library(caret)

# is a package for importing and exporting data between R and other statistical software programs such as SAS, Stata and SPSS. It provides the functions read_sas(), 
library(haven)

# dplyr is a package for data manipulation and transformation. It provides a set of functions for data manipulation such as filtering, selecting, arranging, summarizing, and joining data frames.
library(dplyr)

# Read the lab data
lab <- read_xpt("lab.xpt")

# Read the lifestyle data
lifestyle <- read_xpt("lifestyle.xpt")

# Combine the lab and lifestyle data frames
lab_lifestyle <- left_join(lab, lifestyle, by = "SEQN")

# Combine the demographic and nutritional risk factors data frames
demographic_nutritional_risk_factors <- left_join(demographic, nutritional_risk_factors, by = "SEQN")

# Combine the two data frames created above
nhanes_data <- left_join(lab_lifestyle, demographic_nutritional_risk_factors, by = "SEQN")

# Read the demographic data
demographic <- read_xpt("demographic.xpt")

# Read the nutritional risk factors data
nutritional_risk_factors <- read_xpt("nutritional_risk_factors.xpt")

# Load the full_labs_nhanes Excel file into the R environment and assign it to a variable called full_labs_nhanes
full_labs_nhanes <- read_excel("full_labs_nhanes.xlsx")

# Load the disease_body_measures Excel file into the R environment and assign it to a variable called disease_body_measures
disease_body_measures <- read_excel("nhanes_disease_body_measures_smoking.xlsx")

# Load the NHANES_Behavior_Depression CSV file into the R environment and assign it to a variable called covariates
covariates <- read_csv("NHANES_Behavior_Depression.csv")

# Merge the full_labs_nhanes, disease_body_measures, and covariates data frames by SEQN (the unique identifier for each participant)
full_df <- merge(full_labs_nhanes, disease_body_measures, by = "SEQN")
full_df <- merge(full_df, covariates, by = "SEQN")

#Create a new variable in the full_df data frame that represents whether or not the participant reported having a sleep disorder
full_df$sleep_disorder <- as.numeric(full_df$Told_Doctor_Sleeping_1_yes == 1)

# Get the names of all the columns in the full_df data frame
full_names <- names(full_df)

# Create an empty vector called sig_var that will eventually contain the names of the significant variables
sig_var <- c("significant_variables")

# Loop through each column in the full_df data frame and perform the following operations:
for (j in 1:684) {
  print(j)
  i <- full_names[j]
  
# Check if the column has more than 15 unique values, and if so, convert the column to numeric
  if (dim(table(full_df[,j])) > 15) {
    full_df[,j] <- as.numeric(full_df[,j])
  }

# Check if the column has more than 1.5 unique values, and if so, perform linear regression with sleep_disorder as the outcome variable and the current column as the predictor variable
  if (dim(table(full_df[,j])) > 1.5) {
    line <- lm(sleep_disorder ~ ., data = full_df[, c(i, "sleep_disorder")])
    summary <- summary(line)
    if (min(summary$coefficients[2,4]) < 0.00001) {
      sig_var <- c(sig_var, i)
    }
  }
}

# Write the sig_var vector to a CSV file called sig_var.csv
write.csv(sig_var, "sig_var.csv")


# The sig_short object is a vector that contains short names for a selection of significant variables identified in a data analysis
sig_short <- c(
  'Red blood cell count (million cells/uL)',
  'Red cell distribution width (%)',
  'Cotinine, Serum (ng/mL)',
  'Hydroxycotinine, Serum (ng/mL)',
  'RBC folate (ng/mL)',
  'Glycohemoglobin (%)',
  'HS C-Reactive Protein (mg/L)',
  'Insulin (pmol/L)',
  'Blood cadmium (ug/L)',
  'Albumin, refrigerated serum (g/dL)',
  'Alkaline Phosphatase (ALP) (IU/L)',
  'Blood Urea Nitrogen (mg/dL)',
  'Glucose, refrigerated serum (mg/dL)',
  'Gamma Glutamyl Transferase (GGT) (IU/L)',
  'Total Protein (g/dL)',
  'N-acetyl-S-(n-propyl)-L-cysteine comt',
  'BMXWT - Weight (kg)',
  'BMXBMI - Body Mass Index (kg/m**2)',
  'BMXWAIST - Waist Circumference (cm)',
  'SMQ020 - Smoked at least 100 cigarettes in life',
  'SMQ690A - Used last 5 days - Cigarettes',
  'SMQ856 - Last 7-d worked at job not at home?',
  'MCQ160b - Ever told had congestive heart failure',
  'MCQ160c - Ever told you had coronary heart disease',
  'MCQ160d - Ever told you had angina/angina pectoris',
  'MCQ160e - Ever told you had heart attack',
  'MCQ160f - Ever told you had a stroke',
  'MCQ160m - Ever told you had thyroid problem',
  'MCQ160p - Ever told you had COPD, emphysema, ChB',
  'MCQ160l - Ever told you had any liver condition',
  'MCQ510f - Liver condition: Other liver disease',
  'MCQ520 - Abdominal pain during past 12 months?',
  'MCQ540 - Ever seen a DR about this pain',
  'MCQ550 - Has DR ever said you have gallstones',
  'MCQ560 - Ever had gallbladder surgery?',
  'MCQ220 - Ever told you had cancer or malignancy',
  'MCQ300b - Close relative had asthma?',
  'MCQ300c - Close relative had diabetes?',
  'MCQ300a - Close relative had heart attack?',
  'MCQ366a - Doctor told you to control/lose weight',
  'MCQ366b - Doctor told you to exercise',
  'MCQ366c - Doctor told you to reduce salt in diet',
  'MCQ366d - Doctor told you to reduce fat/calories',
  'MCQ371a - Are you now controlling or losing weight',
  'MCQ371c - Are you now reducing salt in diet',
  'MCQ371d - Are you now reducing fat in diet',
  'Used_liquid_diet',
  'Supplement_lose_weight',
  'Drank_water_lose_weight',
  'Special_diet_lose_weight',
  'Fewer_carbs',
  'Ate_fruits_veg',
  'Changed_eating_habits',
  'Ate_less_sugar',
  'Ate_less_junk_food',
  'Weight_loss_surgery',
  'PHQ_9',
  'Age',
  'Alcohol..gm.',
  'Dietary.fiber..gm.',
  'Food.folate..mcg.',
  'Caffeine..mg..1',
  'LBXTR...Triglyceride..mg.dL.',
  'Gender')

# creates a new variable x that contains only the variables from sig_var that are also in sig_short.
x <- sig_var[sig_var %in% sig_short]

# updates sig_short to contain only the variables that were present in both sig_var and sig_short.
sig_short <- x

# creates a new variable df that contains the data from full_df.
df <- full_df

#  creates a vector ix containing a random sample of 80% of the row indices of df.
ix <- sample(nrow(df), 0.8 * nrow(df))

# creates a xgb.DMatrix object for the training data, using the columns specified in sig_short and the sleep_disorder column as the label.
dtrain <- xgb.DMatrix(data.matrix(df[ix, sig_short]), label = df[ix, ]$sleep_disorder)

# creates a xgb.DMatrix object for the validation data, using the columns specified in sig_short and the sleep_disorder column as the label.
dvalid <- xgb.DMatrix(data.matrix(df[-ix, sig_short]), label = df[-ix, ]$sleep_disorder)

# creates a list of parameters to be used in the xgboost model.
params <- list(
  objective = "binary:logistic",
  learning_rate = 0.01,
  subsample = 0.9,
  colsample_bynode = 1,
  reg_lambda = 2,
  max_depth = 5
)

#  fits an xgboost model using the training data and the specified parameters.
fit_xgb <- xgb.train(
  params,
  data = dtrain,
  watchlist = list(valid = dvalid),
  early_stopping_rounds = 20,
  nrounds = 10000,
  verbose = 2,
  prediction = T
)

#  generates predictions using the fitted model and the validation data.
pred <- predict(fit_xgb, dvalid)

# creates a vector of the actual outcomes for the validation data.
outcome <- df[-ix, ]$sleep_disorder

# creates a tibble containing the actual outcomes and predicted probabilities for the validation data.
d_binomial <- tibble("target" = outcome, "prediction" = pred)

# calculates the confusion matrix using a threshold of 0.2 for the predicted probabilities.
g <- confusionMatrix(as.factor(outcome), as.factor(as.numeric(pred>0.2)))

# generates a smoothed ROC curve for the predicted probabilities.
r1 <- smooth(roc(d_binomial$target, d_binomial$prediction, plot = T))

#  calculates the confidence intervals for the sensitivity at each level of specificity.
sens.ci <- ci.se(r1, specificities = seq(0,1,.01), boot.n=100, conf.level = .95)

# plots the ROC curve with the confidence intervals.
plot(sens.ci, type="shape", col="Purple", xlim = c(0,1))
               
               
               
