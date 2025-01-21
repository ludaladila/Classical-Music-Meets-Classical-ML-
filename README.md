# Classical Music Meets Classical ML 

## Overview

This repository contains the solution to the Kaggle in-class competition "Classical Music Meets Classical ML Fall 2024." The objective of this competition is to predict which previous patrons of an orchestra will purchase a season subscription to the upcoming 2014-15 concert season. The model should produce soft predictions, i.e., the probability that each account will purchase a subscription.

This README provides a detailed explanation of the data pipeline, feature engineering, and model training approach used in this solution.

## Table of Contents

- [Data Pipeline](#data-pipeline)
  - [Loading Data](#loading-data)
  - [Data Cleaning and Feature Engineering](#data-cleaning-and-feature-engineering)
  - [Data Aggregation and Merging](#data-aggregation-and-merging)
- [Modeling Approach](#modeling-approach)
  - [Data Preprocessing](#data-preprocessing)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
- [Prediction](#prediction)
  - [Preparation of Test Data](#preparation-of-test-data)
  - [Generating Predictions](#generating-predictions)

## Data Pipeline

The data pipeline is handled through the `DataPreprocessor` class, which includes the following steps:

### Loading Data

- **Function**: `load_data()`
- **Details**: Loads all the provided data files into pandas DataFrames. These include:
  - `account.csv`: Information about customer accounts.
  - `subscriptions.csv`: Details about subscription purchases.
  - `tickets_all.csv`: Ticket purchase history.
  - `concerts.csv`: Information on various concerts.
  - `concerts_2014-15.csv`: Concert data for the 2014-15 season.
  - `train.csv`: Contains the target variable indicating whether customers purchased subscriptions.

### Data Cleaning and Feature Engineering

- **Functions**: `process_account_data()`, `process_subscriptions_data()`
- **Details**:
  - Cleaned columns that were irrelevant or had too many missing values.
  - Created new features to enhance prediction quality:
    - `has_donated`: Indicates if an account made a donation.
    - `years_since_first_donation`: Captures the duration since the first donation was made.
    - Encoded categorical variables, such as subscription packages and sections.
  - Handled missing values by filling them with the most frequent values or appropriate defaults.

### Data Aggregation and Merging

- **Functions**: `aggregate_subscription_data()`, `merge_data()`
- **Details**:
  - Aggregated subscription data to generate useful summary features, including:
    - Total number of seats, average price levels, and subscription counts per account.
    - Created encoded features for packages and sections.
  - Merged the cleaned account data with aggregated subscription data to create a unified dataset for modeling.

## Modeling Approach

The modeling approach involves using the `ModelTrainer` class for feature preparation, training, and evaluation.

### Data Preprocessing

- **Function**: `preprocess_features()`
- **Details**:
  - The dataset is divided into features (`X`) and target (`y`).
  - Numeric features are scaled using `StandardScaler` to standardize the features.
  - The features and target are prepared for training using machine learning models.

### Training the Model

- **Function**: `train_model()`
- **Details**:
  - **Model**: `XGBClassifier` from XGBoost, used for its excellent performance on structured/tabular data.
  - **Hyperparameter Tuning**:
    - Used `GridSearchCV` to tune hyperparameters such as `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
    - **Cross-Validation**: Stratified K-Fold cross-validation with 5 folds was used to ensure a well-distributed representation of both classes.
  - **Training**: Trained an ensemble of XGBoost models using the best parameters from grid search.

### Evaluation

- **Function**: `evaluate_model()`
- **Details**:
  - The model was evaluated using metrics like accuracy, ROC-AUC score, precision, recall, and F1 score.
  - The ROC-AUC score is the primary metric used in the competition, as it measures the model's ability to distinguish between positive and negative classes, which is key for this prediction problem.

## Prediction

The `Prediction` class is responsible for generating predictions using the trained models.

### Preparation of Test Data

- **Function**: `prepare_test_data()`
- **Details**:
  - Merged the test data (`test.csv`) with the training features to create a consistent feature set.
  - Missing values in the test data were filled with appropriate default values, and numeric features were scaled using the pre-trained scaler.

### Generating Predictions

- **Function**: `make_predictions()`
- **Details**:
  - Used an ensemble of trained XGBoost models to predict the probabilities of subscriptions for the test data.
  - Averaged the predictions from all the models to generate a robust final prediction.
  - Generated a submission file (`submission_ensemble.csv`) containing soft predictions, which are probabilistic outputs.


