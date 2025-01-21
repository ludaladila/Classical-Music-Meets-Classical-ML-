import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

class DataPreprocessor:
    """
    Class for data preprocessing, including loading and cleaning data.
    """
    def __init__(self, base_path):
        """
        Initializes the DataPreprocessor with the base path for data files.
        
        Parameters:
        base_path (str): Path to the directory containing the data files.
        """
        self.base_path = base_path
        self.account = None
        self.subscriptions = None
        self.tickets_all = None
        self.concerts = None
        self.concerts_2014_15 = None
        self.train = None
        self.zipcodes = None
        self.merged_data = None

    def load_data(self):
        """
        Loads all the data files into pandas DataFrames.
        """
        self.account = pd.read_csv(self.base_path + 'account.csv', encoding='latin1')
        self.subscriptions = pd.read_csv(self.base_path + 'subscriptions.csv')
        self.tickets_all = pd.read_csv(self.base_path + 'tickets_all.csv')
        self.concerts = pd.read_csv(self.base_path + 'concerts.csv')
        self.concerts_2014_15 = pd.read_csv(self.base_path + 'concerts_2014-15.csv')
        self.train = pd.read_csv(self.base_path + 'train.csv')
        self.zipcodes = pd.read_csv(self.base_path + 'zipcodes.csv')

    def process_account_data(self):
        """
        Cleans and processes the account data, including handling missing values and creating new features.
        """
        # Remove columns with high missing rates and low utility
        self.account.drop(columns=['shipping.zip.code', 'shipping.city', 'relationship'], inplace=True)
         # Create donation-related features, including binary indicator for donation history
        self.account['has_donated'] = self.account['first.donated'].notnull().astype(int)
        self.account['first.donated'] = pd.to_datetime(
            self.account['first.donated'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        # Calculate years since first donation, fill with 0 for non-donors
        self.account['years_since_first_donation'] = 2015 - self.account['first.donated'].dt.year
        self.account['years_since_first_donation'].fillna(0, inplace=True)
        self.account['amount.donated.2013'] = pd.to_numeric(self.account['amount.donated.2013'], errors='coerce').fillna(0)
        self.account['amount.donated.lifetime'] = pd.to_numeric(self.account['amount.donated.lifetime'], errors='coerce').fillna(0)
        self.account['no.donations.lifetime'] = pd.to_numeric(self.account['no.donations.lifetime'], errors='coerce').fillna(0)
        self.account.drop(columns=['first.donated', 'billing.zip.code', 'billing.city'], inplace=True)
        self.account['account.id'] = self.account['account.id'].astype(str)

    def process_subscriptions_data(self):
        """
        Cleans and processes the subscriptions data, including handling missing values and creating new features.
        """
         # Fill missing values in package and location with mode
        self.subscriptions['package'] = self.subscriptions['package'].fillna(self.subscriptions['package'].mode()[0])
        self.subscriptions['location'] = self.subscriptions['location'].fillna(self.subscriptions['location'].mode()[0])
        section_mode = self.subscriptions.groupby(['price.level', 'location'])['section'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        
        self.subscriptions['section'] = self.subscriptions.apply(
            lambda row: section_mode.get((row['price.level'], row['location']), self.subscriptions['section'].mode()[0])
            if pd.isnull(row['section']) else row['section'], axis=1)
        price_level_mode = self.subscriptions.groupby(['section', 'location'])['price.level'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
          # Fill missing sections based on price level and location combinations
        self.subscriptions['price.level'] = self.subscriptions.apply(
            lambda row: price_level_mode.get((row['section'], row['location']), self.subscriptions['price.level'].mode()[0])
            if pd.isnull(row['price.level']) else row['price.level'], axis=1)
        self.subscriptions['account.id'] = self.subscriptions['account.id'].astype(str)
        self.subscriptions['no.seats'] = pd.to_numeric(self.subscriptions['no.seats'], errors='coerce').fillna(0)
        self.subscriptions['price.level'] = pd.to_numeric(self.subscriptions['price.level'], errors='coerce').fillna(0)
        self.subscriptions['subscription_tier'] = pd.to_numeric(self.subscriptions['subscription_tier'], errors='coerce').fillna(0)
        self.subscriptions[['season_start', 'season_end']] = self.subscriptions['season'].str.split('-', expand=True)
        self.subscriptions['season_start'] = self.subscriptions['season_start'].astype(int)
        self.subscriptions['season_end'] = self.subscriptions['season_end'].astype(int)
        self.subscriptions['years_since_subscription'] = 2015 - self.subscriptions['season_end']
        self.subscriptions['multiple.subs_binary'] = self.subscriptions['multiple.subs'].map({'yes': 1, 'no': 0})
        # Encode categorical features using LabelEncoder

        le = LabelEncoder()
        self.subscriptions['package_encoded'] = le.fit_transform(self.subscriptions['package'])
        self.subscriptions['section_encoded'] = le.fit_transform(self.subscriptions['section'])
        self.aggregate_subscription_data()

    def aggregate_subscription_data(self):
        """
        Aggregates subscription data to create new features for each account.
        """
        # Create basic aggregation features (sum, mean, etc.)
        aggregated_data = self.subscriptions.groupby('account.id').agg({
            'no.seats': 'sum',
            'price.level': 'mean',
            'subscription_tier': 'mean',
            'years_since_subscription': 'mean',
            'multiple.subs_binary': 'sum'
        }).reset_index()
         # Calculate additional subscription metrics
        aggregated_data['subscription_count'] = self.subscriptions.groupby('account.id').size().values
        aggregated_data['subscription_total_seats'] = self.subscriptions.groupby('account.id')['no.seats'].sum().values
        aggregated_data['subscription_avg_price_level'] = self.subscriptions.groupby('account.id')['price.level'].mean().values
        # Create features for most common package and section
        encoded_features = self.subscriptions.groupby('account.id').agg({
            'package_encoded': lambda x: x.mode().iloc[0] if not x.mode().empty else 0,
            'section_encoded': lambda x: x.mode().iloc[0] if not x.mode().empty else 0
        }).reset_index()
        self.aggregated_data = aggregated_data.merge(encoded_features, on='account.id', how='left')

    def merge_data(self):
        """
        Merges account data with aggregated subscription data.
        """
        self.customer_data = self.account.merge(self.aggregated_data, on='account.id', how='left')
        self.customer_data.fillna(0, inplace=True)

    def get_merged_data(self):
        """
        Executes the entire data loading and processing pipeline.
        
        Returns:
        pd.DataFrame: Merged and processed data.
        """
        self.load_data()
        self.process_account_data()
        self.process_subscriptions_data()
        self.merge_data()
        return self.customer_data


class ModelTrainer:
    """
    Class for training a machine learning model using the processed data.
    """
    def __init__(self, data, target_feature='label'):
        """
        Initializes the ModelTrainer with the processed data.
        
        Parameters:
        data (pd.DataFrame): The processed data to be used for training.
        target_feature (str): The name of the target feature.
        """
        self.data = data
        self.target_feature = target_feature
        self.models = []
        self.scaler = StandardScaler()

    def preprocess_features(self):
        """
        Prepares the features and target for training, including scaling numeric features.
        """
        X = self.data.drop(columns=['account.id', self.target_feature])
        y = self.data[self.target_feature]
        # Scale numeric features
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        self.X = X
        self.y = y
        self.numeric_features = numeric_features

    def train_model(self):
        """
        Trains an ensemble of XGBoost models using Stratified K-Fold cross-validation.
        """
        # Define parameter grid for hyperparameter tuning
        self.preprocess_features()
        param_grid = {
            'n_estimators': [100, 200],     # Number of trees in the forest
            'max_depth': [3, 5],            # Maximum depth of each tree
            'learning_rate': [0.01, 0.1],   # Learning rate controls the contribution of each tree
            'subsample': [0.7, 0.8],        # Fraction of samples used for training each tree
            'colsample_bytree': [0.7, 0.8]  # Fraction of features used for training each tree
        }
        # Initialize base model with logloss evaluation metric
        xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
         # Perform grid search to find optimal hyperparameters
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
        grid_search.fit(self.X, self.y)
         # Initialize arrays for storing cross-validation predictions

        self.best_params = grid_search.best_params_
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        val_preds = np.zeros(len(self.X))
        val_preds_proba = np.zeros(len(self.X))

        # Train models using Stratified K-Fold cross-validation
        for train_idx, val_idx in skf.split(self.X, self.y):
            X_train_fold, X_val_fold = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train_fold, y_val_fold = self.y.iloc[train_idx], self.y.iloc[val_idx]
            model = XGBClassifier(random_state=42, eval_metric='logloss', **self.best_params)
            model.fit(X_train_fold, y_train_fold)
            self.models.append(model)
            val_preds[val_idx] = model.predict(X_val_fold)
            val_preds_proba[val_idx] = model.predict_proba(X_val_fold)[:, 1]
        
        # Evaluate the trained model
        self.evaluate_model(val_preds, val_preds_proba)
        joblib.dump(self.models, 'xgb_models_list.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')

    def evaluate_model(self, val_preds, val_preds_proba):
        """
        Evaluates the trained model using accuracy and ROC-AUC score.
        
        Parameters:
        val_preds (np.array): Predicted labels for the validation set.
        val_preds_proba (np.array): Predicted probabilities for the validation set.
        """
        accuracy = accuracy_score(self.y, val_preds)
        roc_auc = roc_auc_score(self.y, val_preds_proba)
        print(f'Model Accuracy: {accuracy:.2f}')
        print(f'Model ROC-AUC Score: {roc_auc:.2f}')
        print(classification_report(self.y, val_preds.astype(int)))


class Prediction:
    """
    Class for making predictions on new test data using the trained models.
    """
    def __init__(self, test_data, merged_data):
        """
        Initializes the Prediction class with the test data and merged training data.
        
        Parameters:
        test_data (pd.DataFrame): The test data for which predictions are to be made.
        merged_data (pd.DataFrame): The merged training data.
        """
        self.test_data = test_data
        self.merged_data = merged_data
        # Load trained models and scaler
        self.models = joblib.load('xgb_models_list.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def prepare_test_data(self):
        """
        Prepares the test data for prediction, including merging with training data and scaling features.
        """
        # Merge test data with training data
        self.test_data.columns = ['account.id']
        if self.test_data.iloc[0]['account.id'] == 'ID':
            self.test_data = self.test_data.drop(0).reset_index(drop=True)
        self.test_data['account.id'] = self.test_data['account.id'].astype(str)
        self.test_merged = pd.merge(self.test_data, self.merged_data, on='account.id', how='left')
        fill_values = dict((col, 0) for col in self.test_merged.columns if col != 'account.id')
        # Fill missing values with 0
        self.test_merged.fillna(fill_values, inplace=True)
        self.X_test = self.test_merged.drop(columns=['account.id'])
        # Scale numeric features
        numeric_features = self.X_test.select_dtypes(include=np.number).columns.tolist()
        self.X_test.loc[:, numeric_features] = self.scaler.transform(self.X_test[numeric_features])

    def make_predictions(self):
        """
        Uses the trained models to make predictions on the prepared test data.
        """
        self.prepare_test_data()
        # Make predictions using the ensemble of models
        test_preds_proba = np.zeros(len(self.X_test))
        for model in self.models:
            test_preds_proba += model.predict_proba(self.X_test)[:, 1]
        test_preds_proba /= len(self.models)
        test_preds_proba = np.clip(test_preds_proba, 1e-6, 1 - 1e-6)
        # Generate submission file
        submission = pd.DataFrame({'ID': self.test_data['account.id'], 'Predicted': test_preds_proba})
        submission.to_csv('submission_ensemble.csv', index=False)
        print('Submission file generated: submission_ensemble.csv')


def main():
    """
    Main function to execute the data preprocessing, model training, and prediction steps.
    """
    # Initialize data preprocessor and model trainer
    base_path = './data/'
    data_preprocessor = DataPreprocessor(base_path)
    merged_data = data_preprocessor.get_merged_data()
    # Merge training labels with merged data
    train = pd.read_csv(base_path + 'train.csv')
    train_merged = pd.merge(merged_data, train, on='account.id', how='inner')
    # Train model and make predictions
    model_trainer = ModelTrainer(train_merged)
    model_trainer.train_model()
    test = pd.read_csv(base_path + 'test.csv', header=None)
    predictor = Prediction(test, merged_data)
    predictor.make_predictions()


if __name__ == '__main__':
    main()
