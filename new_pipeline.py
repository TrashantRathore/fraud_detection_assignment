import pandas as pd
import numpy as np
import hashlib
from datetime import timedelta
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from category_encoders import TargetEncoder
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
        """__init__ method is used as a initializer for each unique record."""
    def __init__(self, target_col='fraud_label', date_col='request_time'):
        self.target_col = target_col
        self.date_col = date_col
        self.champion_model = None
        self.challenger_model = None
        self.registry = [] # Simulates MLflow registry
        self.top_domains = [] # Stores top email domains during training

    def hash_pii(self, val):
        """We need to mask/hash PII data for a customer for complaince purpose and also can track if same customer comes again, beneficial for our model training."""
        if pd.isna(val):
            return "UNKNOWN"
        return hashlib.sha256(str(val).encode('utf-8')).hexdigest()

    def preprocess_data(self, df: pd.DataFrame, is_training=True) -> pd.DataFrame:
        """Cleans data, extracts email domains, and drops unused features."""
        logger.info("Starting preprocessing...")
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # 1.Removing Irrelevant Features 
        """request_status feature is creating target leakage recieved after the fraud check i.e removing this feature,
        customer_name is unused feature having PII which is also not beneficial to the model so removing this feature,
        mcc_code which is correlated with merchant_id will create Multi-collinearity problem afterwards so removing this feature,
        mcc_title which is correlated with merchant_id will create Multi-collinearity problem afterwards so removing this feature,
        request_type which is constant/single value for all rows i.e adding no intrinsic value to the model so removing this feature,
        company_name is correlated with merchant_id will create Multi-collinearity problem afterwards so removing this feature,
        currency_code which is constant/single value for all rows i.e adding no intrinsic value to the model so removing this feature,
        partner_id contains the IFSC code for issuer_bank feature i.e will have Multicollinearity problem/redundant data afterwards while model building so removing this feature,
        """
        cols_to_drop = ['request_status', 'customer_name','mcc_code', 'mcc_title','request_type','company_name','currency_code','partner_id']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            
        # 2. Extracting Email Domain for getting the domains frequently used for transactions and probably fraud
        if 'email' in df.columns:
            # Extract domain (everything after '@')
            df['email_domain'] = df['email'].astype(str).apply(
                lambda x: x.split('@')[-1].lower() if '@' in x else 'unknown'
            )
            
            # Group rare domains i.e Keeping Top 10 Categories of domain, rest all domains added into a single category for preventing column explosion and preserving model relevance
            if is_training:
                # Learn the top 10 most common domains
                self.top_domains = df['email_domain'].value_counts().nlargest(10).index.tolist()
            
            # Apply grouping: if not in top 10, label as 'other'
            df['email_domain'] = df['email_domain'].apply(
                lambda x: x if x in self.top_domains else 'other'
            )
            
            # Drop the original raw email column
            df = df.drop(columns=['email'])

        # 3. Hash remaining PII (Phone, Device ID), Directly calling hash_pii function i.e defined under the same class. Hashing done using SHA256 i.e considered standard for encoding and decoding PII data in Financial Data
        pii_cols = ['phone', 'device_id']
        for col in pii_cols:
            if col in df.columns:
                df[col] = df[col].apply(self.hash_pii)
                
        return df.sort_values(self.date_col).reset_index(drop=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates velocity and temporal features."""
        logger.info("Engineering features...")
        
        # Temporal Features - Creating using the time variable available in this dataset. Useful for getting velocity, trend and season related time series variables
        df['hour_of_day'] = df[self.date_col].dt.hour
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        # Taking 12 AM to 5 AM as high fraud region of the day and curated a new feature using this hour_of_day curated feature
        df['is_high_risk_hour'] = df['hour_of_day'].apply(lambda x: 1 if 0 <= x <= 5 else 0)
        
        # Velocity Features (Simulating a feature store aggregation)
        df = df.set_index(self.date_col)
        if 'device_id' in df.columns:
            df['tx_count_24h_device'] = df.groupby('device_id')['amount'].transform(
                lambda x: x.rolling('24H').count()
            )
        df = df.reset_index()
        
        return df

    def time_aware_split(self, df: pd.DataFrame, test_days=30):
        """Splits data strictly by time to prevent temporal leakage."""
        # Here we are splitting the data based on time intervals and not any random splitting between train data and test data. 
        max_date = df[self.date_col].max()
        split_date = max_date - timedelta(days=test_days)
        
        train_df = df[df[self.date_col] < split_date].copy()
        test_df = df[df[self.date_col] >= split_date].copy()
        
        # Define features matrix (drop identifiers not used by the model)
        cols_to_drop = [self.date_col, self.target_col, 'device_id', 'phone', 'request_id']
        features = [c for c in df.columns if c not in cols_to_drop]
        
        X_train, y_train = train_df[features], train_df[self.target_col]
        X_test, y_test = test_df[features], test_df[self.target_col]
        
        logger.info(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")
        return X_train, X_test, y_train, y_test

    def build_and_train_models(self, X_train, y_train):
        """Builds preprocessing pipelines and trains Champion and Challenger."""
        # Building pipeline for preprocessing which would be reused for train and test dataset and for future test dataset.
        # Define feature groups based on cardinality and type
        
        high_card_categorical = ['merchant_id', 'merchant_city]
        low_card_categorical = ['merchant_type', 'merchant_state', 'service_type', 'issuer_bank', 'email_domain']
        numeric_cols = ['amount', 'hour_of_day', 'tx_count_24h_device', 'day_of_week', 'is_high_risk_hour']
        
        # Filter to only columns that actually exist in the dataframe
        high_card = [c for c in high_card_categorical if c in X_train.columns]
        low_card = [c for c in low_card_categorical if c in X_train.columns]
        num_features = [c for c in numeric_cols if c in X_train.columns]

        # 3-Part Preprocessor: Scaling as we using Logistic Regrssion, OHE for Low Card. features and Target Encoding for High Card. features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat_low', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card),
                ('cat_high', TargetEncoder(min_samples_leaf=20, smoothing=10), high_card)
            ])

        # Challenger: Logistic Regression - Taking Hyperparameters as base parameters only, with taking class_weight as 'balanced' this will penalize the algorithm way more for missing a single fraudulent transaction than for misclassifying a genuine transaction.
        # Not using any Oversampling technique like SMOTE as it distorts with real probabilities and class weights keeps the real data intergrity.
        self.challenger_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])

        # Champion: XGBoost -- using a scaling mechanism for dealing with class imbalance i.e 'scale_pos_weight', taking eval_metric as 'pr-auc' as it helps in class imbalance dataset.
        # Didn't use HPT because boost given to PR-AUC is very limited compared to risk of overfitting on this already class imbalance dataset
        scale_pos = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
        self.champion_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                scale_pos_weight=scale_pos,
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                eval_metric='aucpr',
                n_jobs=-1
            ))
        ])

        logger.info("Training Challenger Model...")
        self.challenger_model.fit(X_train, y_train)
        
        logger.info("Training Champion Model...")
        self.champion_model.fit(X_train, y_train)

    def evaluate_and_register(self, model_name, model, X_test, y_test):
        """Evaluates model, picks optimal threshold, and logs to simulated registry."""
        y_prob = model.predict_proba(X_test)[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        metrics = {
            "model_name": model_name,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "optimal_threshold": round(optimal_threshold, 4),
            "status": "CANDIDATE_READY_FOR_SHADOW"
        }
        self.registry.append(metrics)
        logger.info(f"Registered Version: {metrics}")
        return metrics


# ==========================================
# Execution / Reproduction Script
# ==========================================
if __name__ == "__main__":
    
    # 1. Create Dummy Data (Now including customer_name and realistic emails)
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
    mock_df = pd.DataFrame({
        'request_id': range(1000),
        'request_time': dates,
        'amount': np.random.exponential(1000, 1000),
        'merchant_id': np.random.choice(['M1', 'M2', 'M3'], 1000),
        'mcc_code': np.random.choice(['5411', '5812', '5999'], 1000),
        'service_type': np.random.choice(['upi', 'card', 'wallet'], 1000), # Low Cardinality
        'device_id': np.random.choice(['D1', 'D2', 'D3'], 1000),
        'customer_name': np.random.choice(['John Doe', 'Jane Smith', 'Alice Jones'], 1000), # Will be dropped
        'email': np.random.choice(['user1@gmail.com', 'user2@yahoo.com', 'scammer@tempmail.net', 'user3@outlook.com'], 1000), # Domains will be extracted
        'request_status': ['SUCCESS'] * 1000, # Will be dropped
        'fraud_label': np.random.choice([0, 1], p=[0.98, 0.02], size=1000)
    })

    pipeline = FraudDetectionPipeline()

    # Step 1 & 2: Preprocess (is_training=True so it learns the top domains) and Engineer Features
    clean_df = pipeline.preprocess_data(mock_df, is_training=True)
    feat_df = pipeline.engineer_features(clean_df)

    # Step 3: Time-Aware Split
    X_train, X_test, y_train, y_test = pipeline.time_aware_split(feat_df, test_days=10)

    # Step 4: Train Models (OHE applied to email_domain)
    pipeline.build_and_train_models(X_train, y_train)

    # Step 5: Evaluate
    logger.info("\n--- Pipeline Evaluation ---")
    pipeline.evaluate_and_register("Logistic_Regression_v1", pipeline.challenger_model, X_test, y_test)
    pipeline.evaluate_and_register("XGBoost_Champion_v1", pipeline.champion_model, X_test, y_test)
