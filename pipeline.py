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
    def __init__(self, target_col='fraud_label', date_col='request_time'):
        self.target_col = target_col
        self.date_col = date_col
        self.champion_model = None
        self.challenger_model = None
        self.registry = [] # Simulates MLflow registry

    def hash_pii(self, val):
        """Hashes PII for compliance."""
        if pd.isna(val):
            return "UNKNOWN"
        return hashlib.sha256(str(val).encode('utf-8')).hexdigest()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans data, drops leakage, and hashes identifiers."""
        logger.info("Starting preprocessing...")
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # 1. Remove Target Leakage
        if 'request_status' in df.columns:
            df = df.drop(columns=['request_status'])
            
        # 2. Hash PII fields
        pii_cols = ['phone', 'email', 'customer_name', 'device_id']
        for col in pii_cols:
            if col in df.columns:
                df[col] = df[col].apply(self.hash_pii)
                
        return df.sort_values(self.date_col).reset_index(drop=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates velocity and temporal features."""
        logger.info("Engineering features...")
        
        # Temporal Features
        df['hour_of_day'] = df[self.date_col].dt.hour
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        df['is_high_risk_hour'] = df['hour_of_day'].apply(lambda x: 1 if 0 <= x <= 5 else 0)
        
        # Velocity Features (Simulating a feature store aggregation)
        # Count of transactions by device in the last 24h
        df = df.set_index(self.date_col)
        if 'device_id' in df.columns:
            df['tx_count_24h_device'] = df.groupby('device_id')['amount'].transform(
                lambda x: x.rolling('24H').count()
            )
        df = df.reset_index()
        
        return df

    def time_aware_split(self, df: pd.DataFrame, test_days=30):
        """Splits data strictly by time to prevent temporal leakage."""
        max_date = df[self.date_col].max()
        split_date = max_date - timedelta(days=test_days)
        
        train_df = df[df[self.date_col] < split_date].copy()
        test_df = df[df[self.date_col] >= split_date].copy()
        
        # Define features matrix
        cols_to_drop = [self.date_col, self.target_col, 'device_id', 'phone', 'email', 'customer_name', 'request_id']
        features = [c for c in df.columns if c not in cols_to_drop]
        
        X_train, y_train = train_df[features], train_df[self.target_col]
        X_test, y_test = test_df[features], test_df[self.target_col]
        
        logger.info(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")
        return X_train, X_test, y_train, y_test

    def build_and_train_models(self, X_train, y_train):
        """Builds preprocessing pipelines and trains Champion and Challenger."""
        categorical_cols = ['merchant_id', 'merchant_type', 'service_type', 'issuer_bank', 'mcc_code']
        numeric_cols = ['amount', 'hour_of_day', 'tx_count_24h_device']
        
        # Ensure columns exist in dataframe
        cat_features = [c for c in categorical_cols if c in X_train.columns]
        num_features = [c for c in numeric_cols if c in X_train.columns]

        # Target Encoding for high-cardinality, StandardScaler for numeric
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', TargetEncoder(min_samples_leaf=20, smoothing=10), cat_features)
            ])

        # Challenger: Logistic Regression
        self.challenger_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])

        # Champion: XGBoost (Optimized for latency and imbalance)
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
        
        # Calculate business threshold (e.g., target 90% recall, find required threshold)
        # For safety, picking a threshold that maximizes F1 as a proxy
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

    def calculate_psi(self, expected, actual, buckets=10):
        """
        Calculates Population Stability Index (PSI) to detect Data Drift.
        Demonstrates drift detection capability.
        """
        def build_buckets(data, bins):
            counts, _ = np.histogram(data, bins=bins)
            return np.where(counts == 0, 0.0001, counts) / len(data)

        bins = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=buckets)
        expected_dist = build_buckets(expected, bins)
        actual_dist = build_buckets(actual, bins)

        psi_values = (actual_dist - expected_dist) * np.log(actual_dist / expected_dist)
        psi = np.sum(psi_values)
        
        status = "STABLE" if psi < 0.1 else ("WARNING" if psi < 0.2 else "CRITICAL DRIFT")
        logger.info(f"Drift Check - PSI: {psi:.4f} [{status}]")
        return psi

# ==========================================
# Execution / Reproduction Script
# ==========================================
if __name__ == "__main__":
    # 1. Load Data (Mocking a dataset for execution completeness)
    # df = pd.read_csv('payment_data.csv') 
    
    # Example execution with dummy data to demonstrate pipeline runs without errors
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
    mock_df = pd.DataFrame({
        'request_id': range(1000),
        'request_time': dates,
        'amount': np.random.exponential(1000, 1000),
        'merchant_id': np.random.choice(['M1', 'M2', 'M3'], 1000),
        'mcc_code': np.random.choice(['5411', '5812', '5999'], 1000),
        'device_id': np.random.choice(['D1', 'D2', 'D3'], 1000),
        'email': np.random.choice(['a@a.com', 'b@b.com'], 1000),
        'request_status': ['SUCCESS'] * 1000, # Will be dropped
        'fraud_label': np.random.choice([0, 1], p=[0.98, 0.02], size=1000) # 2% fraud
    })

    # Initialize Pipeline
    pipeline = FraudDetectionPipeline()

    # Step 1 & 2: Preprocess and Engineer Features
    clean_df = pipeline.preprocess_data(mock_df)
    feat_df = pipeline.engineer_features(clean_df)

    # Step 3: Time-Aware Split
    X_train, X_test, y_train, y_test = pipeline.time_aware_split(feat_df, test_days=10)

    # Step 4: Train Models
    pipeline.build_and_train_models(X_train, y_train)

    # Step 5: Evaluate and Register (Model Approval Logic)
    logger.info("\n--- Pipeline Evaluation ---")
    pipeline.evaluate_and_register("Logistic_Regression_v1", pipeline.challenger_model, X_test, y_test)
    pipeline.evaluate_and_register("XGBoost_Champion_v1", pipeline.champion_model, X_test, y_test)

    # Step 6: Simulate Drift Detection (Monitoring)
    logger.info("\n--- Simulating Data Drift on 'Amount' Feature ---")
    train_amount = X_train['amount'].values
    # Simulating a massive shift in production transaction sizes
    production_amount_drifted = np.random.exponential(5000, 200) 
    pipeline.calculate_psi(train_amount, production_amount_drifted)
