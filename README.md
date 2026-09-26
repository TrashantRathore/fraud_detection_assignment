# Fraud Detection ML System for a Payment Gateway

## 1. Problem Statement and Assumptions

**Objective:** Design and implement a near real-time Machine Learning pipeline to score the probability of fraud for payment gateway transactions.

**Assumptions & Business Context:**
* **Imbalance:** The dataset is highly imbalanced, with fraud representing around ~2-5% of total transactions.
* **Latency:** The model will be queried in the critical path of a payment flow. Inference latency must be extremely low (e.g., < 50-100ms).
* **Delayed Labels:** Ground truth labels for fraud (chargebacks) are not instantaneous. They can take 30-90 days to arrive, meaning real-time retraining is impossible, and concept drift must be managed carefully.
* **Compliance:** PII (Personally Identifiable Information) such as `phone`, `email`, and `device_id` cannot be used in plain text due to data privacy regulations.

---

## 2. Approach Explanation

### EDA & Problem Framing
* **Target Metric:** Standard accuracy is useless here. The primary offline metric optimized is **PR-AUC (Precision-Recall Area Under Curve)**, as it focuses strictly on the minority (fraud) class.
* **Business Metric:** In production, the threshold is optimized for a target **False Positive Rate (FPR)** to minimize the "insult rate" (declining legitimate customers), which directly impacts merchant revenue and customer trust.

### Feature Engineering
To maximize predictive power without leaking future information, the following transformations are applied in the `FraudDetectionPipeline`:
* **PII Masking:** `phone` and `device_id` are hashed using SHA-256. This ensures compliance while allowing the model to recognize repeat bad actors.
* **Temporal Features:** Extracted `hour_of_day` and `day_of_week`. Created a high-signal boolean feature `is_high_risk_hour` (e.g., 12 AM to 5 AM) which frequently correlates with unauthorized account takeovers.
* **Features Removed:** request_status feature is creating target leakage recieved after the fraud check i.e removing this feature,
        customer_name is unused feature having PII which is also not beneficial to the model so removing this feature,
        mcc_code which is correlated with merchant_id will create Multi-collinearity problem afterwards so removing this feature,
        mcc_title which is correlated with merchant_id will create Multi-collinearity problem afterwards so removing this feature,
        request_type which is constant/single value for all rows i.e adding no intrinsic value to the model so removing this feature,
        company_name is correlated with merchant_id will create Multi-collinearity problem afterwards so removing this feature,
        currency_code which is constant/single value for all rows i.e adding no intrinsic value to the model so removing this feature,
        partner_id contains the IFSC code for issuer_bank feature i.e will have Multicollinearity problem/redundant data afterwards while model building so removing this feature 
* **Velocity Features:** Created `tx_count_24h_device` to track the number of transactions per device in a rolling 24-hour window. Sudden spikes are strong indicators of card-testing or fraud rings. *(In production, this requires a low-latency Feature Store like Redis).*
* **Domain Extraction:** Extracted the domain from the `email` field. Kept the top 10 most frequent domains and bucketed the rest into "other" to prevent high-cardinality explosions.
* **Leakage Prevention:** Dropped `request_status` (since it is determined *after* the transaction), `customer_name`, and highly correlated/redundant IDs (`mcc_code`, `partner_id`).

### Model Development & Validation
* **Time-Aware Split:** Standard random train/test splits cause data leakage in fraud detection. The data is split strictly sequentially (Out-of-Time validation), holding out the final *N* days for testing to simulate real-world deployment.
* **Handling Imbalance:**
  * Avoided oversampling techniques like SMOTE as they distort actual probability distributions and kept class weights as as 'balanced' this will penalize the algorithm way more for missing a single fraudulent transaction than for misclassifying a genuine transaction.
  * Used algorithmic class weighting: `class_weight='balanced'` for Logistic Regression and dynamically calculated `scale_pos_weight` for XGBoost.
* **Model Selection:**
  * **Champion (XGBoost):** Selected for its ability to capture complex, non-linear interactions between velocity and categorical features while maintaining low inference latency.
  * **Challenger (Logistic Regression):** Serves as a highly interpretable, ultra-fast baseline.
* **Dynamic Thresholding:** The pipeline evaluates the Precision-Recall curve and selects the exact probability cutoff that maximizes the F1-score. F1 Score takes care of Precision and Recall both i.e taking it as primary evaluation metric.

---

## 3. Deployment Strategy

* **Real-Time Serving:** The model will be serialized (e.g., ONNX or PMML format) and deployed behind a low-latency REST API. Numeric and categorical aggregations (velocity features) will be fetched asynchronously from a real-time Feature Store(MLOPS Implementation either using FAST API + Docker/Kubernetes + Clound(AWS/Azure) or using Saas platforms like Databricks which provide inhouse Model End to End Lifecycle management with help of Azure DevOps).
* **Rollout Plan (Safe Deployment):**
  1. **Shadow Mode:** The new model runs in production alongside the live model. It scores transactions and logs predictions to a database, but its outputs do not affect the customer flow.
  2. **Canary Release:** Model is activated for 1-5% of live traffic to monitor business KPI impacts.
  3. **Ramp-Up:** Gradually increase traffic to 100% if stability is confirmed.
* **Fallback Plan:** If the ML service goes down or latency exceeds 200ms, the system will automatically fail open to a **Static Rule-Based Engine** (e.g., blocking known bad IP lists, high-velocity thresholds) to ensure payment processing is not halted.

---

## 4. Monitoring & Drift Detection

Because true labels (chargebacks) are delayed, we must rely on proxy metrics to monitor health.

* **Data/Input Drift (Feature Level):**
  * **Metric:** Population Stability Index (PSI).
  * **Logic:** Compare the distribution of incoming features (e.g., `amount`, `tx_count_24h_device`) against the training baseline. If PSI > 0.2 for a critical feature, trigger a warning.
* **Prediction Drift (Output Level):**
  * **Metric:** Score distribution.
  * **Logic:** If the model historically flags 2% of transactions > 0.8 probability, and suddenly flags 10%, an alert is triggered.
* **Concept Drift (Delayed Labels):**
  * **Metric:** PR-AUC degradation on matured cohorts (data from 30+ days ago).
* **Business KPI Alerts:**
  * **Approval Rate Drop:** If the overall transaction approval rate drops by > 2% in a 1-hour window, trigger a **P0 Alert** to the on-call engineer (potential massive false-positive spike).
  * **Review Queue Size:** Alert if the manual fraud review queue exceeds operational capacity.

---

## 5. Model Versioning, Approval, and Governance

* **Model Registry:** Every model run is logged to a registry (simulated in code via the `self.registry` list, representing a tool like MLflow or Vertex AI). It records hyperparameters, PR-AUC, optimal thresholds, and dataset timestamps.
* **Approval Logic (Order of Operations):**
  1. **Training:** Model is trained on the latest available data.
  2. **Offline Validation:** Must exceed the current production champion's PR-AUC by at least 1%, or maintain it while reducing the False Positive Rate.
  3. **Registration:** Flagged as `CANDIDATE_READY_FOR_SHADOW`.
  4. **Shadow Validation:** Runs in shadow for 7 days. If prediction drift is within acceptable bounds, it is manually approved by a Data Scientist & Risk Manager to go live.
* **Rollback:** Because models are versioned in a central registry, rolling back requires updating a single environment variable or alias (e.g., `model_alias=production`) to point to the previous version ID.

---

## 6. How to Run / Reproduce

To reproduce the pipeline, run the Python script directly. The script includes a synthetic data generator that mimics the provided PG schema.

**python fraud_pipeline.py**

**Expected Output:**
The script will log the step-by-step process to the console:
1. Data cleaning and domain extraction.
2. Feature engineering (Temporal & Velocity).
3. Out-of-Time splitting.
4. Training Champion (XGB) and Challenger (LR) models.
5. Registering the models and outputting their optimal thresholds, ROC-AUC, and PR-AUC scores.

---

## 7. Limitations & Next Steps

Given more time and resources, the following improvements would be prioritized:

* **Graph/Network Features:** Fraudsters often share attributes. Implementing a graph database to create features like `degrees_of_separation_from_known_fraudster` (linking emails to device IDs to IPs) adds immense predictive power.
* **More Granular Velocity:** Implementing shorter-term velocity features (e.g., 5-minute, 1-hour count/sum) and longer-term aggregations (7-day, 30-day average transaction size).
* **Text/NLP Processing:** If `company_name` or `mcc_title` were retained, NLP embeddings (like Word2Vec or TF-IDF) could be used to cluster suspicious, newly registered merchant names.
* **Real Feature Store Integration:** The current velocity logic uses Pandas `.rolling()`. In production, this would be replaced with a streaming ingestion engine (e.g., Kafka + Redis) to maintain state.
**Prerequisites:**
```bash
pip install pandas numpy xgboost scikit-learn category_encoders
