# 📊 Customer Churn Prediction System

<div align="center">

**A comprehensive machine learning solution for predicting customer churn in telecommunications**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

**Version 2.0** | **Production Ready** | **With Smart Recommendations**

</div>

---

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for predicting customer churn in telecommunications. The system uses optimized preprocessing, feature engineering, and ensemble methods to achieve high prediction accuracy while providing actionable business recommendations.

**Key Features:**
- **Simplified Input**: Only 6 critical features required (reduced from 21)
- **Smart Recommendations**: Automatic actionable suggestions to reduce churn risk
- **Production-Ready API**: FastAPI backend with comprehensive validation
- **Interactive Dashboard**: Streamlit frontend for non-technical users

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# 1. Clone or download the project
cd copie-ml

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Terminal 1: Start the API server
uvicorn backend_improved:app --reload

# Terminal 2: Start the dashboard
streamlit run frontend_improved.py
```

**Access Points:**
- 🌐 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs
- 🎨 **Dashboard**: http://localhost:8501

---

## 📊 Dataset

**Source**: Telco Customer Churn Dataset (Kaggle)

**Characteristics:**
- **Size**: 7,043 customers
- **Original Features**: 21 columns
- **Target Variable**: Churn (Yes/No)
- **Class Distribution**: 
  - No Churn: 5,174 (73.46%)
  - Churn: 1,869 (26.54%)

**Why this dataset?**
- Representative of real-world telecom churn patterns
- Moderate class imbalance (manageable with SMOTE)
- Diverse feature types (demographic, behavioral, contractual)
- Sufficient size for robust model training

---

## 🔬 Machine Learning Pipeline

### 1. Data Preprocessing

#### 1.1 Data Cleaning
**Technique**: Missing value imputation for `TotalCharges`

**Why**: New customers (tenure = 0) have missing `TotalCharges` because they haven't been billed yet. These are filled with 0, which is logically correct for new customers.

**Implementation**:
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
```

#### 1.2 Feature Engineering
**Technique**: Creation of 6 domain-specific features

**Why Feature Engineering?**
- Captures non-linear relationships between existing features
- Creates business-interpretable features (e.g., customer loyalty segments)
- Improves model's ability to identify churn patterns

**Features Created:**

1. **MonthlyCharges_to_TotalCharges** (Ratio)
   - Formula: `MonthlyCharges / (TotalCharges + 1)`
   - **Purpose**: Identifies spending efficiency. High ratio indicates new customers or potential dissatisfaction
   - **Business Insight**: New customers or those with low total spending relative to monthly charges are at higher risk

2. **AvgMonthlyCharges** (Average)
   - Formula: `TotalCharges / (tenure + 1)`
   - **Purpose**: Normalizes spending across different customer tenures
   - **Business Insight**: Allows comparison of spending patterns regardless of customer age

3. **TenureGroup** (Categorical Segmentation)
   - Bins: New (0-12), Regular (12-24), Loyal (24-48), VeryLoyal (48+)
   - **Purpose**: Captures non-linear relationship between tenure and churn
   - **Business Insight**: New customers (≤12 months) have highest churn risk (42.7%)

4. **ServiceCount** (Count Feature)
   - Formula: Sum of active services (Phone, Security, Backup, etc.)
   - **Purpose**: Measures customer engagement level
   - **Business Insight**: More services = higher engagement = lower churn risk

5. **ContractValue** (Interaction Feature)
   - Formula: `MonthlyCharges × Contract Duration`
   - **Purpose**: Quantifies total contract value
   - **Business Insight**: Higher contract value indicates stronger commitment

6. **HighRiskProfile** (Composite Indicator)
   - Combines multiple risk factors into binary indicator
   - **Purpose**: Quick identification of high-risk customers
   - **Business Insight**: Enables rapid filtering for retention campaigns

**Result**: 21 original features → 27 features after engineering

#### 1.3 Encoding & Scaling

**One-Hot Encoding**
- **Technique**: `pd.get_dummies()` with `drop_first=True`
- **Why**: 
  - Converts categorical variables to numerical format required by ML algorithms
  - `drop_first=True` prevents multicollinearity (dummy variable trap)
  - Preserves all information while making features interpretable
- **Result**: 36 features after encoding (from 27)

**StandardScaler**
- **Technique**: Z-score normalization: `(x - μ) / σ`
- **Why**:
  - Features have different scales (tenure: 0-72, MonthlyCharges: 18-118)
  - Tree-based models (Random Forest, XGBoost) don't strictly require scaling, but:
    - SMOTE uses distance metrics (KNN) which are scale-sensitive
    - Ensures consistent feature importance interpretation
    - Improves convergence for ensemble methods
- **Critical Order**: Scaling **BEFORE** SMOTE (not after)
  - **Reason**: SMOTE generates synthetic samples using KNN, which requires normalized features for accurate distance calculations

#### 1.4 Class Imbalance Handling

**Technique**: SMOTE (Synthetic Minority Oversampling Technique)

**Why SMOTE?**
- **Problem**: 73.46% No Churn vs 26.54% Churn creates bias toward majority class
- **Why not undersampling?**: Would lose valuable data (reduce from 5,174 to 1,869)
- **Why not class weights?**: SMOTE creates new samples, improving model's understanding of minority class patterns
- **How it works**: 
  - Uses KNN to find similar minority class samples
  - Generates synthetic samples along line segments connecting neighbors
  - Creates balanced dataset without losing information


**Why this order matters**:
1. Split train/test (80/20) - prevents data leakage
2. Scale features - normalizes for distance calculations
3. Apply SMOTE - generates synthetic samples on normalized data
4. Train model - learns from balanced, normalized dataset

---

### 2. Model Selection & Training

#### 2.1 Algorithm Selection

**Three algorithms tested:**

1. **Logistic Regression** (Baseline)
   - **Why included**: 
     - Simple, interpretable baseline
     - Fast training and prediction
     - Provides probability estimates
   - **Limitations**: Assumes linear relationships, may miss complex patterns
   - **Configuration**: `class_weight='balanced'` to handle residual imbalance

2. **Random Forest** (Primary Model)
   - **Why chosen**:
     - Handles non-linear relationships and feature interactions
     - Robust to outliers and missing values
     - Provides feature importance scores
     - Less prone to overfitting than single decision trees
     - Good performance on tabular data
   - **How it works**: 
     - Builds multiple decision trees on bootstrapped samples
     - Uses random feature subsets at each split (reduces correlation)
     - Averages predictions across all trees
   - **Best Hyperparameters** (from GridSearchCV):
     - `n_estimators=200`: More trees = better generalization (diminishing returns after ~200)
     - `max_depth=20`: Prevents overfitting while capturing patterns
     - `min_samples_split=2`: Allows fine-grained splits
     - `min_samples_leaf=1`: Maximum flexibility
     - `class_weight='balanced'`: Additional handling of class imbalance

3. **XGBoost** (Gradient Boosting)
   - **Why tested**:
     - State-of-the-art performance on structured data
     - Handles missing values natively
     - Built-in regularization prevents overfitting
     - Often outperforms Random Forest
   - **How it works**:
     - Sequentially builds trees that correct previous errors
     - Uses gradient descent to minimize loss function
     - Applies regularization (L1/L2) to prevent overfitting
   - **Best Hyperparameters**:
     - `n_estimators=200`: Number of boosting rounds
     - `max_depth=7`: Shallower trees prevent overfitting
     - `learning_rate=0.1`: Balance between speed and performance
     - `subsample=1.0`: Use all samples (no subsampling needed with regularization)
     - `scale_pos_weight=2`: Adjusts for class imbalance



#### 2.2 Hyperparameter Tuning

**Technique**: GridSearchCV with 5-fold cross-validation

**Why GridSearchCV?**
- **Systematic search**: Tests all combinations in parameter grid
- **Cross-validation**: Reduces overfitting to specific train/test split
- **5-fold CV**: Good balance between computational cost and validation reliability
- **Scoring metric**: ROC-AUC (area under ROC curve)
  - **Why ROC-AUC?**: 
    - Works well with imbalanced classes
    - Measures ability to distinguish between classes at all thresholds
    - More informative than accuracy for classification



**Results:**
- Random Forest CV AUC: **0.9266**
- XGBoost CV AUC: **0.9292**

#### 2.3 Model Selection

**Final Model**: Random Forest

**Why Random Forest over XGBoost?**
- **Performance**: Similar AUC (0.8250 vs 0.8104 on test set)
- **Interpretability**: Easier to explain feature importance
- **Stability**: Less sensitive to hyperparameter changes
- **Training time**: Faster for this dataset size
- **Production**: Simpler deployment and maintenance

**Test Set Performance:**
- **AUC-ROC**: 0.8250
- **Accuracy**: 0.7672
- **Precision**: 0.5548
- **Recall**: 0.6230
- **F1-Score**: 0.5869

---

### 3. Threshold Optimization

**Technique**: F1-Score optimization across threshold range

**Why optimize threshold?**
- Default threshold (0.5) assumes equal cost of false positives and false negatives
- In churn prediction: **False negatives are more costly** (losing a customer)
- Lower threshold = higher recall (catch more churners) but lower precision

**Process:**
1. Test thresholds from 0.1 to 0.9 (step 0.05)
2. Calculate F1-Score for each threshold
3. Select threshold with maximum F1-Score

**Result**: Optimal threshold = **0.45** (instead of 0.5)

**Impact:**
- F1-Score improvement: 0.6019 → 0.6125 (+1.76%)
- Better balance between precision and recall
- More churners identified (higher recall) with acceptable precision

**Interpretation**: 
- Probability ≥ 0.45 → Predict "Churn"
- This threshold prioritizes identifying at-risk customers (higher recall)
- Acceptable trade-off: Some false alarms (false positives) to catch more real churners

---

## 📈 Results & Interpretation

### Model Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | 0.8250 | **Good discrimination**: Model can distinguish churners from non-churners 82.5% of the time. AUC > 0.8 is considered good for binary classification. |
| **Accuracy** | 0.7672 | **76.72% correct predictions**: Overall prediction accuracy. Note: Accuracy can be misleading with imbalanced classes. |
| **Precision** | 0.5548 | **55.48% precision**: Of customers predicted to churn, 55.48% actually churn. Lower precision means more false alarms, but acceptable for proactive retention. |
| **Recall** | 0.6230 | **62.30% recall**: Model identifies 62.30% of actual churners. This is the critical metric - we want to catch as many churners as possible. |
| **F1-Score** | 0.5869 | **Harmonic mean**: Balance between precision and recall. F1 > 0.5 indicates reasonable performance. |

### Feature Importance Analysis

**Top 5 Most Important Features:**

1. **TotalCharges (16.63%)**
   - **Interpretation**: Total amount paid is strongest predictor
   - **Insight**: Low total charges = new or dissatisfied customers
   - **Action**: Monitor customers with low total spending

2. **tenure (15.24%)**
   - **Interpretation**: Customer loyalty duration is critical
   - **Insight**: New customers (≤12 months) at highest risk
   - **Action**: Implement early retention programs

3. **MonthlyCharges (13.42%)**
   - **Interpretation**: Monthly billing amount matters
   - **Insight**: Very high or very low charges indicate risk
   - **Action**: Review pricing for high-charge customers

4. **PaymentMethod_Electronic check (10.33%)**
   - **Interpretation**: Payment method indicates risk
   - **Insight**: Electronic check users have higher churn (manual payment = less commitment)
   - **Action**: Encourage automatic payment setup

5. **InternetService_Fiber optic (5.63%)**
   - **Interpretation**: Service type affects churn
   - **Insight**: Fiber optic customers churn at 41.9% (vs 19% for DSL)
   - **Action**: Investigate fiber service quality/price concerns

### Business Insights

**Contract Type Impact:**
- **Month-to-month**: 42.7% churn rate ⚠️
  - **Why**: No commitment, easy to switch
  - **Action**: Aggressively promote long-term contracts
- **One-year**: ~15% churn rate
  - **Why**: Moderate commitment
  - **Action**: Offer incentives to extend to 2-year
- **Two-year**: 2.8% churn rate ✅
  - **Why**: Strong commitment, lower churn
  - **Action**: Maintain competitive 2-year contract offers

**Internet Service Impact:**
- **Fiber optic**: 41.9% churn rate ⚠️
  - **Why**: Possibly price/quality mismatch, high expectations
  - **Action**: Review fiber pricing and service quality
- **DSL**: 19.0% churn rate
  - **Why**: Established service, stable
- **No internet**: 7.4% churn rate ✅
  - **Why**: Phone-only customers are more stable

**Customer Tenure Segments:**
- **New (0-12 months)**: Highest risk
  - **Why**: Still evaluating service, no loyalty established
  - **Action**: Welcome programs, early check-ins
- **Regular (12-24 months)**: Moderate risk
  - **Why**: Some loyalty, but still exploring options
- **Loyal (24-48 months)**: Lower risk
  - **Why**: Established relationship
- **Very Loyal (48+ months)**: Lowest risk
  - **Why**: Strong relationship, unlikely to switch

---

## 🛠 Technical Architecture

### Deployment Stack

**Backend (FastAPI)**
- **Why FastAPI?**
  - Automatic API documentation (Swagger/OpenAPI)
  - Fast performance (async support)
  - Type validation with Pydantic
  - Easy integration with ML models
- **Endpoints**: `/predict`, `/batch_predict`, `/health`, `/model_info`
- **Model Loading**: Joblib serialization for fast model persistence

**Frontend (Streamlit)**
- **Why Streamlit?**
  - Rapid development for data apps
  - Built-in widgets and visualizations
  - No frontend framework knowledge required
  - Easy deployment
- **Features**: Single prediction, batch processing, analytics dashboard

**Model Serialization (Joblib)**
- **Why Joblib?**
  - Optimized for NumPy arrays (faster than pickle)
  - Handles large models efficiently
  - Standard in scikit-learn ecosystem
- **Files Saved**:
  - `churn_prediction_model_improved.pkl`: Trained model
  - `scaler_improved.pkl`: StandardScaler for preprocessing
  - `feature_columns_improved.pkl`: Feature order for consistency
  - `optimal_threshold.pkl`: Optimized decision threshold

### Architecture Flow

```
User Input (6 features)
    ↓
Streamlit Frontend (Validation & UI)
    ↓
FastAPI Backend (Preprocessing)
    ↓
Model Inference (Random Forest)
    ↓
Post-processing (Threshold, Recommendations)
    ↓
Response (Prediction + Recommendations)
```

---

## 📡 API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 1000.0,
    "Contract": "Month-to-month",
    "InternetService": "DSL",
    "PaymentMethod": "Electronic check"
  }'
```

**Response:**
```json
{
  "churn_prediction": true,
  "churn_probability": 0.75,
  "risk_level": "High",
  "key_factors": [
    "Month-to-month contract (high churn risk)",
    "New customer (≤12 months tenure)"
  ],
  "recommendations": [
    {
      "priority": "High",
      "action": "Offer Long-term Contract",
      "description": "Offer a 1-year or 2-year contract with discount.",
      "expected_impact": "Reduce churn probability by 30-40%"
    }
  ]
}
```

---

## 🎯 Key Design Decisions

### Why Only 6 Features?

**Feature Selection Method**: Domain expertise + feature importance analysis

**Selected Features:**
1. **tenure** - Strongest predictor (15.24% importance)
2. **MonthlyCharges** - High importance (13.42%)
3. **TotalCharges** - Highest importance (16.63%)
4. **Contract** - Business-critical (42.7% churn for month-to-month)
5. **InternetService** - Significant impact (41.9% churn for fiber)
6. **PaymentMethod** - Behavioral indicator (10.33% importance)

**Benefits:**
- **Reduced input complexity**: Easier for users to provide data
- **Faster predictions**: Less preprocessing
- **Better UX**: Shorter forms, higher completion rates
- **Maintained accuracy**: 6 features capture 80%+ of predictive power

**Other features**: Set to sensible defaults (e.g., gender="Male", Partner="No")

### Why These Specific Algorithms?

**Random Forest**:
- **Non-linear patterns**: Churn has complex, non-linear relationships
- **Feature interactions**: Contract type × tenure × charges creates interactions
- **Interpretability**: Business stakeholders need to understand predictions
- **Robustness**: Handles outliers and missing values well

**XGBoost** (tested but not selected):
- **Performance**: Slightly better CV score (0.9292 vs 0.9266)
- **Not selected**: Similar test performance, more complex to tune

**Logistic Regression** (baseline):
- **Baseline comparison**: Shows improvement from simple to complex models
- **Interpretability**: Coefficients are directly interpretable

---

## 📊 Model Limitations & Considerations

### Limitations

1. **Historical Data**: Model trained on past data, may not capture future trends
2. **Domain Specificity**: Optimized for telecom, may not generalize to other industries
3. **Temporal Changes**: Customer behavior patterns may evolve over time
4. **Data Quality**: Assumes input data quality matches training data

### Best Practices

1. **Regular Retraining**: Update model with new data quarterly/semi-annually
2. **Performance Monitoring**: Track prediction accuracy over time
3. **Threshold Adjustment**: Re-optimize threshold as business priorities change
4. **A/B Testing**: Validate recommendations before full deployment
5. **Human Oversight**: Use predictions as input, not sole decision-maker

---

## 🚧 Future Enhancements

- **Real-time Integration**: Connect to live customer databases
- **Automated Retraining**: Pipeline for periodic model updates
- **Model Monitoring**: Drift detection and performance tracking
- **Deep Learning**: Test neural networks for complex pattern detection
- **Containerization**: Docker deployment for easier scaling
- **Database Integration**: PostgreSQL/MySQL for prediction history


---

## 🙏 Acknowledgments

- **Dataset**: Telco Customer Churn Dataset (Kaggle)
- **Libraries**: scikit-learn, FastAPI, Streamlit, XGBoost communities
- **Techniques**: SMOTE, GridSearchCV, Feature Engineering best practices

---

<div align="center">


**Version 2.0.0** | **Last Updated**: 2024

</div>
