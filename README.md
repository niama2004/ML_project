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

This project provides a complete end-to-end machine learning solution for predicting customer churn in the telecommunications industry. It includes:

- **📈 Optimized ML Pipeline**: Feature engineering, hyperparameter tuning, and ensemble methods
- **🚀 RESTful API**: FastAPI backend with simplified input (only 6 critical features)
- **💻 Interactive Dashboard**: Streamlit frontend with actionable recommendations
- **💡 Smart Recommendations**: Automatic suggestions to reduce churn risk
- **📊 Advanced Analytics**: Comprehensive visualizations and insights

### Key Highlights

✨ **Simplified Input**: Only 6 critical features required (instead of 18)  
✨ **Smart Recommendations**: Automatic actionable recommendations based on predictions  
✨ **Optimized Model**: Hyperparameter tuning, feature engineering, optimal threshold  
✨ **Production Ready**: Complete deployment pipeline with API and dashboard  

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Performance](#-model-performance)
- [Business Insights](#-business-insights)
- [Technical Details](#-technical-details)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

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

## 📁 Project Structure

```
copie-ml/
│
├── 📓 Notebooks
│   ├── ml1.ipynb                      # Original ML pipeline
│   └── ml1_improved.ipynb             # ⭐ Optimized ML pipeline with all improvements
│
├── 🔧 Backend
│   ├── backend.py                     # Original FastAPI server
│   └── backend_improved.py            # ⭐ Improved API (simplified + recommendations)
│
├── 🎨 Frontend
│   ├── frontend.py                    # Original Streamlit dashboard
│   └── frontend_improved.py           # ⭐ Improved dashboard (enhanced UX)
│
├── 📊 Data & Models
│   ├── Customer-Churn.csv             # Dataset (7,043 customers, 21 features)
│   ├── churn_prediction_model.pkl     # Original trained model
│   ├── churn_prediction_model_improved.pkl  # ⭐ Optimized model
│   ├── scaler.pkl / scaler_improved.pkl
│   ├── feature_columns.pkl / feature_columns_improved.pkl
│   └── optimal_threshold.pkl          # ⭐ Optimal decision threshold
│
├── 📚 Documentation
│   ├── README.md                      # This file
│   ├── GUIDE_UTILISATION_IMPROVED.md  # Detailed usage guide (French)
│   ├── ANALYSE_AMELIORATIONS.md      # Technical improvements analysis
│   └── ameliorations_code.py          # Reusable improvement functions
│
└── ⚙️ Configuration
    └── requirements.txt               # Python dependencies
```

**⭐ = Recommended/Improved versions**

---

## ✨ Key Features

### 🎯 Simplified Input System

**Only 6 critical features required** for predictions:

1. **tenure** - Months with company
2. **MonthlyCharges** - Monthly billing amount
3. **TotalCharges** - Total charges
4. **Contract** - Contract type (Month-to-month/One year/Two year)
5. **InternetService** - Internet service type (DSL/Fiber optic/No)
6. **PaymentMethod** - Payment method

All other fields are optional with sensible defaults.

### 💡 Smart Recommendations Engine

The system automatically generates actionable recommendations:

- **🔴 High Priority**: Offer long-term contracts, new customer retention programs
- **🟡 Medium Priority**: Encourage automatic payments, review pricing
- **🟢 Low Priority**: Maintain service quality

Each recommendation includes:
- Priority level
- Specific action to take
- Description and rationale
- Expected impact on churn probability

### 📊 Enhanced Visualizations

- Interactive gauge charts for churn probability
- Color-coded risk levels (High/Medium/Low)
- Distribution charts and trend analysis
- Batch prediction summaries with statistics

### 🤖 Optimized ML Pipeline

- **Feature Engineering**: 6 new features created
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Optimal Threshold**: Data-driven decision threshold (not fixed 0.5)
- **Ensemble Methods**: Voting Classifier combining multiple models
- **Business Metrics**: Cost analysis and ROI calculations

---

## 💻 Usage Guide

### Single Customer Prediction

1. Open the dashboard at http://localhost:8501
2. Select "🔍 Single Prediction" mode
3. Fill in the **6 critical fields**:
   - Tenure (months)
   - Monthly Charges ($)
   - Total Charges ($)
   - Contract Type
   - Internet Service
   - Payment Method
4. Click "🚀 Predict Churn"
5. View results with:
   - Churn probability and risk level
   - Key risk factors identified
   - Actionable recommendations

### Batch Prediction (Mini Dataset)

**Option 1: Upload CSV**
```csv
tenure,MonthlyCharges,TotalCharges,Contract,InternetService,PaymentMethod
12,70.0,1000.0,Month-to-month,DSL,Electronic check
24,50.0,1200.0,One year,Fiber optic,Bank transfer (automatic)
```

**Option 2: Manual Entry**
- Enter customer data directly in the dashboard
- Support for multiple customers at once

**Results Include:**
- Individual predictions for each customer
- Summary statistics (high/medium/low risk counts)
- Top recommendations across all customers
- Downloadable CSV with all predictions

### Analytics Dashboard

- View prediction history
- Analyze trends over time
- Risk level distributions
- Feature analysis by contract type, service, etc.

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "optimal_threshold": 0.45
}
```

#### 2. Single Prediction (Simplified)

```http
POST /predict
Content-Type: application/json
```

**Request (Only 6 fields required):**
```json
{
  "tenure": 12,
  "MonthlyCharges": 70.0,
  "TotalCharges": 1000.0,
  "Contract": "Month-to-month",
  "InternetService": "DSL",
  "PaymentMethod": "Electronic check"
}
```

**Response:**
```json
{
  "churn_prediction": true,
  "churn_probability": 0.75,
  "risk_level": "High",
  "confidence": "Very confident",
  "key_factors": [
    "Month-to-month contract (high churn risk)",
    "New customer (≤12 months tenure)",
    "Electronic check payment method"
  ],
  "recommendations": [
    {
      "priority": "High",
      "action": "Offer Long-term Contract",
      "description": "Offer a 1-year or 2-year contract with discount. Month-to-month contracts have 42.7% churn rate.",
      "expected_impact": "Reduce churn probability by 30-40%"
    },
    {
      "priority": "High",
      "action": "New Customer Retention Program",
      "description": "New customers (≤12 months) are at highest risk. Implement welcome program and check-in calls.",
      "expected_impact": "Reduce churn probability by 25-35%"
    }
  ]
}
```

#### 3. Full Input Prediction

```http
POST /predict_full
Content-Type: application/json
```

For predictions with all 18 fields (backward compatibility).

#### 4. Batch Prediction

```http
POST /batch_predict
Content-Type: application/json
```

**Request:**
```json
[
  {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 1000.0,
    "Contract": "Month-to-month",
    "InternetService": "DSL",
    "PaymentMethod": "Electronic check"
  },
  {
    "tenure": 24,
    "MonthlyCharges": 50.0,
    "TotalCharges": 1200.0,
    "Contract": "One year",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Bank transfer (automatic)"
  }
]
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_id": 1,
      "churn_prediction": true,
      "churn_probability": 0.75,
      "risk_level": "High",
      "recommendations_count": 2,
      "key_factors": ["Month-to-month contract", "New customer"]
    }
  ],
  "summary": {
    "total_customers": 2,
    "high_risk_count": 1,
    "medium_risk_count": 0,
    "low_risk_count": 1,
    "average_churn_probability": 0.45,
    "predicted_churn_count": 1,
    "top_recommendations": {
      "Offer Long-term Contract": 1,
      "New Customer Retention Program": 1
    }
  }
}
```

#### 5. Model Information

```http
GET /model_info
```

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "features_used": 36,
  "optimal_threshold": 0.45,
  "critical_features": [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod"
  ]
}
```

### Risk Level Classification

- **🔴 High Risk** (>70% probability): Immediate action required
- **🟡 Medium Risk** (30-70% probability): Monitor closely
- **🟢 Low Risk** (<30% probability): Standard service

---

## 🔬 Machine Learning Pipeline

### Dataset

- **Source**: Telco Customer Churn Dataset
- **Size**: 7,043 customers
- **Features**: 21 original features
- **Target**: Binary classification (Churn: Yes/No)
- **Churn Rate**: 26.54%

### Preprocessing (Improved)

1. **Data Cleaning**
   - Handle missing values in `TotalCharges` (filled with 0 for new customers)
   - Type conversion and validation

2. **Feature Engineering** ⭐
   - `MonthlyCharges_to_TotalCharges`: Ratio for spending efficiency
   - `AvgMonthlyCharges`: Average charges per month
   - `TenureGroup`: Customer segmentation (New/Regular/Loyal/VeryLoyal)
   - `ServiceCount`: Number of active services
   - `ContractValue`: Contract value (monthly charges × duration)
   - `HighRiskProfile`: Composite risk indicator

3. **Encoding & Scaling** ⭐
   - **Corrected Order**: Scaling BEFORE SMOTE (critical fix)
   - One-hot encoding for categorical variables
   - StandardScaler for numerical features

4. **Class Imbalance** ⭐
   - SMOTE applied on normalized data
   - Balanced training set: 4,139 samples per class

### Model Training (Improved)

**Algorithms Tested:**
1. Logistic Regression (baseline)
2. Random Forest (with GridSearchCV) ⭐
3. XGBoost (with GridSearchCV) ⭐

**Hyperparameter Tuning** ⭐:
- GridSearchCV with 5-fold cross-validation
- Optimized for ROC-AUC score
- Best parameters selected for each algorithm

**Ensemble Method** ⭐:
- Voting Classifier combining all three models
- Soft voting using probability predictions

### Model Selection

**Best Model**: Random Forest (or Ensemble if better)

**Performance Metrics:**
- Cross-validation AUC: ~0.93
- Test set AUC: ~0.82-0.85
- Optimized threshold: ~0.45 (instead of 0.5) ⭐

### Model Evaluation

**Comprehensive Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Confusion Matrix
- Business metrics (cost, ROI) ⭐

---

## 📈 Model Performance

### Expected Performance (Improved Model)

| Metric | Original | Improved | Improvement |
|--------|----------|----------|-------------|
| **AUC-ROC** | 82.05% | 85-87% | +3-5% |
| **Recall** | 59.89% | 70-75% | +10-15% |
| **F1-Score** | 58.26% | 65-70% | +7-12% |
| **Threshold** | 0.5 (fixed) | 0.45 (optimal) | Optimized |

### Key Improvements Impact

1. **Preprocessing Fix**: Scaling before SMOTE improves synthetic sample quality
2. **Feature Engineering**: 6 new features capture additional patterns
3. **Hyperparameter Tuning**: Optimized parameters for better performance
4. **Optimal Threshold**: Better balance between precision and recall
5. **Ensemble Methods**: Combines strengths of multiple models

---

## 💡 Business Insights

### Top Churn Drivers

1. **TotalCharges** (16.63%): Total amount charged
2. **tenure** (15.24%): Customer loyalty duration
3. **MonthlyCharges** (13.42%): Monthly billing amount
4. **PaymentMethod_Electronic check** (10.33%): Payment method
5. **InternetService_Fiber optic** (5.63%): Internet service type

### Key Patterns

**Contract Type Impact:**
- Month-to-month: **42.7% churn rate** ⚠️
- One-year: ~15% churn rate
- Two-year: **2.8% churn rate** ✅

**Internet Service Impact:**
- Fiber optic: **41.9% churn rate** ⚠️
- DSL: 19.0% churn rate
- No internet: **7.4% churn rate** ✅

**Customer Tenure:**
- New customers (≤12 months): Highest risk
- Long-term customers (>24 months): Lowest risk

### Business Recommendations

1. **Focus on Month-to-Month Customers**: Highest churn risk (42.7%)
2. **Improve Fiber Optic Service**: Address quality/price concerns (41.9% churn)
3. **Develop Loyalty Programs**: Incentivize long-term contracts
4. **Proactive Engagement**: Monitor high monthly charge customers
5. **Early Intervention**: Target customers in first year (critical period)

---

## 🛠 Technical Details

### Technologies Used

**Machine Learning:**
- scikit-learn (model training, evaluation)
- XGBoost (gradient boosting)
- imbalanced-learn (SMOTE)
- joblib (model serialization)

**Data Processing:**
- pandas (data manipulation)
- numpy (numerical computations)

**Visualization:**
- matplotlib, seaborn (static charts)
- plotly (interactive visualizations)

**Web Framework:**
- FastAPI (REST API)
- Streamlit (dashboard)
- uvicorn (ASGI server)

**Utilities:**
- requests (HTTP client)
- pydantic (data validation)

### Architecture

```
┌─────────────────┐
│   User Input    │
│  (6 features)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │
│  (Frontend)      │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  FastAPI Server │
│  (Backend)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Model       │
│  (Random Forest)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Predictions +  │
│  Recommendations│
└─────────────────┘
```

### Model Files

**Original Model:**
- `churn_prediction_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`

**Improved Model** ⭐:
- `churn_prediction_model_improved.pkl`
- `scaler_improved.pkl`
- `feature_columns_improved.pkl`
- `optimal_threshold.pkl`

The improved backend automatically loads the improved model if available, otherwise falls back to the original.

---

## 📚 Documentation

### Available Guides

1. **README.md** (this file) - Complete project overview
2. **GUIDE_UTILISATION_IMPROVED.md** - Detailed usage guide in French
3. **ANALYSE_AMELIORATIONS.md** - Technical analysis of improvements
4. **ameliorations_code.py** - Reusable improvement functions

### API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🎓 Learning Resources

### For Understanding the ML Pipeline

1. **ml1.ipynb** - Original pipeline (educational)
2. **ml1_improved.ipynb** ⭐ - Optimized pipeline with all improvements

### Key Concepts Demonstrated

- Data preprocessing and feature engineering
- Handling class imbalance with SMOTE
- Hyperparameter tuning with GridSearchCV
- Model evaluation and selection
- Threshold optimization
- Ensemble methods
- Model deployment with FastAPI
- Interactive dashboards with Streamlit

---

## 🔄 Version Comparison

| Feature | Original (v1.0) | Improved (v2.0) |
|---------|-----------------|-----------------|
| **Input Fields** | 18 required | 6 required ⭐ |
| **Recommendations** | ❌ None | ✅ Automatic ⭐ |
| **Risk Factors** | ❌ None | ✅ Identified ⭐ |
| **Preprocessing** | Standard | Optimized ⭐ |
| **Hyperparameter Tuning** | ❌ Default | ✅ GridSearchCV ⭐ |
| **Feature Engineering** | ❌ None | ✅ 6 new features ⭐ |
| **Threshold** | 0.5 fixed | Optimal (data-driven) ⭐ |
| **Ensemble Methods** | ❌ None | ✅ Voting Classifier ⭐ |
| **Business Metrics** | ❌ None | ✅ Cost/ROI ⭐ |
| **Visualizations** | Basic | Enhanced ⭐ |

---

## 🚧 Future Enhancements

### Planned Improvements

- [ ] Real-time data integration
- [ ] Automated retraining pipeline
- [ ] Model monitoring and drift detection
- [ ] Customer segmentation analysis
- [ ] Deep learning models (neural networks)
- [ ] Docker containerization
- [ ] Database integration
- [ ] Authentication and authorization
- [ ] Advanced filtering and search
- [ ] Export to multiple formats (Excel, PDF)

### Completed Improvements ✅

- [x] Hyperparameter tuning
- [x] Feature engineering
- [x] Ensemble methods
- [x] Optimal threshold optimization
- [x] Retention strategy recommendations
- [x] Cost-benefit analysis
- [x] Simplified input interface
- [x] Enhanced visualizations
- [x] Recommendations display

---

## 📝 Notes & Limitations

### Model Limitations

- Trained on historical data - may not capture future trends
- Assumes similar customer behavior patterns
- Requires retraining with new data periodically
- Results specific to telecom industry

### Best Practices

- Always validate input data before prediction
- Monitor model performance over time
- Retrain model with new data regularly
- Use predictions as one input among many business factors
- Adjust business costs in recommendations based on your context

---

## 👥 Contributing

This is a learning project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is for educational purposes. The dataset is publicly available.

---

## 🙏 Acknowledgments

- **Dataset**: Telco Customer Churn Dataset
- **Libraries**: scikit-learn, FastAPI, Streamlit communities
- **Machine Learning**: Concepts and best practices

---

## 📧 Support

For questions or feedback:
- Check the documentation files
- Review the API documentation at `/docs`
- Open an issue for bugs or feature requests

---

<div align="center">

**Built with ❤️ using Python, FastAPI, Streamlit, and Scikit-learn**

**Version 2.0.0** | **Last Updated**: 2024

[⬆ Back to Top](#-customer-churn-prediction-system)

</div>
#   M L _ p r o j e c t _ 1  
 