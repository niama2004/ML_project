# backend_improved.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List, Optional
import uvicorn

# Load the trained model and preprocessing objects
try:
    # Try to load improved model first, fallback to original
    try:
        model = joblib.load('churn_prediction_model_improved.pkl')
        scaler = joblib.load('scaler_improved.pkl')
        feature_columns = joblib.load('feature_columns_improved.pkl')
        optimal_threshold = joblib.load('optimal_threshold.pkl')
        print("✅ Improved model loaded successfully!")
    except:
        model = joblib.load('churn_prediction_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        optimal_threshold = 0.5
        print("✅ Original model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    feature_columns = []
    optimal_threshold = 0.5

app = FastAPI(
    title="Customer Churn Prediction API - Improved",
    description="Simplified API for predicting customer churn with recommendations",
    version="2.0.0"
)

# ============================================================================
# SIMPLIFIED INPUT MODEL - Only Critical Features
# ============================================================================

class SimpleCustomerData(BaseModel):
    """Simplified customer data with only critical features"""
    # Most important features only
    tenure: int = 12  # Months with company
    MonthlyCharges: float  # Monthly billing amount
    TotalCharges: float  # Total charges
    Contract: str  # Contract type
    InternetService: str  # Internet service type
    PaymentMethod: str  # Payment method
    
    # Optional features (will use defaults if not provided)
    gender: Optional[str] = "Male"
    SeniorCitizen: Optional[int] = 0
    Partner: Optional[str] = "No"
    Dependents: Optional[str] = "No"
    PhoneService: Optional[str] = "Yes"
    MultipleLines: Optional[str] = "No"
    OnlineSecurity: Optional[str] = "No"
    OnlineBackup: Optional[str] = "No"
    DeviceProtection: Optional[str] = "No"
    TechSupport: Optional[str] = "No"
    StreamingTV: Optional[str] = "No"
    StreamingMovies: Optional[str] = "No"
    PaperlessBilling: Optional[str] = "Yes"

# Full model for backward compatibility
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class Recommendation(BaseModel):
    priority: str  # High, Medium, Low
    action: str
    description: str
    expected_impact: str

class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float
    risk_level: str
    confidence: str
    recommendations: List[Recommendation]
    key_factors: List[str]

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    summary: dict

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_recommendations(customer_data: dict, churn_prob: float, risk_level: str) -> List[Recommendation]:
    """Generate actionable recommendations based on customer data and prediction"""
    recommendations = []
    
    # High priority recommendations for high-risk customers
    if risk_level == "High":
        # Contract recommendation
        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append(Recommendation(
                priority="High",
                action="Offer Long-term Contract",
                description="Offer a 1-year or 2-year contract with discount. Month-to-month contracts have 42.7% churn rate.",
                expected_impact="Reduce churn probability by 30-40%"
            ))
        
        # Internet service recommendation
        if customer_data.get('InternetService') == 'Fiber optic':
            recommendations.append(Recommendation(
                priority="High",
                action="Improve Fiber Service Quality",
                description="Fiber optic customers have 41.9% churn rate. Review service quality and pricing.",
                expected_impact="Reduce churn probability by 20-30%"
            ))
        
        # Payment method recommendation
        if 'Electronic check' in customer_data.get('PaymentMethod', ''):
            recommendations.append(Recommendation(
                priority="Medium",
                action="Encourage Automatic Payment",
                description="Electronic check users have higher churn. Offer incentives for automatic payment setup.",
                expected_impact="Reduce churn probability by 10-15%"
            ))
        
        # Tenure recommendation
        if customer_data.get('tenure', 0) <= 12:
            recommendations.append(Recommendation(
                priority="High",
                action="New Customer Retention Program",
                description="New customers (≤12 months) are at highest risk. Implement welcome program and check-in calls.",
                expected_impact="Reduce churn probability by 25-35%"
            ))
        
        # Charges recommendation
        if customer_data.get('MonthlyCharges', 0) > 70:
            recommendations.append(Recommendation(
                priority="Medium",
                action="Review Pricing",
                description="High monthly charges may indicate dissatisfaction. Consider offering a discount or value package.",
                expected_impact="Reduce churn probability by 15-20%"
            ))
    
    # Medium risk recommendations
    elif risk_level == "Medium":
        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append(Recommendation(
                priority="Medium",
                action="Promote Long-term Contract",
                description="Consider offering contract upgrade incentives.",
                expected_impact="Reduce churn probability by 20-30%"
            ))
        
        if customer_data.get('tenure', 0) <= 24:
            recommendations.append(Recommendation(
                priority="Low",
                action="Regular Check-ins",
                description="Schedule periodic check-ins to ensure satisfaction.",
                expected_impact="Reduce churn probability by 10-15%"
            ))
    
    # Low risk - maintenance recommendations
    else:
        recommendations.append(Recommendation(
            priority="Low",
            action="Maintain Service Quality",
            description="Continue providing excellent service to maintain low churn risk.",
            expected_impact="Maintain low churn probability"
        ))
    
    return recommendations

def identify_key_factors(customer_data: dict, churn_prob: float) -> List[str]:
    """Identify key factors contributing to churn risk"""
    factors = []
    
    if customer_data.get('Contract') == 'Month-to-month':
        factors.append("Month-to-month contract (high churn risk)")
    
    if customer_data.get('tenure', 0) <= 12:
        factors.append("New customer (≤12 months tenure)")
    
    if customer_data.get('InternetService') == 'Fiber optic':
        factors.append("Fiber optic service (higher churn rate)")
    
    if 'Electronic check' in customer_data.get('PaymentMethod', ''):
        factors.append("Electronic check payment method")
    
    if customer_data.get('MonthlyCharges', 0) > 70:
        factors.append("High monthly charges (>$70)")
    
    if customer_data.get('TotalCharges', 0) < 500:
        factors.append("Low total charges (new or dissatisfied customer)")
    
    return factors if factors else ["Standard risk factors"]

def preprocess_customer_data(customer_data: dict) -> pd.DataFrame:
    """Preprocess customer data to match training format"""
    # Create DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # Categorical columns for one-hot encoding
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']
    
    # One-hot encode
    df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Ensure all columns from training are present
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match training
    df_encoded = df_encoded[feature_columns]
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # Add new features if they exist
    for col in ['MonthlyCharges_to_TotalCharges', 'AvgMonthlyCharges', 'ServiceCount', 'ContractValue']:
        if col in df_encoded.columns:
            numerical_cols.append(col)
    
    # Only scale columns that exist
    existing_numerical = [col for col in numerical_cols if col in df_encoded.columns]
    if existing_numerical:
        df_encoded[existing_numerical] = scaler.transform(df_encoded[existing_numerical])
    
    return df_encoded

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Customer Churn Prediction API - Improved Version",
        "version": "2.0.0",
        "features": [
            "Simplified input (only critical features required)",
            "Actionable recommendations",
            "Key risk factors identification",
            "Batch prediction with summary"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (simplified input)",
            "predict_full": "/predict_full (full input)",
            "batch_predict": "/batch_predict",
            "model_info": "/model_info"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "optimal_threshold": optimal_threshold
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn_simple(customer_data: SimpleCustomerData):
    """
    Predict churn with simplified input (only critical features required)
    Returns prediction with actionable recommendations
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Convert to dictionary
        customer_dict = customer_data.dict()
        
        # Preprocess
        processed_data = preprocess_customer_data(customer_dict)
        
        # Make prediction
        probability = model.predict_proba(processed_data)[0][1]
        prediction = (probability >= optimal_threshold)
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High"
            confidence = "Very confident"
        elif probability > 0.4:
            risk_level = "Medium"
            confidence = "Moderately confident"
        else:
            risk_level = "Low"
            confidence = "Low confidence"
        
        # Generate recommendations
        recommendations = generate_recommendations(customer_dict, probability, risk_level)
        
        # Identify key factors
        key_factors = identify_key_factors(customer_dict, probability)
        
        return PredictionResponse(
            churn_prediction=bool(prediction),
            churn_probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            recommendations=recommendations,
            key_factors=key_factors
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_full", response_model=PredictionResponse)
async def predict_churn_full(customer_data: CustomerData):
    """
    Predict churn with full input (all features)
    Returns prediction with actionable recommendations
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        customer_dict = customer_data.dict()
        processed_data = preprocess_customer_data(customer_dict)
        
        probability = model.predict_proba(processed_data)[0][1]
        prediction = (probability >= optimal_threshold)
        
        if probability > 0.7:
            risk_level = "High"
            confidence = "Very confident"
        elif probability > 0.4:
            risk_level = "Medium"
            confidence = "Moderately confident"
        else:
            risk_level = "Low"
            confidence = "Low confidence"
        
        recommendations = generate_recommendations(customer_dict, probability, risk_level)
        key_factors = identify_key_factors(customer_dict, probability)
        
        return PredictionResponse(
            churn_prediction=bool(prediction),
            churn_probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            recommendations=recommendations,
            key_factors=key_factors
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(customers_data: List[SimpleCustomerData]):
    """
    Predict churn for multiple customers with simplified input
    Returns predictions with summary statistics
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        predictions = []
        all_recommendations = []
        
        for i, customer_data in enumerate(customers_data):
            customer_dict = customer_data.dict()
            processed_data = preprocess_customer_data(customer_dict)
            
            probability = model.predict_proba(processed_data)[0][1]
            prediction = (probability >= optimal_threshold)
            
            risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
            recommendations = generate_recommendations(customer_dict, probability, risk_level)
            key_factors = identify_key_factors(customer_dict, probability)
            
            pred_result = {
                "customer_id": i + 1,
                "churn_prediction": bool(prediction),
                "churn_probability": probability,
                "risk_level": risk_level,
                "recommendations_count": len(recommendations),
                "key_factors": key_factors
            }
            predictions.append(pred_result)
            all_recommendations.extend([rec.action for rec in recommendations])
        
        # Summary statistics
        predictions_df = pd.DataFrame(predictions)
        summary = {
            "total_customers": len(predictions),
            "high_risk_count": int((predictions_df['risk_level'] == 'High').sum()),
            "medium_risk_count": int((predictions_df['risk_level'] == 'Medium').sum()),
            "low_risk_count": int((predictions_df['risk_level'] == 'Low').sum()),
            "average_churn_probability": float(predictions_df['churn_probability'].mean()),
            "predicted_churn_count": int(predictions_df['churn_prediction'].sum()),
            "top_recommendations": pd.Series(all_recommendations).value_counts().head(5).to_dict()
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the trained model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features_used": len(feature_columns),
        "optimal_threshold": optimal_threshold,
        "critical_features": [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "InternetService",
            "PaymentMethod"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

