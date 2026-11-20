import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from cachetools import TTLCache


app = FastAPI(
    title="Loan Eligibility & Repayment Risk API",
    description="Two-Stage Decision Support System for Loan Officers with Cost Optimization & Fairness Auditing.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_cache = TTLCache(maxsize=128, ttl=600)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
model_s1 = joblib.load(os.path.join(MODELS_DIR, "stage1_approval_model.joblib"))
scaler_s1 = joblib.load(os.path.join(MODELS_DIR, "stage1_scaler.joblib"))
features_s1 = joblib.load(os.path.join(MODELS_DIR, "stage1_features.joblib"))

model_s2 = joblib.load(os.path.join(MODELS_DIR, "stage2_repayment_model.joblib"))
scaler_s2 = joblib.load(os.path.join(MODELS_DIR, "stage2_scaler.joblib"))
features_s2 = joblib.load(os.path.join(MODELS_DIR, "stage2_features.joblib"))
test_data_s2 = joblib.load(os.path.join(MODELS_DIR, "stage2_test_data.joblib"))
pricing_model = joblib.load(os.path.join(MODELS_DIR, "stage2_pricing_ridge_model.joblib"))


class LoanApplicationRequest(BaseModel):
    loan_amnt: float = Field(..., gt=0, example=15000)
    fico_range_low: float = Field(..., ge=300, le=850, example=710)
    dti: float = Field(..., ge=0, example=16.5)
    emp_length: float = Field(..., ge=0, le=10, example=5.0)
    addr_state: str = Field("CA", example="CA")
    term: int = Field(36, example=36)
    int_rate: float = Field(11.5, gt=0, example=11.5)
    installment: float = Field(495.0, gt=0, example=495.0)
    grade: str = Field("B", example="B")
    home_ownership: str = Field("MORTGAGE", example="MORTGAGE")
    annual_inc: float = Field(85000.0, gt=0, example=85000.0)
    verification_status: str = Field("Verified", example="Verified")
    purpose: str = Field("debt_consolidation", example="debt_consolidation")
    open_acc: float = Field(10.0, ge=0, example=10.0)
    revol_bal: float = Field(12000.0, ge=0, example=12000.0)
    revol_util: float = Field(42.0, ge=0, le=100, example=42.0)
    occupation_category: str = Field("Tech_Engineering", example="Tech_Engineering")
    decision_threshold: float = Field(0.50, ge=0.05, le=0.95, example=0
