import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from cachetools import TTLCache

# 1. Initialize FastAPI App
app = FastAPI(
    title="Loan Eligibility & Repayment Risk API",
    description="Two-Stage Decision Support System for Loan Officers with Cost Optimization & Fairness Auditing.",
    version="1.0.0"
)

# Allow frontend (HTML/JS) to call this API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory TTL Cache (128 items, 10-minute expiry)
api_cache = TTLCache(maxsize=128, ttl=600)

# 2. Load Saved ML Artifacts at Startup
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


# 3. Pydantic Input Schemas (Validates incoming JSON)
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
    decision_threshold: float = Field(0.50, ge=0.05, le=0.95, example=0.50)


class CostMatrixRequest(BaseModel):
    cost_false_approval: float = Field(5.0, gt=0, example=5.0)
    cost_false_rejection: float = Field(1.0, gt=0, example=1.0)


# Helper: Align incoming features to exact trained column order
def build_feature_vector(raw_dict: dict, expected_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([raw_dict])
    return df.reindex(columns=expected_columns, fill_value=0)


# 4. Endpoints (Matches College Guide + Project Features)
@app.get("/")
def root():
    return {"message": "Loan Eligibility & Repayment Risk API is running."}


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict")
def predict_loan(req: LoanApplicationRequest):
    # --- Stage 1: Historical Loan Eligibility ---
    s1_dict = {
        "loan_amnt": req.loan_amnt,
        "fico_range_low": req.fico_range_low,
        "dti": req.dti,
        "emp_length": req.emp_length,
        "dti_to_fico": req.dti / (req.fico_range_low + 1),
        f"addr_state_grouped_{req.addr_state}": 1
    }
    X_s1 = build_feature_vector(s1_dict, features_s1)
    X_s1_scaled = scaler_s1.transform(X_s1)
    approval_prob = float(model_s1.predict_proba(X_s1_scaled)[0][1])
    stage1_approved = approval_prob >= 0.50

    # --- Stage 2: Future Repayment Risk (Includes Occupation) ---
    s2_dict = {
        "loan_amnt": req.loan_amnt,
        "term": req.term,
        "int_rate": req.int_rate,
        "installment": req.installment,
        "emp_length": req.emp_length,
        "annual_inc": req.annual_inc,
        "dti": req.dti,
        "fico_range_low": req.fico_range_low,
        "open_acc": req.open_acc,
        "revol_bal": req.revol_bal,
        "revol_util": req.revol_util,
        "loan_to_income": req.loan_amnt / (req.annual_inc + 1),
        "installment_to_monthly_inc": req.installment / ((req.annual_inc / 12.0) + 1),
        f"grade_{req.grade}": 1,
        f"home_ownership_{req.home_ownership}": 1,
        f"verification_status_{req.verification_status}": 1,
        f"purpose_{req.purpose}": 1,
        f"occupation_category_{req.occupation_category}": 1
    }
    X_s2 = build_feature_vector(s2_dict, features_s2)
    X_s2_scaled = scaler_s2.transform(X_s2)
    default_prob = float(model_s2.predict_proba(X_s2_scaled)[0][1])
    high_risk = default_prob >= req.decision_threshold

    suggested_rate = float(pricing_model.predict(np.array([[req.fico_range_low]]))[0])

    if not stage1_approved:
        decision = "REJECT (Failed Stage 1 Eligibility)"
    elif high_risk:
        decision = "MANUAL REVIEW / REJECT (High Stage 2 Default Risk)"
    else:
        decision = "APPROVE (Eligible & Low Default Risk)"

    return {
        "final_decision": decision,
        "occupation_category": req.occupation_category,
        "stage1_eligibility": {
            "approved": stage1_approved,
            "approval_probability": round(approval_prob, 4)
        },
        "stage2_repayment_risk": {
            "high_default_risk": high_risk,
            "default_probability": round(default_prob, 4),
            "threshold_used": req.decision_threshold,
            "suggested_interest_rate_pct": round(suggested_rate, 2)
        }
    }


@app.post("/optimize-threshold")
def optimize_threshold(req: CostMatrixRequest):
    cache_key = f"cost_{req.cost_false_approval}_{req.cost_false_rejection}"
    if cache_key in api_cache:
        return api_cache[cache_key]

    probs = model_s2.predict_proba(test_data_s2["X_test_scaled"])[:, 1]
    y_test = test_data_s2["y_test"]

    best_thresh, min_cost, curve = 0.50, float("inf"), []
    for t in np.arange(0.10, 0.91, 0.05):
        preds = (probs >= t).astype(int)
        false_approvals = int(np.sum((y_test == 1) & (preds == 0)))
        false_rejections = int(np.sum((y_test == 0) & (preds == 1)))
        total_cost = (false_approvals * req.cost_false_approval) + (false_rejections * req.cost_false_rejection)
        curve.append({"threshold": round(float(t), 2), "total_cost": round(float(total_cost), 2)})
        if total_cost < min_cost:
            min_cost, best_thresh = total_cost, round(float(t), 2)

    result = {"optimal_threshold": best_thresh, "minimum_total_cost": round(float(min_cost), 2), "cost_curve": curve}
    api_cache[cache_key] = result
    return result


@app.get("/fairness-audit/{group_by}")
def fairness_audit(group_by: str = "occupation_category"):
    allowed = ["home_ownership", "grade", "occupation_category"]
    if group_by not in allowed:
        raise HTTPException(status_code=400, detail=f"group_by must be one of {allowed}")

    cache_key = f"fairness_{group_by}"
    if cache_key in api_cache:
        return api_cache[cache_key]

    groups_df = test_data_s2["groups"].copy()
    groups_df["actual"] = test_data_s2["y_test"]
    groups_df["predicted"] = model_s2.predict(test_data_s2["X_test_scaled"])

    report = []
    for val, g in groups_df.groupby(group_by):
        report.append({
            "group": str(val),
            "count": int(len(g)),
            "actual_default_rate": round(float(g["actual"].mean()), 4),
            "predicted_default_rate": round(float(g["predicted"].mean()), 4),
            "error_rate": round(float((g["actual"] != g["predicted"]).mean()), 4)
        })

    result = {"audited_attribute": group_by, "subgroups": report}
    api_cache[cache_key] = result
    return result