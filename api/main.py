# main.py
# ================================================
# FastAPI Prediction Endpoint
# Employee Performance Analytics
#
# USAGE:
#   uvicorn api.main:app --reload --port 8000
# DOCS:
#   http://localhost:8000/docs
# ================================================

import os
import sys
import json
import time
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---- Setup paths ------------------------------------
API_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ---- Import schemas ---------------------------------
from api.schemas import (
    EmployeeInput,
    PredictionResponse,
    BatchInput,
    BatchResponse,
    HealthResponse,
)

# ---- Logging ----------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# ---- Load model artifacts at startup ----------------
MODEL_PATH    = PROJECT_ROOT / 'models' / \
                'xgb_performance_model.joblib'
PIPELINE_PATH = PROJECT_ROOT / 'models' / \
                'preprocessing_pipeline.joblib'
FEAT_PATH     = PROJECT_ROOT / 'models' / \
                'feature_cols.json'
META_PATH     = PROJECT_ROOT / 'models' / \
                'model_metadata.json'

model         = None
pipeline      = None
feature_cols  = None
model_version = "xgb_v1.0"


def load_artifacts():
    global model, pipeline, feature_cols, model_version

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        log.info("Model loaded successfully")
    else:
        log.warning(f"Model not found: {MODEL_PATH}")

    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        log.info("Pipeline loaded successfully")
    else:
        log.warning(f"Pipeline not found: {PIPELINE_PATH}")

    if FEAT_PATH.exists():
        with open(FEAT_PATH) as f:
            feature_cols = json.load(f)
        log.info(f"Feature cols loaded: {len(feature_cols)}")

    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        model_version = meta.get('model_type', 'xgb_v1.0')


load_artifacts()

# ---- Create FastAPI app -----------------------------
app = FastAPI(
    title       = "Employee Performance Prediction API",
    description = (
        "Predicts high performer probability and "
        "promotion readiness for HR planning.\n\n"
        "Model: XGBoost Classifier\n"
        "Target: High Performer (Rating >= 4.0)"
    ),
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# =====================================================
# HELPER FUNCTION 1 - BUILD DATAFRAME WITH FEATURES
# =====================================================
def employee_to_dataframe(emp: EmployeeInput) -> pd.DataFrame:
    """
    Convert EmployeeInput to DataFrame AND
    calculate all engineered features from Phase 5.
    """

    # ── Raw fields ────────────────────────────────────────────
    data = {
        'Department'              : emp.Department,
        'JobLevel'                : emp.JobLevel,
        'Age'                     : emp.Age,
        'Gender'                  : emp.Gender,
        'EducationLevel'          : emp.EducationLevel,
        'LocationType'            : emp.LocationType,
        'YearsAtCompany'          : emp.YearsAtCompany,
        'YearsSinceLastPromotion' : emp.YearsSinceLastPromotion,
        'HistoricalRatingAvg'     : emp.HistoricalRatingAvg,
        'Overall360Score'         : emp.Overall360Score
                                    or emp.HistoricalRatingAvg,
        'SelfRating'              : emp.SelfRating
                                    or emp.HistoricalRatingAvg,
        'PeerAvgRating'           : emp.PeerAvgRating
                                    or emp.HistoricalRatingAvg,
        'SubordinateAvgRating'    : None,
        'SelfOtherGap'            : round(
            (emp.SelfRating or 3.0) -
            (emp.PeerAvgRating or 3.0), 2
        ),
        'Leadership360'           : emp.Overall360Score or 3.5,
        'Collaboration360'        : emp.PeerAvgRating or 3.5,
        'OKRCompletionPct'        : emp.OKRCompletionPct,
        'NumOKRsAssigned'         : emp.NumOKRsAssigned,
        'WeightedGoalAttainment'  : emp.WeightedGoalAttainment,
        'EngagementScore'         : emp.EngagementScore,
        'JobSatisfaction'         : emp.JobSatisfaction,
        'WorkLifeBalanceRating'   : emp.WorkLifeBalanceRating,
        'BurnoutRisk'             : emp.BurnoutRisk,
        'TrainingHoursLastYear'   : emp.TrainingHoursLastYear,
        'OvertimeHoursMonthly'    : emp.OvertimeHoursMonthly,
        'AbsenteeismDays'         : emp.AbsenteeismDays,
        'AvgMonthlyHours'         : emp.AvgMonthlyHours,
        'ProjectsHandled'         : emp.ProjectsHandled,
        'HighPotentialFlag'       : emp.HighPotentialFlag,
        'PIPHistoryFlag'          : emp.PIPHistoryFlag,
        'CalibrationAdjustedFlag' : emp.CalibrationAdjustedFlag,
    }

    df = pd.DataFrame([data])

    # ── Engineered features (matching Phase 5) ────────────────

    # Feature 1: Performance Trend
    df['PerformanceTrend'] = 0.0

    # Feature 2: OKR Efficiency Score
    df['OKREfficiencyScore'] = round(
        float(emp.OKRCompletionPct) /
        max(float(emp.NumOKRsAssigned), 1), 2
    )

    # Feature 3: Rating 360 Variance
    ratings = [
        v for v in [
            emp.SelfRating,
            emp.PeerAvgRating,
            emp.Overall360Score
        ] if v is not None
    ]
    if len(ratings) >= 2:
        mean_r   = sum(ratings) / len(ratings)
        variance = (
            sum((r - mean_r) ** 2 for r in ratings) /
            len(ratings)
        ) ** 0.5
        df['Rating360Variance'] = round(variance, 3)
    else:
        df['Rating360Variance'] = 0.3

    # Feature 4: Workload Ratio
    level_benchmarks = {
        'IC1': 160, 'IC2': 168, 'IC3': 175,
        'M1': 185,  'M2': 195,  'Director': 210
    }
    benchmark = level_benchmarks.get(emp.JobLevel, 170)
    df['WorkloadRatio'] = round(
        float(emp.AvgMonthlyHours) / benchmark, 3
    )

    # Feature 5: Promotion Lag Flag
    df['PromotionLagFlag'] = int(
        emp.HighPotentialFlag == 1 and
        emp.YearsSinceLastPromotion >= 3
    )

    # Feature 6: Burnout Workload Index
    burnout_map    = {'Low': 0.2, 'Medium': 0.6, 'High': 1.0}
    burnout_num    = burnout_map.get(emp.BurnoutRisk, 0.5)
    workload_excess = max(
        float(emp.AvgMonthlyHours) / benchmark - 1.0, 0
    )
    workload_norm  = min(workload_excess / 0.5, 1.0)
    df['BurnoutWorkloadIndex'] = round(
        0.6 * burnout_num + 0.4 * workload_norm, 3
    )

    # Feature 7: Training Efficiency
    df['TrainingEfficiency'] = round(
        0.0 / max(float(emp.TrainingHoursLastYear), 1), 4
    )

    # Feature 8: Leadership Gap
    overall_360 = emp.Overall360Score or emp.HistoricalRatingAvg
    leadership  = emp.Overall360Score or 3.5
    df['LeadershipGap'] = round(
        float(leadership) - float(overall_360), 2
    )

    # Feature 9: Promotion Readiness Score
    perf_norm   = (float(emp.HistoricalRatingAvg) - 1) / 4
    okr_norm    = float(emp.OKRCompletionPct) / 100
    tenure_norm = min(float(emp.YearsAtCompany) / 7, 1.0)
    score360    = (float(emp.Overall360Score or 3.0) - 1) / 4
    promo_lag   = min(
        float(emp.YearsSinceLastPromotion) / 5, 1.0
    )
    df['PromotionReadinessScore'] = round(
        0.30 * perf_norm +
        0.20 * score360 +
        0.20 * okr_norm +
        0.15 * float(emp.HighPotentialFlag) +
        0.10 * tenure_norm +
        0.05 * promo_lag,
        3
    )

    # Feature 10: Engagement Performance Alignment
    engagement_norm = float(emp.EngagementScore) / 100
    df['EngagementPerfAlignment'] = round(
        engagement_norm * perf_norm, 3
    )

    log.info(
        f"Features built for {emp.EmployeeID}: "
        f"OKREfficiency={df['OKREfficiencyScore'].iloc[0]}, "
        f"Readiness={df['PromotionReadinessScore'].iloc[0]}"
    )

    return df


# =====================================================
# HELPER FUNCTION 2 - BUILD RESPONSE OBJECT
# =====================================================
def build_prediction_response(
    emp: EmployeeInput,
    probability: float
) -> PredictionResponse:
    """Build the response object from probability score."""

    predicted_hp = bool(probability >= 0.45)

    if probability >= 0.85:
        confidence = "Very High"
    elif probability >= 0.70:
        confidence = "High"
    elif probability >= 0.50:
        confidence = "Moderate"
    elif probability >= 0.30:
        confidence = "Low"
    else:
        confidence = "Very Low"

    readiness_score = (
        probability * 0.6 +
        emp.HighPotentialFlag * 0.2 +
        min(emp.YearsAtCompany / 7, 1.0) * 0.1 +
        (emp.OKRCompletionPct / 100) * 0.1
    )

    if readiness_score >= 0.75:
        readiness = "Ready Now"
    elif readiness_score >= 0.55:
        readiness = "Ready in 6 Months"
    elif readiness_score >= 0.40:
        readiness = "Ready in 12 Months"
    else:
        readiness = "Not Yet Ready"

    key_risks = {
        "FlightRisk"  : bool(
            emp.EngagementScore <= 45 and
            emp.YearsAtCompany >= 2.0
        ),
        "BurnoutRisk" : bool(emp.BurnoutRisk == "High"),
        "PIPRisk"     : bool(
            probability <= 0.25 and
            emp.HistoricalRatingAvg <= 2.5
        ),
    }

    return PredictionResponse(
        EmployeeID                 = emp.EmployeeID,
        HighPerformerProbability   = round(probability, 4),
        PredictedHighPerformer     = predicted_hp,
        ConfidenceBand             = confidence,
        PromotionReadinessCategory = readiness,
        KeyRiskFlags               = key_risks,
        ModelVersion               = model_version,
        Disclaimer                 = (
            "Model-assisted prediction. "
            "Human review required for all talent decisions."
        )
    )


# =====================================================
# HELPER FUNCTION 3 - RUN PREDICTION
# =====================================================
def run_prediction(emp: EmployeeInput) -> float:
    """Run preprocessing and model inference."""

    if model is None or pipeline is None:
        raise HTTPException(
            status_code = 503,
            detail      = (
                "Model or pipeline not loaded. "
                "Run Phase 4 and Phase 7 notebooks first."
            )
        )

    df_input = employee_to_dataframe(emp)

    try:
        X_processed = pipeline.transform(df_input)
        log.info(
            f"Pipeline transform OK: shape {X_processed.shape}"
        )
    except Exception as e:
        log.warning(f"Transform failed: {e}")
        log.info("Trying fit_transform as fallback")
        try:
            X_processed = pipeline.fit_transform(df_input)
        except Exception as e2:
            raise HTTPException(
                status_code = 422,
                detail      = f"Preprocessing failed: {str(e2)}"
            )

    try:
        probability = float(
            model.predict_proba(X_processed)[0, 1]
        )
        log.info(
            f"Prediction OK: probability={probability:.4f}"
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Model prediction failed: {str(e)}"
        )

    return probability


# =====================================================
# ENDPOINT 1 - HEALTH CHECK
# =====================================================
@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["System"],
    summary        = "Check API health and model status"
)
def health_check():
    return HealthResponse(
        status          = "healthy" if model else "degraded",
        model_loaded    = model is not None,
        pipeline_loaded = pipeline is not None,
        model_version   = model_version,
        api_version     = "1.0.0"
    )


# =====================================================
# ENDPOINT 2 - SINGLE PREDICTION
# =====================================================
@app.post(
    "/predict",
    response_model = PredictionResponse,
    tags           = ["Predictions"],
    summary        = "Predict high performer for one employee"
)
def predict_single(employee: EmployeeInput):
    """
    Predict high performer probability for a single employee.

    Business use cases:
    - HR checks an employee before calibration meeting
    - Manager requests readiness score for promotion discussion
    - HRIS system webhooks new data for automatic scoring
    """
    log.info(
        f"Single prediction request: {employee.EmployeeID}"
    )

    probability = run_prediction(employee)
    response    = build_prediction_response(
        employee, probability
    )

    log.info(
        f"Prediction complete: {employee.EmployeeID} "
        f"probability={probability:.3f} "
        f"HP={response.PredictedHighPerformer}"
    )

    return response


# =====================================================
# ENDPOINT 3 - BATCH PREDICTION
# =====================================================
@app.post(
    "/predict/batch",
    response_model = BatchResponse,
    tags           = ["Predictions"],
    summary        = "Predict high performers for multiple employees"
)
def predict_batch(batch: BatchInput):
    """
    Predict high performer probability for multiple employees.

    Business use cases:
    - Monthly scoring of entire workforce
    - Department calibration preparation
    - Succession planning shortlist generation
    """
    log.info(
        f"Batch prediction: {len(batch.employees)} employees"
    )

    start_time  = time.time()
    predictions = []
    hp_count    = 0

    for emp in batch.employees:
        try:
            probability = run_prediction(emp)
            response    = build_prediction_response(
                emp, probability
            )
            predictions.append(response)
            if response.PredictedHighPerformer:
                hp_count += 1

        except HTTPException as e:
            log.warning(
                f"Prediction failed for "
                f"{emp.EmployeeID}: {e.detail}"
            )

    elapsed_ms = round(
        (time.time() - start_time) * 1000, 2
    )

    log.info(
        f"Batch complete: {len(predictions)} predictions, "
        f"{hp_count} HPs, {elapsed_ms}ms"
    )

    return BatchResponse(
        total_employees    = len(batch.employees),
        predicted_hp_count = hp_count,
        predictions        = predictions,
        processing_time_ms = elapsed_ms
    )


# =====================================================
# ENDPOINT 4 - MODEL INFO
# =====================================================
@app.get(
    "/model/info",
    tags    = ["System"],
    summary = "Get model information and feature list"
)
def model_info():
    meta = {}
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)

    return {
        "model_version"     : model_version,
        "model_loaded"      : model is not None,
        "pipeline_loaded"   : pipeline is not None,
        "feature_count"     : len(feature_cols)
                              if feature_cols else 0,
        "decision_threshold": 0.45,
        "target_definition" : "HighPerformer = Rating >= 4.0",
        "metadata"          : meta,
        "disclaimer"        : (
            "All predictions are model-assisted. "
            "Human review required for talent decisions."
        )
    }


# =====================================================
# ENDPOINT 5 - EXAMPLE REQUEST
# =====================================================
@app.get(
    "/example",
    tags    = ["System"],
    summary = "Get an example request payload"
)
def get_example():
    return {
        "EmployeeID"              : "EMP00042",
        "Department"              : "Engineering",
        "JobLevel"                : "IC3",
        "Age"                     : 31,
        "Gender"                  : "Male",
        "EducationLevel"          : "Master's",
        "LocationType"            : "Hybrid",
        "YearsAtCompany"          : 3.5,
        "YearsSinceLastPromotion" : 2.0,
        "HistoricalRatingAvg"     : 3.8,
        "Overall360Score"         : 4.1,
        "SelfRating"              : 4.2,
        "PeerAvgRating"           : 3.9,
        "OKRCompletionPct"        : 88.5,
        "NumOKRsAssigned"         : 5,
        "WeightedGoalAttainment"  : 85.0,
        "EngagementScore"         : 74.0,
        "JobSatisfaction"         : 4.0,
        "WorkLifeBalanceRating"   : 3.8,
        "BurnoutRisk"             : "Low",
        "TrainingHoursLastYear"   : 52,
        "OvertimeHoursMonthly"    : 12.0,
        "AbsenteeismDays"         : 2,
        "AvgMonthlyHours"         : 175,
        "ProjectsHandled"         : 4,
        "HighPotentialFlag"       : 1,
        "PIPHistoryFlag"          : 0,
        "CalibrationAdjustedFlag" : 0
    }
