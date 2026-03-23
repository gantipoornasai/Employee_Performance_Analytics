# test_api.py
# Simple test script for the FastAPI endpoint
# Run AFTER starting: uvicorn api.main:app --reload

import sys
print("Script started")
print("Python version:", sys.version)

try:
    import requests
    print("requests imported OK")
except ImportError:
    print("ERROR: requests not installed")
    print("Run: pip install requests")
    sys.exit(1)

try:
    import json
    print("json imported OK")
except ImportError:
    print("ERROR: json not available")
    sys.exit(1)

BASE_URL = "http://localhost:8000"
print(f"Testing API at: {BASE_URL}")
print()


# ── TEST 1: Health Check ──────────────────────────────────────
def test_health():
    print("=" * 50)
    print("TEST 1: Health Check")
    print("=" * 50)

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=10
        )
        print(f"Status Code     : {response.status_code}")
        print(f"Raw Response    : {response.text}")

        if response.status_code == 200:
            data = response.json()
            print(f"API Status      : {data.get('status')}")
            print(f"Model Loaded    : {data.get('model_loaded')}")
            print(f"Pipeline Loaded : {data.get('pipeline_loaded')}")
            print(f"Model Version   : {data.get('model_version')}")
            print("RESULT: PASSED")
        else:
            print("RESULT: FAILED")

    except requests.exceptions.ConnectionError:
        print("RESULT: FAILED")
        print("Cannot connect to API server")
        print("Make sure this is running in another terminal:")
        print("uvicorn api.main:app --reload --port 8000")
        return False

    return True


# ── TEST 2: Single Prediction ─────────────────────────────────
def test_single_prediction():
    print()
    print("=" * 50)
    print("TEST 2: Single Prediction")
    print("=" * 50)

    payload = {
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

    print("Sending request...")
    print(f"Payload keys: {list(payload.keys())}")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json    = payload,
            timeout = 30
        )

        print(f"Status Code : {response.status_code}")
        print(f"Raw Response: {response.text[:800]}")

        if response.status_code == 200:
            data = response.json()
            print()
            print(f"EmployeeID               : {data.get('EmployeeID')}")
            print(f"HP Probability           : {data.get('HighPerformerProbability')}")
            print(f"Predicted HP             : {data.get('PredictedHighPerformer')}")
            print(f"Confidence Band          : {data.get('ConfidenceBand')}")
            print(f"Readiness Category       : {data.get('PromotionReadinessCategory')}")
            print(f"Risk Flags               : {data.get('KeyRiskFlags')}")
            print("RESULT: PASSED")

        elif response.status_code == 422:
            print()
            print("RESULT: FAILED (422 Validation Error)")
            print("The API rejected the input data")
            print("Error details:")
            try:
                error = response.json()
                for detail in error.get('detail', []):
                    print(f"  Field   : {detail.get('loc')}")
                    print(f"  Problem : {detail.get('msg')}")
                    print(f"  Type    : {detail.get('type')}")
                    print()
            except Exception:
                print(response.text)

        else:
            print(f"RESULT: FAILED (status {response.status_code})")

    except requests.exceptions.ConnectionError:
        print("RESULT: FAILED - Cannot connect to server")

    except Exception as e:
        print(f"RESULT: FAILED - Unexpected error: {e}")
        import traceback
        traceback.print_exc()


# ── TEST 3: Batch Prediction ──────────────────────────────────
def test_batch_prediction():
    print()
    print("=" * 50)
    print("TEST 3: Batch Prediction (2 employees)")
    print("=" * 50)

    payload = {
        "employees": [
            {
                "EmployeeID"              : "EMP00001",
                "Department"              : "Engineering",
                "JobLevel"                : "IC3",
                "Age"                     : 31,
                "YearsAtCompany"          : 3.5,
                "YearsSinceLastPromotion" : 2.0,
                "HistoricalRatingAvg"     : 3.8,
                "OKRCompletionPct"        : 88.5,
                "NumOKRsAssigned"         : 5,
                "WeightedGoalAttainment"  : 85.0,
                "EngagementScore"         : 74.0,
                "BurnoutRisk"             : "Low",
                "TrainingHoursLastYear"   : 52,
                "OvertimeHoursMonthly"    : 12.0,
                "AbsenteeismDays"         : 2,
                "AvgMonthlyHours"         : 175,
                "ProjectsHandled"         : 4,
                "HighPotentialFlag"       : 1,
                "PIPHistoryFlag"          : 0,
                "CalibrationAdjustedFlag" : 0
            },
            {
                "EmployeeID"              : "EMP00002",
                "Department"              : "Sales",
                "JobLevel"                : "IC2",
                "Age"                     : 27,
                "YearsAtCompany"          : 1.5,
                "YearsSinceLastPromotion" : 1.5,
                "HistoricalRatingAvg"     : 3.2,
                "OKRCompletionPct"        : 65.0,
                "NumOKRsAssigned"         : 3,
                "WeightedGoalAttainment"  : 62.0,
                "EngagementScore"         : 55.0,
                "BurnoutRisk"             : "Medium",
                "TrainingHoursLastYear"   : 25,
                "OvertimeHoursMonthly"    : 18.0,
                "AbsenteeismDays"         : 5,
                "AvgMonthlyHours"         : 178,
                "ProjectsHandled"         : 3,
                "HighPotentialFlag"       : 0,
                "PIPHistoryFlag"          : 0,
                "CalibrationAdjustedFlag" : 0
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json    = payload,
            timeout = 30
        )

        print(f"Status Code : {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Total Employees      : {data.get('total_employees')}")
            print(f"Predicted HP Count   : {data.get('predicted_hp_count')}")
            print(f"Processing Time (ms) : {data.get('processing_time_ms')}")
            print()
            print("Individual Results:")
            for pred in data.get('predictions', []):
                print(
                    f"  {pred.get('EmployeeID'):<12}"
                    f"  Prob={pred.get('HighPerformerProbability'):.3f}"
                    f"  HP={str(pred.get('PredictedHighPerformer')):<6}"
                    f"  {pred.get('ConfidenceBand'):<12}"
                    f"  {pred.get('PromotionReadinessCategory')}"
                )
            print("RESULT: PASSED")

        else:
            print(f"RESULT: FAILED (status {response.status_code})")
            print(f"Response: {response.text[:500]}")

    except requests.exceptions.ConnectionError:
        print("RESULT: FAILED - Cannot connect to server")

    except Exception as e:
        print(f"RESULT: FAILED - Error: {e}")


# ── TEST 4: Model Info ────────────────────────────────────────
def test_model_info():
    print()
    print("=" * 50)
    print("TEST 4: Model Information")
    print("=" * 50)

    try:
        response = requests.get(
            f"{BASE_URL}/model/info",
            timeout=10
        )
        print(f"Status Code : {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Model Version      : {data.get('model_version')}")
            print(f"Feature Count      : {data.get('feature_count')}")
            print(f"Decision Threshold : {data.get('decision_threshold')}")
            print(f"Target Definition  : {data.get('target_definition')}")
            print("RESULT: PASSED")
        else:
            print(f"RESULT: FAILED")
            print(f"Response: {response.text[:300]}")

    except requests.exceptions.ConnectionError:
        print("RESULT: FAILED - Cannot connect to server")


# ── MAIN ──────────────────────────────────────────────────────
print("Starting tests...")
print()

# Run health check first
# If it fails, skip the rest
health_ok = test_health()

if health_ok is not False:
    test_single_prediction()
    test_batch_prediction()
    test_model_info()

print()
print("=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)
print()
print("Check http://localhost:8000/docs")
print("for interactive API documentation")
