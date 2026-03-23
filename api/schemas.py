# schemas.py
# ================================================
# Input and Output data models for the API
# Uses Pydantic for automatic validation
# ================================================

from pydantic import BaseModel, Field
from typing import Optional


class EmployeeInput(BaseModel):
    """
    Input schema for a single employee prediction.
    All fields match our training dataset columns.
    Optional fields have sensible defaults.
    """

    # Identifiers
    EmployeeID: str = Field(
        ...,
        description="Unique employee identifier",
        example="EMP00042"
    )

    # Demographics
    Department: str = Field(
        ...,
        description="Employee department",
        example="Engineering"
    )
    JobLevel: str = Field(
        ...,
        description="Job level (IC1 to Director)",
        example="IC3"
    )
    Age: int = Field(
        ..., ge=18, le=70,
        description="Employee age",
        example=31
    )
    Gender: Optional[str] = Field(
        default="Male",
        description="Gender",
        example="Male"
    )
    EducationLevel: Optional[str] = Field(
        default="Bachelor's",
        description="Highest education level",
        example="Master's"
    )
    LocationType: Optional[str] = Field(
        default="Hybrid",
        description="Work location type",
        example="Hybrid"
    )

    # Tenure
    YearsAtCompany: float = Field(
        ..., ge=0,
        description="Years at company",
        example=3.5
    )
    YearsSinceLastPromotion: float = Field(
        ..., ge=0,
        description="Years since last promotion",
        example=2.0
    )

    # Performance history
    HistoricalRatingAvg: float = Field(
        ..., ge=1, le=5,
        description="Historical performance rating average",
        example=3.8
    )

    # 360 Feedback
    Overall360Score: Optional[float] = Field(
        default=None, ge=1, le=5,
        description="Overall 360 feedback score",
        example=4.1
    )
    SelfRating: Optional[float] = Field(
        default=None, ge=1, le=5,
        description="Self rating score",
        example=4.2
    )
    PeerAvgRating: Optional[float] = Field(
        default=None, ge=1, le=5,
        description="Average peer rating",
        example=3.9
    )

    # OKR
    OKRCompletionPct: float = Field(
        ..., ge=0, le=100,
        description="OKR completion percentage",
        example=88.5
    )
    NumOKRsAssigned: int = Field(
        ..., ge=0,
        description="Number of OKRs assigned",
        example=5
    )
    WeightedGoalAttainment: float = Field(
        ..., ge=0, le=100,
        description="Weighted goal attainment percentage",
        example=85.0
    )

    # Engagement
    EngagementScore: float = Field(
        ..., ge=0, le=100,
        description="Engagement survey score",
        example=74.0
    )
    JobSatisfaction: Optional[float] = Field(
        default=3.5, ge=1, le=5,
        description="Job satisfaction score",
        example=4.0
    )
    WorkLifeBalanceRating: Optional[float] = Field(
        default=3.5, ge=1, le=5,
        description="Work life balance rating",
        example=3.8
    )
    BurnoutRisk: str = Field(
        ...,
        description="Burnout risk level (Low/Medium/High)",
        example="Low"
    )

    # Productivity
    TrainingHoursLastYear: int = Field(
        ..., ge=0,
        description="Training hours completed last year",
        example=52
    )
    OvertimeHoursMonthly: float = Field(
        ..., ge=0,
        description="Average overtime hours per month",
        example=12.0
    )
    AbsenteeismDays: int = Field(
        ..., ge=0,
        description="Number of absent days",
        example=2
    )
    AvgMonthlyHours: int = Field(
        ..., ge=0,
        description="Average monthly hours worked",
        example=175
    )
    ProjectsHandled: int = Field(
        ..., ge=0,
        description="Number of projects handled",
        example=4
    )

    # Flags
    HighPotentialFlag: int = Field(
        default=0, ge=0, le=1,
        description="High potential flag (0 or 1)",
        example=1
    )
    PIPHistoryFlag: int = Field(
        default=0, ge=0, le=1,
        description="PIP history flag (0 or 1)",
        example=0
    )
    CalibrationAdjustedFlag: int = Field(
        default=0, ge=0, le=1,
        description="Calibration adjusted flag (0 or 1)",
        example=0
    )

    class Config:
        json_schema_extra = {
            "example": {
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
                "CalibrationAdjustedFlag" : 0,
            }
        }


class PredictionResponse(BaseModel):
    """
    Output schema returned by the API.
    """
    EmployeeID                : str
    HighPerformerProbability  : float
    PredictedHighPerformer    : bool
    ConfidenceBand            : str
    PromotionReadinessCategory: str
    KeyRiskFlags              : dict
    ModelVersion              : str
    Disclaimer                : str


class BatchInput(BaseModel):
    """
    Input schema for batch predictions.
    Send multiple employees in one request.
    """
    employees: list[EmployeeInput]


class BatchResponse(BaseModel):
    """
    Output schema for batch predictions.
    """
    total_employees      : int
    predicted_hp_count   : int
    predictions          : list[PredictionResponse]
    processing_time_ms   : float


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status        : str
    model_loaded  : bool
    pipeline_loaded: bool
    model_version : str
    api_version   : str
