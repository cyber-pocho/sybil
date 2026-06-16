from enum import Enum
from typing import Optional

from pydantic import BaseModel


class DetectedFrequency(str, Enum):
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    irregular = "irregular"


class DateRange(BaseModel):
    start: str
    end: str


class ColumnStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    skew: float
    kurtosis: float


class ColumnDiagnostic(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    is_numeric: bool
    is_categorical: bool
    stats: Optional[ColumnStats] = None


class DiagnosisReport(BaseModel):
    datetime_column: Optional[str]   # None when datetime lives in the index
    datetime_in_index: bool
    numeric_columns: list[str]
    categorical_columns: list[str]
    row_count: int
    date_range: Optional[DateRange]
    detected_frequency: DetectedFrequency
    duplicate_timestamps: int
    columns: list[ColumnDiagnostic]
