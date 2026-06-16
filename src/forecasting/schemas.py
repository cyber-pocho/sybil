from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ForecastResult(BaseModel):
    timestamps: list[datetime]       # one entry per forecast step
    point_forecast: list[float]      # the model's best guess at each step
    lower_80: list[float]            # 10th percentile — wide enough to be useful in practice
    upper_80: list[float]            # 90th percentile
    lower_95: list[float]            # 2.5th percentile — standard statistical reporting
    upper_95: list[float]            # 97.5th percentile
    model_name: str
    fit_time_seconds: float
    metadata: dict[str, Any] = {}   # model-specific extras (e.g. trend component, ARIMA order)


class ModelCard(BaseModel):
    name: str
    description: str
    best_for: str             # plain-English guidance, e.g. "Short series with strong seasonality"
    handles_missing: bool     # whether the model tolerates NaNs without preprocessing
    min_samples_required: int
    supports_uncertainty: str # "native" | "conformal" | "bootstrap" | "none"
