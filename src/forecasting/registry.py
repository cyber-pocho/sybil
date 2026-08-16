"""
The set of forecasters the service knows how to build, by name.

This lived inside the agent as a private `_MODELS` dict, which was fine while the
agent was the only caller. The task queue is a second caller and must not import
the agent to get at it — `sibyl.agents` pulls in LangGraph, and a Celery worker
that only fits a Prophet model has no business requiring an LLM stack.

So the registry sits in the forecasting layer, next to the models it names, and
both callers import it from here. Adding a forecaster is one row.
"""

from forecasting.base import BaseForecaster
from forecasting.conformal import ConformalWrapper
from forecasting.ets_model import ETSForecaster
from forecasting.prophet_model import ProphetForecaster

MODELS: dict[str, type[BaseForecaster]] = {
    "prophet": ProphetForecaster,
    "ets":     ETSForecaster,
}


def build(name: str, conformal: bool = False) -> BaseForecaster:
    """Instantiate a forecaster by name, wrapped in conformal intervals if asked.

    Raises KeyError for an unknown name so each caller can turn that into whatever
    its own contract wants — a tool result the agent can recover from, or a 422.
    """
    if name not in MODELS:
        raise KeyError(name)
    model = MODELS[name]()
    return ConformalWrapper(model) if conformal else model


def models_that_fit(usable_samples: int) -> list[str]:
    """Names whose card's sample floor is satisfied by `usable_samples`.

    Every card states `min_samples_required` and no forecaster enforces it —
    ProphetForecaster.fit() will fit 60 points against its own stated minimum of
    100. Callers use this to enforce what the card only advertises.
    """
    return sorted(
        name for name, cls in MODELS.items()
        if cls().card.min_samples_required <= usable_samples
    )
