"""
LangGraph agent that diagnoses a series, picks a forecasting model, and says why.

The whole point of the ModelCard on BaseForecaster is that the choice of model is
explainable: each card states what its model is good for, whether it tolerates
NaNs, and how much history it needs. This agent is the consumer of that — it
reads the diagnosis, reads the cards, and reasons about the match rather than
following a hardcoded rule table.

The graph is the standard two-node ReAct loop:

    START → agent ⇄ tools → END

`agent` calls the LLM; `tools_condition` routes to `tools` whenever the reply
carries tool calls, and to END when it doesn't. The loop runs until the model
stops reaching for tools, which is its way of saying it has an answer.

Tools close over one series rather than taking it as an argument: a DataFrame
cannot survive a round trip through a JSON tool call, and passing an opaque
handle would just move the closure somewhere less obvious.

Nothing in this module names a model vendor. The agent needs a chat model that
can call tools; which one, and whether its weights are open, is decided in
llm.py and injectable here for tests.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from diagnosis.pipeline import run_full_diagnosis
from diagnosis.schemas import FullDiagnosisReport
from forecasting.base import BaseForecaster
from forecasting.conformal import ConformalWrapper
from forecasting.ets_model import ETSForecaster
from forecasting.prophet_model import ProphetForecaster
from forecasting.schemas import ForecastResult
from sibyl.agents.llm import bind_tools, build_llm  # noqa: F401  (build_llm re-exported)

# Model registry. Adding a forecaster here is all it takes for the agent to
# consider it — the card supplies the description the LLM reasons over.
_MODELS: dict[str, type[BaseForecaster]] = {
    "prophet": ProphetForecaster,
    "ets":     ETSForecaster,
}

# Tuned against the weakest model the agent supports — a 7B local one — because a
# prompt that holds there holds everywhere. Each rule below exists because the
# first live run broke it: the model skipped straight to an answer, named a
# forecaster that does not exist, ignored a card's sample floor, and reported an
# infinite interval as though it were a result.
_SYSTEM_PROMPT = """You are a time-series forecasting analyst.

Work in this order, and do not skip a step:
1. Call `diagnose`. Never choose a model before you have read its output.
2. Call `list_models`. It is the only source of truth for which models exist and
   what each one needs — never name a model it did not return.
3. Call `run_forecast` exactly once, with the model whose card best fits the
   diagnosis.
4. Only then reply, in two or three sentences: which model you ran, and which
   numbers from the diagnosis led you there.

Reading the cards:
- `minimum samples` is a hard floor. A series shorter than it rules that model
  out, however well the rest of the card fits.
- `handles missing values` matters only when the diagnosis reports some.
- Conformal intervals hold back part of the history to calibrate, so they need
  roughly twice a model's minimum samples. Below that they widen to infinity and
  tell the user nothing — leave `conformal` false when history is short.

You have not answered until `run_forecast` has run. A reply naming a model
without forecasting is a failed answer."""


@dataclass
class AgentRun:
    """Everything the run produced: the answer, the artifacts, and the transcript."""

    explanation: str                              # the model's final prose answer
    diagnosis: FullDiagnosisReport | None = None  # None if the agent skipped diagnose
    forecast: ForecastResult | None = None        # None if it never forecast
    model_used: str | None = None
    messages: list[Any] = field(default_factory=list)   # full transcript, for debugging


# ── tools ─────────────────────────────────────────────────────────────────────

def _build_tools(series: pd.Series, run: AgentRun) -> list[StructuredTool]:
    """Build the three tools, closed over `series`, recording results into `run`.

    Each tool returns a string: tool results re-enter the conversation as text,
    so the LLM reads the same summary a human would.
    """

    def diagnose() -> str:
        """Profile the series and test it for stationarity, seasonality and anomalies.

        Call this first. Returns a six-line summary of what the data looks like.
        """
        # run_full_diagnosis works on a DataFrame with an explicit date column,
        # so rebuild one from the index rather than duplicating its logic here.
        df = pd.DataFrame({
            "date":  series.index.astype(str),
            "value": series.to_numpy(),
        })
        run.diagnosis = run_full_diagnosis(df, target_column="value")
        return run.diagnosis.to_summary()

    def list_models() -> str:
        """List the available forecasting models and what each one is suited to."""
        lines = []
        for name, cls in _MODELS.items():
            card = cls().card
            lines.append(
                f"{name}: {card.description}\n"
                f"  best for: {card.best_for}\n"
                f"  handles missing values: {card.handles_missing}\n"
                f"  minimum samples: {card.min_samples_required}\n"
                f"  uncertainty: {card.supports_uncertainty}"
            )
        lines.append(
            "conformal: any model above can be wrapped in split-conformal intervals, "
            "which are distribution-free and hold in finite samples. Costs one extra "
            "fit and needs roughly twice the history."
        )
        return "\n\n".join(lines)

    def run_forecast(model_name: str, horizon: int, conformal: bool = False) -> str:
        """Fit one model and forecast `horizon` steps ahead.

        Args:
            model_name: one of the names from `list_models`.
            horizon:    number of steps to forecast.
            conformal:  replace the model's native intervals with conformal ones.
        """
        if model_name not in _MODELS:
            # Returned rather than raised: the LLM can correct itself from this,
            # whereas an exception would abort the whole graph.
            return f"Unknown model '{model_name}'. Available: {', '.join(_MODELS)}."

        model: BaseForecaster = _MODELS[model_name]()
        if conformal:
            model = ConformalWrapper(model)

        # The cards advertise a sample floor that no forecaster actually enforces:
        # ProphetForecaster.fit() will fit 60 points against its own stated minimum
        # of 100 without complaint. The first live run showed a model reading that
        # floor, naming it out loud, and fitting anyway — so enforce it here rather
        # than hope the prompt holds for every model anyone plugs in. Returned like
        # the unknown-model case, not raised, so the agent can choose again.
        usable = int(series.notna().sum())
        floor = model.card.min_samples_required
        if usable < floor:
            # Name the models that do fit rather than asking the caller to re-derive
            # them. The registry already knows, and live runs showed a 7B model
            # reasoning its way to the right alternative and then describing the
            # call instead of making it — every step removed is a step it cannot
            # fumble. The closing imperative is aimed at the same weakness.
            fits = sorted(
                name for name, cls in _MODELS.items()
                if cls().card.min_samples_required <= usable
            )
            return (
                f"Refused: {model.card.name} needs at least {floor} usable observations "
                f"and this series has {usable}. "
                + (f"Models that fit: {', '.join(fits)}. " if fits else "No model fits. ")
                + "Call `run_forecast` again now with one of them. Do not answer until it runs."
            )

        model.fit(series)
        result = model.predict(horizon)

        run.forecast = result
        run.model_used = result.model_name

        first, last = result.point_forecast[0], result.point_forecast[-1]
        summary = (
            f"Fitted {result.model_name} in {result.fit_time_seconds}s. "
            f"{horizon}-step forecast runs from {first:.1f} to {last:.1f}, "
            f"with a 95% interval of [{result.lower_95[-1]:.1f}, {result.upper_95[-1]:.1f}] "
            f"at the final step."
        )

        # An infinite interval is the honest answer to too little calibration data
        # — ConformalWrapper widens rather than under-cover — but it is useless to
        # a caller, and a model reading this string will otherwise repeat it as a
        # normal result. Name the problem and the fix instead of leaving "[-inf,
        # inf]" to speak for itself.
        if not np.isfinite([result.lower_95[-1], result.upper_95[-1]]).all():
            summary += (
                " That interval is infinite: the calibration split was too small to bound "
                "this level. Re-run with conformal=false, or use a longer series."
            )
        return summary

    # from_function reads the name, docstring and type hints to build the schema
    # the LLM sees — so the docstrings above are the tool descriptions.
    return [StructuredTool.from_function(f) for f in (diagnose, list_models, run_forecast)]


# ── graph ─────────────────────────────────────────────────────────────────────

def build_graph(llm, tools: list[StructuredTool]):
    """Wire the two-node ReAct loop and compile it."""
    llm_with_tools = bind_tools(llm, tools)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    # tools_condition routes to "tools" when the reply has tool calls, else to END
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_agent(
    series: pd.Series,
    horizon: int = 30,
    llm=None,
    recursion_limit: int = 25,
) -> AgentRun:
    """Diagnose `series`, choose a model, forecast `horizon` steps, and explain why.

    Args:
        series:          observations with a DatetimeIndex — the same contract
                         BaseForecaster.fit() expects.
        horizon:         steps to forecast.
        llm:             any tool-calling chat model, from any provider. Defaults
                         to whatever SIBYL_LLM_PROVIDER / SIBYL_LLM_MODEL select;
                         inject a stub to run the graph without network access.
        recursion_limit: hard ceiling on agent↔tools round trips, so a model that
                         keeps calling tools cannot loop forever.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            "run_agent requires a DatetimeIndex. "
            "Pass series.set_index(<date_column>) before calling."
        )

    llm = llm if llm is not None else build_llm()
    run = AgentRun(explanation="")
    tools = _build_tools(series, run)

    state = build_graph(llm, tools).invoke(
        {"messages": [
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(
                f"Forecast this series {horizon} steps ahead. "
                f"It has {len(series)} observations. Explain your model choice."
            ),
        ]},
        {"recursion_limit": recursion_limit},
    )

    run.messages = state["messages"]

    # The last AI message with no tool calls is the agent's actual answer.
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and not message.tool_calls:
            run.explanation = _text_of(message)
            break

    return run


# Block types that carry a model's private reasoning rather than its answer.
# Vendors each picked their own spelling; the agent must never surface any of them.
_REASONING_BLOCKS = {"thinking", "redacted_thinking", "reasoning", "reasoning_content"}


def _text_of(message: AIMessage) -> str:
    """Flatten message content to plain text, whatever shape the provider chose.

    Providers disagree here. Most return a plain string. Ones with reasoning
    turned on return a list of typed blocks, where the reasoning blocks are not
    the answer and must be dropped.
    """
    if isinstance(message.content, str):
        return message.content

    blocks = [b for b in message.content if isinstance(b, dict)]
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    if text:
        return text

    # Nothing announced itself as a text block. Rather than return an empty
    # answer, accept any block carrying text — except the reasoning ones, which
    # are the whole reason this function exists.
    return "".join(
        b.get("text", "") for b in blocks if b.get("type") not in _REASONING_BLOCKS
    ).strip()
