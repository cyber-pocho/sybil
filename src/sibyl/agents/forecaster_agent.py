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
"""

from dataclasses import dataclass, field
from typing import Any

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

# Model registry. Adding a forecaster here is all it takes for the agent to
# consider it — the card supplies the description the LLM reasons over.
_MODELS: dict[str, type[BaseForecaster]] = {
    "prophet": ProphetForecaster,
    "ets":     ETSForecaster,
}

_SYSTEM_PROMPT = """You are a time-series forecasting analyst.

Your job, in order:
1. Call `diagnose` to understand the series before touching a model.
2. Call `list_models` to see what is available and what each one is suited to.
3. Call `run_forecast` exactly once, with the model whose card best matches the
   diagnosis. Wrap it in conformal intervals when the model's native intervals
   rest on assumptions the diagnosis contradicts.
4. Reply with two or three sentences: which model you chose and which specific
   findings in the diagnosis led you there. Cite the numbers you saw.

Choose on evidence from the diagnosis, not on model popularity. If the series
has missing values, prefer a model whose card says it handles them."""


@dataclass
class AgentRun:
    """Everything the run produced: the answer, the artifacts, and the transcript."""

    explanation: str                              # the model's final prose answer
    diagnosis: FullDiagnosisReport | None = None  # None if the agent skipped diagnose
    forecast: ForecastResult | None = None        # None if it never forecast
    model_used: str | None = None
    messages: list[Any] = field(default_factory=list)   # full transcript, for debugging


def build_llm(model: str = "claude-opus-5", max_tokens: int = 8192):
    """Claude with adaptive thinking — the model decides how much to think per call.

    Model selection is a judgement call over several competing signals, which is
    exactly the shape of problem adaptive thinking is for. Imported lazily so the
    rest of this module stays importable without the Anthropic dependency.
    """
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
    )


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

        model.fit(series)
        result = model.predict(horizon)

        run.forecast = result
        run.model_used = result.model_name

        first, last = result.point_forecast[0], result.point_forecast[-1]
        return (
            f"Fitted {result.model_name} in {result.fit_time_seconds}s. "
            f"{horizon}-step forecast runs from {first:.1f} to {last:.1f}, "
            f"with a 95% interval of [{result.lower_95[-1]:.1f}, {result.upper_95[-1]:.1f}] "
            f"at the final step."
        )

    # from_function reads the name, docstring and type hints to build the schema
    # the LLM sees — so the docstrings above are the tool descriptions.
    return [StructuredTool.from_function(f) for f in (diagnose, list_models, run_forecast)]


# ── graph ─────────────────────────────────────────────────────────────────────

def build_graph(llm, tools: list[StructuredTool]):
    """Wire the two-node ReAct loop and compile it."""
    llm_with_tools = llm.bind_tools(tools)

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
        llm:             any tool-calling chat model. Defaults to Claude; inject a
                         stub to run the graph without network access.
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


def _text_of(message: AIMessage) -> str:
    """Flatten message content to plain text.

    With thinking enabled the content is a list of blocks rather than a string,
    and the thinking blocks are not the answer — keep only the text ones.
    """
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
