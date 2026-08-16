"""
Tests for the LangGraph forecasting agent.

The LLM is stubbed. A real call to any provider would make these tests
non-deterministic, slow, and dependent on that vendor's credential being set in
CI — none of which tells us anything about the graph. What we actually want to
know is that the wiring holds: tools fire in the order the model asks for,
results land in AgentRun, and the loop terminates. StubChatModel replays a fixed
script of AIMessages to pin all of that down.

The stub is also the honest model of what the agent requires. It implements the
chat interface and bind_tools and nothing else, which is exactly the contract
llm.py hands over — so anything passing here works on any provider that meets it.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("langgraph", reason="agent layer requires the [agents] extra")

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from sibyl.agents.forecaster_agent import AgentRun, run_agent  # noqa: E402

RNG = np.random.default_rng(11)


class StubChatModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per invocation.

    bind_tools() returns self: the stub's replies are scripted, so the tool
    schemas it would otherwise be given have no effect on what it says.
    """

    replies: list[AIMessage]
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1

        # Every reply needs a fresh id. LangGraph's add_messages reducer merges by
        # id, so replaying one message object would overwrite the previous turn
        # instead of appending — the transcript would silently stop growing.
        fresh = reply.model_copy(update={
            "id": f"stub-{self.calls}",
            "tool_calls": [
                {**call, "id": f"{call['id']}-{self.calls}"} for call in reply.tool_calls
            ],
        })
        return ChatResult(generations=[ChatGeneration(message=fresh)])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "stub"


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture(scope="module")
def series() -> pd.Series:
    """Two years of daily data with a weekly cycle — enough for every stage."""
    n = 500
    t = np.arange(n)
    y = 100.0 + 0.1 * t + 6.0 * np.sin(2 * np.pi * t / 7) + RNG.normal(0, 1.5, n)
    return pd.Series(y, index=pd.date_range("2023-01-01", periods=n, freq="D"))


@pytest.fixture(scope="module")
def full_run(series) -> AgentRun:
    """The happy path: diagnose → list_models → run_forecast → explain."""
    llm = StubChatModel(replies=[
        _tool_call("diagnose", {}, "c1"),
        _tool_call("list_models", {}, "c2"),
        _tool_call("run_forecast", {"model_name": "prophet", "horizon": 14}, "c3"),
        AIMessage(content="Chose prophet: 500 daily points with a clear weekly cycle."),
    ])
    return run_agent(series, horizon=14, llm=llm)


# ── the loop terminates and returns the answer ────────────────────────────────

def test_returns_agent_run(full_run):
    assert isinstance(full_run, AgentRun)


def test_explanation_is_the_final_message(full_run):
    assert full_run.explanation.startswith("Chose prophet")


def test_transcript_is_retained(full_run):
    # system + human + 3 tool-call turns + 3 tool results + final answer
    assert len(full_run.messages) == 9


# ── each tool's result lands on the run ───────────────────────────────────────

def test_diagnosis_recorded(full_run):
    assert full_run.diagnosis is not None
    assert full_run.diagnosis.target_column == "value"


def test_diagnosis_detects_daily_frequency(full_run):
    assert full_run.diagnosis.profile.detected_frequency == "daily"


def test_diagnosis_finds_weekly_seasonality(full_run):
    assert full_run.diagnosis.seasonality.dominant_period == 7


def test_forecast_recorded(full_run):
    assert full_run.forecast is not None
    assert len(full_run.forecast.point_forecast) == 14


def test_model_used_recorded(full_run):
    assert full_run.model_used == "prophet"


def test_forecast_intervals_ordered(full_run):
    f = full_run.forecast
    assert all(lo <= pt <= hi for lo, pt, hi in zip(f.lower_95, f.point_forecast, f.upper_95))


# ── the agent can wrap a model in conformal intervals ─────────────────────────

def test_conformal_flag_is_honoured(series):
    llm = StubChatModel(replies=[
        _tool_call(
            "run_forecast",
            {"model_name": "ets", "horizon": 7, "conformal": True},
            "c1",
        ),
        AIMessage(content="Used ETS with conformal intervals."),
    ])
    run = run_agent(series, horizon=7, llm=llm)
    assert run.model_used == "conformal(ets)"


# ── the card's sample floor is enforced, not merely advertised ────────────────

@pytest.fixture(scope="module")
def short_series() -> pd.Series:
    """60 monthly points: above ETS's floor of 24, below prophet's of 100."""
    n = 60
    return pd.Series(
        500.0 + 2.5 * np.arange(n) + RNG.normal(0, 8.0, n),
        index=pd.date_range("2021-01-31", periods=n, freq="ME"),
    )


def test_model_below_its_sample_floor_is_refused(short_series):
    # No forecaster enforces its own card, so the tool has to. The first live run
    # showed a model quoting prophet's 100-sample minimum and fitting 60 anyway.
    llm = StubChatModel(replies=[
        _tool_call("run_forecast", {"model_name": "prophet", "horizon": 12}, "c1"),
        AIMessage(content="unused"),
    ])
    run = run_agent(short_series, horizon=12, llm=llm)

    refusal = run.messages[3].content
    assert run.forecast is None
    assert "Refused" in refusal
    assert "at least 100" in refusal
    # The refusal must name the way out, not just the problem.
    assert "Models that fit: ets" in refusal


def test_refusal_is_recoverable(short_series):
    # Same shape as the unknown-model path: the agent is told, and picks again.
    llm = StubChatModel(replies=[
        _tool_call("run_forecast", {"model_name": "prophet", "horizon": 12}, "c1"),
        _tool_call("run_forecast", {"model_name": "ets", "horizon": 12}, "c2"),
        AIMessage(content="Prophet needs 100 points; used ETS."),
    ])
    run = run_agent(short_series, horizon=12, llm=llm)
    assert run.model_used == "ets"


def test_model_above_its_floor_is_fitted(series):
    llm = StubChatModel(replies=[
        _tool_call("run_forecast", {"model_name": "prophet", "horizon": 5}, "c1"),
        AIMessage(content="ok"),
    ])
    assert run_agent(series, horizon=5, llm=llm).model_used == "prophet"


# ── a degenerate interval is called out, not reported as a result ─────────────

def test_infinite_conformal_interval_is_flagged_to_the_model():
    """A short series leaves too few calibration points to bound 95%.

    ConformalWrapper correctly returns infinite width rather than under-covering,
    but "[-inf, inf]" reads like a normal number to a model summarising the tool
    result — the first live run repeated it as though it were a finding. The tool
    now says what went wrong and what to do about it.
    """
    n = 60
    short = pd.Series(
        500.0 + 2.5 * np.arange(n) + RNG.normal(0, 8.0, n),
        index=pd.date_range("2021-01-31", periods=n, freq="ME"),
    )
    llm = StubChatModel(replies=[
        _tool_call("run_forecast", {"model_name": "ets", "horizon": 12, "conformal": True}, "c1"),
        AIMessage(content="done"),
    ])
    run = run_agent(short, horizon=12, llm=llm)

    # system, human, tool-call turn, tool result, answer
    tool_result = run.messages[3].content
    assert "interval is infinite" in tool_result
    assert "conformal=false" in tool_result


# ── a bad tool argument is recoverable, not fatal ──────────────────────────────

def test_unknown_model_returns_error_string_and_agent_recovers(series):
    llm = StubChatModel(replies=[
        _tool_call("run_forecast", {"model_name": "nope", "horizon": 7}, "c1"),
        _tool_call("run_forecast", {"model_name": "ets", "horizon": 7}, "c2"),
        AIMessage(content="First choice was invalid; used ETS instead."),
    ])
    run = run_agent(series, horizon=7, llm=llm)

    # The bad call must not raise — the agent gets told and picks again
    assert run.model_used == "ets"
    assert "Unknown model" in run.messages[3].content


# ── the model may answer without calling anything ─────────────────────────────

def test_agent_can_answer_without_tools(series):
    llm = StubChatModel(replies=[AIMessage(content="No forecast needed.")])
    run = run_agent(series, llm=llm)

    assert run.explanation == "No forecast needed."
    assert run.forecast is None
    assert run.diagnosis is None


# ── input validation ──────────────────────────────────────────────────────────

def test_integer_index_raises():
    llm = StubChatModel(replies=[AIMessage(content="unused")])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        run_agent(pd.Series([1.0, 2.0, 3.0]), llm=llm)


# ── runaway loops are bounded ─────────────────────────────────────────────────

def test_recursion_limit_stops_a_looping_model(series):
    from langgraph.errors import GraphRecursionError

    # A model that only ever calls tools would otherwise loop forever
    llm = StubChatModel(replies=[_tool_call("diagnose", {}, "c1")])
    with pytest.raises(GraphRecursionError):
        run_agent(series, llm=llm, recursion_limit=6)


# ── reasoning blocks are not the answer, whichever vendor emitted them ────────

@pytest.mark.parametrize("reasoning_type", ["thinking", "redacted_thinking", "reasoning"])
def test_reasoning_blocks_stripped_from_explanation(series, reasoning_type):
    # A reasoning model returns a list of blocks rather than a string, and each
    # vendor picked its own name for the private ones. None may reach the user.
    llm = StubChatModel(replies=[AIMessage(content=[
        {"type": reasoning_type, "text": "internal reasoning the user must not see"},
        {"type": "text", "text": "ETS fits this series best."},
    ])])
    run = run_agent(series, llm=llm)

    assert run.explanation == "ETS fits this series best."
    assert "internal reasoning" not in run.explanation


def test_plain_string_content_is_passed_through(series):
    # The common case: a provider with no reasoning blocks at all.
    llm = StubChatModel(replies=[AIMessage(content="Prophet, for the weekly cycle.")])
    assert run_agent(series, llm=llm).explanation == "Prophet, for the weekly cycle."
