"""
First live run of the forecaster agent against a real model — any real model.

Everything so far has run against a stub, which tests the graph but says nothing
about whether the system prompt actually elicits the reasoning we want. This
script finds out, and because the agent is provider-agnostic it finds out for
whichever model you point it at:

    SIBYL_LLM_PROVIDER=anthropic SIBYL_LLM_MODEL=claude-opus-5   python run_agent_live.py
    SIBYL_LLM_PROVIDER=openai    SIBYL_LLM_MODEL=gpt-5           python run_agent_live.py
    SIBYL_LLM_PROVIDER=ollama    SIBYL_LLM_MODEL=llama3.1:8b     python run_agent_live.py

The last one costs nothing and needs no account, which makes it the cheapest way
to shake out prompt problems before spending anything on a hosted model.

Two scenarios, because one happy path only proves the plumbing works:

  A. 500 daily points, weekly cycle, a handful of NaNs.
     ETS's card says handles_missing=False, prophet's says True. A correct
     agent picks prophet and *says the NaNs are why*.

  B. 60 monthly points, trend only, no NaNs.
     Prophet's card says min_samples_required=100, ETS's says 24. The series is
     below prophet's floor. Note that floor is advisory — ProphetForecaster.fit()
     never enforces it — so the agent is the only thing honouring the card here.

They discriminate on different card fields, so between them they distinguish
"reads the cards" from "picks the name it has seen most often".
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sibyl.agents.forecaster_agent import run_agent  # noqa: E402

RNG = np.random.default_rng(0)


def series_with_missing() -> pd.Series:
    """500 daily points, weekly cycle, 5 NaNs punched in at fixed positions."""
    n = 500
    t = np.arange(n)
    y = 100.0 + 0.1 * t + 6.0 * np.sin(2 * np.pi * t / 7) + RNG.normal(0, 1.5, n)
    y[[97, 143, 201, 322, 410]] = np.nan  # fixed indices so runs stay comparable
    return pd.Series(y, index=pd.date_range("2023-01-01", periods=n, freq="D"))


def series_too_short_for_prophet() -> pd.Series:
    """60 monthly points — above ETS's floor of 24, below prophet's of 100."""
    n = 60
    t = np.arange(n)
    y = 500.0 + 2.5 * t + RNG.normal(0, 8.0, n)
    return pd.Series(y, index=pd.date_range("2021-01-31", periods=n, freq="ME"))


def token_usage(messages) -> tuple[int, int]:
    """Sum input and output tokens across the transcript.

    usage_metadata is LangChain's provider-neutral accounting, so this works the
    same whether the model was hosted or running on the machine next to you.
    Deliberately no dollar figure: every vendor prices differently, and a local
    model does not price at all, so tokens are the only honest common unit.
    """
    tokens_in = tokens_out = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        tokens_in += usage.get("input_tokens", 0)
        tokens_out += usage.get("output_tokens", 0)
    return tokens_in, tokens_out


def report(label: str, expected: str, series: pd.Series, horizon: int) -> tuple[int, int]:
    print(f"\n{'=' * 74}\n{label}\n  expecting: {expected}\n{'=' * 74}")

    run = run_agent(series, horizon=horizon)

    print(f"\nmodel_used:  {run.model_used}")
    print(f"diagnosis:   {'yes' if run.diagnosis else 'MISSING — agent skipped diagnose'}")
    print(f"forecast:    {'yes' if run.forecast else 'MISSING — agent never forecast'}")
    print(f"\nexplanation:\n{run.explanation}\n")

    # The transcript is the point: it shows what order the agent called tools in
    # and what it saw, which is what prompt tuning actually acts on.
    print("-" * 74)
    print("transcript:")
    for message in run.messages:
        kind = type(message).__name__
        calls = getattr(message, "tool_calls", None)
        if calls:
            for call in calls:
                print(f"  {kind:<15} → {call['name']}({call['args']})")
            continue
        content = getattr(message, "content", "")
        if isinstance(content, list):  # reasoning models return typed blocks
            content = " ".join(
                b.get("text", f"<{b.get('type')}>") for b in content if isinstance(b, dict)
            )
        print(f"  {kind:<15}   {str(content)[:160].replace(chr(10), ' ')}")

    tokens_in, tokens_out = token_usage(run.messages)
    print(f"\ntokens: {tokens_in} in / {tokens_out} out")
    return tokens_in, tokens_out


if __name__ == "__main__":
    provider = os.environ.get("SIBYL_LLM_PROVIDER")
    model = os.environ.get("SIBYL_LLM_MODEL")
    if not provider or not model:
        sys.exit(
            "Set SIBYL_LLM_PROVIDER and SIBYL_LLM_MODEL first. There is no default —\n"
            "  hosted: SIBYL_LLM_PROVIDER=anthropic SIBYL_LLM_MODEL=claude-opus-5\n"
            "  local:  SIBYL_LLM_PROVIDER=ollama    SIBYL_LLM_MODEL=llama3.1:8b\n"
            "and install the matching extra, e.g. pip install -e '.[agents,ollama]'."
        )

    print(f"provider: {provider}    model: {model}")

    totals = [
        report(
            "A — 500 daily points, weekly cycle, 5 missing values",
            "prophet, justified by the NaNs (ETS's card says it cannot handle them)",
            series_with_missing(),
            horizon=30,
        ),
        report(
            "B — 60 monthly points, trend only, no missing values",
            "ets, justified by length (60 < prophet's advisory floor of 100)",
            series_too_short_for_prophet(),
            horizon=12,
        ),
    ]
    print(f"\n{'=' * 74}")
    print(f"total: {sum(t[0] for t in totals)} in / {sum(t[1] for t in totals)} out")
