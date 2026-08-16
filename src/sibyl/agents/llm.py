"""
Provider-agnostic chat-model construction.

The agent asks exactly two things of a model: that it speaks the LangChain
chat-message interface, and that it can call tools. Everything else — who hosts
it, whether the weights are open, what a token costs — is somebody else's
concern, and none of it belongs in the graph.

So the choice of provider is data, not code. `_PROVIDERS` maps a name to the
four facts that actually differ between vendors:

    package         which pip package supplies the LangChain integration
    class_name      which chat-model class lives inside it
    api_key_env     which variable holds the credential (None if self-hosted)
    max_tokens_arg  what that vendor calls the output cap

Adding a provider is one row here and one line in pyproject. Nothing in
forecaster_agent.py changes, because nothing there knows a vendor exists.

Anything genuinely vendor-specific — Anthropic's `thinking`, OpenAI's
`reasoning_effort`, Ollama's `num_ctx` — is passed straight through as **kwargs
by the caller. That keeps this table small: it holds what every provider has,
not the union of what any provider might have.
"""

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class Provider:
    """The four things that differ between one vendor's chat model and another's."""

    package: str
    class_name: str
    api_key_env: str | None
    max_tokens_arg: str
    extra: str          # the pyproject extra that installs `package`


#                    package, class_name, api_key_env, max_tokens_arg, extra
_PROVIDERS: dict[str, Provider] = {
    # Hosted, closed weights.
    "anthropic": Provider(
        "langchain_anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY", "max_tokens", "anthropic"),
    "openai": Provider(
        "langchain_openai", "ChatOpenAI", "OPENAI_API_KEY", "max_tokens", "openai"),
    "google": Provider(
        "langchain_google_genai", "ChatGoogleGenerativeAI", "GOOGLE_API_KEY",
        "max_output_tokens", "google"),
    "mistral": Provider(
        "langchain_mistralai", "ChatMistralAI", "MISTRAL_API_KEY", "max_tokens", "mistral"),
    "groq": Provider(
        "langchain_groq", "ChatGroq", "GROQ_API_KEY", "max_tokens", "groq"),

    # Open weights. Ollama runs models on your own machine and needs no
    # credential; openai_compatible covers anything exposing an OpenAI-shaped
    # /v1 endpoint — vLLM, llama.cpp's server, LM Studio, TGI, OpenRouter,
    # Together. Both reuse an integration class from above, which is the point:
    # an open-weights model is not a different kind of thing to this agent.
    "ollama": Provider(
        "langchain_ollama", "ChatOllama", None, "num_predict", "ollama"),
    "openai_compatible": Provider(
        "langchain_openai", "ChatOpenAI", "OPENAI_API_KEY", "max_tokens", "openai"),
}

# Self-hosted runtimes still want *something* in the api_key slot, because the
# OpenAI client refuses to construct without one, then never sends it anywhere
# that checks. A placeholder is friendlier than an authentication error from a
# server that does not authenticate.
_PLACEHOLDER_KEY = "not-needed"

_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
}


def available_providers() -> list[str]:
    """Provider names accepted by `build_llm`, for error messages and docs."""
    return sorted(_PROVIDERS)


def resolve_provider(provider: str | None = None) -> tuple[str, Provider]:
    """Pick a provider from the argument, else the environment, else fail loudly.

    There is deliberately no default vendor. A default would make one provider
    the quiet norm and the rest opt-in, which is the exact asymmetry this module
    exists to remove — and it would send a request to a paid endpoint that the
    caller never actually chose.
    """
    name = provider or os.environ.get("SIBYL_LLM_PROVIDER")
    if not name:
        raise ValueError(
            "No model provider selected. Set SIBYL_LLM_PROVIDER (and SIBYL_LLM_MODEL), "
            f"or pass provider= to build_llm. Available: {', '.join(available_providers())}."
        )

    name = name.strip().lower()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Available: {', '.join(available_providers())}. "
            "For any server exposing an OpenAI-shaped /v1 endpoint, use 'openai_compatible'."
        )
    return name, _PROVIDERS[name]


def _model_kwargs(
    name: str,
    spec: Provider,
    model: str,
    max_tokens: int | None,
    base_url: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble constructor kwargs for one provider's chat-model class.

    Split out from `build_llm` so the mapping can be tested without any provider
    package installed — the part that varies between vendors is exactly the part
    worth pinning down in tests.
    """
    kwargs: dict[str, Any] = {"model": model}

    # Every vendor caps output, and every vendor spells it differently. Omit it
    # entirely when unset rather than inventing a ceiling on the caller's behalf.
    if max_tokens is not None:
        kwargs[spec.max_tokens_arg] = max_tokens

    base_url = base_url or _DEFAULT_BASE_URLS.get(name)
    if base_url is not None:
        kwargs["base_url"] = base_url

    if spec.api_key_env and name == "openai_compatible" and not os.environ.get(spec.api_key_env):
        kwargs["api_key"] = _PLACEHOLDER_KEY

    # Caller-supplied vendor extras win: they are the escape hatch for anything
    # this table is too small to know about.
    kwargs.update(extra)
    return kwargs


def _chat_model_class(spec: Provider) -> type:
    """Import the integration class, turning a missing package into install advice."""
    try:
        module = import_module(spec.package)
    except ImportError as exc:
        raise ImportError(
            f"{spec.package} is not installed. Install it with: "
            f'pip install -e ".[agents,{spec.extra}]"'
        ) from exc
    return getattr(module, spec.class_name)


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
    **kwargs: Any,
):
    """Construct a chat model for `provider`, reading defaults from the environment.

    Args:
        provider:   a key of `_PROVIDERS`; falls back to SIBYL_LLM_PROVIDER.
        model:      the vendor's model id; falls back to SIBYL_LLM_MODEL.
        max_tokens: output cap, mapped to whatever this vendor calls it.
        base_url:   endpoint override; falls back to SIBYL_LLM_BASE_URL. Required
                    for openai_compatible, defaulted for ollama, ignored elsewhere.
        **kwargs:   passed to the integration class untouched — this is where
                    vendor-specific settings go, e.g. thinking={"type": "adaptive"}
                    for Anthropic or reasoning_effort="high" for OpenAI.

    Deliberately absent: `temperature`. It is not universal — current Anthropic
    models reject it outright — so it goes through **kwargs like any other
    vendor-specific knob rather than sitting in the signature pretending to be
    portable.
    """
    name, spec = resolve_provider(provider)

    model = model or os.environ.get("SIBYL_LLM_MODEL")
    if not model:
        raise ValueError(
            f"No model id for provider {name!r}. Set SIBYL_LLM_MODEL or pass model=. "
            "The id is the vendor's own, e.g. 'claude-opus-5', 'gpt-5', 'llama3.1:8b'."
        )

    if max_tokens is None and os.environ.get("SIBYL_LLM_MAX_TOKENS"):
        max_tokens = int(os.environ["SIBYL_LLM_MAX_TOKENS"])

    if spec.api_key_env and name != "openai_compatible" and not os.environ.get(spec.api_key_env):
        raise ValueError(
            f"{spec.api_key_env} is not set, which provider {name!r} needs to authenticate. "
            "Set it, or switch to a self-hosted provider ('ollama', 'openai_compatible')."
        )

    cls = _chat_model_class(spec)
    return cls(**_model_kwargs(
        name, spec, model, max_tokens, base_url or os.environ.get("SIBYL_LLM_BASE_URL"), kwargs
    ))


def bind_tools(llm, tools):
    """Bind tools to a model, turning "this one can't" into advice rather than a stack trace.

    Tool calling is the single capability the agent cannot work around — the loop
    *is* the model choosing tools. Plenty of open-weights checkpoints ship without
    it, and LangChain reports that as a bare NotImplementedError from somewhere
    deep in an integration, which is a confusing place for a user to land.
    """
    try:
        return llm.bind_tools(tools)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"{type(llm).__name__} does not support tool calling, which this agent requires: "
            "it works by letting the model choose which tool to call next. Pick a model "
            "variant that advertises tool or function calling."
        ) from exc
