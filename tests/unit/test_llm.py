"""
Tests for provider resolution.

No provider package is installed in CI, and that is deliberate: what needs
pinning down here is the mapping from a provider name to a vendor's constructor,
not any vendor's behaviour. So these tests exercise the pure parts — name
resolution, the per-vendor kwarg spellings, and the error paths — and never
construct a real client.

The environment is scrubbed per test. A developer with a real API key exported
must get the same results as CI, or these tests are measuring their shell.
"""

import pytest

from sibyl.agents.llm import (
    _PROVIDERS,
    Provider,
    _chat_model_class,
    _model_kwargs,
    available_providers,
    bind_tools,
    build_llm,
    resolve_provider,
)

_ENV_VARS = [
    "SIBYL_LLM_PROVIDER", "SIBYL_LLM_MODEL", "SIBYL_LLM_MAX_TOKENS", "SIBYL_LLM_BASE_URL",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove every variable this module reads, so tests cannot inherit a real setup."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ── the registry is coherent ──────────────────────────────────────────────────

def test_registry_covers_closed_and_open_weights():
    assert {"anthropic", "openai", "google"} <= set(available_providers())
    assert {"ollama", "openai_compatible"} <= set(available_providers())


@pytest.mark.parametrize("name", sorted(_PROVIDERS))
def test_every_row_is_fully_populated(name):
    spec = _PROVIDERS[name]
    assert isinstance(spec, Provider)
    assert spec.package and spec.class_name and spec.max_tokens_arg and spec.extra


def test_self_hosted_providers_need_no_credential():
    # Ollama talks to a local daemon; requiring a key would be a lie.
    assert _PROVIDERS["ollama"].api_key_env is None


# ── resolving which provider to use ───────────────────────────────────────────

def test_explicit_provider_wins():
    name, spec = resolve_provider("openai")
    assert name == "openai"
    assert spec.class_name == "ChatOpenAI"


def test_provider_read_from_environment(monkeypatch):
    monkeypatch.setenv("SIBYL_LLM_PROVIDER", "groq")
    assert resolve_provider()[0] == "groq"


def test_argument_beats_environment(monkeypatch):
    monkeypatch.setenv("SIBYL_LLM_PROVIDER", "groq")
    assert resolve_provider("mistral")[0] == "mistral"


def test_provider_name_is_normalised():
    assert resolve_provider("  Anthropic  ")[0] == "anthropic"


def test_no_provider_is_an_error_not_a_default():
    # The absence of a default is the feature — no vendor is privileged.
    with pytest.raises(ValueError, match="No model provider selected"):
        resolve_provider()


def test_unknown_provider_lists_the_alternatives():
    with pytest.raises(ValueError, match="Unknown provider") as exc:
        resolve_provider("gpt4all")
    assert "openai_compatible" in str(exc.value)


# ── each vendor spells its arguments differently ──────────────────────────────

def _kwargs(name, **over):
    args = {"model": "m", "max_tokens": None, "base_url": None, "extra": {}} | over
    return _model_kwargs(name, _PROVIDERS[name], **args)


def test_max_tokens_uses_the_vendors_own_name():
    assert _kwargs("anthropic", max_tokens=100)["max_tokens"] == 100
    assert _kwargs("google", max_tokens=100)["max_output_tokens"] == 100
    assert _kwargs("ollama", max_tokens=100)["num_predict"] == 100


def test_max_tokens_omitted_when_unset():
    # Better no ceiling than one this layer invented on the caller's behalf.
    assert not {"max_tokens", "max_output_tokens", "num_predict"} & set(_kwargs("anthropic"))


def test_ollama_gets_a_local_base_url_by_default():
    assert _kwargs("ollama")["base_url"] == "http://localhost:11434"


def test_explicit_base_url_overrides_the_default():
    assert _kwargs("ollama", base_url="http://gpu-box:11434")["base_url"] == "http://gpu-box:11434"


def test_hosted_providers_get_no_base_url():
    assert "base_url" not in _kwargs("anthropic")


def test_openai_compatible_gets_a_placeholder_key():
    # vLLM and llama.cpp ignore the credential, but the OpenAI client insists on one.
    assert _kwargs("openai_compatible")["api_key"] == "not-needed"


def test_real_key_is_left_alone_for_openai_compatible(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert "api_key" not in _kwargs("openai_compatible")


def test_vendor_specific_kwargs_pass_through_untouched():
    # This is the escape hatch for anything the registry is too small to know.
    kwargs = _kwargs("anthropic", extra={"thinking": {"type": "adaptive"}})
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_caller_kwargs_win_over_computed_ones():
    assert _kwargs("ollama", base_url="http://a", extra={"base_url": "http://b"})["base_url"] \
        == "http://b"


# ── failure paths say what to do next ─────────────────────────────────────────

def test_missing_package_names_the_install_command():
    absent = Provider("sibyl_no_such_integration", "ChatNothing", None, "max_tokens", "openai")
    with pytest.raises(ImportError, match=r'pip install -e ".\[agents,openai\]"'):
        _chat_model_class(absent)


def test_missing_model_id_is_an_error():
    with pytest.raises(ValueError, match="No model id"):
        build_llm(provider="anthropic")


def test_missing_api_key_is_caught_before_the_import(monkeypatch):
    # Fails on the credential, not on langchain_anthropic being absent — the
    # actionable error is the one the user can fix without reading a traceback.
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        build_llm(provider="anthropic", model="claude-opus-5")


def test_self_hosted_provider_needs_no_credential(monkeypatch):
    """Ollama must clear the credential gate with no key set.

    Stub the import rather than asserting on a missing-package ImportError: that
    version of this test passed only on machines without langchain_ollama, which
    is passing for the wrong reason. Substituting dict for the model class lets
    the call complete and returns the kwargs for inspection.
    """
    monkeypatch.setattr("sibyl.agents.llm._chat_model_class", lambda spec: dict)
    built = build_llm(provider="ollama", model="llama3.1:8b")
    assert built["model"] == "llama3.1:8b"


def test_hosted_provider_builds_once_its_key_is_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr("sibyl.agents.llm._chat_model_class", lambda spec: dict)
    built = build_llm(provider="anthropic", model="claude-opus-5", max_tokens=8192)
    assert built == {"model": "claude-opus-5", "max_tokens": 8192}


def test_environment_supplies_provider_model_and_cap(monkeypatch):
    monkeypatch.setenv("SIBYL_LLM_PROVIDER", "google")
    monkeypatch.setenv("SIBYL_LLM_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("SIBYL_LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setattr("sibyl.agents.llm._chat_model_class", lambda spec: dict)
    built = build_llm()
    assert built == {"model": "gemini-2.5-pro", "max_output_tokens": 4096}


# ── tool calling is non-negotiable ────────────────────────────────────────────

class _NoToolsModel:
    def bind_tools(self, tools):
        raise NotImplementedError


class _ToolsModel:
    def bind_tools(self, tools):
        return ("bound", tools)


def test_model_without_tool_calling_explains_itself():
    with pytest.raises(NotImplementedError, match="does not support tool calling"):
        bind_tools(_NoToolsModel(), [])


def test_model_with_tool_calling_is_bound_normally():
    assert bind_tools(_ToolsModel(), ["t"]) == ("bound", ["t"])
