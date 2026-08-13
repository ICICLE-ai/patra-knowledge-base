import httpx
import pytest

from rest_server.features.shared.openai_compat import (
    build_chat_completions_url,
    build_models_url,
    chat_text,
    chat_text_with_model_fallback,
    extract_message_text,
    is_generation_model,
    list_available_models,
    model_priority,
)


# ---------------------------------------------------------------------------
# build_chat_completions_url / build_models_url
# ---------------------------------------------------------------------------

def test_build_chat_completions_url_already_suffixed():
    assert build_chat_completions_url("https://api.example.com/v1/chat/completions") == (
        "https://api.example.com/v1/chat/completions"
    )


def test_build_chat_completions_url_v1_suffix():
    assert build_chat_completions_url("https://api.example.com/v1") == (
        "https://api.example.com/v1/chat/completions"
    )


def test_build_chat_completions_url_bare_host():
    assert build_chat_completions_url("https://api.example.com") == (
        "https://api.example.com/v1/chat/completions"
    )


def test_build_chat_completions_url_strips_trailing_slash():
    assert build_chat_completions_url("https://api.example.com/v1/") == (
        "https://api.example.com/v1/chat/completions"
    )


def test_build_models_url_already_suffixed():
    assert build_models_url("https://api.example.com/v1/models") == "https://api.example.com/v1/models"


def test_build_models_url_litellm_tapis_host_strips_v1():
    assert build_models_url("https://litellm.pods.tacc.tapis.io/v1") == (
        "https://litellm.pods.tacc.tapis.io/models"
    )


def test_build_models_url_v1_suffix():
    assert build_models_url("https://api.example.com/v1") == "https://api.example.com/v1/models"


def test_build_models_url_bare_host():
    assert build_models_url("https://api.example.com") == "https://api.example.com/v1/models"


# ---------------------------------------------------------------------------
# extract_message_text
# ---------------------------------------------------------------------------

def test_extract_message_text_string_passthrough():
    assert extract_message_text("hello") == "hello"


def test_extract_message_text_list_of_strings():
    assert extract_message_text(["a", "b"]) == "a\nb"


def test_extract_message_text_list_of_dicts():
    assert extract_message_text([{"text": "a"}, {"content": "b"}]) == "a\nb"


def test_extract_message_text_dict_text_key():
    assert extract_message_text({"text": "hello"}) == "hello"


def test_extract_message_text_dict_content_key():
    assert extract_message_text({"content": "hello"}) == "hello"


def test_extract_message_text_unsupported_type_returns_empty():
    assert extract_message_text(42) == ""


def test_extract_message_text_none_returns_empty():
    assert extract_message_text(None) == ""


# ---------------------------------------------------------------------------
# is_generation_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("gpt-4o-mini", True),
        ("text-embedding-3-small", False),
        ("whisper-1", False),
        ("tts-1", False),
        ("rerank-english-v3.0", False),
        ("text-moderation-latest", False),
        ("llama-3.1-70b", True),
    ],
)
def test_is_generation_model(model_id, expected):
    assert is_generation_model(model_id) is expected


# ---------------------------------------------------------------------------
# model_priority
# ---------------------------------------------------------------------------

def test_model_priority_ordering():
    models = ["glm-4", "some-other-model", "qwen2.5", "llama-3.1", "gemma-2"]
    ordered = sorted(models, key=model_priority)
    assert ordered == ["gemma-2", "llama-3.1", "qwen2.5", "glm-4", "some-other-model"]


def test_model_priority_is_case_insensitive():
    assert model_priority("GEMMA-2") == model_priority("gemma-2")


# ---------------------------------------------------------------------------
# list_available_models -- mocked httpx.Client
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


class _FakeClient:
    def __init__(self, get_response=None, post_response=None):
        self._get_response = get_response
        self._post_response = post_response
        self.get_calls = []
        self.post_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, headers=None):
        self.get_calls.append((url, headers))
        return self._get_response

    def post(self, url, headers=None, json=None):
        self.post_calls.append((url, headers, json))
        return self._post_response


def test_list_available_models_filters_and_sorts(monkeypatch):
    fake_response = _FakeResponse(
        {
            "data": [
                {"id": "text-embedding-3-small"},
                {"id": "llama-3.1-70b"},
                {"id": "gemma-2-9b"},
                {"id": "  "},
                {"id": 123},
            ]
        }
    )
    fake_client = _FakeClient(get_response=fake_response)
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.httpx.Client",
        lambda timeout: fake_client,
    )

    result = list_available_models(api_base="https://api.example.com")

    assert result == ["gemma-2-9b", "llama-3.1-70b"]


def test_list_available_models_sends_bearer_token(monkeypatch):
    fake_response = _FakeResponse({"data": []})
    fake_client = _FakeClient(get_response=fake_response)
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.httpx.Client",
        lambda timeout: fake_client,
    )

    list_available_models(api_base="https://api.example.com", api_key="secret-key")

    _url, headers = fake_client.get_calls[0]
    assert headers["Authorization"] == "Bearer secret-key"


def test_list_available_models_no_api_key_omits_auth_header(monkeypatch):
    fake_response = _FakeResponse({"data": []})
    fake_client = _FakeClient(get_response=fake_response)
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.httpx.Client",
        lambda timeout: fake_client,
    )

    list_available_models(api_base="https://api.example.com")

    _url, headers = fake_client.get_calls[0]
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# chat_text -- mocked httpx.Client
# ---------------------------------------------------------------------------

def test_chat_text_returns_content_and_model(monkeypatch):
    fake_response = _FakeResponse(
        {"choices": [{"message": {"content": "  Hello there  "}}]}
    )
    fake_client = _FakeClient(post_response=fake_response)
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.httpx.Client",
        lambda timeout: fake_client,
    )

    content, model = chat_text(
        api_base="https://api.example.com", model="gemma-2", messages=[{"role": "user", "content": "hi"}]
    )

    assert content == "Hello there"
    assert model == "gemma-2"


def test_chat_text_falls_back_to_reasoning_content(monkeypatch):
    fake_response = _FakeResponse(
        {"choices": [{"message": {"content": "", "reasoning_content": "reasoned answer"}}]}
    )
    fake_client = _FakeClient(post_response=fake_response)
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.httpx.Client",
        lambda timeout: fake_client,
    )

    content, _model = chat_text(api_base="https://api.example.com", model="gemma-2", messages=[])

    assert content == "reasoned answer"


def test_chat_text_no_content_or_reasoning_raises_value_error(monkeypatch):
    fake_response = _FakeResponse({"choices": [{"message": {}}]})
    fake_client = _FakeClient(post_response=fake_response)
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.httpx.Client",
        lambda timeout: fake_client,
    )

    with pytest.raises(ValueError):
        chat_text(api_base="https://api.example.com", model="gemma-2", messages=[])


def test_chat_text_sends_content_type_and_payload_shape(monkeypatch):
    fake_response = _FakeResponse({"choices": [{"message": {"content": "ok"}}]})
    fake_client = _FakeClient(post_response=fake_response)
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.httpx.Client",
        lambda timeout: fake_client,
    )

    chat_text(
        api_base="https://api.example.com",
        model="gemma-2",
        messages=[{"role": "user", "content": "hi"}],
        api_key="secret",
    )

    _url, headers, payload = fake_client.post_calls[0]
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer secret"
    assert payload["model"] == "gemma-2"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# chat_text_with_model_fallback
# ---------------------------------------------------------------------------

def test_fallback_uses_explicit_model_without_listing(monkeypatch):
    calls = []

    def fake_chat_text(**kwargs):
        calls.append(kwargs["model"])
        return "answer", kwargs["model"]

    monkeypatch.setattr("rest_server.features.shared.openai_compat.chat_text", fake_chat_text)
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.list_available_models",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not list models when model is given")),
    )

    content, model = chat_text_with_model_fallback(
        api_base="https://api.example.com", model="gemma-2", messages=[]
    )

    assert content == "answer"
    assert model == "gemma-2"
    assert calls == ["gemma-2"]


def test_fallback_lists_models_when_none_given(monkeypatch):
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.list_available_models",
        lambda **kwargs: ["gemma-2", "llama-3.1"],
    )
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.chat_text",
        lambda **kwargs: ("answer", kwargs["model"]),
    )

    content, model = chat_text_with_model_fallback(api_base="https://api.example.com", model=None, messages=[])

    assert content == "answer"
    assert model == "gemma-2"


def test_fallback_tries_next_candidate_on_failure(monkeypatch):
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.list_available_models",
        lambda **kwargs: ["gemma-2", "llama-3.1"],
    )

    def fake_chat_text(**kwargs):
        if kwargs["model"] == "gemma-2":
            raise RuntimeError("model unavailable")
        return "answer", kwargs["model"]

    monkeypatch.setattr("rest_server.features.shared.openai_compat.chat_text", fake_chat_text)

    content, model = chat_text_with_model_fallback(api_base="https://api.example.com", model=None, messages=[])

    assert content == "answer"
    assert model == "llama-3.1"


def test_fallback_all_candidates_fail_raises_runtime_error_with_attempted_models(monkeypatch):
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.list_available_models",
        lambda **kwargs: ["gemma-2", "llama-3.1"],
    )
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.chat_text",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError) as ctx:
        chat_text_with_model_fallback(api_base="https://api.example.com", model=None, messages=[])

    assert "gemma-2" in str(ctx.value)
    assert "llama-3.1" in str(ctx.value)


def test_fallback_no_candidates_available_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(
        "rest_server.features.shared.openai_compat.list_available_models",
        lambda **kwargs: [],
    )

    with pytest.raises(RuntimeError, match="No text-generation models available"):
        chat_text_with_model_fallback(api_base="https://api.example.com", model=None, messages=[])
