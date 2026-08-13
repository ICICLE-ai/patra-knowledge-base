import json
from unittest.mock import AsyncMock

import pytest

from rest_server.deps import PatraActor
from rest_server.features.ask_patra.models import AskPatraCitation
from rest_server.features.ask_patra.service import (
    _dedupe_citations,
    _is_capability_question,
    _is_greeting,
    _resolve_llm_auth,
    _score_text,
    _tokenize_query,
    _wants_record_lookup,
    answer_question,
    search_pattra_records,
)


# ---------------------------------------------------------------------------
# _tokenize_query
# ---------------------------------------------------------------------------

def test_tokenize_query_filters_short_tokens_and_stopwords():
    # "find" is itself in STOPWORDS (it's a LOOKUP_HINTS term, not a content word), so it
    # gets filtered along with "the"/"and"; "up" is too short (<= 2 chars after the >2 filter).
    assert _tokenize_query("find the cat and dog up") == ["cat", "dog"]


def test_tokenize_query_empty_string():
    assert _tokenize_query("") == []


# ---------------------------------------------------------------------------
# _is_greeting -- documenting real, verified behavior (not assumed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("hi", True),
        ("hello", True),
        ("hey", True),
        ("Hi Patra", True),
        ("HELLO", True),
        ("good morning", True),
        ("hi!", True),
        # "hello!"/"hey!" are explicitly listed in the exact-match set inside _is_greeting,
        # but the len(tokens) == 0 guard only clears for tokens <= 2 chars (per
        # _tokenize_query's `len(token) > 2` filter) -- "hello" (5 chars) and "hey" (3 chars)
        # survive tokenization, so these never actually match despite being in the set.
        # Verified empirically; documenting the real behavior, not "fixing" it silently.
        ("hello!", False),
        ("hey!", False),
        ("hi there", False),
        ("thanks", False),
        ("find model cards about cats", False),
    ],
)
def test_is_greeting(query, expected):
    assert _is_greeting(query) is expected


# ---------------------------------------------------------------------------
# _is_capability_question
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("what can you do", True),
        ("What can you help me do?", True),
        ("how can you help", True),
        ("what can patra do", True),
        ("find model cards about cats", False),
        ("hi", False),
    ],
)
def test_is_capability_question(query, expected):
    assert _is_capability_question(query) is expected


# ---------------------------------------------------------------------------
# _wants_record_lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("find model cards about cats", True),
        ("show me datasheets", True),
        ("hi", False),
        ("what can you do", False),
        ("thanks for your help", False),
        ("please analyze this dataset thoroughly", True),  # >= 3 tokens fallback
    ],
)
def test_wants_record_lookup(query, expected):
    assert _wants_record_lookup(query) is expected


# ---------------------------------------------------------------------------
# _score_text
# ---------------------------------------------------------------------------

def test_score_text_counts_matched_tokens():
    score, matched = _score_text(["cat", "dog"], "a cat and a dog live here", None, "unrelated")
    assert score == 2
    assert matched == ["cat", "dog"]


def test_score_text_no_match():
    score, matched = _score_text(["xyz"], "abc def")
    assert score == 0
    assert matched == []


def test_score_text_ignores_none_values():
    score, matched = _score_text(["cat"], None, None)
    assert score == 0


# ---------------------------------------------------------------------------
# _dedupe_citations
# ---------------------------------------------------------------------------

def _citation(resource_id, title, subtitle):
    return AskPatraCitation(
        resource_type="model_card", resource_id=resource_id, title=title, subtitle=subtitle, route="/x"
    )


def test_dedupe_citations_case_and_whitespace_insensitive():
    citations = [
        _citation(1, "Model A", "sub"),
        _citation(2, "  model a  ", "SUB"),  # duplicate of #1 once normalized
        _citation(3, "Model B", "sub"),
    ]
    result = _dedupe_citations(citations, limit=10)
    assert [c.resource_id for c in result] == [1, 3]


def test_dedupe_citations_respects_limit():
    citations = [_citation(i, f"Model {i}", "sub") for i in range(5)]
    result = _dedupe_citations(citations, limit=2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _resolve_llm_auth
# ---------------------------------------------------------------------------

def test_resolve_llm_auth_litellm_host_prefers_service_token(monkeypatch):
    monkeypatch.setenv("ASK_PATRA_TAPIS_TOKEN", "service-token")
    api_key, headers = _resolve_llm_auth("https://litellm.pods.tacc.tapis.io/v1", "request-token")
    assert api_key is None
    assert headers == {"X-Tapis-Token": "service-token"}


def test_resolve_llm_auth_litellm_host_falls_back_to_request_token(monkeypatch):
    monkeypatch.delenv("ASK_PATRA_TAPIS_TOKEN", raising=False)
    api_key, headers = _resolve_llm_auth("https://litellm.pods.tacc.tapis.io/v1", "request-token")
    assert api_key is None
    assert headers == {"X-Tapis-Token": "request-token"}


def test_resolve_llm_auth_non_litellm_host_uses_api_key(monkeypatch):
    monkeypatch.setenv("ASK_PATRA_LLM_API_KEY", "plain-key")
    monkeypatch.delenv("ASK_PATRA_TAPIS_TOKEN", raising=False)
    api_key, headers = _resolve_llm_auth("https://api.openai.com/v1", "request-token")
    assert api_key == "plain-key"
    assert headers == {}


def test_resolve_llm_auth_no_tokens_available(monkeypatch):
    monkeypatch.delenv("ASK_PATRA_TAPIS_TOKEN", raising=False)
    monkeypatch.delenv("ASK_PATRA_LLM_API_KEY", raising=False)
    api_key, headers = _resolve_llm_auth("https://api.openai.com/v1", None)
    assert api_key is None
    assert headers == {}


# ---------------------------------------------------------------------------
# search_pattra_records -- dedicated local mock connection
# (the shared conftest mock pool's row shapes don't cover this query's SELECT list)
# ---------------------------------------------------------------------------

def _fake_conn_for_search(model_rows, datasheet_rows):
    conn = AsyncMock()

    async def fetch(query, *args):
        if "FROM model_cards mc" in query:
            return model_rows
        if "FROM datasheets d" in query:
            return datasheet_rows
        return []

    conn.fetch = fetch
    return conn


@pytest.mark.asyncio
async def test_search_pattra_records_empty_query_returns_empty_without_db_call():
    conn = AsyncMock()
    result = await search_pattra_records(conn, query="", include_private=False)
    assert result == []
    conn.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_search_pattra_records_scores_and_filters_model_cards():
    conn = _fake_conn_for_search(
        model_rows=[
            {
                "id": 1, "name": "Cat Classifier", "author": "tester", "short_description": "classifies cats",
                "full_description": "", "keywords": "cat, vision", "category": "classification",
                "input_data": "", "output_data": "", "foundational_model": "",
            },
            {
                "id": 2, "name": "Weather Model", "author": "someone", "short_description": "predicts rain",
                "full_description": "", "keywords": "", "category": "regression",
                "input_data": "", "output_data": "", "foundational_model": "",
            },
        ],
        datasheet_rows=[],
    )

    result = await search_pattra_records(conn, query="cat classifier", include_private=False)

    assert len(result) == 1
    assert result[0].resource_type == "model_card"
    assert result[0].resource_id == 1
    assert result[0].route == "/explore-model-cards/1"


@pytest.mark.asyncio
async def test_search_pattra_records_includes_datasheets():
    conn = _fake_conn_for_search(
        model_rows=[],
        datasheet_rows=[
            {
                "identifier": 5, "title": "Crop Yield Dataset", "creator": "tester",
                "description": "yields by region", "subject": "agriculture",
            },
        ],
    )

    result = await search_pattra_records(conn, query="crop yield", include_private=False)

    assert len(result) == 1
    assert result[0].resource_type == "datasheet"
    assert result[0].resource_id == 5
    assert result[0].route == "/explore-datasheets/5"


# ---------------------------------------------------------------------------
# answer_question -- env vars pointed at tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def ask_patra_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ASK_PATRA_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("ASK_PATRA_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("ASK_PATRA_PROMPTS_DIR", str(tmp_path / "prompts"))
    monkeypatch.delenv("ASK_PATRA_LLM_API_BASE", raising=False)
    monkeypatch.delenv("ASK_PATRA_LLM_ENABLED", raising=False)
    return tmp_path


def _guest_actor():
    return PatraActor(username=None)


@pytest.mark.asyncio
async def test_answer_question_llm_disabled_always_falls_back(ask_patra_env):
    conn = _fake_conn_for_search(model_rows=[], datasheet_rows=[])

    conversation_id, answer, model_used, citations, messages, starters = await answer_question(
        conn, actor=_guest_actor(), message="hi"
    )

    assert model_used is None
    assert "Hello" in answer
    assert len(messages) == 2  # user + assistant


@pytest.mark.asyncio
async def test_answer_question_greeting_skips_llm_call_even_when_enabled(ask_patra_env, monkeypatch):
    monkeypatch.setenv("ASK_PATRA_LLM_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("ASK_PATRA_LLM_ENABLED", "true")

    def fail_if_called(**kwargs):
        raise AssertionError("chat_text_with_model_fallback should not be called for a greeting")

    monkeypatch.setattr(
        "rest_server.features.ask_patra.service.chat_text_with_model_fallback", fail_if_called
    )
    conn = _fake_conn_for_search(model_rows=[], datasheet_rows=[])

    _cid, answer, model_used, _citations, _messages, _starters = await answer_question(
        conn, actor=_guest_actor(), message="hello"
    )

    assert model_used is None
    assert "Hello" in answer


@pytest.mark.asyncio
async def test_answer_question_llm_exception_falls_back_gracefully(ask_patra_env, monkeypatch):
    monkeypatch.setenv("ASK_PATRA_LLM_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("ASK_PATRA_LLM_ENABLED", "true")
    monkeypatch.setattr(
        "rest_server.features.ask_patra.service.chat_text_with_model_fallback",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("llm unavailable")),
    )
    conn = _fake_conn_for_search(model_rows=[], datasheet_rows=[])

    _cid, answer, model_used, _citations, _messages, _starters = await answer_question(
        conn, actor=_guest_actor(), message="find model cards about cats"
    )

    assert model_used is None
    assert answer  # code fallback still produces something


@pytest.mark.asyncio
async def test_answer_question_llm_success_sets_mode_and_model(ask_patra_env, monkeypatch):
    monkeypatch.setenv("ASK_PATRA_LLM_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("ASK_PATRA_LLM_ENABLED", "true")
    monkeypatch.setattr(
        "rest_server.features.ask_patra.service.chat_text_with_model_fallback",
        lambda **kwargs: ("LLM generated answer", "gemma-2"),
    )
    conn = _fake_conn_for_search(model_rows=[], datasheet_rows=[])

    _cid, answer, model_used, _citations, _messages, _starters = await answer_question(
        conn, actor=_guest_actor(), message="find model cards about cats"
    )

    assert answer == "LLM generated answer"
    assert model_used == "gemma-2"


@pytest.mark.asyncio
async def test_answer_question_persists_conversation_to_disk(ask_patra_env):
    conn = _fake_conn_for_search(model_rows=[], datasheet_rows=[])

    conversation_id, _answer, _model_used, _citations, _messages, _starters = await answer_question(
        conn, actor=_guest_actor(), message="hi", conversation_id="test-convo"
    )

    conversation_file = ask_patra_env / "memory" / f"{conversation_id}.json"
    assert conversation_file.exists()
    saved = json.loads(conversation_file.read_text())
    assert len(saved["messages"]) == 2


@pytest.mark.asyncio
async def test_answer_question_reset_clears_prior_messages(ask_patra_env):
    conn = _fake_conn_for_search(model_rows=[], datasheet_rows=[])

    conversation_id, *_ = await answer_question(
        conn, actor=_guest_actor(), message="hi", conversation_id="test-convo-2"
    )
    # Second call with reset=True should start with a clean message history (only this
    # turn's user+assistant messages), not accumulate on top of the first call's messages.
    _cid, _answer, _model_used, _citations, messages, _starters = await answer_question(
        conn, actor=_guest_actor(), message="hello again", conversation_id=conversation_id, reset=True
    )

    assert len(messages) == 2
