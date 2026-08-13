import importlib
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import rest_server.main as main_module
from rest_server.features.ask_patra.models import AskPatraStarter
from tests.conftest import _make_mock_pool


@asynccontextmanager
async def _no_op_lifespan(_app):
    yield


@pytest.fixture
def ask_patra_app(monkeypatch):
    """Reload rest_server.main with ENABLE_ASK_PATRA=true so the router is
    registered. is_ask_patra_enabled() is checked at module-import time in
    main.py, so exercising the flag requires reloading the module rather than
    monkeypatching the function directly -- same pattern as
    tests/test_hf_import_api.py's hf_import_app fixture. Reloading only
    reassigns rest_server.main.app going forward; other test files' own
    bound references to the pre-reload app object are unaffected.
    """
    from rest_server.database import get_pool

    monkeypatch.setenv("ENABLE_ASK_PATRA", "true")
    importlib.reload(main_module)
    main_module.app.router.lifespan_context = _no_op_lifespan
    main_module.app.dependency_overrides[get_pool] = lambda: _make_mock_pool()
    yield main_module.app
    monkeypatch.setenv("ENABLE_ASK_PATRA", "false")
    importlib.reload(main_module)


def _answer_question_result(model_used=None, citations=None, messages=None, starters=None):
    return (
        "conv-123",
        "This is the answer.",
        model_used,
        citations or [],
        messages or [],
        starters or [],
    )


# ---------------------------------------------------------------------------
# GET /api/ask-patra/bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_returns_provider_and_starters(ask_patra_app):
    starters = [AskPatraStarter(title="Find models", prompt="Find models about cats")]
    with patch("rest_server.routes.ask_patra.ensure_ask_patra_storage", return_value=starters), patch(
        "rest_server.routes.ask_patra._provider_label", return_value="PATRA AI"
    ):
        with TestClient(ask_patra_app) as client:
            response = client.get("/api/ask-patra/bootstrap")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["provider"] == "PATRA AI"
    assert body["starter_prompts"] == [{"title": "Find models", "prompt": "Find models about cats"}]


def test_bootstrap_reflects_litellm_provider_label(ask_patra_app):
    with patch("rest_server.routes.ask_patra.ensure_ask_patra_storage", return_value=[]), patch(
        "rest_server.routes.ask_patra._provider_label", return_value="SambaNova via LiteLLM"
    ):
        with TestClient(ask_patra_app) as client:
            response = client.get("/api/ask-patra/bootstrap")

    assert response.json()["provider"] == "SambaNova via LiteLLM"


# ---------------------------------------------------------------------------
# POST /api/ask-patra/chat
# ---------------------------------------------------------------------------

def test_chat_success_mode_code_fallback_when_no_model_used(ask_patra_app):
    with patch(
        "rest_server.routes.ask_patra.answer_question", return_value=_answer_question_result(model_used=None)
    ), patch("rest_server.routes.ask_patra._provider_label", return_value="PATRA AI (code fallback)"):
        with TestClient(ask_patra_app) as client:
            response = client.post("/api/ask-patra/chat", json={"message": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "code_fallback"
    assert body["model_used"] is None
    assert body["conversation_id"] == "conv-123"
    assert body["answer"] == "This is the answer."


def test_chat_success_mode_llm_when_model_used(ask_patra_app):
    with patch(
        "rest_server.routes.ask_patra.answer_question",
        return_value=_answer_question_result(model_used="gemma-2"),
    ), patch("rest_server.routes.ask_patra._provider_label", return_value="PATRA AI"):
        with TestClient(ask_patra_app) as client:
            response = client.post("/api/ask-patra/chat", json={"message": "find model cards about cats"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "llm"
    assert body["model_used"] == "gemma-2"


def test_chat_forwards_tapis_token_header_to_service(ask_patra_app):
    captured_kwargs = {}

    async def fake_answer_question(conn, **kwargs):
        captured_kwargs.update(kwargs)
        return _answer_question_result()

    with patch("rest_server.routes.ask_patra.answer_question", side_effect=fake_answer_question):
        with TestClient(ask_patra_app) as client:
            response = client.post(
                "/api/ask-patra/chat",
                json={"message": "hi"},
                headers={"X-Tapis-Token": "  my-token  "},
            )

    assert response.status_code == 200
    assert captured_kwargs["request_tapis_token"] == "my-token"


def test_chat_missing_tapis_token_passes_none(ask_patra_app):
    captured_kwargs = {}

    async def fake_answer_question(conn, **kwargs):
        captured_kwargs.update(kwargs)
        return _answer_question_result()

    with patch("rest_server.routes.ask_patra.answer_question", side_effect=fake_answer_question):
        with TestClient(ask_patra_app) as client:
            response = client.post("/api/ask-patra/chat", json={"message": "hi"})

    assert response.status_code == 200
    assert captured_kwargs["request_tapis_token"] is None


def test_chat_passes_conversation_id_and_reset_through(ask_patra_app):
    captured_kwargs = {}

    async def fake_answer_question(conn, **kwargs):
        captured_kwargs.update(kwargs)
        return _answer_question_result()

    with patch("rest_server.routes.ask_patra.answer_question", side_effect=fake_answer_question):
        with TestClient(ask_patra_app) as client:
            response = client.post(
                "/api/ask-patra/chat",
                json={"message": "hi", "conversation_id": "existing-convo", "reset": True},
            )

    assert response.status_code == 200
    assert captured_kwargs["conversation_id"] == "existing-convo"
    assert captured_kwargs["reset"] is True


def test_chat_rejects_empty_message(ask_patra_app):
    with TestClient(ask_patra_app) as client:
        response = client.post("/api/ask-patra/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_rejects_unknown_fields(ask_patra_app):
    with TestClient(ask_patra_app) as client:
        response = client.post("/api/ask-patra/chat", json={"message": "hi", "unexpected_field": "value"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Flag-disabled behavior
# ---------------------------------------------------------------------------

def test_ask_patra_routes_absent_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_ASK_PATRA", "false")
    importlib.reload(main_module)
    main_module.app.router.lifespan_context = _no_op_lifespan

    with TestClient(main_module.app) as client:
        response = client.get("/api/ask-patra/bootstrap")

    assert response.status_code == 404
