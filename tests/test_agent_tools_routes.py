"""Route/HTTP-layer tests for /agent-tools/*.

The real internals of patra_agent_service.py depend on a `src/` package that does not
exist anywhere in this repo checkout (confirmed: `from src.hybrid_schema_matcher import
...` etc., no `src/` directory present) -- so these tests mock the 5 imported service
functions directly at the `rest_server.routes.agent_tools` namespace boundary, and only
exercise the route wiring: request validation, response shape, and error mapping.
"""

from unittest.mock import patch

from rest_server.patra_agent_service import AgentServiceError
from rest_server.patra_synthesis_service import SynthesisServiceError


def _schema_pool_item():
    return {
        "dataset_id": "ds-1",
        "title": "Dataset One",
        "source_family": "test",
        "source_url": "https://example.com/ds-1",
        "public_access": "public",
        "task_tags": {},
    }


def _paper_schema_search_response():
    return {
        "status": "ok",
        "message": "done",
        "query_schema": {},
        "extraction": {
            "confidence": "high",
            "rejected": False,
            "rejection_reason": "",
            "source_kind": "pdf",
            "grouped_fields": [],
            "unresolved_fields": [],
            "grouped_schema": {},
            "machine_schema": {},
            "provenance": [],
        },
        "candidate_count": 1,
        "winner_dataset_id": "ds-1",
        "ranking": [],
    }


def _missing_column_response():
    return {
        "status": "ok",
        "message": "done",
        "dataset_id": "ds-1",
        "title": "Dataset One",
        "source_family": "test",
        "source_url": "https://example.com/ds-1",
        "summary": {},
        "rows": [],
    }


def _synthesize_response():
    return {
        "status": "ok",
        "message": "done",
        "artifact": {
            "artifact_key": "artifact-1",
            "title": "Artifact One",
            "source_dataset_id": "ds-1",
            "planner_mode": "deterministic",
            "row_count": 10,
            "generated_fields": [],
            "rejected_fields": [],
            "output_csv_download_url": "https://example.com/out.csv",
            "output_schema_download_url": "https://example.com/out-schema.json",
        },
        "plan": {
            "planner_mode": "deterministic",
            "group_by_fields": [],
            "direct_fields": [],
            "derived_fields": [],
            "rejected_fields": [],
            "planner_notes": [],
        },
        "validation_issues": [],
        "preview_rows": [],
        "storage": {"internal": "detail"},
    }


# ---------------------------------------------------------------------------
# GET /agent-tools/schema-pool
# ---------------------------------------------------------------------------

def test_get_schema_pool_success(client):
    with patch("rest_server.routes.agent_tools.list_schema_pool", return_value=[_schema_pool_item()]):
        response = client.get("/agent-tools/schema-pool")

    assert response.status_code == 200
    assert response.json() == [_schema_pool_item()]


def test_get_schema_pool_agent_service_error_returns_400(client):
    with patch(
        "rest_server.routes.agent_tools.list_schema_pool",
        side_effect=AgentServiceError("schema pool unavailable"),
    ):
        response = client.get("/agent-tools/schema-pool")

    assert response.status_code == 400
    assert response.json()["detail"] == "schema pool unavailable"


# ---------------------------------------------------------------------------
# POST /agent-tools/paper-schema-search
# ---------------------------------------------------------------------------

def test_paper_schema_search_success(client):
    with patch(
        "rest_server.routes.agent_tools.run_paper_schema_search",
        return_value=_paper_schema_search_response(),
    ) as mock_search:
        response = client.post(
            "/agent-tools/paper-schema-search",
            json={"document_url": "https://example.com/paper.pdf", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_search.assert_called_once()


def test_paper_schema_search_agent_service_error_returns_400(client):
    with patch(
        "rest_server.routes.agent_tools.run_paper_schema_search",
        side_effect=AgentServiceError("could not parse paper"),
    ):
        response = client.post("/agent-tools/paper-schema-search", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == "could not parse paper"


# ---------------------------------------------------------------------------
# POST /agent-tools/paper-schema-search-upload
# ---------------------------------------------------------------------------

def test_paper_schema_search_upload_success(client):
    with patch(
        "rest_server.routes.agent_tools.run_uploaded_paper_schema_search",
        return_value=_paper_schema_search_response(),
    ) as mock_search:
        response = client.post(
            "/agent-tools/paper-schema-search-upload",
            files={"file": ("paper.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_search.assert_called_once()


def test_paper_schema_search_upload_empty_file_returns_400_without_calling_service(client):
    with patch("rest_server.routes.agent_tools.run_uploaded_paper_schema_search") as mock_search:
        response = client.post(
            "/agent-tools/paper-schema-search-upload",
            files={"file": ("paper.pdf", b"", "application/pdf")},
        )

    assert response.status_code == 400
    mock_search.assert_not_called()


def test_paper_schema_search_upload_agent_service_error_returns_400(client):
    with patch(
        "rest_server.routes.agent_tools.run_uploaded_paper_schema_search",
        side_effect=AgentServiceError("bad upload"),
    ):
        response = client.post(
            "/agent-tools/paper-schema-search-upload",
            files={"file": ("paper.pdf", b"content", "application/pdf")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad upload"


# ---------------------------------------------------------------------------
# POST /agent-tools/missing-column-analysis
# ---------------------------------------------------------------------------

def test_missing_column_analysis_success(client):
    with patch(
        "rest_server.routes.agent_tools.analyze_missing_columns_for_candidate",
        return_value=_missing_column_response(),
    ) as mock_analyze:
        response = client.post(
            "/agent-tools/missing-column-analysis",
            json={"query_schema": {}, "candidate_dataset_id": "ds-1"},
        )

    assert response.status_code == 200
    assert response.json()["dataset_id"] == "ds-1"
    mock_analyze.assert_called_once()


def test_missing_column_analysis_agent_service_error_returns_400(client):
    with patch(
        "rest_server.routes.agent_tools.analyze_missing_columns_for_candidate",
        side_effect=AgentServiceError("candidate not found"),
    ):
        response = client.post(
            "/agent-tools/missing-column-analysis",
            json={"query_schema": {}, "candidate_dataset_id": "ds-1"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "candidate not found"


# ---------------------------------------------------------------------------
# POST /agent-tools/generate-synthesized-dataset
# ---------------------------------------------------------------------------

def test_generate_synthesized_dataset_success_pops_storage_key(client):
    with patch(
        "rest_server.routes.agent_tools.generate_synthesized_dataset",
        return_value=_synthesize_response(),
    ):
        response = client.post(
            "/agent-tools/generate-synthesized-dataset",
            json={"query_schema": {}, "candidate_dataset_id": "ds-1"},
        )

    assert response.status_code == 200
    assert "storage" not in response.json()


def test_generate_synthesized_dataset_agent_service_error_returns_400(client):
    with patch(
        "rest_server.routes.agent_tools.generate_synthesized_dataset",
        side_effect=AgentServiceError("could not synthesize"),
    ):
        response = client.post(
            "/agent-tools/generate-synthesized-dataset",
            json={"query_schema": {}, "candidate_dataset_id": "ds-1"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "could not synthesize"


def test_generate_synthesized_dataset_synthesis_service_error_returns_400(client):
    with patch(
        "rest_server.routes.agent_tools.generate_synthesized_dataset",
        side_effect=SynthesisServiceError("validation failed"),
    ):
        response = client.post(
            "/agent-tools/generate-synthesized-dataset",
            json={"query_schema": {}, "candidate_dataset_id": "ds-1"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "validation failed"
