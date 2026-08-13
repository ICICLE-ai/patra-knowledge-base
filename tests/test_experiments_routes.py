import importlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import rest_server.main as main_module
from rest_server.database import get_pool
from rest_server.routes.experiments import _validate_domain


@asynccontextmanager
async def _no_op_lifespan(_app):
    yield


@pytest.fixture
def experiments_app(monkeypatch):
    """Reload rest_server.main with ENABLE_DOMAIN_EXPERIMENTS=true so the router
    is registered -- is_domain_experiments_enabled() is checked at
    module-import time in main.py, same reload requirement as
    tests/test_hf_import_api.py's hf_import_app and
    tests/test_ask_patra_routes.py's ask_patra_app fixtures.
    """
    monkeypatch.setenv("ENABLE_DOMAIN_EXPERIMENTS", "true")
    importlib.reload(main_module)
    main_module.app.router.lifespan_context = _no_op_lifespan
    yield main_module.app
    monkeypatch.setenv("ENABLE_DOMAIN_EXPERIMENTS", "false")
    importlib.reload(main_module)


def _make_pool(fetch_rows=None, fetchrow_row="__unset__"):
    """A local mock pool for experiments.py's query shapes: each test only
    exercises one endpoint, so this dispatches purely on which of
    fetch/fetchrow is called rather than matching SQL text -- the shared
    conftest mock pool's query-shape matching doesn't cover events/
    power_summary tables at all.
    """
    conn = AsyncMock()

    async def fetch(query, *args):
        return fetch_rows if fetch_rows is not None else []

    async def fetchrow(query, *args):
        return None if fetchrow_row == "__unset__" else fetchrow_row

    conn.fetch = fetch
    conn.fetchrow = fetchrow

    pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


TS = datetime(2026, 1, 15, 12, 30, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# _validate_domain
# ---------------------------------------------------------------------------

def test_validate_domain_accepts_known_domain():
    assert _validate_domain("animal-ecology") == "animal-ecology"


def test_validate_domain_rejects_unknown_domain():
    with pytest.raises(HTTPException) as ctx:
        _validate_domain("not-a-real-domain")
    assert ctx.value.status_code == 404
    assert "not-a-real-domain" in ctx.value.detail


def test_unknown_domain_returns_404_through_endpoint(experiments_app):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool()
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/not-a-real-domain/users")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /experiments/{domain}/users
# ---------------------------------------------------------------------------

def test_list_experiment_users_success(experiments_app):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(
        fetch_rows=[{"user_id": "alice", "username": "alice"}]
    )
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/users")
    assert response.status_code == 200
    assert response.json() == [{"user_id": "alice", "username": "alice"}]


# ---------------------------------------------------------------------------
# GET /experiments/{domain}/users/{user_id}/summary
# ---------------------------------------------------------------------------

def test_user_experiment_summary_shapes_row_and_handles_none_metrics(experiments_app):
    row = {
        "experiment_id": "exp-1",
        "user_id": "alice",
        "model_id": "model-1",
        "model_name": "Resnet",
        "device_id": "cam-1",
        "start_at": TS,
        "total_images": 10,
        "saved_images": 4,
        "precision": None,
        "recall": None,
        "f1_score": None,
    }
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetch_rows=[row])
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/users/alice/summary")

    assert response.status_code == 200
    body = response.json()[0]
    assert body["start_at"] == TS.isoformat()
    assert body["saved_images"] == 4
    assert body["precision"] is None
    assert body["recall"] is None
    assert body["f1_score"] is None


def test_user_experiment_summary_start_at_none_when_missing(experiments_app):
    row = {
        "experiment_id": "exp-1",
        "user_id": "alice",
        "model_id": "model-1",
        "model_name": None,
        "device_id": None,
        "start_at": None,
        "total_images": None,
        "saved_images": 0,
        "precision": 0.5,
        "recall": 0.6,
        "f1_score": 0.55,
    }
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetch_rows=[row])
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/users/alice/summary")

    body = response.json()[0]
    assert body["start_at"] is None
    assert body["precision"] == 0.5


# ---------------------------------------------------------------------------
# GET /experiments/{domain}/users/{user_id}/list
# ---------------------------------------------------------------------------

def test_list_user_experiments_success(experiments_app):
    row = {
        "experiment_id": "exp-1",
        "start_at": TS,
        "device_id": "cam-1",
        "model_id": "model-1",
        "model_name": "Resnet",
    }
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetch_rows=[row])
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/users/alice/list")

    assert response.status_code == 200
    body = response.json()[0]
    assert body["experiment_id"] == "exp-1"
    assert body["start_at"] == TS.isoformat()


# ---------------------------------------------------------------------------
# GET /experiments/{domain}/{experiment_id}
# ---------------------------------------------------------------------------

def _experiment_detail_row(**overrides):
    row = {
        "experiment_id": "exp-1",
        "model_id": "model-1",
        "model_name": "Resnet",
        "device_id": "cam-1",
        "image_receiving_timestamp": TS,
        "total_images": 100,
        "total_predictions": 90,
        "total_ground_truth_objects": 95,
        "true_positives": 80,
        "false_positives": 10,
        "false_negatives": 15,
        "precision": 0.8,
        "recall": 0.7,
        "f1_score": 0.75,
        "map_50": 0.9,
        "map_50_95": 0.6,
        "mean_iou": 0.5,
    }
    row.update(overrides)
    return row


def test_get_experiment_detail_found(experiments_app):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(
        fetchrow_row=_experiment_detail_row()
    )
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/exp-1")

    assert response.status_code == 200
    body = response.json()
    assert body["experiment_id"] == "exp-1"
    assert body["start_at"] == TS.isoformat()
    assert body["precision"] == 0.8


def test_get_experiment_detail_not_found_returns_404(experiments_app):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetchrow_row=None)
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/does-not-exist")

    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def test_get_experiment_detail_null_timestamp_and_metrics(experiments_app):
    row = _experiment_detail_row(image_receiving_timestamp=None, precision=None, map_50=None)
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetchrow_row=row)
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/exp-1")

    body = response.json()
    assert body["start_at"] is None
    assert body["precision"] is None
    assert body["map_50"] is None


# ---------------------------------------------------------------------------
# GET /experiments/{domain}/{experiment_id}/images
# ---------------------------------------------------------------------------

def _image_row(**overrides):
    row = {
        "image_name": "img-1.jpg",
        "ground_truth": "deer",
        "label": "deer",
        "probability": 0.95,
        "image_decision": "Save",
        "flattened_scores": "deer:0.95",
        "image_receiving_timestamp": TS,
        "image_scoring_timestamp": TS,
    }
    row.update(overrides)
    return row


def test_get_experiment_images_success(experiments_app):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetch_rows=[_image_row()])
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/exp-1/images")

    assert response.status_code == 200
    body = response.json()[0]
    assert body["image_name"] == "img-1.jpg"
    assert body["probability"] == 0.95
    assert body["image_receiving_timestamp"] == TS.isoformat()


def test_get_experiment_images_null_timestamps(experiments_app):
    row = _image_row(image_receiving_timestamp=None, image_scoring_timestamp=None, probability=None)
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetch_rows=[row])
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/exp-1/images")

    body = response.json()[0]
    assert body["image_receiving_timestamp"] is None
    assert body["image_scoring_timestamp"] is None
    assert body["probability"] is None


@pytest.mark.parametrize("limit", [0, 501])
def test_get_experiment_images_limit_out_of_range_returns_422(experiments_app, limit):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool()
    with TestClient(experiments_app) as client:
        response = client.get(f"/experiments/animal-ecology/exp-1/images?limit={limit}")
    assert response.status_code == 422


def test_get_experiment_images_negative_skip_returns_422(experiments_app):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool()
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/exp-1/images?skip=-1")
    assert response.status_code == 422


def test_get_experiment_images_default_pagination(experiments_app):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetch_rows=[])
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/exp-1/images")
    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# GET /experiments/{domain}/{experiment_id}/power
# ---------------------------------------------------------------------------

def _power_row(**overrides):
    row = {
        "experiment_id": "exp-1",
        "image_generating_plugin_cpu_power_consumption": 1.1,
        "image_generating_plugin_gpu_power_consumption": 2.2,
        "power_monitor_plugin_cpu_power_consumption": 3.3,
        "power_monitor_plugin_gpu_power_consumption": 4.4,
        "image_scoring_plugin_cpu_power_consumption": 5.5,
        "image_scoring_plugin_gpu_power_consumption": 6.6,
        "total_cpu_power_consumption": 9.9,
        "total_gpu_power_consumption": 13.2,
    }
    row.update(overrides)
    return row


def test_get_experiment_power_found(experiments_app):
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetchrow_row=_power_row())
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/exp-1/power")

    assert response.status_code == 200
    body = response.json()
    assert body["experiment_id"] == "exp-1"
    assert body["total_cpu_power_consumption"] == 9.9


def test_get_experiment_power_missing_returns_null_not_404(experiments_app):
    # Documenting real (not "fixed") behavior: unlike /experiments/{domain}/{experiment_id},
    # a missing power_summary row returns HTTP 200 with a JSON null body, not a 404.
    experiments_app.dependency_overrides[get_pool] = lambda: _make_pool(fetchrow_row=None)
    with TestClient(experiments_app) as client:
        response = client.get("/experiments/animal-ecology/exp-1/power")

    assert response.status_code == 200
    assert response.json() is None


# ---------------------------------------------------------------------------
# Flag-disabled behavior
# ---------------------------------------------------------------------------

def test_experiments_routes_absent_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_DOMAIN_EXPERIMENTS", "false")
    importlib.reload(main_module)
    main_module.app.router.lifespan_context = _no_op_lifespan
    main_module.app.dependency_overrides[get_pool] = lambda: _make_pool()

    with TestClient(main_module.app) as client:
        response = client.get("/experiments/animal-ecology/users")

    assert response.status_code == 404
