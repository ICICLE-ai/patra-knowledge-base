"""POST /users: open self-registration, no auth required."""


def test_register_user_succeeds(client):
    resp = client.post("/users", json={"username": "newlab_member"})
    assert resp.status_code == 200
    assert resp.json() == {"username": "newlab_member"}


def test_register_user_idempotent(client):
    first = client.post("/users", json={"username": "newlab_member"})
    second = client.post("/users", json={"username": "newlab_member"})
    assert first.status_code == 200
    assert second.status_code == 200


def test_register_user_rejects_empty_username(client):
    resp = client.post("/users", json={"username": ""})
    assert resp.status_code == 400


def test_register_user_rejects_missing_body(client):
    resp = client.post("/users")
    assert resp.status_code == 422
