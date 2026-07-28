"""POST /users: self-registration, requires an authenticated caller."""

USER_HEADERS = {"X-Tapis-Token": "fake-tapis-access-token-for-testing", "X-Patra-Username": "newlab_member"}


def test_register_user_rejects_unauthenticated(client):
    resp = client.post("/users")
    assert resp.status_code == 401


def test_register_user_rejects_missing_username(client, tapis_headers):
    resp = client.post("/users", headers=tapis_headers)
    assert resp.status_code == 400


def test_register_user_succeeds(client):
    resp = client.post("/users", headers=USER_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"username": "newlab_member"}


def test_register_user_idempotent(client):
    first = client.post("/users", headers=USER_HEADERS)
    second = client.post("/users", headers=USER_HEADERS)
    assert first.status_code == 200
    assert second.status_code == 200
