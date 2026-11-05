import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import public_mode
from api.public_mode import install_public_mode, mask_coordinates
from simulation.dispatch_engine import DispatchManager


PASSWORD = "mật-khẩu-thử"


def auth(password=PASSWORD):
    token = base64.b64encode(f"demo:{password}".encode("utf-8")).decode("ascii")

    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def dist(tmp_path):
    (tmp_path / "index.html").write_text("<html>app</html>", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("js", encoding="utf-8")
    (tmp_path.parent / "secret.txt").write_text("secret", encoding="utf-8")

    return tmp_path


@pytest.fixture
def client(dist):
    app = FastAPI()

    @app.get("/api/point")
    def point():
        return {
            "lat": 10.7744123,
            "route_geometry": [[106.6575123, 10.7744123]],
            "distance_km": 3.14159,
            "name": "Quận 3",
        }

    @app.get("/")
    def root():
        return {"status": "api"}

    assert install_public_mode(app, password=PASSWORD, dist=dist)

    return TestClient(app)


def test_not_installed_without_password(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)

    assert not install_public_mode(FastAPI())


def test_refuses_to_expose_raw_deliveries(monkeypatch):
    monkeypatch.setattr(public_mode, "ANONYMIZE_DELIVERIES", False)

    with pytest.raises(RuntimeError):
        install_public_mode(FastAPI(), password=PASSWORD)


@pytest.mark.parametrize("path", ["/", "/api/point", "/assets/app.js", "/docs"])
def test_every_path_needs_the_password(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers=auth("sai")).status_code == 401


def test_malformed_authorization_is_rejected(client):
    for value in ("Basic !!!", "Bearer abc", "Basic " + base64.b64encode(b"x").decode()):
        response = client.get("/api/point", headers={"Authorization": value})

        assert response.status_code == 401


def test_coordinates_are_rounded_and_other_values_kept(client):
    body = client.get("/api/point", headers=auth()).json()

    assert body["lat"] == 10.774
    assert body["route_geometry"] == [[106.658, 10.774]]
    assert body["distance_km"] == 3.14159
    assert body["name"] == "Quận 3"


def test_mask_leaves_integers_alone():
    assert mask_coordinates({"orders": 10, "code": 106}) == {"orders": 10, "code": 106}


def test_frontend_files_and_client_side_routes(client):
    assert client.get("/assets/app.js", headers=auth()).text == "js"
    assert "app" in client.get("/", headers=auth()).text
    assert "app" in client.get("/dispatch/board", headers=auth()).text


def test_frontend_cannot_escape_dist(client):
    response = client.get("/..%2Fsecret.txt", headers=auth())

    assert "secret" not in response.text


def test_oldest_dispatch_session_is_dropped_at_the_limit(monkeypatch):
    manager = DispatchManager()
    monkeypatch.setattr("simulation.dispatch_engine.MAX_DISPATCH_SESSIONS", 2)

    first = manager.create_session(total_days=1)
    second = manager.create_session(total_days=1)
    third = manager.create_session(total_days=1)

    assert manager.get_session(first.session_id) is None
    assert manager.get_session(second.session_id) is second
    assert manager.get_session(third.session_id) is third
