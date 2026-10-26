from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_deliveries_search_filter_sort():
    r = client.get(
        "/api/deliveries",
        params={"limit": 5, "is_late": True, "sort_by": "weight", "sort_dir": "desc"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    weights = [i["weight"] for i in body["items"] if i["weight"] is not None]
    assert weights == sorted(weights, reverse=True)
    assert all(str(i["isLate"]).lower() == "true" for i in body["items"])


def test_deliveries_have_cluster_id():
    body = client.get("/api/deliveries", params={"limit": 3}).json()
    assert all(i.get("cluster_id") is not None for i in body["items"])


def test_deliveries_cluster_filter():
    body = client.get("/api/deliveries", params={"cluster_id": 1, "limit": 200}).json()
    assert body["total"] == 83
    assert {i["cluster_id"] for i in body["items"]} == {1}


def test_deliveries_rejects_unknown_sort_column():
    assert client.get("/api/deliveries", params={"sort_by": "senderAddress"}).status_code == 422


def test_stability_endpoint():
    body = client.get("/api/results/stability").json()
    assert body["overview"]["random_seeds"] == len(body["seeds"])
    assert "exact_best_route" not in body["overview"]
    assert body["hyperparameters"]


def test_clustering_quality_endpoint():
    body = client.get("/api/results/clustering-quality").json()
    assert any(r["k"] == 10 for r in body["kmeans"])
    assert len(body["dbscan"]) > 0
    assert body["selected"]["kmeans_k"] == 10
