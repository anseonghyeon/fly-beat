from fastapi.testclient import TestClient

from flybeat.api.main import create_app

client = TestClient(create_app())


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_charts_list_and_get():
    ids = {c["id"] for c in client.get("/api/charts").json()}
    assert "easy-groove" in ids
    chart = client.get("/api/charts/easy-groove").json()
    assert chart["lanes"] == 4 and chart["notes"]


def test_chart_traversal_is_404():
    for bad in ["..%2F..%2Fetc%2Fpasswd", "EASY", "a.json", "%2e%2e"]:
        assert client.get(f"/api/charts/{bad}").status_code == 404


def test_random_chart():
    a = client.get("/api/charts/random/5").json()
    b = client.get("/api/charts/random/5").json()
    assert a == b
    assert client.get("/api/charts/random/-1").status_code == 422
