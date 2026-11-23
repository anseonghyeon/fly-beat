import time

import pytest
from fastapi.testclient import TestClient

from flybeat import paths
from flybeat.api import play
from flybeat.api.main import create_app

ORIGIN = {"origin": "http://localhost:5173"}


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_charts_list_and_get(client):
    ids = {c["id"] for c in client.get("/api/charts").json()}
    assert "easy-groove" in ids
    chart = client.get("/api/charts/easy-groove").json()
    assert chart["lanes"] == 4 and chart["notes"]


def test_chart_traversal_is_404(client):
    for bad in ["..%2F..%2Fetc%2Fpasswd", "EASY", "a.json", "%2e%2e"]:
        assert client.get(f"/api/charts/{bad}").status_code == 404


def test_random_chart(client):
    a = client.get("/api/charts/random/5").json()
    assert a == client.get("/api/charts/random/5").json()
    assert client.get("/api/charts/random/-1").status_code == 422


def test_untrusted_host_rejected(client):
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 400


def test_post_guards(client):
    body = {"mode": "A", "steps": 1000}
    r = client.post("/api/train", json=body, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/train", content='{"mode":"A"}', headers={"content-type": "text/plain"})
    assert r.status_code == 415


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "Z"},
        {"mode": "A", "steps": 10},
        {"mode": "A", "steps": 10**9},
        {"mode": "C", "steps": 5000},
        {"mode": "A", "episodes": 3},
        {"mode": "C", "episodes": 0},
    ],
)
def test_train_request_validation(client, body):
    assert client.post("/api/train", json=body, headers=ORIGIN).status_code == 422


def test_unknown_ids_404(client):
    assert client.get("/api/train/nope").status_code == 404
    assert client.get("/api/checkpoints/..%2Fx").status_code == 404
    assert client.get("/api/brain/plasticity/nope").status_code == 404


def test_train_job_then_watch(client, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYBEAT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", tmp_path)
    monkeypatch.setattr(play, "LEAD_MS", 10**9)  # 테스트에서는 실시간으로 기다리지 않음

    r = client.post("/api/train", json={"mode": "A", "steps": 2048}, headers=ORIGIN)
    assert r.status_code == 201
    job_id = r.json()["id"]
    # 동시 실행 제한 (기본 1개)
    assert client.post("/api/train", json={"mode": "A"}, headers=ORIGIN).status_code == 409

    deadline = time.time() + 180
    while time.time() < deadline:
        j = client.get(f"/api/train/{job_id}").json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    ckpt = j["checkpoint"]
    assert ckpt in {c["id"] for c in client.get("/api/checkpoints").json()}

    with client.websocket_connect("/ws/play", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "checkpoint": ckpt, "chart": "tutorial-steps"})
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["mode"] == "A"
        n_frames = 0
        while True:
            m = ws.receive_json()
            if m["type"] == "end":
                break
            n_frames += len(m["frames"])
        assert n_frames > 100 and m["stats"]["total"] > 0


def test_watch_errors(client):
    with client.websocket_connect("/ws/play", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "checkpoint": "../etc", "chart": "tutorial-steps"})
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect("/ws/play", headers=ORIGIN) as ws:
        ws.send_text("x" * 10000)
        assert ws.receive_json()["type"] == "error"


def test_plasticity_endpoint(synthetic_data, client):
    from flybeat.agents.dopamine import DopamineConfig, train

    cid = train(DopamineConfig(profile=synthetic_data, episodes=1, n_envs=2, lr=1.0))
    p = client.get(f"/api/brain/plasticity/{cid}").json()
    assert p["synapses"] > 0 and len(p["histogram"]["counts"]) == 24
