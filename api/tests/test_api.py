
from fastapi.testclient import TestClient
from api.main import app



client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    r = client.post("/predict", json={"text": "this airline is awful"})
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["negative_proba"] <= 1.0
    assert data["negative_label"] in [0, 1]
