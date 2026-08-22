from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_and_models():
    res_h = client.get("/health")
    assert res_h.status_code == 200
    res_m = client.get("/v1/models")
    assert res_m.status_code == 200
    assert len(res_m.json()["data"]) >= 3

def test_openai_chat_completions():
    payload = {
        "model": "clinical-llama3-dora-8b",
        "messages": [{"role": "user", "content": "What is the door-to-balloon time for STEMI?"}]
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
