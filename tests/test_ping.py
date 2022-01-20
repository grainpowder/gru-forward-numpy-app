from fastapi.testclient import TestClient

from npgru.main import app

client = TestClient(app)


def test_ping():
    response = client.get("/npgru/health/ping")
    assert response.status_code == 200
    assert response.text.strip('"') == "pong"
