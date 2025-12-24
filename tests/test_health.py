from pathlib import Path
import os
from fastapi.testclient import TestClient
from mon_mlops_project.serving.api import app

client = TestClient(app)

def test_health_serving():
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["MLFLOW_MODEL_URI"] = (repo_root / "artifacts" / "model").as_uri()

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_uri" in data
