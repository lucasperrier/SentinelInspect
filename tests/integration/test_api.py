"""API integration tests.

These drive the real application through FastAPI's TestClient: the lifespan
handler runs, the model loads, and requests go through the actual routes. They
are the only tests here that exercise transport, validation and inference
together -- which is where the unit tests, by construction, cannot look.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sentinelinspect import __version__
from sentinelinspect.inference_service.app import create_app


@pytest.fixture(scope="module")
def client(tiny_checkpoint):
    app = create_app(checkpoint_path=str(tiny_checkpoint))
    with TestClient(app) as test_client:   # `with` is what runs the lifespan
        yield test_client


# ---- health -------------------------------------------------------------

def test_health_reports_the_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["package_version"] == __version__
    assert len(body["checkpoint_sha256"]) == 16
    assert len(body["review_band"]) == 2


def test_health_identifies_which_weights_are_serving(client):
    """'Is it up?' is not the useful question during an incident; 'which model
    is up?' is."""
    body = client.get("/health").json()
    assert body["checkpoint_sha256"]
    assert body["model_name"] == "resnet50"


# ---- predict, happy path ------------------------------------------------

def test_predict_returns_the_full_contract(client, image_bytes):
    response = client.post(
        "/predict", files={"file": ("sample.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_label"] in ("crack", "no_crack")
    assert 0.0 <= body["confidence_score"] <= 1.0
    assert set(body["probabilities"]) == {"crack", "no_crack"}
    assert isinstance(body["needs_review"], bool)
    assert body["model_metadata"]["checkpoint_sha256"]
    assert body["latency_ms"] >= 0.0


def test_api_and_core_agree(client, image_bytes):
    """Same bytes through the HTTP route and through the Predictor the app is
    holding. Not two identically-configured predictors -- the *same object*.
    If these ever differ, 'shared inference core' is not true."""
    http = client.post("/predict", files={"file": ("s.jpg", image_bytes, "image/jpeg")}).json()
    direct = client.app.state.predictor.predict_image(image_bytes)

    assert http["predicted_label"] == direct.predicted_label
    assert http["probabilities"]["crack"] == pytest.approx(
        direct.probabilities["crack"], abs=1e-4
    )


def test_png_is_accepted_too(client):
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(buffer, format="PNG")
    response = client.post(
        "/predict", files={"file": ("s.png", buffer.getvalue(), "image/png")}
    )
    assert response.status_code == 200


# ---- predict, failure modes ---------------------------------------------

def test_missing_file_is_a_validation_error(client):
    assert client.post("/predict").status_code == 422


def test_non_image_content_type_is_rejected(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415
    assert "content type" in response.json()["detail"].lower()


def test_empty_upload_is_rejected(client):
    response = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_corrupt_image_is_a_client_error_not_a_server_error(client):
    """A 500 here would say the server broke. It did not: the caller sent
    something undecodable, and the status code should say so."""
    response = client.post(
        "/predict", files={"file": ("broken.jpg", b"\xff\xd8\xff\xe0 not really a jpeg", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setenv("SENTINELINSPECT_MAX_UPLOAD_MB", "0.001")   # 1 KB
    buffer = io.BytesIO()
    Image.fromarray(
        np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    ).save(buffer, format="JPEG")

    response = client.post(
        "/predict", files={"file": ("big.jpg", buffer.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()


# ---- startup ------------------------------------------------------------

def test_startup_fails_loudly_when_the_checkpoint_is_missing(tmp_path):
    """Better a crash loop than a service that reports healthy and 500s on
    every request."""
    app = create_app(checkpoint_path=str(tmp_path / "absent.ckpt"))
    with pytest.raises(Exception, match="not found"):
        with TestClient(app):
            pass


def test_openapi_documents_the_contract(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    responses = schema["paths"]["/predict"]["post"]["responses"]
    for code in ("400", "413", "415", "503"):
        assert code in responses, f"undocumented failure mode: {code}"
