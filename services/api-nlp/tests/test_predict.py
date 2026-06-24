from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
import pytest

valid_payload = {
  "texte": "Personnel charmant, chambre impeccable, petit-déjeuner copieux. On reviendra !"
}


def test_predict_valid_sample() -> None:
    """POST /predict retourne 200 et une structure valide."""
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert "sentiment" in body
    assert "scores_5_stars" in body

def test_predict_empty_sample() -> None:
    """POST /predict retourne une erreur si le texte envoyé est vide."""
    payload = valid_payload.copy().pop("texte")
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    print(response.status_code)
    assert response.status_code == 422
#    body = response.json()
#    assert "sentiment" in body
#    assert "scores_5_stars" in body


def test_predict_large_sample() -> None:
    """POST /predict retourne une erreur si le texte envoyé est trop long."""
    payload = {
        "texte": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
#    body = response.json()
#    assert "sentiment" in body
#    assert "scores_5_stars" in body

def test_predict_valid_sample_positive():
    payload = {
        "texte": "Personnel charmant, chambre impeccable, petit-déjeuner copieux. On reviendra !"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "positif"

def test_predict_valid_sample_neutral():
    payload = {
        "texte": "Petit déjeuner buffet correct mais sans surprise. Chambre propre, literie ferme. Standard."
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "neutre"

def test_predict_valid_sample_negative():
    payload = {
        "texte": "Chambre sale, personnel désagréable, expérience décevante. Je ne reviendrai pas."
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "négatif"