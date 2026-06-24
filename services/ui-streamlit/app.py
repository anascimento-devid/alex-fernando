"""UI Streamlit pour le service NLP Aubergine Hôtels.

Au clone, l'UI s'affiche mais ne consomme PAS encore l'API. À toi de :

1. Implémenter l'appel à `POST /predict` via httpx (Tâche 4 du brief).
2. Gérer les erreurs (API down, timeout 10 s).
3. Afficher le sentiment avec une couleur selon la classe :
   - 🔴 négatif → couleur rouge
   - 🟠 neutre  → couleur orange
   - 🟢 positif → couleur verte
4. Afficher les probabilités 5 étoiles brutes en barres (st.bar_chart).

L'URL de l'API est dans la variable d'environnement `API_URL`
(injectée par docker-compose, vaut `http://api-nlp:8000`).
"""
from __future__ import annotations

import os
import streamlit as st
from typing import Any
import httpx
import pandas as pd


API_URL: str = os.getenv("API_URL", "http://api-nlp:8000")
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"
REQUEST_TIMEOUT_SECONDS = 10.0

SENTIMENT_DISPLAY: dict[str, tuple[str, str]] = {
    "négatif": ("🔴", "Négatif"),
    "neutre": ("🟠", "Neutre"),
    "positif": ("🟢", "Positif"),
}


def check_api_health() -> bool:
    """Vérifie rapidement si l'API NLP est disponible."""

    try:
        response = httpx.get(HEALTH_URL, timeout=2.0)
        response.raise_for_status()

        payload = response.json()

        return (
            payload.get("status") == "ok"
            and payload.get("model_loaded") is True
        )
    except (httpx.HTTPError, ValueError):
        return False


def call_predict_api(texte: str) -> dict[str, Any]:
    """Envoie une review à l'API de prédiction."""

    response = httpx.post(
        PREDICT_URL,
        json={"texte": texte},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    return response.json()


def display_prediction(result: dict[str, Any]) -> None:
    """Affiche le sentiment et les scores bruts retournés par l'API."""

    sentiment = result.get("sentiment")

    if sentiment not in SENTIMENT_DISPLAY:
        st.error("L’API a retourné un sentiment inconnu.")
        return

    emoji, label = SENTIMENT_DISPLAY[sentiment]
    message = f"{emoji} Sentiment détecté : **{label}**"

    match sentiment:
        case "positif":
            st.success(message)
        case "neutre":
            st.warning(message)
        case "négatif":
            st.error(message)
        case _:
            st.error("Sentiment inconnu.")

    scores = result.get("scores_5_stars", {})

    if not scores:
        st.warning("Les probabilités détaillées ne sont pas disponibles.")
        return

    scores_dataframe = pd.DataFrame.from_dict(
        scores,
        orient="index",
        columns=["Probabilité"],
    )

    scores_dataframe.index.name = "Classe"

    st.subheader("Probabilités brutes du modèle")
    st.bar_chart(scores_dataframe)

    with st.expander("Détails techniques"):
        st.write(f"**Modèle :** `{result.get('model_name', 'inconnu')}`")
        st.write(
            f"**Latence serveur :** "
            f"{result.get('latence_ms', 'inconnue')} ms"
        )
        st.json(result)

st.set_page_config(
    page_title="Aubergine Hôtels — sentiment FR",
    page_icon="🍆",
    layout="centered",
)

st.title("🍆 Aubergine Hôtels — qualification du sentiment")
st.caption(
    "Démo interne : copie une review FR, le service NLP renvoie son sentiment "
    "en 3 classes (négatif / neutre / positif)."
)

texte = st.text_area(
    "Texte de la review",
    height=150,
    placeholder="Ex : Personnel charmant, chambre impeccable, on reviendra !",
)

if st.button(
    "Analyser",
    type="primary",
    disabled=not texte.strip(),
):
    try:
        with st.spinner("Analyse en cours..."):
            prediction = call_predict_api(texte.strip())

        display_prediction(prediction)

    except httpx.TimeoutException:
        st.error(
            "L’analyse a dépassé le délai maximal de 10 secondes. "
            "Réessaie dans quelques instants."
        )

    except httpx.ConnectError:
        st.error(
            "Impossible de joindre l’API NLP. "
            "Vérifie que le conteneur api-nlp est démarré."
        )

    except httpx.HTTPStatusError as error:
        status_code = error.response.status_code

        try:
            payload = error.response.json()
            detail = payload.get(
                "detail",
                "Erreur retournée par l’API.",
            )
        except ValueError:
            detail = "Réponse d’erreur API invalide."

        match status_code:
            case 422:
                st.warning(f"Review invalide : {detail}")

            case 501:
                st.warning(
                    "L’interface fonctionne, mais l’endpoint "
                    "`/predict` n’est pas encore implémenté."
                )

            case code if code >= 500:
                st.error(
                    "Le service NLP rencontre une erreur interne. "
                    "Consulte les logs du conteneur api-nlp."
                )

            case _:
                st.error(
                    f"Erreur API {status_code} : {detail}"
                )

    except httpx.RequestError:
        st.error(
            "Une erreur réseau empêche de contacter le service NLP."
        )

    except ValueError:
        st.error(
            "La réponse de l’API n’est pas un JSON valide."
        )

with st.sidebar:
    st.header("État du service")

    if check_api_health():
        st.success("API NLP disponible")
    else:
        st.error("API NLP indisponible")

    st.markdown(f"**API URL :** `{API_URL}`")
    st.markdown(
        f"**Timeout :** "
        f"`{REQUEST_TIMEOUT_SECONDS:.0f} secondes`"
    )

    st.divider()

    st.markdown(
        """
        **Résultat attendu**

        - sentiment en trois classes ;
        - probabilités brutes de 1 à 5 étoiles ;
        - modèle utilisé ;
        - latence d’inférence.
        """
    )