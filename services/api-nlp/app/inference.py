"""Logique d'inférence pour la classification de sentiment.

Le modèle CamemBERT chargé sort des labels 5 étoiles
(`'1 star'`, `'2 stars'`, ..., `'5 stars'`). Le métier (Aubergine Hôtels)
veut 3 classes (`négatif`, `neutre`, `positif`).

→ Ton travail dans ce fichier :
   1. Implémenter `predict_sentiment()` (récupère scores 5★ + appelle map).
   2. Implémenter `map_stars_to_sentiment()` (mapping 5★ → 3 classes).
   3. Justifier le seuil retenu en commentaire (cf. brief).

Le pipeline transformers est chargé une seule fois au démarrage, dans
`main.py` (lifespan), et stocké dans `state["pipeline"]`. Tu le récupères
en argument.
"""
from __future__ import annotations

import time
from typing import Any

from app.schemas import Sentiment, SentimentOut


def map_stars_to_sentiment(star_label: str) -> Sentiment:
    """Mappe un label 5 étoiles ('1 star', ..., '5 stars') en 3 classes métier.

    À compléter par l'apprenant. **Le choix du mapping est un arbitrage
    métier**, pas une recette imposée : plusieurs découpages sont valides
    (cf. mini-cours `02_HuggingFace_Transformers_essentiel.md`, section
    "Justification du seuil de mapping").

    Ton travail : proposer **ton** mapping et **le justifier** dans le
    README perso async (coût d'un faux positif / faux négatif côté
    métier Aubergine Hôtels).

    Args:
        star_label: label produit par le modèle (ex: '4 stars').

    Returns:
        Sentiment 3 classes.

    Raises:
        ValueError: si `star_label` n'est pas dans le format attendu.
    """
    # TODO Tâche 3 — implémenter le mapping de ton choix et documenter
    if star_label == '1 star' or star_label == '2 stars':
        return "négatif"
    elif star_label == '3 stars':
        return "neutre"
    else:
        return "positif"
    # le raisonnement métier dans le README perso async.


def predict_sentiment(pipeline: Any, text: str, model_name: str) -> SentimentOut:
    """Inférence de sentiment sur un texte FR.

    Args:
        pipeline: pipeline `transformers.pipeline("text-classification", ...)`
            chargé au démarrage de l'API.
        text: texte FR de la review.
        model_name: identifiant HF du modèle (passé pour traçabilité).

    Returns:
        SentimentOut avec sentiment 3 classes, scores 5★ bruts, et latence ms.
    """
    start_time = time.perf_counter()
    probabilities = pipeline(text, top_k=None)
    scores_5_stars = {entry['label']: entry['score'] for entry in probabilities}
    """
    low_score = scores_5_stars["1 star"] + scores_5_stars["2 stars"] + scores_5_stars["3 stars"]
    mid_score = scores_5_stars["2 stars"] + scores_5_stars["3 stars"] + scores_5_stars["4 stars"]
    high_score = scores_5_stars["3 stars"] + scores_5_stars["4 stars"] + scores_5_stars["5 stars"]

    group_scores = {
        "1 star": low_score,
        "3 stars": mid_score,
        "5 stars": high_score,
    }
    label_argmax = max(group_scores, key=group_scores.get)
    """

    label_argmax = max(scores_5_stars, key=scores_5_stars.get)
    sentiment = map_stars_to_sentiment(label_argmax)
    request_length = (time.perf_counter() - start_time) * 1000
    return SentimentOut(sentiment=sentiment, scores_5_stars=scores_5_stars, model_name=model_name, latence_ms=request_length)
    # TODO Tâche 3 — compléter :
    #
    # 1. Mesurer le temps d'inférence (time.perf_counter() avant/après).
    # 2. Appeler `pipeline(text, top_k=None)` pour récupérer toutes les
    #    probabilités (5 entrées, une par étoile).
    # 3. Construire `scores_5_stars: dict[str, float]` à partir du résultat.
    # 4. Identifier le label argmax (la plus haute proba).
    # 5. Appeler `map_stars_to_sentiment(label_argmax)` pour obtenir la
    #    classe métier.
    # 6. Renvoyer un `SentimentOut(...)`.
    raise NotImplementedError("Compléter `predict_sentiment` (Tâche 3).")