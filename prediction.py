"""Laad de drie overlevingsmodellen en bereken hun uitkomsten voor één patiënt."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_FILE = Path(__file__).resolve().parent / "survival_models.joblib"


def load_model_bundle():
    """Lees survival_models.joblib (gemaakt door train_model.py) en controleer de inhoud."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "survival_models.joblib ontbreekt. Train de drie modellen met: python train_model.py"
        )
    bundle = joblib.load(MODEL_FILE)
    if not all(years in bundle.get("models", {}) for years in (1, 5, 10)):
        raise ValueError("Het modelbestand bevat niet alle drie uitkomsten (1, 5 en 10 jaar).")
    return bundle


def predict_all(bundle, values):
    """Geef per termijn de overlevingskans en de binaire uitkomst (grens 50%).

    `values` is een dictionary {kenmerk: waarde}; None betekent onbekend.
    """
    schema = bundle["schema"]
    # Onbekende waarden (None) moeten als NaN binnenkomen, anders worden ze niet geïmputeerd.
    row = pd.DataFrame(
        [{column: np.nan if values.get(column) is None else values[column] for column in schema["columns"]}],
        columns=schema["columns"],
    )
    results = {}
    for years, model in bundle["models"].items():
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError(f"Model voor {years} jaar kent geen overlevingsklasse.")
        probability = float(model.predict_proba(row)[0][classes.index(1)])
        results[int(years)] = {
            "survival_probability": probability,
            "survives": probability >= 0.5,
        }
    return results
