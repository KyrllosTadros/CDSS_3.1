"""Load the locally trained scikit-learn model and predict metastasis."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_FILE = Path(__file__).resolve().parent / "model.joblib"


def load_model_bundle():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("model.joblib ontbreekt. Train eerst met: python train_model.py")
    return joblib.load(MODEL_FILE)


def predict_one(bundle, values):
    schema = bundle["schema"]
    row = pd.DataFrame(
        [{column: values.get(column) for column in schema["columns"]}],
        columns=schema["columns"],
    )
    model = bundle["model"]
    label = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
    return label, probability
