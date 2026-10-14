"""Train and save the breast-cancer metastasis MLP using the local CSV."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "NKR_IKNL_breast_syntheticdata.csv"
MODEL_FILE = ROOT / "model.joblib"
TARGET = "uitkomst"
IDENTIFIERS = ["key_nkr", "key_eid", "topo"]

MISSING_CODES = {
    "diffgrad": [9], "stadium": ["M"], "pstadium": ["M"],
    "cstadium": ["M"], "ond_lymf": [98, 99], "pos_lymf": [99],
    "er_stat": [9], "pr_stat": [9], "her2_stat": [4, 7],
    "multifoc": [9], "swk": [8.0], "swk_uitslag": [9.0],
}

CATEGORICAL_HINTS = {
    "diag_basis", "ct", "cn", "pt", "pn", "stadium", "cstadium", "pstadium",
    "diffgrad", "uitgebr_chir_code", "gedrag", "gesl", "her2_stat", "dcis_comp",
    "okd", "org_chir", "dir_reconstr", "meta_rt", "meta_chir", "multifoc",
    "later", "morf", "er_stat", "pr_stat", "swk", "chemo", "target", "horm",
    "rt", "topo_sublok", "tumsoort", "vit_stat",
}


def load_training_data(path: Path = DATA_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset niet gevonden: {path}\n"
            "Plaats het CSV-bestand lokaal in de projectmap."
        )
    df = pd.read_csv(path, sep=";")
    missing = {"pm", "cm"} - set(df.columns)
    if missing:
        raise ValueError(f"De CSV mist vereiste doelkolommen: {', '.join(sorted(missing))}")

    # Use pm, falling back to cm when pm is missing, as in the original script.
    df[TARGET] = df["pm"].combine_first(df["cm"])
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    labels = set(df[TARGET].unique())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"Doelkolom moet beide klassen 0 en 1 bevatten; gevonden: {sorted(labels)}")

    # pm/cm determine the target and must never be model inputs.
    X = df.drop(columns=[TARGET, "pm", "cm", *IDENTIFIERS], errors="ignore")
    for column, codes in MISSING_CODES.items():
        if column in X.columns:
            X[column] = X[column].replace(codes, np.nan)
    X = X.loc[:, X.isna().mean() <= 0.25]
    if X.empty:
        raise ValueError("Na het verwijderen van kolommen met >25% missende waarden zijn er geen invoervariabelen over.")

    categorical = [
        column for column in X.columns
        if column in CATEGORICAL_HINTS
        or X[column].dtype.name in {"object", "category", "string"}
    ]
    numeric = [column for column in X.columns if column not in categorical]
    for column in categorical:
        X[column] = X[column].map(lambda value: np.nan if pd.isna(value) else str(value))
    for column in numeric:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    keep = [column for column in X.columns if not X[column].isna().all()]
    X = X[keep]
    categorical = [column for column in categorical if column in keep]
    numeric = [column for column in numeric if column in keep]
    if X.empty:
        raise ValueError("Geen bruikbare invoervariabelen gevonden.")
    return X, df[TARGET], categorical, numeric


def build_pipeline(categorical, numeric):
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", MLPClassifier(hidden_layer_sizes=(300,), max_iter=1000,
                                      random_state=42, early_stopping=True)),
    ])


def main():
    X, y, categorical, numeric = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=42, stratify=y
    )
    model = build_pipeline(categorical, numeric)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    schema = {
        "columns": list(X.columns),
        "categorical": categorical,
        "numeric": numeric,
        "categories": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in categorical},
        "numeric_defaults": {c: float(pd.to_numeric(X[c], errors="coerce").median()) for c in numeric},
    }
    joblib.dump({"model": model, "schema": schema}, MODEL_FILE)
    print(f"Model opgeslagen: {MODEL_FILE}")
    print(f"A-priori aandeel klasse 1 in deze dataset: {y.mean():.3f}")
    print(f"Nauwkeurigheid op testset: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix (rijen=werkelijk, kolommen=voorspeld):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
