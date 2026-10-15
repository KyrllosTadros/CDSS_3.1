"""Train the 1-, 5-, and 10-year breast-cancer survival models.

Alle kolommen, min-max normalisatie, one-hot met
drop_first, 1-10 neuronen gekozen op de Brier-score van de tune-set, 60/20/20
split), maar zonder data leakage: alle leer-stappen (kolomselectie op missings,
imputatie, normalisatie, one-hot-encoding
worden uitsluitend op trainingsdata gefit en daarna toegepast op tune- en testdata.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "NKR_IKNL_breast_syntheticdata.csv"
MODEL_FILE = ROOT / "survival_models.joblib"

YEARS = (1, 5, 10)
NEURON_RANGE = range(1, 11)
MAX_MISSING = 0.25  # kolommen met meer dan 25% missings (op trainingsdata) vallen af
RANDOM_STATE = 42

# topo is overal gelijk, de keys zijn identifiers; vit_stat(_int) vormen de uitkomst.
DROP_COLUMNS = ["topo", "key_nkr", "key_eid"]
TARGET_SOURCES = ["vit_stat", "vit_stat_int"]

CATEGORICAL = [
    "diag_basis", "ct", "cn", "pt", "pn",
    "stadium", "cstadium", "pstadium",
    "diffgrad", "uitgebr_chir_code", "gedrag", "gesl",
    "her2_stat", "dcis_comp", "okd", "org_chir",
    "dir_reconstr", "meta_rt", "meta_chir", "multifoc",
    "later", "morf", "er_stat", "pr_stat", "swk",
    "chemo", "target", "horm", "rt", "topo_sublok",
    "tumsoort",
]

# Codes die "onbekend" betekenen. Dit is een vaste hercodering (geen leren uit
# data), dus veilig om vóór de split te doen.
MISSING_CODES = {
    "diffgrad": [9],
    "stadium": ["M"], "pstadium": ["M"], "cstadium": ["M"],
    "ond_lymf": [98, 99],
    "pos_lymf": [99],
    "er_stat": [9], "pr_stat": [9],
    "her2_stat": [4, 7],
    "multifoc": [9],
    "swk": [8.0],
    "swk_uitslag": [9.0],
}


def make_survival_target(frame: pd.DataFrame, years: int) -> pd.Series:
    """1 = minstens `years` jaar overleefd, 0 = overleden binnen `years`, NA = onbekend."""
    status = pd.to_numeric(frame["vit_stat"], errors="coerce")
    days = pd.to_numeric(frame["vit_stat_int"], errors="coerce")
    limit = years * 365.25
    target = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    target.loc[(status == 1) & (days <= limit)] = 0
    target.loc[((status == 0) & (days >= limit)) | ((status == 1) & (days > limit))] = 1
    return target


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Vaste opschoning: geen enkele statistiek wordt hier uit de data geleerd."""
    X = frame.drop(columns=DROP_COLUMNS + TARGET_SOURCES, errors="ignore").copy()
    for column, codes in MISSING_CODES.items():
        if column in X.columns:
            X[column] = X[column].replace(codes, np.nan)
    for column in X.columns:
        if column in CATEGORICAL:
            X[column] = X[column].map(lambda value: np.nan if pd.isna(value) else str(value))
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    return X


def select_columns(X_train: pd.DataFrame) -> list:
    """Behoud kolommen met hooguit 25% missings, bepaald op de trainingsset."""
    return X_train.columns[X_train.isna().mean() <= MAX_MISSING].tolist()


def build_preprocessor(columns: list) -> ColumnTransformer:
    categorical = [c for c in columns if c in CATEGORICAL]
    numeric = [c for c in columns if c not in CATEGORICAL]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", MinMaxScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), categorical),
    ])  # remainder="drop": overige kolommen worden genegeerd


def positive_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def train_horizon(X: pd.DataFrame, y: pd.Series, years: int):
    # 60/20/20 split
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=RANDOM_STATE, stratify=y_rest
    )

    # Kolomselectie en preprocessing: alleen gefit op de trainingsset
    columns = select_columns(X_train)
    preprocessor = build_preprocessor(columns)
    Z_train = preprocessor.fit_transform(X_train)
    Z_tune = preprocessor.transform(X_tune)

    # Aantal neuronen kiezen op de Brier-score van de tune-set
    brier_tune = {}
    for neurons in NEURON_RANGE:
        nn = MLPClassifier(hidden_layer_sizes=(neurons,), max_iter=1000,
                           random_state=RANDOM_STATE)
        nn.fit(Z_train, y_train)
        brier_tune[neurons] = brier_score_loss(y_tune, positive_proba(nn, Z_tune))
    best = min(brier_tune, key=brier_tune.get)

    # Eindmodel: train + tune samen, preprocessing opnieuw gefit op die data
    X_fit = pd.concat([X_train, X_tune])
    y_fit = pd.concat([y_train, y_tune])
    model = Pipeline([
        ("preprocess", build_preprocessor(columns)),
        ("classifier", MLPClassifier(hidden_layer_sizes=(best,), max_iter=1000,
                                     random_state=RANDOM_STATE)),
    ])
    model.fit(X_fit, y_fit)

    # Evaluatie op de nooit eerder gebruikte testset
    p_test = positive_proba(model, X_test)
    pred_test = (p_test >= 0.5).astype(int)
    pred_fit = (positive_proba(model, X_fit) >= 0.5).astype(int)
    summary = {
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "accuracy_train": float(accuracy_score(y_fit, pred_fit)),
        "auc": float(roc_auc_score(y_test, p_test)),
        "brier": float(brier_score_loss(y_test, p_test)),
        "best_neurons": int(best),
        "brier_tune": {int(k): float(v) for k, v in brier_tune.items()},
        "survival_share": float(y.mean()),
        "training_rows": int(len(X_fit)),
        "columns_used": columns,
    }
    return model, summary


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset niet gevonden: {DATA_FILE}")
    frame = pd.read_csv(DATA_FILE, sep=";")
    X_all = prepare_features(frame)
    models = {}
    summaries = {}

    for years in YEARS:
        y_all = make_survival_target(frame, years)
        known = y_all.notna()
        X, y = X_all.loc[known], y_all.loc[known].astype(int)
        model, summary = train_horizon(X, y, years)
        models[years] = model
        summaries[years] = summary
        print(
            f"{years}-jaarsuitkomst: {len(X)} dossiers, "
            f"aandeel overleving={summary['survival_share']:.3f}, "
            f"neuronen={summary['best_neurons']}, "
            f"acc train={summary['accuracy_train']:.3f}, "
            f"acc test={summary['accuracy']:.3f}, "
            f"test-AUC={summary['auc']:.3f}, "
            f"test-Brier={summary['brier']:.3f}"
        )

    columns = X_all.columns.tolist()
    categorical = [c for c in columns if c in CATEGORICAL]
    numeric = [c for c in columns if c not in CATEGORICAL]
    schema = {
        "columns": columns,
        "categorical": categorical,
        "numeric": numeric,
        "categories": {
            column: sorted(X_all[column].dropna().astype(str).unique().tolist())
            for column in categorical
        },
        "numeric_defaults": {
            column: float(X_all[column].median()) for column in numeric
        },
        "summaries": summaries,
    }
    joblib.dump({"models": models, "schema": schema, "years": YEARS}, MODEL_FILE)
    print(f"Drie modellen opgeslagen in {MODEL_FILE}")


if __name__ == "__main__":
    main()