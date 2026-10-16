"""Train de 1-, 5- en 10-jaars overlevingsmodellen met data-gedreven kenmerkselectie.

Alles gebeurt zonder data leakage: kenmerkselectie gebruikt alleen de trainings-
en tune-set, de testset wordt pas aan het eind eenmalig gebruikt.

Stappen
1. Kandidaten: verwijder ID's, uitkomst, niet-klinische en afgeleide kolommen
   en kolommen met meer dan 25% missings (bepaald op de trainingsset).
2. Screening: univariate AUC per kenmerk (fit op train, AUC op tune).
3. Lasso: L1-logistische regressie op de trainingsset; kenmerken waarvan alle
   coëfficiënten op nul staan vallen af.
4. Permutation importance: MLP op train, daling in AUC op tune per kenmerk.
   Alleen duidelijke dalingen (gemiddelde > 2 x standaardafwijking) tellen mee.
5. Setvergelijking: top-K kenmerken; de kleinste K binnen 1% van de beste
   Brier-score op de tune-set wordt gekozen. De selectie geldt voor alle
   drie de termijnen, zodat de app één invoerformulier kan tonen.
6. Eindmodellen: net als model.py (1-10 neuronen via Brier op tune, daarna
   hertrainen op train + tune, evaluatie op test).
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "NKR_IKNL_breast_syntheticdata.csv"
MODEL_FILE = ROOT / "survival_models.joblib"
REPORT_FILE = ROOT / "feature_selection_report.csv"

YEARS = (1, 5, 10)
NEURON_RANGE = range(1, 11)
RANDOM_STATE = 42

# --- Instellingen voor de kenmerkselectie -----------------------------------
MAX_MISSING = 0.25           # kolommen met meer missings (op train) vallen af
MIN_UNIVARIATE_AUC = 0.52    # minimaal signaal in één van de drie termijnen
LASSO_C = 0.05               # kleiner = strengere lasso
PERMUTATION_REPEATS = 10
K_OPTIONS = (1, 2, 3, 5, 8, 10, 15, 20)
SELECTION_TOLERANCE = 0.01   # kleinste set binnen 1% van de beste Brier-score

# Kenmerken die je om klinische redenen altijd in het model wilt houden, ook als de
# data ze niet kiest (bijv. ["cstadium", "chemo", "horm"]). Leeg = puur data-gedreven.
FORCE_INCLUDE = []

# --- Kolommen die nooit als voorspeller worden gebruikt ---------------------
DROP_COLUMNS = ["key_nkr", "key_eid", "topo"]   # ID's en een constante kolom
TARGET_SOURCES = ["vit_stat", "vit_stat_int"]   # de uitkomst zelf
NOT_CLINICAL = ["incjr"]                        # incidentiejaar: bepaalt vooral de lengte van follow-up
DERIVED = ["stadium", "ct", "cn", "cm", "pt", "pn", "pm"]  # afgeleid van / onderdeel van de stadiumgroepen
EXCLUDED = DROP_COLUMNS + TARGET_SOURCES + NOT_CLINICAL + DERIVED

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


def log(message):
    print(message, flush=True)


# --- Data voorbereiden --------------------------------------------------------
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
    X = frame.drop(columns=EXCLUDED, errors="ignore").copy()
    for column, codes in MISSING_CODES.items():
        if column in X.columns:
            X[column] = X[column].replace(codes, np.nan)
    for column in X.columns:
        if column in CATEGORICAL:
            X[column] = X[column].map(lambda value: np.nan if pd.isna(value) else str(value))
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    return X


def split_horizon(X: pd.DataFrame, y: pd.Series) -> dict:
    """60/20/20 split, gestratificeerd op de uitkomst."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=RANDOM_STATE, stratify=y_rest
    )
    return {"X_train": X_train, "X_tune": X_tune, "X_test": X_test,
            "y_train": y_train, "y_tune": y_tune, "y_test": y_test}


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
    ])  # remainder="drop"


def positive_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def quick_mlp(columns: list) -> Pipeline:
    """MLP voor de selectiestappen (vaste, kleine architectuur).

    Zonder early stopping en met lichte regularisatie (alpha): met early stopping
    gaf dit model op scheve uitkomsten (zoals 1-jaarsoverleving, 97% klasse 1)
    instabiele, soms omgekeerde voorspellingen, waardoor de importance onbetrouwbaar was.
    """
    return Pipeline([
        ("preprocess", build_preprocessor(columns)),
        ("classifier", MLPClassifier(hidden_layer_sizes=(10,), max_iter=500,
                                     alpha=1e-2, random_state=RANDOM_STATE)),
    ])


# --- Kenmerkselectie ----------------------------------------------------------
def univariate_auc(split: dict, columns: list) -> pd.Series:
    """Stap 2: AUC van een model met alleen dit ene kenmerk (fit op train, AUC op tune)."""
    result = {}
    for column in columns:
        model = Pipeline([
            ("preprocess", build_preprocessor([column])),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        model.fit(split["X_train"][[column]], split["y_train"])
        result[column] = roc_auc_score(split["y_tune"], positive_proba(model, split["X_tune"][[column]]))
    return pd.Series(result)


def lasso_columns(split: dict, columns: list) -> set:
    """Stap 3: kenmerken waarvan minstens één coëfficiënt na L1-straf niet nul is."""
    preprocessor = build_preprocessor(columns)
    Z = preprocessor.fit_transform(split["X_train"][columns])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 'penalty' is in nieuwere scikit-learn-versies verouderd
        lasso = LogisticRegression(penalty="l1", solver="liblinear", C=LASSO_C, max_iter=1000)
        lasso.fit(Z, split["y_train"])
    kept = set()
    for name, coefficient in zip(preprocessor.get_feature_names_out(), lasso.coef_.ravel()):
        if coefficient != 0:
            stem = name.split("__", 1)[1]
            matches = [c for c in columns if stem == c or stem.startswith(c + "_")]
            kept.add(max(matches, key=len))
    return kept


def permutation_scores(split: dict, columns: list) -> pd.DataFrame:
    """Stap 4: daling in AUC op tune als één kenmerk wordt gehusseld.

    Geeft per kenmerk het gemiddelde ("mean"), de spreiding ("std") en of de daling
    duidelijk boven nul ligt ("significant": mean > 2 * std).
    """
    model = quick_mlp(columns)
    model.fit(split["X_train"][columns], split["y_train"])
    result = permutation_importance(
        model, split["X_tune"][columns], split["y_tune"],
        scoring="roc_auc", n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_STATE,
    )
    scores = pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std}, index=columns)
    scores["significant"] = scores["mean"] > 2 * scores["std"]
    return scores


def tune_brier(split: dict, columns: list) -> float:
    model = quick_mlp(columns)
    model.fit(split["X_train"][columns], split["y_train"])
    return float(brier_score_loss(split["y_tune"], positive_proba(model, split["X_tune"][columns])))


def select_features(splits: dict, all_columns: list):
    """Voer stap 1-5 uit. Geeft de gekozen kenmerken, een verslag en details terug."""
    # 1. Kandidaten
    candidates = [
        c for c in all_columns
        if all(splits[y]["X_train"][c].isna().mean() <= MAX_MISSING for y in YEARS)
    ]
    dropped = sorted(set(all_columns) - set(candidates))
    log(f"Stap 1: {len(candidates)} kandidaten (afgevallen door >{MAX_MISSING:.0%} missings: {', '.join(dropped) or '-'})")

    # 2. Screening
    auc = pd.DataFrame({y: univariate_auc(splits[y], candidates) for y in YEARS})
    passed = [c for c in candidates if auc.loc[c].max() >= MIN_UNIVARIATE_AUC]
    log(f"Stap 2: {len(passed)} kenmerken met univariate AUC >= {MIN_UNIVARIATE_AUC} "
        f"(afgevallen: {', '.join(c for c in candidates if c not in passed) or '-'})")

    # 3. Lasso
    lasso = {y: lasso_columns(splits[y], passed) for y in YEARS}
    pool = [c for c in passed if any(c in lasso[y] for y in YEARS)]
    log(f"Stap 3: {len(pool)} kenmerken met niet-nul lasso-coëfficiënt in minstens één termijn "
        f"(afgevallen: {', '.join(c for c in passed if c not in pool) or '-'})")

    # 4. Permutation importance
    perm = {y: permutation_scores(splits[y], pool) for y in YEARS}
    importance = pd.DataFrame({y: perm[y]["mean"] for y in YEARS})
    significant = pd.DataFrame({y: perm[y]["significant"] for y in YEARS})
    # Alleen duidelijke dalingen tellen mee; de rest is ruis.
    score = importance.where(significant, 0.0).mean(axis=1).sort_values(ascending=False)
    ranking = [c for c in score.index if score[c] > 0] or [score.index[0]]
    log("Stap 4: kenmerken met duidelijke permutation importance (in rangorde): " + ", ".join(ranking)
        + f"  (ruis, niet meegenomen: {', '.join(c for c in pool if c not in ranking) or '-'})")

    # 5. Setvergelijking
    ks = sorted({k for k in K_OPTIONS if k < len(ranking)} | {len(ranking)})
    briers = {k: {y: tune_brier(splits[y], ranking[:k]) for y in YEARS} for k in ks}
    best = {y: min(briers[k][y] for k in ks) for y in YEARS}
    relative = {k: float(np.mean([briers[k][y] / best[y] for y in YEARS])) for k in ks}
    chosen_k = next(k for k in ks if relative[k] <= 1 + SELECTION_TOLERANCE)
    for k in ks:
        log(f"  top {k:>2}: gem. relatieve Brier {relative[k]:.4f} "
            + " ".join(f"({y}j {briers[k][y]:.4f})" for y in YEARS))
    selected = ranking[:chosen_k]
    log(f"Stap 5: gekozen set = top {chosen_k}: {', '.join(selected)}")

    unknown = [c for c in FORCE_INCLUDE if c not in all_columns]
    if unknown:
        raise ValueError(f"FORCE_INCLUDE bevat onbekende of uitgesloten kolommen: {', '.join(unknown)}")
    forced = [c for c in FORCE_INCLUDE if c not in selected]
    if forced:
        selected = selected + forced
        log(f"Handmatig toegevoegd via FORCE_INCLUDE: {', '.join(forced)}")

    report = pd.DataFrame(index=all_columns)
    for y in YEARS:
        report[f"univariate_auc_{y}j"] = auc[y]
        report[f"lasso_{y}j"] = pd.Series({c: c in lasso[y] for c in passed})
        report[f"importance_{y}j"] = importance[y]
        report[f"significant_{y}j"] = significant[y]
    report["gemiddelde_importance"] = score
    report["geselecteerd"] = report.index.isin(selected)
    report = report.sort_values(["geselecteerd", "gemiddelde_importance"], ascending=False)
    report.index.name = "kenmerk"

    details = {
        "selected": selected,
        "ranking": [{"feature": c, "score": float(score[c])} for c in ranking],
        "noise": [c for c in pool if c not in ranking],
        "k_curve": [{"k": k, "mean_relative_brier": relative[k],
                     "brier": {int(y): briers[k][y] for y in YEARS}} for k in ks],
        "chosen_k": chosen_k,
        "forced": forced,
        "settings": {"max_missing": MAX_MISSING, "min_univariate_auc": MIN_UNIVARIATE_AUC,
                     "lasso_c": LASSO_C, "tolerance": SELECTION_TOLERANCE,
                     "excluded": EXCLUDED},
    }
    return selected, report, details


# --- Eindmodellen -------------------------------------------------------------
def train_horizon(split: dict, columns: list):
    """Net als model.py: neuronen kiezen op de tune-set, hertrainen op train + tune, testen."""
    preprocessor = build_preprocessor(columns)
    Z_train = preprocessor.fit_transform(split["X_train"][columns])
    Z_tune = preprocessor.transform(split["X_tune"][columns])

    brier_tune = {}
    for neurons in NEURON_RANGE:
        nn = MLPClassifier(hidden_layer_sizes=(neurons,), max_iter=1000, random_state=RANDOM_STATE)
        nn.fit(Z_train, split["y_train"])
        brier_tune[neurons] = brier_score_loss(split["y_tune"], positive_proba(nn, Z_tune))
    best = min(brier_tune, key=brier_tune.get)

    X_fit = pd.concat([split["X_train"], split["X_tune"]])[columns]
    y_fit = pd.concat([split["y_train"], split["y_tune"]])
    model = Pipeline([
        ("preprocess", build_preprocessor(columns)),
        ("classifier", MLPClassifier(hidden_layer_sizes=(best,), max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_fit, y_fit)

    X_test, y_test = split["X_test"][columns], split["y_test"]
    p_test = positive_proba(model, X_test)
    summary = {
        "accuracy": float(accuracy_score(y_test, (p_test >= 0.5).astype(int))),
        "accuracy_train": float(accuracy_score(y_fit, (positive_proba(model, X_fit) >= 0.5).astype(int))),
        "auc": float(roc_auc_score(y_test, p_test)),
        "brier": float(brier_score_loss(y_test, p_test)),
        "best_neurons": int(best),
        "brier_tune": {int(k): float(v) for k, v in brier_tune.items()},
        "training_rows": int(len(X_fit)),
    }
    return model, summary


def main():
    started = time.time()
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset niet gevonden: {DATA_FILE}")
    frame = pd.read_csv(DATA_FILE, sep=";")
    X_all = prepare_features(frame)

    splits, targets = {}, {}
    for years in YEARS:
        y_all = make_survival_target(frame, years)
        known = y_all.notna()
        X, y = X_all.loc[known], y_all.loc[known].astype(int)
        splits[years] = split_horizon(X, y)
        targets[years] = float(y.mean())
        log(f"{years}-jaarsuitkomst: {len(X)} dossiers, aandeel overleving={targets[years]:.3f}")

    log("\n=== Kenmerkselectie (alleen train + tune) ===")
    selected, report, details = select_features(splits, list(X_all.columns))
    report.to_csv(REPORT_FILE, sep=";", decimal=",")
    log(f"Verslag opgeslagen in {REPORT_FILE.name}")

    log("\n=== Eindmodellen ===")
    models, summaries = {}, {}
    for years in YEARS:
        model, summary = train_horizon(splits[years], selected)
        summary["survival_share"] = targets[years]
        models[years], summaries[years] = model, summary
        log(f"{years}-jaar: neuronen={summary['best_neurons']}, acc train={summary['accuracy_train']:.3f}, "
            f"acc test={summary['accuracy']:.3f}, test-AUC={summary['auc']:.3f}, test-Brier={summary['brier']:.3f}")

    categorical = [c for c in selected if c in CATEGORICAL]
    numeric = [c for c in selected if c not in CATEGORICAL]
    schema = {
        "columns": selected,
        "categorical": categorical,
        "numeric": numeric,
        "categories": {c: sorted(X_all[c].dropna().astype(str).unique().tolist()) for c in categorical},
        "numeric_defaults": {c: float(X_all[c].median()) for c in numeric},
        "numeric_ranges": {c: (float(X_all[c].min()), float(X_all[c].max())) for c in numeric},
        "summaries": summaries,
        "feature_selection": details,
    }
    joblib.dump({"models": models, "schema": schema, "years": YEARS}, MODEL_FILE)
    log(f"\nDrie modellen opgeslagen in {MODEL_FILE.name} ({time.time() - started:.0f} seconden)")


if __name__ == "__main__":
    main()