"""Streamlit interface voor de overlevingsmodellen na 1, 5 en 10 jaar.

De huisarts kiest een (fictieve) patiënt uit het dossier. De app toont precies
de kenmerken die het getrainde model gebruikt (gekozen door train_model.py).
Kenmerken die bij diagnose vastliggen worden alleen getoond; kenmerken die
tussen follow-up momenten kunnen veranderen zijn aanpasbaar. Aanpassingen worden
niet in het dossier opgeslagen; ze gelden alleen voor de berekening in deze sessie.
"""
import pandas as pd
import streamlit as st

import ehr
import prediction

st.set_page_config(page_title="Borstkanker – overlevingsmodel", page_icon="🩺", layout="wide")
st.title("Borstkankermodel: overleving na 1, 5 en 10 jaar")
st.caption("Technische demonstratie op synthetische data. Niet gebruiken voor diagnose of behandelkeuzes.")

try:
    bundle = prediction.load_model_bundle()
except Exception as error:
    st.error(f"De drie overlevingsmodellen zijn nog niet beschikbaar: {error}")
    st.info("Train ze eenmalig vanuit de projectmap met `python train_model.py`.")
    st.stop()

try:
    patients = ehr.list_patients()
except Exception as error:
    st.error(f"Het patiëntendossier is niet beschikbaar: {error}")
    st.stop()

schema = bundle["schema"]
columns = schema["columns"]

# Kenmerken die tussen follow-up momenten kunnen veranderen (aanpasbaar door de
# huisarts). Alle andere kenmerken liggen bij diagnose vast en zijn alleen-lezen.
EDITABLE = {"cstadium", "tum_afm", "pos_lymf", "chemo", "target", "horm", "rt", "meta_rt", "meta_chir", "gedrag"}

FIELD_LABELS = {
    "leeft": "Leeftijd bij diagnose (jaar)",
    "gesl": "Geslacht",
    "tumsoort": "Tumorsoort (datasetcode)",
    "diag_basis": "Basis van diagnose",
    "topo_sublok": "Sublokalisatie tumor",
    "later": "Lateraliteit",
    "morf": "Morfologiecode",
    "gedrag": "Gedrag van de tumor",
    "diffgrad": "Tumorgraad (diffgrad)",
    "cstadium": "Klinisch stadium",
    "pstadium": "Pathologisch stadium",
    "ond_lymf": "Aantal onderzochte lymfeklieren",
    "pos_lymf": "Aantal positieve lymfeklieren",
    "er_stat": "ER-status",
    "pr_stat": "PR-status",
    "her2_stat": "HER2-status",
    "dcis_comp": "DCIS-component",
    "multifoc": "Multifocaliteit",
    "tum_afm": "Tumorgrootte (mm)",
    "swk": "Schildwachtklier",
    "okd": "okd",
    "org_chir": "Chirurgie (datasetcode)",
    "uitgebr_chir_code": "Chirurgiecode",
    "dir_reconstr": "Reconstructie",
    "chemo": "Chemotherapie (datasetcode)",
    "target": "Doelgerichte therapie (datasetcode)",
    "horm": "Hormoontherapie (datasetcode)",
    "rt": "Radiotherapie (datasetcode)",
    "meta_rt": "Radiotherapie bij metastasen",
    "meta_chir": "Chirurgie bij metastasen",
}


# Leesbare weergave van datasetcodes. Ontbreekt een code hier, dan toont de app de code zelf.
VALUE_LABELS = {
    "gesl": {"1": "Man", "2": "Vrouw"},
    "gedrag": {"2": "In situ", "3": "Maligne"},
}


def label(column):
    return FIELD_LABELS.get(column, column)


def display_value(column, value):
    """Toon de leesbare tekst bij een code (bijv. '2' bij gesl -> 'Vrouw')."""
    if value is None:
        return "Onbekend"
    names = VALUE_LABELS.get(column, {})
    for candidate in (str(value), str(value).removesuffix(".0")):
        if candidate in names:
            return names[candidate]
    return str(value)


def to_category(value, categories):
    """Zet een databasewaarde (bijv. 2) om naar de categorie-tekst van het model (bijv. '2.0')."""
    if value is None or pd.isna(value):
        return None
    candidates = [str(value)]
    try:
        number = float(value)
        candidates += [str(number), str(int(number))]
    except (TypeError, ValueError):
        pass
    return next((c for c in candidates if c in categories), None)


def rows_of(items, per_row=5):
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        yield chunk, st.columns(per_row)[:len(chunk)]


fixed = [c for c in columns if c not in EDITABLE]
editable = [c for c in columns if c in EDITABLE]

# --- Patiëntkeuze -----------------------------------------------------------
names = dict(patients)
patient_id = st.selectbox("Patiënt", list(names), format_func=lambda pid: names[pid])
record = ehr.get_patient(patient_id)

missing_in_record = [c for c in columns if c not in record]
if missing_in_record:
    st.warning(
        "Het dossier bevat deze modelkenmerken niet en ze worden als onbekend behandeld: "
        + ", ".join(missing_in_record) + ". Vul `database.sql` aan en draai `python setup_database.py`."
    )

if "reset_count" not in st.session_state:
    st.session_state["reset_count"] = 0
key_prefix = f"{patient_id}_{st.session_state['reset_count']}"

values = {}

# --- Vaste gegevens ---------------------------------------------------------
st.subheader("Vaste gegevens (bekend bij diagnose)")
if not fixed:
    st.write("Het model gebruikt geen vaste kenmerken.")
for chunk, cards in rows_of(fixed):
    for card, column in zip(cards, chunk):
        raw = record.get(column)
        if column in schema["categorical"]:
            value = to_category(raw, schema["categories"].get(column, []))
            shown = display_value(column, value)
        else:
            value = None if raw is None or pd.isna(raw) else float(raw)
            shown = "Onbekend" if value is None else f"{value:g}"
        values[column] = value
        card.metric(label(column), shown)

# --- Wijzigbare gegevens ----------------------------------------------------
header_left, header_right = st.columns([4, 1])
header_left.subheader("Gegevens bij follow-up (aanpasbaar)")
if editable and header_right.button("Terug naar dossierwaarden"):
    st.session_state["reset_count"] += 1
    st.rerun()

if not editable:
    st.info(
        "Geen van de kenmerken in het model verandert doorgaans tussen follow-up momenten, dus er is niets "
        "aan te passen. Kenmerken die wel kunnen veranderen (zoals behandeling) zijn niet door de selectie gekozen."
    )

ranges = schema.get("numeric_ranges", {})
for chunk, containers in rows_of(editable):
    for container, column in zip(containers, chunk):
        widget_key = f"{key_prefix}_{column}"
        with container:
            if column in schema["categorical"]:
                categories = schema["categories"].get(column, [])
                options = ["Onbekend"] + categories
                current = to_category(record.get(column), categories)
                selected = st.selectbox(
                    label(column), options, index=options.index(current) if current in options else 0,
                    key=widget_key,
                    format_func=lambda option, column=column: display_value(column, None if option == "Onbekend" else option),
                )
                values[column] = None if selected == "Onbekend" else selected
            else:
                raw = record.get(column)
                is_unknown = raw is None or pd.isna(raw)
                low, high = ranges.get(column, (0.0, 250.0))
                default = float(schema["numeric_defaults"].get(column, low)) if is_unknown else float(raw)
                unknown = st.checkbox("Onbekend", value=is_unknown, key=f"{widget_key}_unknown")
                number = st.number_input(
                    label(column), min_value=float(low), max_value=float(high),
                    value=min(max(default, float(low)), float(high)), step=1.0,
                    key=widget_key, disabled=unknown,
                )
                values[column] = None if unknown else number

# --- Berekening -------------------------------------------------------------
if st.button("Bereken de drie modeluitkomsten", type="primary"):
    try:
        results = prediction.predict_all(bundle, values)
        st.subheader(f"Modeluitkomsten voor {names[patient_id]}")
        for card, years in zip(st.columns(3), (1, 5, 10)):
            result = results[years]
            card.metric(f"Overleving na {years} jaar", f"{result['survival_probability']:.1%}")
            card.write("Modeluitkomst: overleving" if result["survives"] else "Modeluitkomst: geen overleving tot deze termijn")
        st.caption(
            "De percentages zijn modeluitkomsten op basis van de trainingsdata; ze zijn niet klinisch "
            "gevalideerd of gekalibreerd en mogen niet als individuele prognose worden gebruikt. "
            "De binaire tekst gebruikt een modelgrens van 50%."
        )
    except Exception as error:
        st.error(f"Berekenen is mislukt: {error}")

with st.expander("Hoe zijn deze uitkomsten gedefinieerd?"):
    st.write(
        "Per termijn betekent klasse 1 dat iemand die termijn heeft overleefd; klasse 0 betekent "
        "overlijden binnen of op die termijn. Dossiers zonder bekende uitkomst voor die termijn "
        "worden voor dat afzonderlijke model niet gebruikt. De drie uitkomsten worden door drie "
        "aparte modellen berekend."
    )
with st.expander("Welke gegevens kan ik aanpassen?"):
    st.write(
        "Aanpasbaar zodra het model ze gebruikt (kunnen tussen follow-up momenten veranderen): "
        + ", ".join(label(c) for c in sorted(EDITABLE)) + ". Alle andere kenmerken liggen bij diagnose "
        "vast. Aanpassingen worden niet in het dossier opgeslagen."
    )