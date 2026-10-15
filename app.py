"""Streamlit interface for the 1-, 5-, and 10-year survival models."""
import streamlit as st

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

schema = bundle["schema"]
st.write(
    "Vul de belangrijkste patiënt- en tumorkenmerken in. De app toont drie afzonderlijke "
    "modeluitkomsten: overleving tot 1, 5 en 10 jaar."
)

FIELD_LABELS = {
    "leeft": "Leeftijd bij diagnose (jaar)",
    "cstadium": "Klinisch stadium (cstadium)",
    "tum_afm": "Tumorgrootte (mm)",
    "pos_lymf": "Aantal positieve lymfeklieren",
    "diffgrad": "Tumorgraad (diffgrad)",
    "er_stat": "ER-status",
    "pr_stat": "PR-status",
    "her2_stat": "HER2-status",
    "chemo": "Chemotherapie (datasetcode)",
    "horm": "Hormoontherapie (datasetcode)",
}
FIELD_HELP = {
    "cstadium": "Klinische TNM-stadiumgroep zoals gecodeerd in de trainingsdata.",
    "tum_afm": "Tumorgrootte in millimeters; 0 kan in de brondata ook onbekend betekenen.",
    "pos_lymf": "Aantal positieve lymfeklieren; onbekend wordt door het model geïmputeerd.",
    "diffgrad": "Histologische graad: 1, 2 of 3; kies Onbekend als die niet bekend is.",
    "er_stat": "ER-statuscode uit de dataset; kies Onbekend als die niet bekend is.",
    "pr_stat": "PR-statuscode uit de dataset; kies Onbekend als die niet bekend is.",
    "her2_stat": "HER2-statuscode uit de dataset; kies Onbekend als die niet bekend is.",
    "chemo": "Behandelcode uit de dataset. De invoer hangt af van de gebruikte codering.",
    "horm": "Behandelcode uit de dataset. De invoer hangt af van de gebruikte codering.",
}

values = {}
with st.form("survival_prediction_form"):
    left, right = st.columns(2)
    with left:
        for column in ("leeft", "cstadium", "tum_afm", "pos_lymf", "diffgrad"):
            label = FIELD_LABELS[column]
            if column in schema["categorical"]:
                options = ["Onbekend"] + schema["categories"].get(column, [])
                selected = st.selectbox(label, options, help=FIELD_HELP.get(column), key=f"input_{column}")
                values[column] = None if selected == "Onbekend" else selected
            else:
                default = float(schema["numeric_defaults"].get(column, 0))
                upper = 105.0 if column == "leeft" else (250.0 if column == "tum_afm" else 60.0)
                minimum = 18.0 if column == "leeft" else 0.0
                unknown = st.checkbox("Onbekend", key=f"unknown_{column}")
                values[column] = st.number_input(
                    label, min_value=minimum, max_value=upper, value=min(max(default, minimum), upper),
                    step=1.0, help=FIELD_HELP.get(column), key=f"input_{column}", disabled=unknown,
                )
                if unknown:
                    values[column] = None
    with right:
        for column in ("er_stat", "pr_stat", "her2_stat", "chemo", "horm"):
            options = ["Onbekend"] + schema["categories"].get(column, [])
            selected = st.selectbox(
                FIELD_LABELS[column], options, help=FIELD_HELP.get(column), key=f"input_{column}"
            )
            values[column] = None if selected == "Onbekend" else selected

    submitted = st.form_submit_button("Bereken de drie modeluitkomsten", type="primary")

if submitted:
    try:
        results = prediction.predict_all(bundle, values)
        st.subheader("Modeluitkomsten")
        cards = st.columns(3)
        for card, years in zip(cards, (1, 5, 10)):
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
