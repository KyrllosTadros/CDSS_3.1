"""Local Streamlit interface for the breast-cancer metastasis MLP."""
import streamlit as st

import prediction

st.set_page_config(page_title="Borstkankermodel", page_icon="🩺")
st.title("Borstkankermodel voor metastasen")
st.caption("Technische demonstratie. Niet gebruiken voor medische beslissingen.")

try:
    bundle = prediction.load_model_bundle()
except Exception as error:
    st.error(f"Model niet beschikbaar: {error}")
    st.info("Voer eerst `python train_model.py` uit om het lokale model te trainen.")
    st.stop()

schema = bundle["schema"]
st.write("Vul de kenmerken van één patiënt in. De verwerking komt overeen met de training.")
values = {}
with st.form("prediction_form"):
    for column in schema["columns"]:
        if column in schema["categorical"]:
            options = schema["categories"].get(column, [])
            if options:
                values[column] = st.selectbox(column, options, key=f"input_{column}")
            else:
                values[column] = st.text_input(column, key=f"input_{column}")
        else:
            default = schema["numeric_defaults"].get(column, 0.0)
            values[column] = st.number_input(column, value=float(default), key=f"input_{column}")
    submitted = st.form_submit_button("Voorspel")

if submitted:
    try:
        label, probability = prediction.predict_one(bundle, values)
        st.subheader("Modeluitkomst")
        st.write("Metastase op afstand volgens het model" if label == 1 else "Geen metastase op afstand volgens het model")
        if probability is not None:
            st.metric("Modelkans voor klasse 1", f"{probability:.1%}")
        st.caption("Dit is een modeluitkomst, geen klinisch gevalideerde of gekalibreerde kans.")
    except Exception as error:
        st.error(f"Voorspellen is mislukt: {error}")
