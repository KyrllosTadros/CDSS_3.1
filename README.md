# Borstkankermodel voor metastasen

Deze Streamlit-app gebruikt een scikit-learn `MLPClassifier`. De dataset wordt lokaal gelezen uit `NKR_IKNL_breast_syntheticdata.csv`. CSV-bestanden en het getrainde model zijn uitgesloten van Git.

## Eenmalige installatie

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Trainen en starten

Voer vanuit deze projectmap uit:

```bash
python train_model.py
streamlit run app.py
```

Het trainingsscript maakt lokaal `model.joblib`. De Streamlit-app laadt dit bestand en biedt invoervelden voor de kenmerken. Train opnieuw met `python train_model.py` om het model te vervangen.

## Modelverwerking

De target `uitkomst` wordt afgeleid uit `pm`, met `cm` als terugval als `pm` ontbreekt. `pm`, `cm`, `key_nkr`, `key_eid` en `topo` worden niet als modelinvoer gebruikt. Kolommen met meer dan 25% ontbrekende waarden worden weggelaten. Imputatie, schaling en categorische codering worden binnen de trainingspipeline geleerd en bij voorspellen opnieuw toegepast.

Deze app is uitsluitend een technische demonstratie; niet gebruiken voor diagnose of behandelkeuzes. Een modeluitkomst is geen klinisch gevalideerde of gekalibreerde kans.
