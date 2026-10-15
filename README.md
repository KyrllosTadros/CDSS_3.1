# Borstkankermodel voor overleving

Deze lokale Streamlit-app toont drie afzonderlijke modeluitkomsten: overleving na 1, 5 en 10 jaar. De modellen worden getraind op `NKR_IKNL_breast_syntheticdata.csv`.

## Installeren en starten

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

`train_model.py` maakt `survival_models.joblib` met de drie modellen en de invoerschema's. Train opnieuw na wijzigingen aan de dataset of modelcode.

## Invoer en uitkomst

De interface vraagt leeftijd, klinisch stadium, tumorgrootte, positieve lymfeklieren, tumorgraad, ER-, PR- en HER2-status en de codes voor chemo- en hormoontherapie. Dit zijn de belangrijkste beschikbare patiënt-, tumor- en behandelkenmerken in het gebruikte model. Een onbekende waarde kan als zodanig worden gekozen; de trainingspipeline vult die in met een waarde uit de trainingsdata.

Per termijn is klasse 1 gedefinieerd als overleving tot die termijn. Klasse 0 betekent overlijden binnen of op die termijn. Dossiers zonder bekende uitkomst op de betreffende termijn worden voor dat model uitgesloten. De modellen hebben elk een eigen trainingsdoel.

De percentages zijn niet klinisch gevalideerd of gekalibreerd en zijn geen individuele prognose. Gebruik de app niet voor diagnose of behandelkeuzes.
