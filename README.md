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

`train_model.py` kiest eerst data-gedreven de voorspellers (zie hieronder), traint daarna de drie modellen en schrijft `survival_models.joblib` met de modellen en het invoerschema. De tussenresultaten van elke selectiestap worden tijdens het trainen in de terminal getoond. Train opnieuw na wijzigingen aan de dataset of modelcode. De joblib hangt af van de scikit-learn-versie; train opnieuw als je die wijzigt.

## Kenmerkselectie

Alles gebeurt zonder data leakage: de selectie gebruikt alleen de trainings- en tune-set; de testset wordt pas aan het eind gebruikt.

1. **Kandidaten:** ID's, uitkomstkolommen, `incjr` (incidentiejaar), afgeleide kolommen (`stadium`, TNM-onderdelen) en kolommen met meer dan 25% missings op de trainingsset vallen af (instelbaar bovenin `train_model.py`).
2. **Screening:** univariate AUC per kenmerk; minimaal 0,52 in één van de drie termijnen.
3. **Lasso:** L1-logistische regressie op de trainingsset; kenmerken zonder niet-nul coëfficiënt vallen af.
4. **Permutation importance:** daling in AUC op de tune-set met een MLP (10 herhalingen). Alleen dalingen die duidelijk boven nul liggen (gemiddelde > 2 x standaardafwijking) tellen mee.
5. **Setvergelijking:** de kleinste top-K binnen 1% van de beste Brier-score op de tune-set, gemiddeld over de drie termijnen. De selectie is gedeeld, zodat de app één formulier toont.

Wil je een klinisch belangrijk kenmerk toch in het model houden, voeg het dan toe aan `FORCE_INCLUDE` bovenin `train_model.py` en train opnieuw.

## Patiënten en dossier

`database.sql` bevat vijf fictieve patiënten met alle kandidaat-kenmerken (elk gebaseerd op één rij uit de synthetische dataset). `python setup_database.py` maakt daaruit `ehr.db`; `ehr.py` leest die database voor de app. In de app kiest de huisarts een patiënt. De app toont alleen de kenmerken die het getrainde model gebruikt. Kenmerken die bij diagnose vastliggen worden alleen getoond; kenmerken die tussen follow-up momenten kunnen veranderen (`EDITABLE` bovenin `app.py`) kunnen worden aangepast zodra het model ze gebruikt. Aanpassingen worden niet in het dossier opgeslagen.

## Invoer en uitkomst

De interface toont de kenmerken die door de kenmerkselectie zijn gekozen. Een onbekende waarde kan als zodanig worden gekozen; de trainingspipeline vult die in met een waarde uit de trainingsdata.

Per termijn is klasse 1 gedefinieerd als overleving tot die termijn. Klasse 0 betekent overlijden binnen of op die termijn. Dossiers zonder bekende uitkomst op de betreffende termijn worden voor dat model uitgesloten. De modellen hebben elk een eigen trainingsdoel.

De percentages zijn niet klinisch gevalideerd of gekalibreerd en zijn geen individuele prognose. Gebruik de app niet voor diagnose of behandelkeuzes.