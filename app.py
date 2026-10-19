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

st.set_page_config(
    page_title="Borstkanker – overlevingsmodel", page_icon="🩺", layout="wide"
)
st.title("Borstkankermodel: overleving na 1, 5 en 10 jaar")
st.caption(
    "Technische demonstratie op synthetische data. Niet gebruiken voor diagnose of behandelkeuzes."
)

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
EDITABLE = {"gedrag"}

FIELD_LABELS = {
    "leeft": "Leeftijd bij diagnose (jaar)",
    "gesl": "Geslacht",
    "tumsoort": "Tumorsoort",
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
    "tumsoort": {
        "501300": "Invasief mammacarcinoom",
        "502200": "Ductaal carcinoma in situ (DCIS)",
        "503200": "Lobulair carcinoma in situ (LCIS)",
    },
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
        chunk = items[start : start + per_row]
        yield chunk, st.columns(per_row)[: len(chunk)]


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
        + ", ".join(missing_in_record)
        + ". Vul `database.sql` aan en draai `python setup_database.py`."
    )

if "reset_count" not in st.session_state:
    st.session_state["reset_count"] = 0
key_prefix = f"{patient_id}_{st.session_state['reset_count']}"

# --- TAB NAVIGATIE TUSSEN MODEL EN KENNISONDERDEEL --------------------------
tab_prognose, tab_protocol = st.tabs(
    ["Overlevingsmodel (Prognose)", "Klinisch Consultprotocol"]
)


with tab_prognose:
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
                        label(column),
                        options,
                        index=options.index(current) if current in options else 0,
                        key=widget_key,
                        format_func=lambda option, column=column: display_value(
                            column, None if option == "Onbekend" else option
                        ),
                    )
                    values[column] = None if selected == "Onbekend" else selected
                else:
                    raw = record.get(column)
                    is_unknown = raw is None or pd.isna(raw)
                    low, high = ranges.get(column, (0.0, 250.0))
                    default = (
                        float(schema["numeric_defaults"].get(column, low))
                        if is_unknown
                        else float(raw)
                    )
                    unknown = st.checkbox(
                        "Onbekend", value=is_unknown, key=f"{widget_key}_unknown"
                    )
                    number = st.number_input(
                        label(column),
                        min_value=float(low),
                        max_value=float(high),
                        value=min(max(default, float(low)), float(high)),
                        step=1.0,
                        key=widget_key,
                        disabled=unknown,
                    )
                    values[column] = None if unknown else number

    # --- Berekening -------------------------------------------------------------
    if st.button("Bereken de drie modeluitkomsten", type="primary"):
        try:
            results = prediction.predict_all(bundle, values)
            st.subheader(f"Modeluitkomsten voor {names[patient_id]}")
            for card, years in zip(st.columns(3), (1, 5, 10)):
                result = results[years]
                card.metric(
                    f"Overleving na {years} jaar",
                    f"{result['survival_probability']:.1%}",
                )
                card.write(
                    "Modeluitkomst: overleving"
                    if result["survives"]
                    else "Modeluitkomst: geen overleving tot deze termijn"
                )
            st.caption(
                "De percentages zijn modeluitkomsten op basis van de trainingsdata; ze zijn niet klinisch "
                "gevalideerd of gekalibreerd en mogen niet als individuele prognose worden gebruikt. "
                "De binaire tekst gebruikt een modelgrens van 50%."
            )
        except Exception as error:
            st.error(f"Berekenen is mislukt: {error}")

    with st.expander("Welke variabelen gebruikt elk model?"):
        # Per termijn gelezen uit het getrainde model zelf, zodat dit klopt na opnieuw trainen.
        used = {
            years: set(bundle["models"][years].feature_names_in_)
            for years in (1, 5, 10)
        }
        all_used = [c for c in columns if any(c in used[y] for y in used)]
        st.dataframe(
            pd.DataFrame(
                {
                    "Variabele": [label(c) for c in all_used],
                    "1 jaar": ["✓" if c in used[1] else "–" for c in all_used],
                    "5 jaar": ["✓" if c in used[5] else "–" for c in all_used],
                    "10 jaar": ["✓" if c in used[10] else "–" for c in all_used],
                    "Aanpasbaar": [
                        "Ja" if c in EDITABLE else "Nee (vast bij diagnose)"
                        for c in all_used
                    ],
                }
            ),
            hide_index=True,
        )
        st.caption(
            f"Aantal variabelen per model: 1 jaar = {len(used[1])}, 5 jaar = {len(used[5])}, 10 jaar = {len(used[10])}."
        )
    with st.expander("Welke gegevens kan ik aanpassen?"):
        st.write(
            "De volgende variabelen zijn aanpasbaar (deze kunnen tussen follow-up momenten veranderen): "
            + ", ".join(label(c) for c in sorted(EDITABLE))
            + ". Alle andere kenmerken liggen bij diagnose "
            "vast. Aanpassingen in dit portaal worden niet in het dossier opgeslagen."
        )

# TAB 2: KLINISCH CONSULTPROTOCOL
# Opzet (Gestalt): elk beslismoment staat in een eigen kader (gemeenschappelijk gebied), het advies staat
# direct onder de vraag (nabijheid), kleur betekent urgentie (gelijkheid): rood = spoed, oranje = actie,
# blauw = informatie, groen = geen actie. Details staan in een uitklapper zodat de actie het eerst opvalt.
NHG = (
    "NHG-Standaard Borstkanker (M103)",
    "https://richtlijnen.nhg.org/standaarden/borstkanker",
)
PW = "https://palliaweb.nl/richtlijnen-palliatieve-zorg/richtlijn/"
WERVEL = ("Richtlijn Wervelmetastasen", PW + "wervelmetastasen")
PIJN = ("Richtlijn Pijn bij patiënten met kanker", PW + "pijn-bij-patienten-met-kanker")
DYSPNEU = ("Richtlijn Dyspneu in de palliatieve fase", PW + "dyspneu")
CALCIUM = ("Richtlijn Hypercalciëmie in de palliatieve fase", PW + "hypercalciemie")
VERMOEID = ("Richtlijn Vermoeidheid bij kanker", PW + "vermoeidheid-bij-kanker")
DEPRESSIE = ("Richtlijn Depressie in de palliatieve fase", PW + "depressie")
BBEWUST = ("Borstkankervereniging Nederland – B-bewust", "https://www.b-bewust.nl")
THUISARTS = ("Thuisarts.nl – Borstkanker", "https://www.thuisarts.nl/borstkanker")


def toon_advies(niveau, kop, details=None, bronnen=()):
    """Toont de kernactie in een kleurvlak; onderbouwing en bronlinks staan ingeklapt eronder."""
    {"spoed": st.error, "actie": st.warning, "info": st.info, "geen": st.success}[
        niveau
    ](kop)
    if details or bronnen:
        with st.expander("Details en onderbouwing"):
            if details:
                st.markdown(details)
            if bronnen:
                st.caption(
                    "Bron: " + " · ".join(f"[{naam}]({url})" for naam, url in bronnen)
                )


with tab_protocol:
    st.subheader(f"Oncologisch nazorgconsult: {names[patient_id]}")
    st.caption(
        "Eerstelijns beslisondersteuning conform NHG-Standaard Borstkanker en Palliaweb/IKNL. "
        "Ondersteuning bij het oordeel van de huisarts, geen vervanging ervan."
    )
    k = f"{patient_id}_consult"  # sleutelvoorvoegsel: antwoorden horen bij de gekozen patiënt

    # ---------------- Beslismoment 1 ----------------
    with st.container(border=True):
        st.markdown("#### 1. Zijn er nieuwe klachten of alarmsignalen?")
        st.caption(
            "Vink aan wat tijdens anamnese en lichamelijk onderzoek aanwezig is."
        )
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("**Mamma & lokaal gebied**")
            s_lokaal1 = st.checkbox(
                "Nieuwe palpabele zwelling in aangedane/contralaterale borst, litteken of oksel",
                key=f"{k}_s1",
            )
            s_lokaal2 = st.checkbox(
                "Huid- of tepelafwijkingen (intrekking, oedeem/peau d'orange, eczeem, bloederige tepeluitvloed)",
                key=f"{k}_s2",
            )
            st.markdown("**Skelet & wervelkolom**")
            s_bot = st.checkbox(
                "Nieuwe aanhoudende of progressieve (nachtelijke) rug- of botpijn",
                key=f"{k}_s3",
            )
            s_neuro = st.checkbox(
                "Neurologische prikkeling/uitval (radiculaire pijn, beenzwakte, ganginstabiliteit)",
                key=f"{k}_s4",
            )
        with col_k2:
            st.markdown("**Luchtwegen**")
            s_long = st.checkbox(
                "Onverklaarde progressieve dyspneu of aanhoudende droge hoest",
                key=f"{k}_s5",
            )
            st.markdown("**Systeem- & medicatiecomplicaties**")
            s_calcium = st.checkbox(
                "Sufheid, polydipsie (extreme dorst), polyurie en obstipatie (verdenking hypercalciëmie)",
                key=f"{k}_s6",
            )
            s_tamoxifen = st.checkbox(
                "Pijnlijk gezwollen onderbeen (DVT) of vaginaal bloedverlies bij tamoxifengebruik",
                key=f"{k}_s7",
            )

        st.divider()
        if not any(
            [s_lokaal1, s_lokaal2, s_bot, s_neuro, s_long, s_calcium, s_tamoxifen]
        ):
            toon_advies(
                "geen",
                "**Geen alarmsignalen.** Geen directe indicatie voor aanvullende beeldvorming of acute "
                "tweedelijnsconsultatie. Ga verder met beslismoment 2.",
            )
        else:
            st.markdown(
                "**Advies: beoordeel de klachten en overleg of verwijs gericht** (hoogste urgentie bovenaan)"
            )
            if s_neuro:
                toon_advies(
                    "spoed",
                    "**SPOED: verdenking ruggenmergcompressie.** Overleg vandaag nog met de dienstdoende neuroloog en vraag "
                    "binnen 12 uur een MRI van de gehele wervelkolom aan.",
                    "- Bij beenzwakte, loopstoornissen of radiculair bandgevoel bestaat verdenking op ruggenmergcompressie (Richtlijn Wervelmetastasen).\n"
                    "- *Medicamenteus:* start direct dexamethason 10 mg i.v. bolus, gevolgd door 1 dd 16 mg oraal, met maagbescherming (protonpompremmer).",
                    bronnen=[WERVEL],
                )
            if s_calcium:
                toon_advies(
                    "actie",
                    "**Verdenking hypercalciëmie:** bepaal met spoed serumcalcium (gecorrigeerd voor albumine) en nierfunctie.",
                    "- Bij gecorrigeerd calcium > 3,0 mmol/l: direct overleggen over intraveneuze rehydratie (NaCl 0,9%) en een "
                    "bisfosfonaat (zoledroninezuur 4 mg i.v. in 15 minuten). Richtlijn Hypercalciëmie (PZNL).",
                    bronnen=[CALCIUM],
                )
            if s_tamoxifen:
                toon_advies(
                    "actie",
                    "**Complicatie bij tamoxifen:** verwijs bij een dik, pijnlijk been voor echo-doppler (DVT); verwijs bij "
                    "vaginaal bloedverlies naar de gynaecoloog.",
                    "- Langdurig tamoxifengebruik geeft een verhoogd risico op endometriumcarcinoom (NHG-Standaard Borstkanker).",
                    bronnen=[NHG],
                )
            if s_long:
                toon_advies(
                    "actie",
                    "**Kortademigheid/hoest:** vraag een X-thorax aan om pleuravocht, een infiltraat of lymphangitis carcinomatosa uit te sluiten.",
                    "- Bij refractaire dyspneu is de eerste keus systemisch morfine (slow release oraal 2dd 10 mg of 6dd 2,5 mg s.c./i.v.).\n"
                    "- Zuurstof is pas geïndiceerd bij aangetoonde diffusiestoornissen. Richtlijn Dyspneu.",
                    bronnen=[DYSPNEU],
                )
            if s_bot and not s_neuro:
                toon_advies(
                    "actie",
                    "**Aanhoudende rug-/botpijn:** vraag röntgendiagnostiek aan of overleg met de specialist over een botscan/MRI.",
                    "- Vermijd langdurige conservatieve fysiotherapie zonder onderliggende diagnostiek.\n"
                    "- *Pijnbestrijding (Richtlijn Pijn bij kanker):* stap 1 paracetamol 4dd 1000 mg; bij onvoldoende effect "
                    "laagdrempelig stap 2: sterkwerkend opioïd (morfine of oxycodon), zwakwerkende opioïden overslaan.",
                    bronnen=[PIJN, WERVEL],
                )
            if s_lokaal1 or s_lokaal2:
                toon_advies(
                    "actie",
                    "**Verdenking lokaal recidief:** verwijs naar de mammapoli voor mammografie en echografie, zo nodig met cytologie/histologie.",
                    "- Conform NHG-Standaard Borstkanker.",
                    bronnen=[NHG],
                )

    # ---------------- Beslismoment 2 ----------------
    with st.container(border=True):
        st.markdown("#### 2. Is het follow-upplan duidelijk?")
        fu = st.radio(
            "Is er een duidelijk en actueel follow-up- en nazorgplan vastgelegd tussen ziekenhuis en huisarts?",
            ["Ja", "Nee"],
            index=None,
            horizontal=True,
            key=f"{k}_fu",
        )
        st.caption(
            "Ja: het plan is actueel en de afspraken over beeldvorming en taakverdeling zijn helder. "
            "Nee: er ontbreekt een recent ontslagbericht of nazorgplan, of het is onduidelijk wie de regie voert."
        )
        if fu == "Ja":
            toon_advies(
                "info",
                "**Volg het bestaande follow-upplan.**",
                "- **Controleschema:** handhaaf de geplande periodieke controles (mammografie en klinisch borstonderzoek in de 2e lijn).\n"
                "- **Eerste lijn:** bij > 5 jaar na een borstsparende behandeling en leeftijd > 60 jaar kan de nacontrole naar de huisarts "
                "(jaarlijkse palpatie, tweejaarlijks mammogram in het ziekenhuis).\n"
                "- **Medicatie:** controleer therapietrouw en bijwerkingen van adjuvante antihormonale therapie (tamoxifen of aromataseremmers).",
                bronnen=[NHG],
            )
        elif fu == "Nee":
            toon_advies(
                "actie",
                "**Overleg met het behandelteam over de follow-up.**",
                "- Neem contact op met de regievoerend medisch specialist of mammacareverpleegkundige.\n"
                "- Vraag het oncologisch nazorgplan op en leg afspraken over taakverdeling en over wie de periodieke beeldvorming "
                "aanvraagt vast in het HIS.",
                bronnen=[NHG],
            )

    # ---------------- Beslismoment 3 ----------------
    with st.container(border=True):
        st.markdown("#### 3. Is extra ondersteuning nodig?")
        ond = st.radio(
            "Heeft de patiënt behoefte aan extra psychosociale, fysieke of praktische ondersteuning?",
            ["Ja", "Nee"],
            index=None,
            horizontal=True,
            key=f"{k}_ond",
        )
        st.caption(
            "Denk aan aanhoudende vermoeidheid, somberheid, arm-/schouderklachten of lymfoedeem, vragen over voeding "
            "of behoefte aan gespreksvoorbereiding."
        )
        if ond == "Ja":
            toon_advies(
                "info",
                "**Bied ondersteuning of verwijs indien nodig.**",
                "- **Gespreksvoorbereiding:** reik de B-bewust checklist aan van Borstkankervereniging Nederland "
                "([www.b-bewust.nl](https://www.b-bewust.nl)).\n"
                "- **Fysiek:** verwijs bij arm-/schouderklachten of lymfoedeem naar een oedeem- of oncologiefysiotherapeut "
                "([Verwijsgids Kanker](https://verwijsgidskanker.nl)); pas bij vermoeidheid energiemanagement toe.\n"
                "- **Mentaal:** POH-GGZ of eerstelijnspsycholoog bij aanhoudende somberheid of angst; wijs op inloophuizen (IPSO) "
                "voor lotgenotencontact.",
                bronnen=[BBEWUST, VERMOEID, DEPRESSIE],
            )
        elif ond == "Nee":
            toon_advies(
                "info",
                "**Bespreek beschikbare nazorg en contactmogelijkheden.**",
                "- **Voorlichting:** wijs op betrouwbare informatie via [Thuisarts.nl](https://www.thuisarts.nl/borstkanker) en "
                "[Kanker.nl](https://www.kanker.nl).\n"
                "- **Bereikbaarheid:** spreek af bij welke veranderende klachten of toenemende ongerustheid de patiënt laagdrempelig "
                "contact opneemt met de praktijk.",
                bronnen=[THUISARTS],
            )

    with st.expander("Geraadpleegde medische richtlijnen & bronnen"):
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            st.markdown("""
            * [NHG-Standaard Borstkanker (M103)](https://richtlijnen.nhg.org/standaarden/borstkanker)
            * [Richtlijn Wervelmetastasen](https://palliaweb.nl/richtlijnen-palliatieve-zorg/richtlijn/wervelmetastasen)
            * [Richtlijn Pijn bij patiënten met kanker](https://palliaweb.nl/richtlijnen-palliatieve-zorg/richtlijn/pijn-bij-patienten-met-kanker)
            * [Richtlijn Dyspneu in de palliatieve fase](https://palliaweb.nl/richtlijnen-palliatieve-zorg/richtlijn/dyspneu)
            * [Richtlijn Hypercalciëmie in de palliatieve fase](https://palliaweb.nl/richtlijnen-palliatieve-zorg/richtlijn/hypercalciemie)
            """)
        with c_b2:
            st.markdown("""
            * [Borstkankervereniging Nederland – B-bewust](https://www.b-bewust.nl)
            * [Handreiking Complementaire Zorg](https://palliaweb.nl/richtlijnen-palliatieve-zorg/richtlijn/complementaire-zorg)
            * [Richtlijn Vermoeidheid bij kanker](https://palliaweb.nl/richtlijnen-palliatieve-zorg/richtlijn/vermoeidheid-bij-kanker)
            * [Richtlijn Depressie in de palliatieve fase](https://palliaweb.nl/richtlijnen-palliatieve-zorg/richtlijn/depressie)
            * [Thuisarts.nl](https://www.thuisarts.nl/borstkanker) | [Overpalliatievezorg.nl](https://overpalliatievezorg.nl)
            """)
