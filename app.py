"""Streamlit interface voor de overlevingsmodellen na 1, 5 en 10 jaar.

Tab 1 (prognose): de huisarts kiest een (fictieve) patiënt uit het dossier. De drie
modellen gebruiken alle drie dezelfde kenmerken: leeftijd en geslacht (vast bij diagnose)
en het gedrag van de tumor (aanpasbaar bij follow-up). Aanpassingen worden niet in het
dossier opgeslagen; ze gelden alleen voor de berekening in deze sessie.

Tab 2 (consultprotocol): beslisondersteuning voor het nazorgconsult in de eerste lijn.
"""

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
    patients = dict(ehr.list_patients())  # {id: naam}
except Exception as error:
    st.error(f"Het patiëntendossier is niet beschikbaar: {error}")
    st.stop()

# --- Modelkenmerken -----------------------------------------------------------
# Leesbare tekst bij de datasetcodes van de twee categorische kenmerken.
GESLACHT = {"1": "Man", "2": "Vrouw"}
GEDRAG = {"2": "In situ", "3": "Maligne"}

# Deze app is geschreven voor precies deze drie kenmerken. Kiest train_model.py na
# opnieuw trainen andere kenmerken, dan moet de interface hieronder mee veranderen.
MODEL_FEATURES = {"leeft", "gesl", "gedrag"}
if set(bundle["schema"]["columns"]) != MODEL_FEATURES:
    st.error(
        "Het getrainde model gebruikt andere kenmerken dan deze app verwacht: "
        + ", ".join(bundle["schema"]["columns"])
        + ". Pas het modelgedeelte van app.py aan."
    )
    st.stop()


def code(value):
    """Databasewaarde (bijv. 2) als categorie-tekst van het model ('2'); None = onbekend."""
    return None if value is None else str(int(value))


# --- Patiëntkeuze -------------------------------------------------------------
patient_id = st.selectbox("Patiënt", list(patients), format_func=patients.get)
record = ehr.get_patient(patient_id)

# De sleutel van het aanpasbare veld bevat de patiënt en een teller. Bij een andere
# patiënt of na "Terug naar dossierwaarden" krijgt het veld een nieuwe sleutel en dus
# weer de dossierwaarde.
st.session_state.setdefault("reset_count", 0)
key_prefix = f"{patient_id}_{st.session_state['reset_count']}"

tab_prognose, tab_protocol = st.tabs(
    ["Overlevingsmodel (Prognose)", "Klinisch Consultprotocol"]
)

# --- TAB 1: OVERLEVINGSMODEL ----------------------------------------------------
with tab_prognose:
    # Vaste gegevens: alleen tonen, niet aanpasbaar.
    st.subheader("Vaste gegevens (bekend bij diagnose)")
    leeftijd, geslacht = record["leeft"], code(record["gesl"])
    col_leeftijd, col_geslacht, _ = st.columns([1, 1, 3])
    col_leeftijd.metric(
        "Leeftijd bij diagnose (jaar)", "Onbekend" if leeftijd is None else leeftijd
    )
    col_geslacht.metric("Geslacht", GESLACHT.get(geslacht, "Onbekend"))

    # Aanpasbaar gegeven: het gedrag van de tumor kan tussen follow-up momenten veranderen.
    header_left, header_right = st.columns([4, 1])
    header_left.subheader("Gegevens bij follow-up (aanpasbaar)")
    if header_right.button("Terug naar dossierwaarden"):
        st.session_state["reset_count"] += 1
        st.rerun()

    opties = [
        None,
        *GEDRAG,
    ]  # None = onbekend; het model vult dit dan zelf in (imputatie)
    with st.columns([1, 4])[0]:
        gedrag = st.selectbox(
            "Gedrag van de tumor",
            opties,
            index=opties.index(code(record["gedrag"])),
            format_func=lambda c: GEDRAG.get(c, "Onbekend"),
            key=f"{key_prefix}_gedrag",
        )

    # --- Berekening ---
    if st.button("Bereken de drie modeluitkomsten", type="primary"):
        try:
            results = prediction.predict_all(
                bundle, {"leeft": leeftijd, "gesl": geslacht, "gedrag": gedrag}
            )
            st.subheader(f"Modeluitkomsten voor {patients[patient_id]}")
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

    with st.expander("Welke gegevens gebruikt het model?"):
        st.write(
            "Alle drie de modellen (1, 5 en 10 jaar) gebruiken dezelfde drie kenmerken: leeftijd bij diagnose "
            "en geslacht (vast bij diagnose) en het gedrag van de tumor (aanpasbaar, omdat dit tussen follow-up "
            "momenten kan veranderen). Aanpassingen in dit portaal worden niet in het dossier opgeslagen."
        )

# --- TAB 2: KLINISCH CONSULTPROTOCOL ----------------------------------------
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
COMPLEMENTAIR = ("Handreiking Complementaire Zorg", PW + "complementaire-zorg")
OVERPZ = ("Overpalliatievezorg.nl", "https://overpalliatievezorg.nl")

# Volledige bronnenlijst onderaan het protocol, verdeeld over twee kolommen.
BRONNEN_LINKS = [NHG, WERVEL, PIJN, DYSPNEU, CALCIUM]
BRONNEN_RECHTS = [BBEWUST, COMPLEMENTAIR, VERMOEID, DEPRESSIE, THUISARTS, OVERPZ]


def toon_advies(niveau, kop, details=None, bronnen=()):
    """Toont de kernactie in een kleurvlak; onderbouwing en bronlinks staan ingeklapt eronder.

    niveau: "spoed" (rood), "actie" (oranje), "info" (blauw) of "geen" (groen).
    """
    kleurvlak = {
        "spoed": st.error,
        "actie": st.warning,
        "info": st.info,
        "geen": st.success,
    }[niveau]
    kleurvlak(kop)
    if details or bronnen:
        with st.expander("Details en onderbouwing"):
            if details:
                st.markdown(details)
            if bronnen:
                st.caption(
                    "Bron: " + " · ".join(f"[{naam}]({url})" for naam, url in bronnen)
                )


with tab_protocol:
    st.subheader(f"Oncologisch nazorgconsult: {patients[patient_id]}")
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
                    "**SPOED: neurologische symptomen passend bij mogelijke wervelmetastasen of ruggenmergcompressie.** "
                    "Overleg dezelfde dag met de behandelend specialist of dienstdoende neuroloog en verricht met spoed MRI-diagnostiek.",
                    "- Beenzwakte, ganginstabiliteit, progressieve neurologische uitval of bilaterale radiculaire klachten kunnen wijzen op metastatische epidurale spinale myelumcompressie (MESMC).\n"
                    "- Bij verdenking op MESMC is volgens de richtlijn beeldvorming binnen 12 uur geïndiceerd.\n"
                    "- *Medicamenteus:* overweeg dexamethason conform de richtlijn Wervelmetastasen: 10 mg i.v. bolus, gevolgd door 1 dd 8-16 mg oraal. Geef bij risicofactoren tevens een protonpompremmer.",
                    bronnen=[WERVEL],
                )
            if s_calcium:
                toon_advies(
                    "actie",
                    "**Verdenking hypercalciëmie:** bepaal met spoed serumcalcium (gecorrigeerd voor albumine) en nierfunctie. Overleg bij verhoogd calcium direct met de behandelend oncoloog/interne geneeskunde.",
                    "- Denk aan hypercalciëmie bij sufheid, verwardheid, polydipsie, polyurie en obstipatie.\n"
                    "- Bepaal serumcalcium (of geïoniseerd calcium indien beschikbaar) en nierfunctie.\n"
                    "- Bij ernstige hypercalciëmie (gecorrigeerd calcium > 3,0 mmol/L) of klinische achteruitgang: overleg dezelfde dag met de tweede lijn over behandeling, waaronder intraveneuze rehydratie en een bisfosfonaat",
                    bronnen=[CALCIUM],
                )
            if s_tamoxifen:
                toon_advies(
                    "actie",
                    "**Complicatie bij tamoxifen:** verwijs bij een dik, pijnlijk been voor echo-doppler (DVT); verwijs bij "
                    "vaginaal bloedverlies naar de gynaecoloog.",
                    "- Langdurig tamoxifengebruik geeft een verhoogd risico op endometriumcarcinoom.",
                    bronnen=[NHG],
                )
            if s_long:
                toon_advies(
                    "actie",
                    "**Kortademigheid/hoest:** vraag een X-thorax aan om pleuravocht, een infiltraat of lymphangitis carcinomatosa uit te sluiten.",
                    "- Bij refractaire dyspneu kan systemische morfine worden toegepast. Bij opioïdnaïeve patiënten is slow release morfine 2dd 10 mg een aanbevolen startdosering conform de richtlijn.\n"
                    "- Chronisch zuurstofgebruik wordt alleen aanbevolen bij een aangetoonde diffusiestoornis (bijvoorbeeld lymphangitis carcinomatosa of longfibrose); evalueer het effect op de dyspneu.",
                    bronnen=[DYSPNEU],
                )

            if s_bot and not s_neuro:
                toon_advies(
                    "actie",
                    "**Nieuwe of progressieve rug-/botpijn:** overleg laagdrempelig met de behandelend specialist en verricht aanvullende diagnostiek bij verdenking op bot- of wervelmetastasen.",
                    "- Vermijd langdurige conservatieve fysiotherapie zonder onderliggende diagnostiek.\n"
                    "- Bij klinische verdenking op wervelmetastasen heeft MRI van de wervelkolom de voorkeur; de urgentie is afhankelijk van de aanwezigheid van neurologische symptomen.\n"
                    "- Let op alarmsignalen zoals toenemende nachtelijke pijn, pijn bij belasting of bewegen van de wervelkolom en progressieve klachten.\n"
                    "- *Pijnbestrijding conform richtlijn Pijn bij kanker:* start met paracetamol 3 dd 1000 mg, eventueel gecombineerd met een NSAID indien niet gecontra-indiceerd. Bij onvoldoende effect kan opschaling naar een sterkwerkend opioïd worden overwogen conform de oncologische richtlijn.\n"
                    "- Overweeg vroege afstemming met het behandelteam of palliatief team bij complexe pijnklachten.",
                    bronnen=[PIJN, WERVEL],
                )

            if s_lokaal1 or s_lokaal2:
                bevinding = " én ".join(
                    tekst
                    for aanwezig, tekst in (
                        (s_lokaal1, "palpabele zwelling"),
                        (s_lokaal2, "huid-/tepelafwijking"),
                    )
                    if aanwezig
                )
                toon_advies(
                    "actie",
                    "**Verdenking (recidief van) mammacarcinoom:** verwijs naar de mammapoli voor verdere diagnostiek wegens verdenking op (recidief van) mammacarcinoom.",
                    f"- **Geregistreerde bevinding voor verwijsbrief (HIS):** {bevinding}.\n"
                    "- Beleid conform NHG-Standaard Borstkanker.",
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
        for kolom, bronnen in zip(st.columns(2), (BRONNEN_LINKS, BRONNEN_RECHTS)):
            kolom.markdown("\n".join(f"* [{naam}]({url})" for naam, url in bronnen))
