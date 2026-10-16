-- Een kleine, fictieve "elektronische patiëntendossier"-database met vijf
-- (verzonnen) borstkankerpatiënten, bedoeld als demonstratie voor de app.
--
-- De tabel bevat alle kandidaat-kenmerken waaruit train_model.py zijn
-- voorspellers kiest (kolommen met meer dan 25% missings en afgeleide/ID-
-- kolommen zijn weggelaten). De app toont daarvan alleen de kenmerken die het
-- getrainde model gebruikt. Codes zijn de datasetcodes uit
-- NKR_IKNL_breast_syntheticdata.csv; NULL betekent onbekend.
--
-- Elke patiënt is gebaseerd op één rij uit de synthetische dataset die het
-- best bij het beoogde profiel past (stadium, tumorgraad, leeftijd, ...),
-- zodat de combinaties van codes onderling kloppen. Onbekend-codes (zoals 9)
-- zijn als NULL opgeslagen.
--
-- Dit bestand is voor SQLite. Draai `python setup_database.py` om er de
-- database ehr.db van te maken.

DROP TABLE IF EXISTS patient;

CREATE TABLE patient (
  id                INTEGER PRIMARY KEY,   -- wordt automatisch ingevuld
  display_name      TEXT NOT NULL,
  leeft             INTEGER,  -- leeftijd bij diagnose (jaar),
  gesl              INTEGER,  -- geslacht,
  tumsoort          INTEGER,  -- tumorsoort (datasetcode),
  diag_basis        INTEGER,  -- basis van diagnose,
  topo_sublok       TEXT,  -- sublokalisatie van de tumor,
  later             INTEGER,  -- lateraliteit,
  morf              INTEGER,  -- morfologiecode,
  gedrag            INTEGER,  -- gedragscode (datasetcode),
  diffgrad          INTEGER,  -- differentiatiegraad,
  cstadium          TEXT,  -- klinisch stadium,
  pstadium          TEXT,  -- pathologisch stadium,
  ond_lymf          INTEGER,  -- aantal onderzochte lymfeklieren,
  pos_lymf          INTEGER,  -- aantal positieve lymfeklieren,
  er_stat           INTEGER,  -- ER-status,
  pr_stat           INTEGER,  -- PR-status,
  her2_stat         INTEGER,  -- HER2-status,
  dcis_comp         INTEGER,  -- DCIS-component (datasetcode),
  multifoc          INTEGER,  -- multifocaliteit (datasetcode),
  tum_afm           INTEGER,  -- tumorgrootte (mm),
  swk               INTEGER,  -- schildwachtklier (datasetcode),
  okd               INTEGER,  -- okd (datasetcode),
  org_chir          INTEGER,  -- chirurgie (datasetcode),
  uitgebr_chir_code TEXT,  -- chirurgiecode,
  dir_reconstr      INTEGER,  -- reconstructie (datasetcode),
  chemo             INTEGER,  -- chemotherapie,
  target            INTEGER,  -- doelgerichte therapie,
  horm              INTEGER,  -- hormoontherapie,
  rt                INTEGER,  -- radiotherapie,
  meta_rt           INTEGER,  -- radiotherapie bij metastasen,
  meta_chir         INTEGER  -- chirurgie bij metastasen
);

INSERT INTO patient
  (display_name, leeft, gesl, tumsoort, diag_basis, topo_sublok, later, morf, gedrag, diffgrad, cstadium, pstadium, ond_lymf, pos_lymf, er_stat, pr_stat, her2_stat, dcis_comp, multifoc, tum_afm, swk, okd, org_chir, uitgebr_chir_code, dir_reconstr, chemo, target, horm, rt, meta_rt, meta_chir)
VALUES
  ('Anna de Vries', 54, 2, 501300, 7, 'C504', 2, 8500, 3, 2, '1A', '1A', 7, 0, 1, 1, 0, 0, 0, 12, 1, 0, 1, '131C50', 0, 0, 0, 2, 2, 0, 1),
  ('Fatima El Amrani', 46, 2, 501300, 7, 'C502', 2, 8500, 3, 3, '2B', '1A', 0, 0, 0, 0, 0, 1, 0, 28, 1, 0, 1, '131C50', 0, 2, 4, 0, 2, 0, 0),
  ('Joke van den Berg', 72, 2, 502200, 7, 'C502', 2, 8500, 2, NULL, '0', '0', 8, 1, NULL, NULL, NULL, NULL, 0, 70, 1, 1, 1, '141C50', 0, 0, 1, 0, 0, 0, 0),
  ('Maria Santos', 61, 2, 501300, 7, 'C501', 1, 8500, 3, 3, '3A', '3A', 12, 9, 1, 0, 1, NULL, 0, 88, NULL, 1, 1, '142C50', 0, 2, 2, 2, 0, 0, 0),
  ('Els Jansen', 57, 2, 501300, 7, 'C508', 2, 8520, 3, 3, '4', '1B', 21, 6, 1, 1, 0, 1, 0, 27, 1, 1, 1, '131C50', 0, 0, 0, 2, 2, 0, 0),
  ('Peter Bakker', 67, 1, 501300, 7, 'C508', 2, 8520, 3, 3, '2A', '2B', 2, 0, 0, 0, 0, 1, 0, 51, 1, 0, 1, '141C50', 1, 0, 0, 0, 2, 0, 0);