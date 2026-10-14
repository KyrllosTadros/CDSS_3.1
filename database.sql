-- A tiny "electronic health record" with five (fictional) patients.
-- The values are the 8 inputs the model expects, in the same order as the
-- training data: pregnancies, glucose, diastolic blood pressure, skin
-- thickness, insulin, BMI, diabetes pedigree function, age.
--
-- This file is written for SQLite. Run `python setup_database.py` to turn it
-- into the database file ehr.db.

DROP TABLE IF EXISTS patient;

CREATE TABLE patient (
  id                  INTEGER PRIMARY KEY,   -- filled in automatically
  display_name        TEXT NOT NULL,
  num_pregnancies     INTEGER,
  glucose             INTEGER,
  blood_pressure_dias INTEGER,
  skin_thickness      INTEGER,
  insulin             INTEGER,
  BMI                 REAL,
  diab_pedigree       REAL,
  age                 INTEGER
);

INSERT INTO patient
  (display_name, num_pregnancies, glucose, blood_pressure_dias, skin_thickness,
   insulin, BMI, diab_pedigree, age)
VALUES
  ('Susan Foreman',    6, 148, 72, 35,   0, 33.6, 0.627, 50),
  ('Sarah Jane Smith', 2, 106, 56, 27, 165, 29.0, 0.426, 22),
  ('Tegan Jovanka',    2, 174, 88, 37, 120, 44.5, 0.646, 24),
  ('Amy Pond',         4,  95, 60, 32,   0, 35.4, 0.284, 28),
  ('Yasmin Khan',      0, 126, 86, 27, 120, 27.4, 0.515, 21);