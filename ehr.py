"""Lees de fictieve patiënten uit de SQLite-database ehr.db."""
import sqlite3
from contextlib import closing
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "ehr.db"


def _connect():
    if not DB_FILE.exists():
        raise FileNotFoundError("ehr.db ontbreekt. Maak de database met: python setup_database.py")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def list_patients():
    """Geef een lijst van (id, naam) voor de patiëntenkeuze."""
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT id, display_name FROM patient ORDER BY id").fetchall()
    return [(row["id"], row["display_name"]) for row in rows]


def get_patient(patient_id):
    """Geef alle gegevens van één patiënt als dictionary."""
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM patient WHERE id = ?", (patient_id,)).fetchone()
    if row is None:
        raise KeyError(f"Patiënt {patient_id} bestaat niet.")
    return dict(row)