"""Maak de SQLite-database ehr.db aan op basis van database.sql.

Gebruik:  python setup_database.py

Opnieuw draaien maakt de tabel leeg en begint opnieuw, dus je kunt database.sql
aanpassen en dit script zo vaak draaien als je wilt.
"""
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
SQL_FILE = HERE / "database.sql"
DB_FILE = HERE / "ehr.db"

with sqlite3.connect(DB_FILE) as conn:
    # database.sql begint met DROP TABLE, dus dit overschrijft een bestaande tabel.
    conn.executescript(SQL_FILE.read_text(encoding="utf-8"))
    rows = conn.execute("SELECT id, display_name FROM patient ORDER BY id").fetchall()

print(f"{DB_FILE.name} aangemaakt met {len(rows)} patiënten:")
for patient_id, name in rows:
    print(f"  {patient_id}: {name}")
