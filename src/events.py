"""Smart Tagging: adattamento di starter/load_smart_tagging.py.

Copyright (c) 2026-2027 Metrica Sports. Licenza in LICENSE-METRICA.txt.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def load_events(data_dir="data", fps=25):
    """Intervalli SportsCode, in secondi e frame della registrazione tattica.

    start_frame incluso; end_frame escluso. Un intervallo descrive una fase,
    non necessariamente un possesso o l'istante esatto della ripresa.
    """
    rows = []
    path = Path(data_dir) / "BAR-ATH_tactical_smart_tagging.xml"
    for instance in ET.parse(path).iter("instance"):
        start, end = float(instance.findtext("start")), float(instance.findtext("end"))
        if start > end:
            raise ValueError("Intervallo Smart Tagging invertito")
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        rows.append({"code": instance.findtext("code"), "start_s": start, "end_s": end,
                     "start_frame": round(start * fps), "end_frame": round(end * fps),
                     "Team": tags.get("Team"), "Half": tags.get("Half"), "Type": tags.get("Type")})
    return pd.DataFrame(rows, columns=["code", "start_s", "end_s", "start_frame", "end_frame", "Team", "Half", "Type"])


def synthetic_events():
    """Eventi inventati per illustrare lo schema senza materiali riservati."""
    return pd.DataFrame([
        {"code": "BUILD UP", "Team": "Casa", "Half": "1st Half", "Type": None,
         "start_s": 0., "end_s": 4., "start_frame": 0, "end_frame": 100},
        {"code": "SET PIECES", "Team": "N/A", "Half": "1st Half", "Type": "Throw-in",
         "start_s": 5., "end_s": 7., "start_frame": 125, "end_frame": 175},
    ])
