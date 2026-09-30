"""Shared paths, data loaders and helpers for the analysis scripts."""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
ESTIMATES = ROOT / "results" / "estimates"
for _d in (TABLES, FIGURES, ESTIMATES):
    _d.mkdir(parents=True, exist_ok=True)

VERSIONS = ["no_cue", "cue"]
PROMPTS = ["baseline", "base_rate", "rare_or_serious"]
PROMPT_LABEL = {"baseline": "Baseline", "base_rate": "Base-rate prompt", "rare_or_serious": "Rare-or-serious prompt"}
FRONTIER = ["anthropic/claude-opus-4.8", "openai/gpt-5.5", "google/gemini-3.1-pro-preview"]


def cases():
    return pd.read_csv(DATA / "cases.csv")


def form_families():
    """The 21 case families on the clinician form (the matched comparison)."""
    c = cases()
    return sorted(c.loc[c.clinician_form == 1, "family"])


def clinician_responses():
    d = pd.read_csv(DATA / "clinician_responses.csv", keep_default_na=False)
    d["cue_associated"] = (d.chosen == "cue_associated").astype(int)
    d["intended"] = (d.chosen == "intended").astype(int)
    return d.merge(pd.read_csv(DATA / "clinician_participants.csv"), on="participant", how="left")


def model_responses(prompt=None):
    d = pd.read_csv(DATA / "model_responses.csv.gz", keep_default_na=False)
    if prompt is not None:
        d = d[d.prompt == prompt].copy()
    d["cue_associated"] = (d.selected_role == "cue_associated").astype(int)
    d["intended"] = (d.selected_role == "intended").astype(int)
    d["valid"] = (d.selected_role != "no_valid_answer").astype(int)
    return d


def explanation_responses():
    d = pd.read_csv(DATA / "explanation_responses.csv.gz", keep_default_na=False)
    d["cue_associated"] = (d.selected_role == "cue_associated").astype(int)
    d["intended"] = (d.selected_role == "intended").astype(int)
    return d


def physician_ratings():
    return pd.read_csv(DATA / "physician_ratings.csv")


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d
