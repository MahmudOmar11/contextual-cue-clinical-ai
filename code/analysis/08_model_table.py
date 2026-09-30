"""Table S1: the models evaluated, from config/models.json.

Gemini 3.1 Pro served in both panels and is listed once.
"""
import json

import pandas as pd

from common import ROOT, TABLES

PANEL_LABEL = {"core": "Core-20", "frontier": "Frontier-3"}
RUN_DATES = {"core": "7-8 June 2026", "frontier": "21 June 2026"}
TIER_ORDER = ["open efficient", "open strong", "closed efficient", "closed flagship", "frontier"]


def main():
    entries = json.loads((ROOT / "config" / "models.json").read_text(encoding="utf-8"))["models"]
    by_model = {}
    for e in entries:
        by_model.setdefault(e["openrouter_id"], []).append(e)

    rows = []
    for es in by_model.values():
        panels = [e["panel"] for e in es]
        first = es[0]
        rows.append({
            "model": first["name"],
            "provider": first["provider"],
            "tier": first["tier"].capitalize(),
            "panel": " and ".join(PANEL_LABEL[p] for p in panels),
            "run_dates": RUN_DATES[panels[0]] if len(panels) == 1 else "7-8 and 21 June 2026",
            "returned_version": first["returned_version"],
            "_order": (TIER_ORDER.index(first["tier"]), first["provider"].lower(), first["name"]),
        })
    table = pd.DataFrame(rows).sort_values("_order").drop(columns="_order")
    table.to_csv(TABLES / "table_s1.csv", index=False)
    print(f"wrote table_s1.csv ({len(table)} models, {len(entries)} panel entries)")


if __name__ == "__main__":
    main()
