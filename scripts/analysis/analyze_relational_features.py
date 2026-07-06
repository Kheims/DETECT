"""Describe the relational features and score the simple Feature Envy rule.

The rule flags a method when its most accessed foreign type is accessed more
often than its own class.

Usage:
    python scripts/analysis/analyze_relational_features.py
"""

import csv
import json
import statistics as st
from pathlib import Path

from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "data/relational_features.csv")))
    samples = json.loads((ROOT / "data/MLCQCodeSmellSamples.normalized.json").read_text())
    y = [int(bool(s["y"][0])) for s in samples]

    for i, s in enumerate(samples):
        if "isBelowLoadLevel" in s["code_snippet"][:300]:
            print("listing 2.1 example:", {k: v for k, v in rows[i].items() if k.startswith("l1_")})
            break

    meth = [i for i, r in enumerate(rows) if r["is_method"] == "1"]
    print(f"method snippets {len(meth)}, Feature Envy positives among them {sum(y[i] for i in meth)}, total {sum(y)}")
    for lvl in ("l0", "l1"):
        for f in ("own_total", "foreign_project", "foreign_external", "foreign_unresolved",
                  "max_single_project_type", "envy_ratio", "envy_rule"):
            pos = [float(rows[i][f"{lvl}_{f}"]) for i in meth if y[i]]
            neg = [float(rows[i][f"{lvl}_{f}"]) for i in meth if not y[i]]
            print(f"{lvl} {f:24s} pos mean {st.mean(pos):6.2f} median {st.median(pos):5.2f} | "
                  f"neg mean {st.mean(neg):6.2f} median {st.median(neg):5.2f}")
    for lvl in ("l0", "l1"):
        pred = [int(rows[i][f"{lvl}_envy_rule"]) if rows[i]["is_method"] == "1" else 0 for i in range(len(rows))]
        print(f"{lvl} envy rule: P {precision_score(y, pred):.3f} R {recall_score(y, pred):.3f} "
              f"F1 {f1_score(y, pred):.3f} MCC {matthews_corrcoef(y, pred):.3f} flagged {sum(pred)}")


if __name__ == "__main__":
    main()
