"""
Apriori on CICIDS-style tabular data (same family as Random Forest training).
---------------------------------------------------------------------------
CICIDS rows are numeric; Apriori needs categorical "items". We discretise
each selected flow feature into quantile bins (low / mid / high), then each
**row** is one **transaction** = { binned features..., outcome_attack|outcome_normal }.

This answers association questions like: "which binned feature combinations
co-occur with outcome_attack?" — complementary to Random Forest classification.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from dataset_loader import SELECTED_FEATURES, load_dataset

try:
    from mlxtend.frequent_patterns import apriori, association_rules

    _MLXTEND = True
except ImportError:
    _MLXTEND = False

_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CICIDS2017_WebAttacks.csv")


def _safe_qcut(series: pd.Series, col_name: str) -> Optional[pd.Series]:
    """3-bin quantile discretisation; fewer bins if degenerate."""
    s = series.astype(float)
    if s.nunique() < 2:
        return None
    for q, labels in ((3, ["lo", "mid", "hi"]), (2, ["lo", "hi"])):
        try:
            cats = pd.qcut(s, q=q, labels=labels[:q], duplicates="drop")
            return col_name.replace(" ", "_") + "=" + cats.astype(str)
        except ValueError:
            continue
    return None


def build_cicids_transactions(
    df: pd.DataFrame,
    max_features: int = 8,
) -> tuple[List[List[str]], List[str]]:
    """
    Each row → list of item strings (binned features + outcome label).
    """
    cand = [f for f in SELECTED_FEATURES if f in df.columns][:max_features]
    work = df[cand + ["is_attack"]].copy()
    item_cols: List[str] = []

    for c in cand:
        b = _safe_qcut(work[c], c)
        if b is None:
            continue
        ic = f"__item_{c}"
        work[ic] = b
        item_cols.append(ic)

    transactions: List[List[str]] = []
    for _, row in work.iterrows():
        items = [row[ic] for ic in item_cols]
        items.append("outcome_attack" if int(row["is_attack"]) == 1 else "outcome_normal")
        transactions.append(items)
    return transactions, item_cols


def run_cicids_apriori(
    csv_path: Optional[str] = None,
    max_rows: int = 12000,
    min_support: Optional[float] = None,
    min_confidence: float = 0.5,
    max_rules: int = 25,
) -> Dict[str, Any]:
    """
    Load CICIDS (or synthetic via dataset_loader), discretise, run Apriori.
    """
    path = csv_path or _DEFAULT_CSV
    result: Dict[str, Any] = {
        "algorithm": "apriori_cicids",
        "mlxtend_available": _MLXTEND,
        "csv_path": path,
        "max_rows_requested": max_rows,
        "transaction_definition": "Each CICIDS flow row → quantile bins of flow features + outcome_attack/outcome_normal",
        "frequent_itemsets": [],
        "association_rules": [],
        "note": None,
        "reading_guide": (
            "CICIDS is numeric; we **discretise** each column into lo/mid/hi (quantiles) so Apriori can run. "
            "Rules relate **binned flow features** to **outcome_attack** or **outcome_normal**. "
            "Random Forest learns a **nonlinear boundary** on the **raw numbers**; Apriori finds **frequent co-occurring categories**."
        ),
    }

    if not _MLXTEND:
        result["note"] = "Install mlxtend: pip install mlxtend"
        result["algorithm"] = "unavailable"
        return result

    csv_ok = os.path.isfile(path)
    df = load_dataset(path if csv_ok else None, max_rows=max_rows)
    result["csv_file_found"] = csv_ok
    result["csv_path_used"] = path if csv_ok else None
    if len(df) < 50:
        result["note"] = "Not enough rows after load — add CICIDS CSV or increase max_rows."
        return result

    transactions, _ = build_cicids_transactions(df)
    n_tx = len(transactions)
    if n_tx == 0:
        result["note"] = "No transactions built."
        return result

    all_items = sorted({x for t in transactions for x in t})
    rows = [{item: (item in set(t)) for item in all_items} for t in transactions]
    oht = pd.DataFrame(rows)

    if min_support is None:
        min_support = max(0.05, min(0.25, 3.0 / n_tx))

    result["min_support_used"] = round(min_support, 4)
    result["num_transactions"] = n_tx
    result["num_distinct_items"] = len(all_items)

    try:
        fi = apriori(oht, min_support=min_support, use_colnames=True, verbose=0)
    except Exception as e:
        result["note"] = f"Apriori failed: {e}"
        return result

    if fi is None or fi.empty:
        result["note"] = "No frequent itemsets — lower min_support or increase max_rows."
        return result

    result["frequent_itemsets"] = [
        {"items": list(row["itemsets"]), "support": round(float(row["support"]), 4)}
        for _, row in fi.iterrows()
    ][:40]

    try:
        rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
    except Exception as e:
        result["note"] = f"association_rules failed: {e}"
        return result

    if rules is None or rules.empty:
        result["note"] = f"No rules above confidence {min_confidence}. Try min_confidence=0.35."
        return result

    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(max_rules)
    for _, row in rules.iterrows():
        result["association_rules"].append(
            {
                "antecedents": list(row["antecedents"]),
                "consequents": list(row["consequents"]),
                "support": round(float(row["support"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4) if "lift" in row else None,
            }
        )

    result["used_synthetic"] = not csv_ok
    return result
