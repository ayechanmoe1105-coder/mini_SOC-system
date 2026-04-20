"""
Apriori association-rule mining over SOC threat co-occurrence.
----------------------------------------------------------------
Each *transaction* = distinct threat types in the same time bucket (UTC **hour**
or UTC **calendar day**). Apriori finds frequent itemsets; association_rules
derive rules like: {brute_force} -> {suspicious_pattern} with support & confidence.
With ``bucket=auto``, if every **hour** has at most one type, the miner retries
using **daily** buckets so co-occurrence can still appear.

Requires: mlxtend (see requirements.txt). If missing, falls back to pairwise
co-occurrence so the dashboard still shows something.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

try:
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules

    _MLXTEND_AVAILABLE = True
except ImportError:
    _MLXTEND_AVAILABLE = False
    pd = None  # type: ignore


def _threat_rows_to_hourly_transactions(
    threat_rows: Sequence[Any],
    hours: int = 168,
) -> List[List[str]]:
    """Build transactions: one per UTC hour, items = distinct threat_type."""
    cutoff = datetime.utcnow() - timedelta(hours=max(1, hours))
    buckets: Dict[datetime, set] = defaultdict(set)

    for t in threat_rows:
        ts = getattr(t, "timestamp", None) or getattr(t, "ingested_at", None)
        if ts is None or ts < cutoff:
            continue
        tt = (getattr(t, "threat_type", None) or "").strip()
        if not tt:
            continue
        key = ts.replace(minute=0, second=0, microsecond=0)
        buckets[key].add(tt)

    return [sorted(s) for s in buckets.values()]


def _threat_rows_to_daily_transactions(
    threat_rows: Sequence[Any],
    hours: int = 168,
) -> List[List[str]]:
    """Build transactions: one per UTC calendar date, items = distinct threat_type."""
    cutoff = datetime.utcnow() - timedelta(hours=max(1, hours))
    buckets: Dict[datetime, set] = defaultdict(set)

    for t in threat_rows:
        ts = getattr(t, "timestamp", None) or getattr(t, "ingested_at", None)
        if ts is None or ts < cutoff:
            continue
        tt = (getattr(t, "threat_type", None) or "").strip()
        if not tt:
            continue
        key = datetime(ts.year, ts.month, ts.day)
        buckets[key].add(tt)

    return [sorted(s) for s in buckets.values()]


def _transactions_to_onehot_df(transactions: List[List[str]]):
    all_items = sorted({x for trans in transactions for x in trans})
    rows = []
    for trans in transactions:
        s = set(trans)
        rows.append({item: (item in s) for item in all_items})
    return pd.DataFrame(rows), all_items


def _fallback_pairwise_rules(
    transactions: List[List[str]],
    min_support: float,
    min_confidence: float,
    max_rules: int,
) -> List[Dict[str, Any]]:
    """Simple 2-item co-occurrence when mlxtend is not installed."""
    n = len(transactions)
    if n == 0:
        return []

    item_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for trans in transactions:
        uniq = sorted(set(trans))
        for a in uniq:
            item_counts[a] += 1
        for i, a in enumerate(uniq):
            for b in uniq[i + 1 :]:
                pair_counts[tuple(sorted((a, b)))] += 1

    rules_out: List[Dict[str, Any]] = []
    for (a, b), c in pair_counts.most_common(200):
        sup = c / n
        if sup < min_support:
            continue
        conf_ab = c / item_counts[a] if item_counts[a] else 0.0
        conf_ba = c / item_counts[b] if item_counts[b] else 0.0
        if conf_ab >= min_confidence:
            rules_out.append(
                {
                    "antecedents": [a],
                    "consequents": [b],
                    "support": round(sup, 4),
                    "confidence": round(conf_ab, 4),
                    "lift": round(conf_ab / (item_counts[b] / n), 4) if n and item_counts[b] else 0.0,
                }
            )
        if conf_ba >= min_confidence and conf_ba != conf_ab:
            rules_out.append(
                {
                    "antecedents": [b],
                    "consequents": [a],
                    "support": round(sup, 4),
                    "confidence": round(conf_ba, 4),
                    "lift": round(conf_ba / (item_counts[a] / n), 4) if n and item_counts[a] else 0.0,
                }
            )

    rules_out.sort(key=lambda r: (-r["confidence"], -r["support"]))
    return rules_out[:max_rules]


def _apriori_core(
    transactions: List[List[str]],
    transaction_definition: str,
    bucket_used: str,
    hours: int,
    min_support: Optional[float],
    min_confidence: float,
    max_rules: int,
    bucket_fallback_from_hour: bool,
) -> Dict[str, Any]:
    """Run mlxtend Apriori + association_rules; caller ensures multi-type buckets exist."""
    n_tx = len(transactions)
    multi = sum(1 for t in transactions if len(t) >= 2)
    if min_support is None:
        min_support = max(0.02, min(0.5, 2.0 / n_tx)) if n_tx else 0.02

    result: Dict[str, Any] = {
        "algorithm": "apriori",
        "mlxtend_available": _MLXTEND_AVAILABLE,
        "lookback_hours": hours,
        "bucket_used": bucket_used,
        "bucket_fallback_from_hour": bucket_fallback_from_hour,
        "transaction_definition": transaction_definition,
        "num_transactions": n_tx,
        "transactions_with_2plus_items": multi,
        "min_support_used": round(min_support, 4),
        "min_confidence": min_confidence,
        "frequent_itemsets": [],
        "association_rules": [],
        "note": None,
    }

    if bucket_fallback_from_hour:
        result["note"] = (
            "Hourly UTC buckets each had at most one threat type; **daily UTC buckets** were used "
            "so co-occurring types in the same calendar day can form transactions."
        )

    if not _MLXTEND_AVAILABLE:
        result["algorithm"] = "pairwise_cooccurrence_fallback"
        fb = (
            "Install mlxtend for full Apriori: pip install mlxtend "
            "(see requirements.txt). Using pairwise fallback."
        )
        result["note"] = f"{result['note'] + ' ' if result.get('note') else ''}{fb}".strip()
        result["association_rules"] = _fallback_pairwise_rules(
            transactions, min_support, min_confidence, max_rules
        )
        if result["association_rules"]:
            bw = "day" if bucket_used == "day" else "hour"
            result["reading_guide"] = (
                f"Confidence(A→B) = fraction of UTC-{bw} buckets that contain A which also contain B. "
                "If that is 1.0, every bucket with A also has B. The reverse rule can have lower "
                f"confidence if B appears in more {bw}s than A. "
                "Lift near 1 means the link is not much stronger than B's overall frequency. "
                "Install mlxtend and re-run for classic Apriori itemsets + association_rules."
            )
        return result

    df, _ = _transactions_to_onehot_df(transactions)
    try:
        fi = apriori(df, min_support=min_support, use_colnames=True, verbose=0)
    except Exception as e:
        result["algorithm"] = "error"
        err = f"Apriori failed: {e}"
        result["note"] = f"{result['note'] + ' ' if result.get('note') else ''}{err}".strip()
        return result

    if fi is None or fi.empty:
        extra = (
            "No frequent itemsets at current min_support — try lowering min_support "
            "or collecting more varied threat_types in the same time buckets."
        )
        result["note"] = f"{result['note'] + ' ' if result.get('note') else ''}{extra}".strip()
        return result

    result["frequent_itemsets"] = [
        {
            "items": list(row["itemsets"]),
            "support": round(float(row["support"]), 4),
        }
        for _, row in fi.iterrows()
    ][:50]

    try:
        rules = association_rules(
            fi,
            metric="confidence",
            min_threshold=min_confidence,
        )
    except Exception as e:
        err = f"association_rules failed: {e}"
        result["note"] = f"{result['note'] + ' ' if result.get('note') else ''}{err}".strip()
        return result

    if rules is None or rules.empty:
        extra = (
            f"No rules above confidence {min_confidence}. "
            "Try min_confidence=0.25 or lower min_support."
        )
        result["note"] = f"{result['note'] + ' ' if result.get('note') else ''}{extra}".strip()
        return result

    rules = rules.sort_values(["confidence", "support"], ascending=False).head(max_rules)

    for _, row in rules.iterrows():
        ant = list(row["antecedents"])
        cons = list(row["consequents"])
        result["association_rules"].append(
            {
                "antecedents": ant,
                "consequents": cons,
                "support": round(float(row["support"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4) if "lift" in row else None,
            }
        )

    if result["association_rules"]:
        bw = "days" if bucket_used == "day" else "hours"
        result["reading_guide"] = (
            "From mlxtend: Support = P(A and B); Confidence(A→B) = P(B|A); "
            "Lift = confidence / P(B). Lift > 1 means B is more likely when A is present than overall. "
            f"Lift ≈ 1 often means the consequent B appears in many {bw} (marginal support ≈ 1), "
            f"or A and B are nearly independent here — confidence still tells you P(B|A) among A-{bw}."
        )

    return result


def run_apriori_on_threats(
    threat_rows: Sequence[Any],
    hours: int = 168,
    min_support: Optional[float] = None,
    min_confidence: float = 0.35,
    max_rules: int = 30,
    bucket: str = "auto",
) -> Dict[str, Any]:
    """
    Run Apriori + association rules on threat co-occurrence in time buckets.

    ``bucket``: ``hour`` | ``day`` | ``auto`` (default). With ``auto``, if every UTC **hour**
    has at most one distinct ``threat_type``, transactions are rebuilt using UTC **calendar days**
    so different types on the same day can co-occur.

    ``min_support``: if None, uses max(0.02, min(0.5, 2/n_transactions)) for small samples.
    """
    b = (bucket or "auto").strip().lower()
    if b not in ("hour", "day", "auto"):
        b = "auto"

    attempts: List[tuple[str, List[List[str]], str]] = []
    if b == "hour":
        attempts.append(
            (
                "hour",
                _threat_rows_to_hourly_transactions(threat_rows, hours=hours),
                "UTC hour bucket → set of distinct threat_type values",
            )
        )
    elif b == "day":
        attempts.append(
            (
                "day",
                _threat_rows_to_daily_transactions(threat_rows, hours=hours),
                "UTC calendar day → set of distinct threat_type values",
            )
        )
    else:
        attempts.append(
            (
                "hour",
                _threat_rows_to_hourly_transactions(threat_rows, hours=hours),
                "UTC hour bucket → set of distinct threat_type values",
            )
        )
        attempts.append(
            (
                "day",
                _threat_rows_to_daily_transactions(threat_rows, hours=hours),
                "UTC calendar day → set of distinct threat_type values",
            )
        )

    empty_note: Optional[str] = None
    for mode, transactions, tx_def in attempts:
        n_tx = len(transactions)
        multi = sum(1 for t in transactions if len(t) >= 2)
        if n_tx == 0:
            empty_note = empty_note or "No threats in the selected time window."
            continue
        if multi == 0:
            continue
        used_fallback = b == "auto" and mode == "day" and len(attempts) > 1
        out = _apriori_core(
            transactions,
            tx_def,
            mode,
            hours,
            min_support,
            min_confidence,
            max_rules,
            bucket_fallback_from_hour=used_fallback,
        )
        out["bucket_requested"] = b
        return out

    last_mode, last_transactions, last_def = attempts[-1]
    n_tx = len(last_transactions)
    multi = sum(1 for t in last_transactions if len(t) >= 2)
    min_s = min_support if min_support is not None else (max(0.02, min(0.5, 2.0 / n_tx)) if n_tx else 0.02)
    return {
        "algorithm": "apriori",
        "mlxtend_available": _MLXTEND_AVAILABLE,
        "lookback_hours": hours,
        "bucket_requested": b,
        "bucket_used": last_mode,
        "bucket_fallback_from_hour": False,
        "transaction_definition": last_def,
        "num_transactions": n_tx,
        "transactions_with_2plus_items": multi,
        "min_support_used": round(min_s, 4),
        "min_confidence": min_confidence,
        "frequent_itemsets": [],
        "association_rules": [],
        "note": empty_note
        or (
            "No co-occurring threat types in the same time bucket. "
            + (
                "Tried UTC hour then UTC calendar day — each bucket still had at most one threat type. "
                if b == "auto" and len(attempts) > 1
                else f"Tried UTC {last_mode} buckets only — each had at most one threat type. "
            )
            + "Generate logs with varied threat_types, increase `hours` (e.g. 720), "
            + "or use **Apriori (CICIDS flows)** for flow-based rules."
        ),
    }
