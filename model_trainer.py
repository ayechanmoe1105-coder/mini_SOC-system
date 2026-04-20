"""
ML Model Trainer — CICIDS 2017 Dataset
=======================================
Trains a Random Forest classifier (supervised) on CICIDS 2017 labelled data.
Also trains an Isolation Forest (unsupervised) as secondary anomaly detector.
Saves both models to trained_model.pkl for use in working_app.py.

Usage:
  1. Run model_trainer.py — uses the downloaded CICIDS2017_WebAttacks.csv
  2. This saves:  trained_model.pkl

Run this with the same Python interpreter you use for working_app.py (e.g. Thonny or ``py -3``).
Pickle compatibility depends on matching NumPy / scikit-learn versions at train time and load time.

If you open the SOC from Thonny: run this file in Thonny (F5) only. Do not use train_model.bat for training
if you run the dashboard from Thonny — batch training uses NumPy 2 and causes ``No module named 'numpy._core'`` on Thonny.
"""

import os, pickle, warnings
import numpy  as np
import pandas as pd
import sklearn

from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     precision_score, recall_score,
                                     f1_score, accuracy_score)

from dataset_loader import load_dataset, SELECTED_FEATURES

warnings.filterwarnings('ignore')

# ── CONFIGURATION (paths relative to this file — works on any PC / folder) ──
_BASE = os.path.dirname(os.path.abspath(__file__))
CICIDS_CSV_PATH = os.path.join(_BASE, 'data', 'CICIDS2017_WebAttacks.csv')
MODEL_SAVE_PATH = os.path.join(_BASE, 'models', 'trained_model.pkl')
RANDOM_STATE    = 42
MAX_ROWS        = 170000    # use all rows from the real file
# ───────────────────────────────────────────────────────────────────────────


def train_and_evaluate():
    print("=" * 60)
    print("  AI-Assisted SOC  --  ML Model Trainer")
    print("  Dataset  : CICIDS 2017  (Real Network Traffic)")
    print("  Method   : Random Forest (Supervised) + Isolation Forest")
    print(f"  NumPy    : {np.__version__}  |  scikit-learn : {sklearn.__version__}")
    print("=" * 60)

    used_real_csv = os.path.isfile(CICIDS_CSV_PATH)
    if not used_real_csv:
        print("\n" + "!" * 60)
        print("  WARNING: Real CSV not found at:")
        print(f"    {CICIDS_CSV_PATH}")
        print("  Training will use SYNTHETIC demo data — not real CICIDS 2017.")
        print("  The dashboard will show a warning. Place the CSV under data\\ then retrain.")
        print("!" * 60 + "\n")

    # ── 1. Load data ───────────────────────────────────────────────────────
    df = load_dataset(CICIDS_CSV_PATH, max_rows=MAX_ROWS)

    features = [f for f in SELECTED_FEATURES if f in df.columns]
    X = df[features].values
    y = df['is_attack'].values          # 0 = Normal, 1 = Attack

    print(f"\n[1] Dataset summary")
    print(f"    Source        : CICIDS 2017 (Thursday Web Attacks)")
    print(f"    Total records : {len(df):,}")
    print(f"    Features used : {len(features)}")
    print(f"    Normal  (0)   : {(y==0).sum():,}")
    print(f"    Attack  (1)   : {(y==1).sum():,}")
    print(f"    Attack types  :")
    for atype, count in df['Attack_Type'].value_counts().items():
        print(f"      {atype:30s} {count:6,}")

    # ── 2. Split ───────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[2] Train / Test split  (75% / 25%)")
    print(f"    Training : {len(X_train):,} records")
    print(f"    Testing  : {len(X_test):,} records")

    # ── 3. Scale ───────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 4. Train Random Forest (supervised — uses labels) ──────────────────
    print(f"\n[3] Training Random Forest Classifier ...")
    print(f"    n_estimators = 200  |  class_weight = balanced")
    print(f"    (class_weight=balanced corrects for class imbalance)")
    rf = RandomForestClassifier(
        n_estimators  = 200,
        class_weight  = 'balanced',   # handles imbalanced Normal vs Attack
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    rf.fit(X_train_s, y_train)
    print(f"    Training done.")

    # ── 5. Train Isolation Forest (unsupervised — backup detector) ─────────
    print(f"\n[4] Training Isolation Forest (anomaly backup) ...")
    iso = IsolationForest(
        n_estimators  = 100,
        contamination = 0.02,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    iso.fit(X_train_s[y_train == 0])   # train on normal traffic only
    print(f"    Training done.")

    # ── 6. Evaluate Random Forest ──────────────────────────────────────────
    y_pred = rf.predict(X_test_s)
    y_prob = rf.predict_proba(X_test_s)[:, 1]   # probability of attack

    acc  = accuracy_score (y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec  = recall_score   (y_test, y_pred, zero_division=0) * 100
    f1   = f1_score       (y_test, y_pred, zero_division=0) * 100
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n[5] Random Forest Evaluation Results")
    print(f"    +---------------------------------------+")
    print(f"    |  Metric          |  Score            |")
    print(f"    +---------------------------------------+")
    print(f"    |  Accuracy        |  {acc:6.2f} %         |")
    print(f"    |  Precision       |  {prec:6.2f} %         |")
    print(f"    |  Recall          |  {rec:6.2f} %         |")
    print(f"    |  F1-Score        |  {f1:6.2f} %         |")
    print(f"    +---------------------------------------+")
    print(f"\n    Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Normal  Attack")
    print(f"    Actual Normal  {tn:6,}  {fp:6,}")
    print(f"    Actual Attack  {fn:6,}  {tp:6,}")
    print(f"\n    True Positives  (attacks caught)  : {tp:,}")
    print(f"    False Positives (false alarms)    : {fp:,}")
    print(f"    True Negatives  (correct normal)  : {tn:,}")
    print(f"    False Negatives (missed attacks)  : {fn:,}")

    print(f"\n[6] Detailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Attack'],
                                zero_division=0))

    # ── 7. Feature importance ──────────────────────────────────────────────
    print(f"[7] Top 5 Most Important Features:")
    importances = rf.feature_importances_
    top5 = sorted(zip(features, importances), key=lambda x: -x[1])[:5]
    for fname, score in top5:
        bar = '#' * int(score * 200)
        print(f"    {fname:35s}  {score:.4f}  {bar}")

    # ── 8. Save model ──────────────────────────────────────────────────────
    model_data = {
        'model'        : rf,            # primary: Random Forest
        'iso_model'    : iso,           # secondary: Isolation Forest
        'scaler'       : scaler,
        'features'     : features,
        'model_type'   : 'RandomForest + IsolationForest',
        'metrics'      : {
            'accuracy'  : round(acc,  2),
            'precision' : round(prec, 2),
            'recall'    : round(rec,  2),
            'f1_score'  : round(f1,   2),
        },
        # For reports / dashboard (TN, FP, FN, TP on test set — binary Normal vs Attack)
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'dataset'      : f'CICIDS 2017 -- {os.path.basename(CICIDS_CSV_PATH)}',
        'n_train'      : len(X_train),
        'attack_types' : df['Attack_Type'].value_counts().to_dict(),
        'top_features' : [f for f, _ in top5],
        # Dashboard / debugging: synthetic training is easy to mistake for “real” CICIDS
        'training_source'   : 'real_csv' if used_real_csv else 'synthetic_demo',
        'csv_expected_path' : CICIDS_CSV_PATH,
    }
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n[8] Model saved  ->  {MODEL_SAVE_PATH}")
    print(f"    working_app.py will auto-load this model.")
    print("=" * 60)
    print("  Training complete!")
    print("=" * 60)
    return model_data


if __name__ == '__main__':
    import sys
    import traceback
    try:
        train_and_evaluate()
    except Exception as _err:
        print("\n[FATAL] Training failed — fix the error below, then run again.\n")
        traceback.print_exc()
        sys.exit(1)
