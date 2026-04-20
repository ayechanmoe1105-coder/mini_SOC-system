"""
verify_project.py — Quick health check for the SOC project.
Run:  py verify_project.py

Checks: critical files exist, imports work, Flask responds, ML model loads.
"""
from __future__ import annotations

import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

def ok(msg: str) -> None:
    print(f"  [OK] {msg}")

def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")

def main() -> int:
    print("SOC Project — verification\n")
    errors = 0

    # 1. Required files
    required = [
        "working_app.py",
        "config.py",
        "log_parser.py",
        "threat_detector.py",
        "risk_scorer.py",
        "alert_system.py",
        "ai_explainer.py",
        "risk_scorer.py",
        "dataset_loader.py",
        "templates/dashboard.html",
        "models/trained_model.pkl",
        "data/CICIDS2017_WebAttacks.csv",
    ]
    for name in required:
        path = os.path.join(BASE, name)
        if os.path.isfile(path):
            ok(f"file: {name}")
        else:
            fail(f"missing: {name}")
            errors += 1

    # 2. Import working_app (may take a few seconds)
    print()
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from working_app import app, CICIDS_MODEL_AVAILABLE, db
        ok("import working_app")
        if CICIDS_MODEL_AVAILABLE:
            ok("ML model loaded (trained_model.pkl)")
        else:
            fail("ML model NOT loaded — run: py model_trainer.py")
            errors += 1
    except Exception as e:
        fail(f"import working_app: {e}")
        return 1

    # 3. Flask test client
    print()
    try:
        with app.app_context():
            c = app.test_client()
            for path, method in [("/api/stats", "GET"), ("/api/model/info", "GET"), ("/", "GET")]:
                r = c.get(path)
                if r.status_code == 200:
                    ok(f"{method} {path} -> 200")
                else:
                    fail(f"{method} {path} -> {r.status_code}")
                    errors += 1
    except Exception as e:
        fail(f"Flask test: {e}")
        errors += 1

    print()
    if errors == 0:
        print("All checks passed. Start the server with:  py working_app.py")
        print("Then open the URL printed in the console (default port is SOC_PORT=15500, not 5000).")
        return 0
    print(f"Finished with {errors} issue(s). Fix the items above.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
