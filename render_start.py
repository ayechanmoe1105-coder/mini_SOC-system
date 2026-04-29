"""
render_start.py
---------------
Render.com startup shim for working_app.py.

PURPOSE: working_app.py only runs its full startup (db.create_all, log parsing,
threat detection, synthetic log generator) inside  if __name__ == '__main__':
That block NEVER runs when gunicorn imports the module.

This file:
1. Forces HOST = 0.0.0.0 (Render requires this, not 127.0.0.1)
2. Sets PORT from Render's $PORT env var
3. Enables the synthetic log generator (so dashboard has live data)
4. Runs db.create_all() so tables exist
5. Starts the log monitor background threads
6. Exposes `app` for gunicorn to import

Usage in Render:
  Start Command:  gunicorn render_start:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
"""

import os
import sys
import threading
import time
import random

# ── 1. Point to the Final/ folder ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FINAL = os.path.join(_HERE, 'Final')
if _FINAL not in sys.path:
    sys.path.insert(0, _FINAL)
os.chdir(_FINAL)          # working_app.py builds paths relative to CWD

# ── 2. Force env vars BEFORE working_app is imported ─────────────────────────
os.environ.setdefault('ENABLE_SYNTHETIC_LOG_GENERATOR', 'true')
os.environ.setdefault('SYNTHETIC_LOG_INTERVAL_SECONDS', '15')
os.environ.setdefault('SOC_PORT', os.environ.get('PORT', '15500'))
os.environ['SOC_PUBLIC_PORT'] = os.environ.get('PORT', '15500')

# ── 3. Import the Flask app object (does NOT run __main__ block) ──────────────
from working_app import app, db

# ── 4. Ensure all DB tables exist ─────────────────────────────────────────────
with app.app_context():
    db.create_all()
    print("✅ DB tables created/verified")

    # Ensure the ingested_at column exists (schema migration helper in working_app)
    try:
        from working_app import ensure_recent_activity_schema
        ensure_recent_activity_schema()
        print("✅ Schema migration applied")
    except Exception as e:
        print(f"⚠️  Schema migration skipped: {e}")

# ── 5. Create logs/ folder and seed sample log files ─────────────────────────
_logs_dir = os.path.join(_FINAL, 'logs')
os.makedirs(_logs_dir, exist_ok=True)

def _seed_sample_logs():
    """Write realistic sample log lines so the system has data to detect."""
    
    apache_log = os.path.join(_logs_dir, 'apache_access.log')
    ssh_log    = os.path.join(_logs_dir, 'ssh_auth.log')
    fw_log     = os.path.join(_logs_dir, 'firewall.log')

    # Only seed if logs are empty / missing
    if os.path.exists(apache_log) and os.path.getsize(apache_log) > 100:
        print("📋 Sample logs already present — skipping seed")
        return

    print("📋 Seeding sample log files for Render demo...")

    # ── Apache access log (SQL injection attempts) ────────────────────────────
    apache_lines = []
    for i in range(80):
        ts = f"15/Apr/2026:{8+i//10:02d}:{(i*3)%60:02d}:00 +0000"
        if i % 5 == 0:
            url = f"/search?q=1%20UNION%20SELECT%20user,password,database%20FROM%20users--"
        elif i % 7 == 0:
            url = f"/login?user=admin'%20OR%20'1'='1"
        elif i % 9 == 0:
            url = f"/page?id=1;DROP%20TABLE%20users--"
        elif i % 3 == 0:
            url = f"/index.php?page=../../etc/passwd"
        else:
            url = f"/index.html?id={i}"
        status = "404" if i % 4 == 0 else "200"
        apache_lines.append(f'10.0.0.1 - - [{ts}] "GET {url} HTTP/1.1" {status} 512\n')

    # Normal traffic mixed in
    for i in range(30):
        ts = f"15/Apr/2026:{9+i//15:02d}:{(i*2)%60:02d}:00 +0000"
        apache_lines.append(f'192.168.1.100 - - [{ts}] "GET /about.html HTTP/1.1" 200 1234\n')

    with open(apache_log, 'w') as f:
        f.writelines(apache_lines)

    # ── SSH auth log (brute force) ────────────────────────────────────────────
    ssh_lines = []
    for i in range(60):
        ts = f"Apr 15 {9+i//20:02d}:{(i*2)%60:02d}:{(i*3)%60:02d}"
        if i % 3 != 0:
            ssh_lines.append(f"{ts} server sshd[1234]: Failed password for root from 10.0.0.5 port {40000+i} ssh2\n")
        else:
            ssh_lines.append(f"{ts} server sshd[1234]: Accepted password for deploy from 192.168.1.10 port 22 ssh2\n")

    with open(ssh_log, 'w') as f:
        f.writelines(ssh_lines)

    # ── Firewall log (port scan) ──────────────────────────────────────────────
    fw_lines = []
    ports = [22, 80, 443, 3306, 5432, 6379, 8080, 8443, 9200, 27017]
    for i, port in enumerate(ports * 3):
        ts = f"Apr 15 {10+i//10:02d}:{(i*3)%60:02d}:{(i*2)%60:02d}"
        fw_lines.append(f"{ts} kernel: [UFW BLOCK] SRC=10.0.0.5 DST=192.168.1.1 PROTO=TCP DPT={port}\n")

    with open(fw_log, 'w') as f:
        f.writelines(fw_lines)

    print(f"✅ Sample logs written to {_logs_dir}")

_seed_sample_logs()

# ── 6. Run initial parse + threat detection ───────────────────────────────────
def _run_initial_detection():
    """Parse the seeded logs and detect threats so dashboard has data immediately."""
    time.sleep(3)  # wait for gunicorn to finish binding
    with app.app_context():
        try:
            from working_app import (
                log_parser, threat_detector, risk_scorer,
                LogEntry, Threat, db
            )
            import glob

            all_entries = []
            for log_file in glob.glob(os.path.join(_logs_dir, '*.log')):
                parsed = log_parser.parse_log_file(log_file)
                for p in parsed:
                    norm = log_parser.normalize_log_entry(p)
                    # Save to DB (skip duplicates)
                    exists = LogEntry.query.filter_by(
                        source_ip=norm.get('source_ip',''),
                        raw_log=norm.get('raw_log','')
                    ).first()
                    if not exists:
                        entry = LogEntry(
                            source_ip=norm.get('source_ip',''),
                            destination_port=norm.get('destination_port'),
                            protocol=norm.get('protocol',''),
                            action=norm.get('action',''),
                            raw_log=norm.get('raw_log',''),
                        )
                        db.session.add(entry)
                        all_entries.append(norm)
            db.session.commit()
            print(f"✅ Initial parse: {len(all_entries)} log entries")

            if all_entries:
                detected = threat_detector.detect_threats(all_entries)
                saved = 0
                for t in detected:
                    analysis = risk_scorer.calculate_comprehensive_risk_score(t, [])
                    threat = Threat(
                        threat_type=t.get('threat_type','unknown'),
                        source_ip=t.get('source_ip',''),
                        risk_score=analysis.get('final_score', t.get('risk_score', 5.0)),
                        description=t.get('description',''),
                        status='active',
                    )
                    db.session.add(threat)
                    saved += 1
                db.session.commit()
                print(f"✅ Initial detection: {saved} threats saved")

        except Exception as e:
            print(f"⚠️  Initial detection error: {e}")
            import traceback; traceback.print_exc()

threading.Thread(target=_run_initial_detection, daemon=True).start()

# ── 7. Start synthetic log generator (keeps dashboard live) ───────────────────
def _start_synthetic():
    """Start the built-in synthetic log generator."""
    time.sleep(5)
    try:
        from working_app import SyntheticLogGenerator
        gen = SyntheticLogGenerator(
            logs_dir=_logs_dir,
            interval=int(os.environ.get('SYNTHETIC_LOG_INTERVAL_SECONDS', '15'))
        )
        gen.start()
        print("✅ Synthetic log generator started")
    except Exception as e:
        print(f"⚠️  Could not start synthetic generator: {e}")
        # Fallback: simple loop that appends log lines
        def _fallback_gen():
            apache = os.path.join(_logs_dir, 'apache_access.log')
            ssh    = os.path.join(_logs_dir, 'ssh_auth.log')
            interval = int(os.environ.get('SYNTHETIC_LOG_INTERVAL_SECONDS', '15'))
            attack_ips = ['10.0.0.1', '10.0.0.5', '172.16.0.50']
            sqli_payloads = [
                "/search?q=UNION+SELECT+*+FROM+users",
                "/login?user=admin'--",
                "/api?id=1+OR+1=1",
                "/page?x=SELECT+password+FROM+admin",
            ]
            while True:
                time.sleep(interval)
                try:
                    now = time.strftime("%d/%b/%Y:%H:%M:%S +0000")
                    ip = random.choice(attack_ips)
                    url = random.choice(sqli_payloads)
                    status = random.choice(['200','404','500'])
                    with open(apache, 'a') as f:
                        f.write(f'{ip} - - [{now}] "GET {url} HTTP/1.1" {status} 512\n')
                    # SSH brute force bursts
                    if random.random() < 0.4:
                        now_ssh = time.strftime("%b %d %H:%M:%S")
                        with open(ssh, 'a') as f:
                            for _ in range(random.randint(3, 7)):
                                f.write(f'{now_ssh} server sshd[999]: Failed password for root from {ip} port {random.randint(40000,60000)} ssh2\n')
                except Exception:
                    pass
        threading.Thread(target=_fallback_gen, daemon=True).start()
        print("✅ Fallback log generator started")

threading.Thread(target=_start_synthetic, daemon=True).start()

# ── 8. Start the continuous log monitor ───────────────────────────────────────
def _start_monitor():
    time.sleep(8)
    try:
        from working_app import ContinuousLogMonitor
        monitor = ContinuousLogMonitor(_logs_dir, scan_interval=15)
        monitor.start()
        print("✅ ContinuousLogMonitor started")
    except Exception as e:
        print(f"⚠️  Monitor start error: {e}")

threading.Thread(target=_start_monitor, daemon=True).start()

print("🚀 render_start.py loaded — gunicorn will serve the app on 0.0.0.0")
print(f"   PORT={os.environ.get('PORT', '15500')}")
print(f"   SYNTHETIC LOGS: {os.environ.get('ENABLE_SYNTHETIC_LOG_GENERATOR','true')}")

# gunicorn imports `app` from this module
__all__ = ['app']
