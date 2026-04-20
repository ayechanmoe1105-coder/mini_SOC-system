#!/usr/bin/env python3
"""Comprehensive API Diagnostic Test"""

import os
import requests
import sys

def test_all_apis():
    # Match working_app.py default (SOC_PORT, usually 15500 — not 5000)
    port = os.environ.get("SOC_PORT", "15500")
    base_url = f"http://127.0.0.1:{port}"
    
    endpoints = [
        ('GET', '/api/stats', 'Stats API'),
        ('GET', '/api/threats?limit=5', 'Threats API'),
        ('GET', '/api/timeline?hours=24', 'Timeline API'),
        ('GET', '/api/incidents', 'Incidents API'),
        ('GET', '/api/attack-patterns', 'Attack Patterns API'),
        ('GET', '/api/alert-rules', 'Alert Rules API'),
        ('GET', '/api/audit-logs', 'Audit Logs API'),
        ('GET', '/api/compliance/gdpr', 'GDPR API'),
        ('GET', '/api/iocs', 'IOCs API'),
        ('GET', '/api/threat-intelligence/config', 'Threat Intel API'),
        ('GET', '/api/geolocation-stats', 'Geolocation API'),
        ('GET', '/api/performance/metrics', 'Performance API'),
        ('GET', '/api/testing/results', 'Test Results API'),
    ]
    
    print("="*60)
    print("SECURITY MONITORING SYSTEM - API DIAGNOSTIC")
    print("="*60)
    
    working = []
    failed = []
    
    for method, endpoint, name in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            if method == 'GET':
                resp = requests.get(url, timeout=5)
            else:
                resp = requests.post(url, timeout=5)
            
            if resp.status_code == 200:
                print(f"✅ {name}: OK")
                working.append(name)
            else:
                print(f"❌ {name}: ERROR {resp.status_code}")
                try:
                    error_data = resp.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Response: {resp.text[:100]}")
                failed.append((name, resp.status_code))
        except Exception as e:
            print(f"❌ {name}: FAILED - {str(e)[:50]}")
            failed.append((name, str(e)[:50]))
    
    print("="*60)
    print(f"RESULTS: {len(working)} working, {len(failed)} failed")
    print("="*60)
    
    if failed:
        print("\nFailed endpoints:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = test_all_apis()
    sys.exit(0 if success else 1)
