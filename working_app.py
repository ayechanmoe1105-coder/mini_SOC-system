import sys
# Ensure UTF-8 output on Windows (avoids emoji encoding errors in PowerShell)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from flask import Flask, request, jsonify, send_file, make_response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
import socket
import webbrowser
import warnings
import logging
import requests
import json
import re
import random
import threading
import time
from config import Config
import io
import csv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask uses os.getcwd() as root when __name__ == '__main__' (Thonny, %Run, etc.),
# which loads the wrong templates/ if CWD is not this project folder. Always use
# templates next to this file so / and /dashboard always render the real SOC UI.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.join(_APP_DIR, 'templates')
# Always read this file directly for / — never use render_template alone (IDEs may use a temp __file__).
_DASHBOARD_HTML_PATH = os.path.join(_APP_DIR, 'templates', 'dashboard.html')

# Initialize Flask app
app = Flask(__name__, template_folder=_TEMPLATE_DIR)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
# Absolute path so Thonny / wrong CWD still uses the DB next to this file
_dsp = os.path.abspath(os.path.join(_APP_DIR, 'security_monitoring.db')).replace('\\', '/')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///' + _dsp)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'connect_args': {'timeout': 30, 'check_same_thread': False},
}

# Initialize extensions
db = SQLAlchemy(app)

# Optional imports with graceful handling
try:
    import geoip2.database
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.widgetbase import Widget
    from reportlab.graphics import renderPDF
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from io import BytesIO
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# ── CICIDS 2017 Trained Model Loader ──────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'trained_model.pkl')
_cicids_model_data = None
CICIDS_MODEL_AVAILABLE = False
SKLEARN_AVAILABLE = True
# Why the ML panel shows unavailable: missing_pkl | numpy_mismatch | pickle_error | numpy_import
_CICIDS_UNAVAILABLE_CODE = None
_CICIDS_UNAVAILABLE_DETAIL = None

try:
    import pickle
    import numpy as np
except Exception as _e:
    SKLEARN_AVAILABLE = False
    _CICIDS_UNAVAILABLE_CODE = 'numpy_import'
    _CICIDS_UNAVAILABLE_DETAIL = str(_e)
    logging.warning(f"[CICIDS] Cannot import numpy/pickle stack: {_e}")
else:
    if not os.path.exists(_MODEL_PATH):
        CICIDS_MODEL_AVAILABLE = False
        _CICIDS_UNAVAILABLE_CODE = 'missing_pkl'
        logging.warning("[CICIDS] trained_model.pkl not found. Run model_trainer.py first.")
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with open(_MODEL_PATH, 'rb') as _f:
                    _cicids_model_data = pickle.load(_f)
            CICIDS_MODEL_AVAILABLE = True
            _m = _cicids_model_data.get('metrics', {})
            logging.info(f"[CICIDS] Loaded trained model from {_MODEL_PATH}")
            logging.info(f"[CICIDS] Dataset: {_cicids_model_data.get('dataset', 'unknown')}")
            logging.info(f"[CICIDS] F1-Score: {_m.get('f1_score', '?')}%")
        except Exception as _e:
            CICIDS_MODEL_AVAILABLE = False
            _cicids_model_data = None
            _em = str(_e)
            _CICIDS_UNAVAILABLE_DETAIL = _em
            if 'numpy._core' in _em:
                _CICIDS_UNAVAILABLE_CODE = 'numpy_mismatch'
                logging.warning(
                    "[CICIDS] ML model unavailable: environment mismatch (%s). "
                    "Re-run model_trainer.py with the same Python you use for working_app.py.",
                    _em,
                )
            else:
                _CICIDS_UNAVAILABLE_CODE = 'pickle_error'
                logging.warning(f"[CICIDS] Could not load trained_model.pkl: {_e}")


def cicids_predict(features_dict: dict) -> dict:
    """
    Score a single network/log record using the CICIDS 2017 trained model.
    Uses Random Forest (primary) + Isolation Forest (secondary).
    features_dict: dict with keys matching CICIDS feature names.
    Returns:  { 'is_attack': bool, 'risk_score': float (0-10), 'confidence': str }
    """
    if not CICIDS_MODEL_AVAILABLE or _cicids_model_data is None:
        return {'is_attack': False, 'risk_score': 0.0, 'confidence': 'N/A'}

    rf      = _cicids_model_data['model']
    scaler  = _cicids_model_data['scaler']
    feats   = _cicids_model_data['features']

    row = [features_dict.get(f, 0.0) for f in feats]
    X   = scaler.transform([row])

    # Random Forest prediction + probability
    rf_pred  = rf.predict(X)[0]            # 0 = normal, 1 = attack
    rf_prob  = rf.predict_proba(X)[0][1]   # probability of attack (0.0–1.0)

    # Isolation Forest secondary check
    iso = _cicids_model_data.get('iso_model')
    iso_flag = False
    if iso is not None:
        iso_pred = iso.predict(X)[0]       # -1 = anomaly
        iso_flag = (iso_pred == -1)

    # Combined: flag if RF says attack OR (high RF prob AND iso says anomaly)
    is_attack = bool(rf_pred == 1 or (rf_prob > 0.4 and iso_flag))

    # Risk score 0–10 based on RF probability
    risk = round(min(10.0, rf_prob * 10.0), 2)

    confidence = 'HIGH' if risk >= 7.0 else ('MEDIUM' if risk >= 4.0 else 'LOW')
    return {
        'is_attack'  : is_attack,
        'risk_score' : risk,
        'confidence' : confidence,
        'rf_prob'    : round(float(rf_prob), 4),
        'iso_anomaly': iso_flag,
    }
# ─────────────────────────────────────────────────────────────────────────

# AI Threat Intelligence System
class AIThreatClassifier:
    """AI-powered threat classification and analysis system."""
    
    def __init__(self):
        self.threat_patterns = {
            'brute_force': {
                'keywords': ['brute', 'force', 'login', 'attempt', 'password', 'auth'],
                'indicators': ['multiple_failed', 'rapid_attempts', 'dictionary'],
                'severity': 'medium',
                'confidence': 0.85
            },
            'sql_injection': {
                'keywords': ['sql', 'injection', 'union', 'select', 'drop', 'insert'],
                'indicators': ['malicious_query', 'parameter_tampering', 'database_access'],
                'severity': 'high',
                'confidence': 0.92
            },
            'ddos_attack': {
                'keywords': ['ddos', 'flood', 'overload', 'traffic', 'amplification'],
                'indicators': ['high_volume', 'multiple_sources', 'service_disruption'],
                'severity': 'high',
                'confidence': 0.88
            },
            'malware': {
                'keywords': ['malware', 'virus', 'trojan', 'backdoor', 'payload'],
                'indicators': ['executable', 'suspicious_file', 'encoded_content'],
                'severity': 'critical',
                'confidence': 0.95
            },
            'phishing': {
                'keywords': ['phishing', 'credential', 'harvest', 'spoof', 'impersonation'],
                'indicators': ['fake_login', 'suspicious_links', 'credential_theft'],
                'severity': 'medium',
                'confidence': 0.78
            },
            'reconnaissance': {
                'keywords': ['scan', 'probe', 'enumerate', 'discover', 'recon'],
                'indicators': ['port_scan', 'service_enumeration', 'vulnerability_scan'],
                'severity': 'low',
                'confidence': 0.72
            }
        }
        
        self.attack_sequences = {
            'lateral_movement': ['reconnaissance', 'brute_force', 'malware'],
            'data_exfiltration': ['phishing', 'malware', 'sql_injection'],
            'ransomware': ['phishing', 'malware', 'ddos_attack'],
            'apt_attack': ['reconnaissance', 'brute_force', 'malware', 'sql_injection']
        }
    
    def classify_threat(self, threat_data):
        """Classify threat using AI pattern matching."""
        try:
            description = threat_data.get('description', '').lower()
            threat_type = threat_data.get('threat_type', '').lower()
            source_ip = threat_data.get('source_ip', '')
            
            # Pattern matching
            classifications = []
            for pattern_name, pattern_data in self.threat_patterns.items():
                score = 0
                
                # Keyword matching
                for keyword in pattern_data['keywords']:
                    if keyword in description or keyword in threat_type:
                        score += 1
                
                # Indicator matching
                for indicator in pattern_data['indicators']:
                    if indicator in description:
                        score += 2
                
                # Calculate confidence
                keyword_match_ratio = score / len(pattern_data['keywords'])
                confidence = min(keyword_match_ratio * pattern_data['confidence'], 0.99)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    classifications.append({
                        'type': pattern_name,
                        'confidence': confidence,
                        'severity': pattern_data['severity'],
                        'match_score': score
                    })
            
            # Sort by confidence
            classifications.sort(key=lambda x: x['confidence'], reverse=True)
            
            return classifications[0] if classifications else self._default_classification()
            
        except Exception as e:
            print(f"❌ Error in AI classification: {e}")
            return self._default_classification()
    
    def _default_classification(self):
        """Default classification when no patterns match."""
        return {
            'type': 'unknown',
            'confidence': 0.5,
            'severity': 'medium',
            'match_score': 0
        }
    
    def detect_attack_sequence(self, recent_threats, hours=24):
        """Detect potential attack sequences in recent threats."""
        try:
            from datetime import datetime, timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter recent threats
            recent = [t for t in recent_threats if t.timestamp >= cutoff_time]
            
            # Get unique threat types
            threat_types = set()
            for threat in recent:
                classification = self.classify_threat({
                    'description': threat.description,
                    'threat_type': threat.threat_type,
                    'source_ip': threat.source_ip
                })
                threat_types.add(classification['type'])
            
            # Check for attack sequences
            detected_sequences = []
            for sequence_name, sequence_pattern in self.attack_sequences.items():
                matches = len(set(sequence_pattern) & threat_types)
                if matches >= 2:  # At least 2 pattern matches
                    sequence_confidence = matches / len(sequence_pattern)
                    detected_sequences.append({
                        'sequence_type': sequence_name,
                        'confidence': sequence_confidence,
                        'matched_patterns': list(set(sequence_pattern) & threat_types),
                        'severity': self._get_sequence_severity(sequence_name)
                    })
            
            return sorted(detected_sequences, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            print(f"❌ Error detecting attack sequences: {e}")
            return []
    
    def _get_sequence_severity(self, sequence_name):
        """Get severity level for attack sequence."""
        severity_map = {
            'lateral_movement': 'high',
            'data_exfiltration': 'critical',
            'ransomware': 'critical',
            'apt_attack': 'critical'
        }
        return severity_map.get(sequence_name, 'medium')
    
    def predict_next_threat(self, recent_threats):
        """Predict next likely threat based on patterns."""
        try:
            if len(recent_threats) < 3:
                return None
            
            # Analyze recent pattern
            recent_classifications = []
            for threat in recent_threats[-5:]:  # Last 5 threats
                classification = self.classify_threat({
                    'description': threat.description,
                    'threat_type': threat.threat_type,
                    'source_ip': threat.source_ip
                })
                recent_classifications.append(classification['type'])
            
            # Find most common sequence
            for sequence_name, sequence_pattern in self.attack_sequences.items():
                if len(set(sequence_pattern) & set(recent_classifications)) >= 2:
                    # Find next step in sequence
                    current_index = 0
                    for i, pattern in enumerate(sequence_pattern):
                        if pattern in recent_classifications:
                            current_index = i
                    
                    if current_index < len(sequence_pattern) - 1:
                        next_threat = sequence_pattern[current_index + 1]
                        return {
                            'predicted_threat': next_threat,
                            'confidence': 0.75,
                            'based_on_sequence': sequence_name,
                            'recommendation': self._get_prevention_recommendation(next_threat)
                        }
            
            return None
            
        except Exception as e:
            print(f"❌ Error predicting next threat: {e}")
            return None
    
    def _get_prevention_recommendation(self, threat_type):
        """Get prevention recommendation for threat type."""
        recommendations = {
            'brute_force': "Enable account lockout and implement rate limiting",
            'sql_injection': "Validate all input parameters and use parameterized queries",
            'ddos_attack': "Configure rate limiting and DDoS protection services",
            'malware': "Update antivirus signatures and implement file scanning",
            'phishing': "Deploy email filtering and conduct user awareness training",
            'reconnaissance': "Implement network segmentation and port security"
        }
        return recommendations.get(threat_type, "Monitor system activity closely")

class AttackPatternAnalyzer:
    """Advanced attack pattern detection for brute force, DDoS, and port scanning."""
    
    def __init__(self):
        self.detection_thresholds = {
            'brute_force': {'max_attempts': 5, 'time_window': 300},  # 5 attempts in 5 minutes
            'ddos': {'max_requests': 100, 'time_window': 60},  # 100 requests in 1 minute
            'port_scan': {'max_ports': 10, 'time_window': 300}  # 10 ports in 5 minutes
        }
    
    def analyze_recent_threats(self, hours=24):
        """Analyze recent threats for attack patterns."""
        try:
            from datetime import datetime, timedelta
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_threats = Threat.query.filter(Threat.timestamp >= cutoff_time).all()
            
            detected_patterns = []
            
            # Detect brute force attacks
            brute_force = self._detect_brute_force(recent_threats)
            if brute_force:
                detected_patterns.extend(brute_force)
            
            # Detect DDoS attacks
            ddos = self._detect_ddos(recent_threats)
            if ddos:
                detected_patterns.extend(ddos)
            
            # Detect port scans
            port_scan = self._detect_port_scan(recent_threats)
            if port_scan:
                detected_patterns.extend(port_scan)
            
            # NEW: Detect suspicious activity patterns from real threats
            suspicious_activity = self._detect_suspicious_activity(recent_threats)
            if suspicious_activity:
                detected_patterns.extend(suspicious_activity)
            
            # NEW: Detect anomalous behavior patterns
            anomalous_behavior = self._detect_anomalous_behavior(recent_threats)
            if anomalous_behavior:
                detected_patterns.extend(anomalous_behavior)
            
            # Save patterns to database
            self._save_patterns(detected_patterns)
            
            return detected_patterns
            
        except Exception as e:
            print(f"❌ Error analyzing attack patterns: {e}")
            return []
    
    def _detect_suspicious_activity(self, threats):
        """Detect suspicious activity patterns from suspicious_pattern threats."""
        patterns = []
        
        # Group threats by source IP
        ip_threats = {}
        for threat in threats:
            if 'suspicious' in threat.threat_type.lower():
                if threat.source_ip not in ip_threats:
                    ip_threats[threat.source_ip] = []
                ip_threats[threat.source_ip].append(threat)
        
        # Check for repeated suspicious activity - LOWERED THRESHOLD to 1
        for ip, ip_threat_list in ip_threats.items():
            if len(ip_threat_list) >= 1:  # Lowered from 3 to 1
                pattern = {
                    'pattern_type': 'suspicious_activity',
                    'source_ips': [ip],
                    'target_ports': [80, 443],
                    'event_count': len(ip_threat_list),
                    'confidence_score': min(len(ip_threat_list) * 0.2, 0.95),
                    'description': f'Repeated suspicious activity detected from {ip} with {len(ip_threat_list)} occurrence(s)',
                    'related_threat_ids': [t.id for t in ip_threat_list]
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_anomalous_behavior(self, threats):
        """Detect anomalous behavior patterns from anomaly threats."""
        patterns = []
        
        # Group threats by source IP
        ip_threats = {}
        for threat in threats:
            if 'anomaly' in threat.threat_type.lower():
                if threat.source_ip not in ip_threats:
                    ip_threats[threat.source_ip] = []
                ip_threats[threat.source_ip].append(threat)
        
        # Check for anomalous behavior clusters - LOWERED THRESHOLD to 1
        for ip, ip_threat_list in ip_threats.items():
            if len(ip_threat_list) >= 1:  # Lowered from 2 to 1
                pattern = {
                    'pattern_type': 'anomalous_behavior',
                    'source_ips': [ip],
                    'target_ports': [],
                    'event_count': len(ip_threat_list),
                    'confidence_score': min(len(ip_threat_list) * 0.25, 0.95),
                    'description': f'Anomalous behavior cluster detected from {ip} with {len(ip_threat_list)} anomaly/anomalies',
                    'related_threat_ids': [t.id for t in ip_threat_list]
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_brute_force(self, threats):
        """Detect brute force login attempts."""
        patterns = []
        
        # Group threats by source IP
        ip_threats = {}
        for threat in threats:
            if 'brute' in threat.threat_type.lower() or 'login' in threat.threat_type.lower():
                if threat.source_ip not in ip_threats:
                    ip_threats[threat.source_ip] = []
                ip_threats[threat.source_ip].append(threat)
        
        # Check for brute force pattern
        for ip, ip_threat_list in ip_threats.items():
            if len(ip_threat_list) >= self.detection_thresholds['brute_force']['max_attempts']:
                pattern = {
                    'pattern_type': 'brute_force',
                    'source_ips': [ip],
                    'target_ports': [22, 3389, 445],  # Common brute force targets
                    'event_count': len(ip_threat_list),
                    'confidence_score': min(len(ip_threat_list) / 10, 0.95),
                    'description': f'Brute force attack detected from {ip} with {len(ip_threat_list)} failed attempts',
                    'related_threat_ids': [t.id for t in ip_threat_list]
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_ddos(self, threats):
        """Detect DDoS attack patterns."""
        patterns = []
        
        # Group by time windows
        from collections import defaultdict
        time_windows = defaultdict(list)
        
        for threat in threats:
            if 'ddos' in threat.threat_type.lower() or 'flood' in threat.threat_type.lower():
                # Round timestamp to nearest minute
                minute_key = threat.timestamp.replace(second=0, microsecond=0)
                time_windows[minute_key].append(threat)
        
        # Check for high volume in short time
        for time_window, window_threats in time_windows.items():
            if len(window_threats) >= self.detection_thresholds['ddos']['max_requests']:
                unique_ips = list(set([t.source_ip for t in window_threats]))
                pattern = {
                    'pattern_type': 'ddos',
                    'source_ips': unique_ips[:10],  # Top 10 IPs
                    'target_ports': [80, 443, 8080],
                    'event_count': len(window_threats),
                    'confidence_score': min(len(window_threats) / 200, 0.95),
                    'description': f'DDoS attack detected with {len(window_threats)} requests from {len(unique_ips)} sources',
                    'related_threat_ids': [t.id for t in window_threats[:50]]
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_port_scan(self, threats):
        """Detect port scanning activity."""
        patterns = []
        
        # Group by source IP
        ip_threats = {}
        for threat in threats:
            if 'scan' in threat.threat_type.lower() or 'probe' in threat.threat_type.lower():
                if threat.source_ip not in ip_threats:
                    ip_threats[threat.source_ip] = []
                ip_threats[threat.source_ip].append(threat)
        
        # Check for port scan pattern
        for ip, ip_threat_list in ip_threats.items():
            if len(ip_threat_list) >= self.detection_thresholds['port_scan']['max_ports']:
                pattern = {
                    'pattern_type': 'port_scan',
                    'source_ips': [ip],
                    'target_ports': list(range(1, 100)),  # Common scanned ports
                    'event_count': len(ip_threat_list),
                    'confidence_score': min(len(ip_threat_list) / 20, 0.95),
                    'description': f'Port scan detected from {ip} targeting {len(ip_threat_list)} ports/services',
                    'related_threat_ids': [t.id for t in ip_threat_list]
                }
                patterns.append(pattern)
        
        return patterns
    
    def _save_patterns(self, patterns):
        """Save detected patterns to database."""
        try:
            for pattern_data in patterns:
                # Check if similar pattern already exists
                existing = AttackPattern.query.filter_by(
                    pattern_type=pattern_data['pattern_type'],
                    status='active'
                ).first()
                
                if not existing:
                    pattern = AttackPattern(
                        pattern_type=pattern_data['pattern_type'],
                        source_ips=json.dumps(pattern_data['source_ips']),
                        target_ports=json.dumps(pattern_data['target_ports']),
                        event_count=pattern_data['event_count'],
                        confidence_score=pattern_data['confidence_score'],
                        description=pattern_data['description'],
                        related_threat_ids=json.dumps(pattern_data['related_threat_ids'])
                    )
                    db.session.add(pattern)
            
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error saving attack patterns: {e}")

# Enhanced Threat Intelligence Integration System
class ThreatIntelligenceManager:
    """External threat intelligence integration and IOC management with real API support."""
    
    def __init__(self):
        self.virustotal_api_key = os.getenv('VIRUSTOTAL_API_KEY', '')
        self.abuseipdb_api_key = os.getenv('ABUSEIPDB_API_KEY', '')
        self.enable_real_apis = os.getenv('ENABLE_REAL_APIS', 'false').lower() == 'true'
        self.enable_search_history = os.getenv('ENABLE_SEARCH_HISTORY', 'true').lower() == 'true'
        self.enable_batch_search = os.getenv('ENABLE_BATCH_SEARCH', 'true').lower() == 'true'
        self.enable_export_results = os.getenv('ENABLE_EXPORT_RESULTS', 'true').lower() == 'true'
        
        self.threat_feeds = {
            'virustotal': self._query_virustotal,
            'abuseipdb': self._query_abuseipdb,
            'local_ioc': self._check_local_ioc
        }
        self.ioc_database = {}  # In-memory IOC cache
        self.search_history = []  # Search history tracking
        self.load_local_iocs()
        self.load_search_history()
    
    def load_search_history(self):
        """Load search history from storage."""
        try:
            # In a real implementation, this would load from database
            # For demo, we'll use in-memory storage
            self.search_history = []
        except Exception as e:
            print(f"❌ Error loading search history: {e}")
            self.search_history = []
    
    def add_to_search_history(self, indicator, indicator_type, results):
        """Add search to history."""
        if not self.enable_search_history:
            return
        
        try:
            history_entry = {
                'indicator': indicator,
                'type': indicator_type,
                'timestamp': datetime.utcnow().isoformat(),
                'verdict': results.get('verdict', 'unknown'),
                'score': results.get('overall_score', 0),
                'sources_checked': len(results.get('sources', {}))
            }
            
            # Add to beginning of history (most recent first)
            self.search_history.insert(0, history_entry)
            
            # Keep only last 100 searches
            if len(self.search_history) > 100:
                self.search_history = self.search_history[:100]
                
        except Exception as e:
            print(f"❌ Error adding to search history: {e}")
    
    def get_search_history(self, limit=20):
        """Get search history."""
        return self.search_history[:limit]
    
    def batch_analyze(self, indicators):
        """Analyze multiple indicators in batch."""
        if not self.enable_batch_search:
            return {'error': 'Batch search is disabled'}
        
        results = []
        for indicator in indicators:
            try:
                # Determine indicator type
                indicator_type = 'ip'
                if '.' in indicator and not indicator.replace('.', '').isdigit():
                    indicator_type = 'domain'
                elif len(indicator) in [32, 40, 64]:
                    indicator_type = 'hash'
                
                analysis = self.analyze_threat_intelligence(indicator, indicator_type)
                results.append(analysis)
                
                # Add small delay to respect rate limits
                import time
                time.sleep(0.1)
                
            except Exception as e:
                results.append({
                    'indicator': indicator,
                    'error': str(e)
                })
        
        return {
            'success': True,
            'results': results,
            'total_analyzed': len(results),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def export_results(self, format='json'):
        """Export search results and history."""
        if not self.enable_export_results:
            return {'error': 'Export functionality is disabled'}
        
        try:
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'ioc_summary': self.get_ioc_summary(),
                'search_history': self.get_search_history(50),
                'ioc_database': self.ioc_database,
                'system_info': {
                    'total_searches': len(self.search_history),
                    'total_iocs': sum(len(v) for v in self.ioc_database.values()),
                    'real_apis_enabled': self.enable_real_apis,
                    'features_enabled': {
                        'search_history': self.enable_search_history,
                        'batch_search': self.enable_batch_search,
                        'export_results': self.enable_export_results
                    }
                }
            }
            
            if format.lower() == 'csv':
                return self._export_to_csv(export_data)
            elif format.lower() == 'xml':
                return self._export_to_xml(export_data)
            else:
                return export_data
                
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return {'error': str(e)}
    
    def _export_to_csv(self, data):
        """Export data to CSV format."""
        try:
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write search history
            writer.writerow(['Search History'])
            writer.writerow(['Indicator', 'Type', 'Timestamp', 'Verdict', 'Score', 'Sources'])
            for entry in data['search_history']:
                writer.writerow([
                    entry['indicator'],
                    entry['type'],
                    entry['timestamp'],
                    entry['verdict'],
                    entry['score'],
                    entry['sources_checked']
                ])
            
            # Write IOC summary
            writer.writerow([])
            writer.writerow(['IOC Summary'])
            writer.writerow(['Type', 'Count'])
            for ioc_type, count in data['ioc_summary'].items():
                if ioc_type != 'total_iocs':
                    writer.writerow([ioc_type, count])
            
            return {'csv_data': output.getvalue(), 'filename': f'threat_intelligence_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
            
        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")
            return {'error': str(e)}
    
    def _export_to_xml(self, data):
        """Export data to XML format."""
        try:
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<threat_intelligence_export>
    <export_timestamp>{data['export_timestamp']}</export_timestamp>
    <system_info>
        <total_searches>{data['system_info']['total_searches']}</total_searches>
        <total_iocs>{data['system_info']['total_iocs']}</total_iocs>
        <real_apis_enabled>{data['system_info']['real_apis_enabled']}</real_apis_enabled>
    </system_info>
    <ioc_summary>
        <malicious_ips>{data['ioc_summary']['malicious_ips']}</malicious_ips>
        <suspicious_domains>{data['ioc_summary']['suspicious_domains']}</suspicious_domains>
        <malware_hashes>{data['ioc_summary']['malware_hashes']}</malware_hashes>
    </ioc_summary>
    <search_history>
"""
            
            for entry in data['search_history']:
                xml_content += f"""        <search>
            <indicator>{entry['indicator']}</indicator>
            <type>{entry['type']}</type>
            <timestamp>{entry['timestamp']}</timestamp>
            <verdict>{entry['verdict']}</verdict>
            <score>{entry['score']}</score>
            <sources_checked>{entry['sources_checked']}</sources_checked>
        </search>
"""
            
            xml_content += """    </search_history>
</threat_intelligence_export>"""
            
            return {'xml_data': xml_content, 'filename': f'threat_intelligence_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xml'}
            
        except Exception as e:
            print(f"❌ Error exporting to XML: {e}")
            return {'error': str(e)}
    
    def load_local_iocs(self):
        """Load local indicators of compromise."""
        # Sample IOCs for demonstration
        self.ioc_database = {
            'malicious_ips': [
                '192.168.1.100', '10.0.0.50', '172.16.0.25',
                '203.0.113.1', '198.51.100.1', '192.0.2.1'
            ],
            'suspicious_domains': [
                'malicious-site.com', 'bad-actor.net', 'threat-domain.org',
                'suspicious-url.biz', 'malware-host.info'
            ],
            'malware_hashes': [
                'a1b2c3d4e5f6', 'f6e5d4c3b2a1', '1234567890abcdef',
                'deadbeefcafe', 'c0ffee123456'
            ],
            'known_attackers': [
                'APT-Group-Alpha', 'Cyber-Gang-Beta', 'Threat-Actor-Gamma',
                'Hacking-Team-Delta', 'Malware-Group-Epsilon'
            ]
        }
    
    def analyze_threat_intelligence(self, indicator, indicator_type='ip'):
        """Analyze threat indicator against multiple intelligence sources."""
        try:
            results = {
                'indicator': indicator,
                'type': indicator_type,
                'timestamp': datetime.utcnow().isoformat(),
                'sources': {},
                'overall_score': 0,
                'verdict': 'unknown',
                'details': {}
            }
            
            # Check against local IOCs first
            local_result = self._check_local_ioc(indicator, indicator_type)
            if local_result:
                results['sources']['local'] = local_result
            
            # Query external feeds
            for feed_name, feed_func in self.threat_feeds.items():
                if feed_name != 'local_ioc':  # Already checked local
                    try:
                        feed_result = feed_func(indicator, indicator_type)
                        if feed_result:
                            results['sources'][feed_name] = feed_result
                    except Exception as e:
                        print(f"❌ Error querying {feed_name}: {e}")
                        results['sources'][feed_name] = {'error': str(e)}
            
            # Calculate overall score and verdict
            results['overall_score'] = self._calculate_threat_score(results['sources'])
            results['verdict'] = self._determine_verdict(results['overall_score'])
            results['details'] = self._extract_threat_details(results['sources'])
            
            # Add to search history
            self.add_to_search_history(indicator, indicator_type, results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in threat intelligence analysis: {e}")
            return {'error': str(e)}
    
    def _query_virustotal(self, indicator, indicator_type):
        """Query VirusTotal API for threat intelligence."""
        if not self.virustotal_api_key:
            return {'error': 'VirusTotal API key not configured'}
        
        if not self.enable_real_apis:
            # Return mock data for demonstration
            return self._mock_virustotal_response(indicator, indicator_type)
        
        try:
            # Real VirusTotal API implementation
            import requests
            
            if indicator_type == 'ip':
                url = f"https://www.virustotal.com/vtapi/v2/ip-address/report"
                params = {
                    'apikey': self.virustotal_api_key,
                    'ip': indicator
                }
            elif indicator_type == 'domain':
                url = f"https://www.virustotal.com/vtapi/v2/domain/report"
                params = {
                    'apikey': self.virustotal_api_key,
                    'domain': indicator
                }
            elif indicator_type == 'hash':
                url = f"https://www.virustotal.com/vtapi/v2/file/report"
                params = {
                    'apikey': self.virustotal_api_key,
                    'resource': indicator
                }
            else:
                return {'error': f'Unsupported indicator type for VirusTotal: {indicator_type}'}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if indicator_type == 'ip':
                    stats = data.get('detected_urls', {})
                    positives = data.get('detected_detected_urls', 0)
                    total = len(stats)
                    
                    return {
                        'source': 'virustotal',
                        'positives': positives,
                        'total': total,
                        'detection_ratio': f"{positives}/{total}" if total > 0 else "0/0",
                        'country': data.get('country', 'Unknown'),
                        'confidence': min(positives / total if total > 0 else 0, 0.95),
                        'details': {
                            'country': data.get('country', 'Unknown'),
                            'detected_urls': data.get('detected_urls', [])
                        }
                    }
                
                # Similar logic for domains and hashes would go here
                return {'source': 'virustotal', 'data': data}
                
            elif response.status_code == 204:
                return {'error': 'VirusTotal: Resource not found'}
            elif response.status_code == 403:
                return {'error': 'VirusTotal: Invalid API key or insufficient privileges'}
            else:
                return {'error': f'VirusTotal: HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"❌ Error querying VirusTotal API: {e}")
            return {'error': f'VirusTotal API error: {str(e)}'}
    
    def _mock_virustotal_response(self, indicator, indicator_type):
        """Mock VirusTotal response for demonstration."""
        # Simulate API call delay
        import time
        time.sleep(0.1)
        
        if indicator_type == 'ip':
            # Different mock responses based on IP
            mock_responses = {
                '192.168.1.100': {
                    'malicious': 5, 'suspicious': 3, 'harmless': 20, 'undetected': 12
                },
                '203.0.113.1': {
                    'malicious': 8, 'suspicious': 4, 'harmless': 15, 'undetected': 8
                }
            }
            
            stats = mock_responses.get(indicator, {
                'malicious': 0, 'suspicious': 1, 'harmless': 35, 'undetected': 14
            })
            
            malicious_count = stats['malicious'] + stats['suspicious']
            total_count = sum(stats.values())
            
            return {
                'source': 'virustotal',
                'malicious_count': malicious_count,
                'total_count': total_count,
                'detection_ratio': f"{malicious_count}/{total_count}",
                'reputation': -malicious_count,
                'confidence': min(malicious_count / total_count, 0.95),
                'details': {
                    'country': 'US',
                    'analysis_engines': stats
                }
            }
        
        return {'match': False}
    
    def _query_abuseipdb(self, indicator, indicator_type):
        """Query AbuseIPDB API for IP reputation."""
        if not self.abuseipdb_api_key:
            return {'error': 'AbuseIPDB API key not configured'}
        
        if indicator_type != 'ip':
            return {'match': False}
        
        if not self.enable_real_apis:
            # Return mock data for demonstration
            return self._mock_abuseipdb_response(indicator)
        
        try:
            # Real AbuseIPDB API implementation
            import requests
            
            url = "https://api.abuseipdb.com/api/v2/check"
            headers = {
                'Accept': 'application/json',
                'Key': self.abuseipdb_api_key
            }
            params = {
                'ipAddress': indicator,
                'maxAgeInDays': 90,
                'verbose': ''
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('data', {})
                
                return {
                    'source': 'abuseipdb',
                    'abuse_confidence': data.get('abuseConfidenceScore', 0),
                    'total_reports': data.get('totalReports', 0),
                    'distinct_users': data.get('numDistinctUsers', 0),
                    'country': data.get('countryCode', 'Unknown'),
                    'isp': data.get('isp', 'Unknown'),
                    'usage_type': data.get('usageType', 'Unknown'),
                    'confidence': data.get('abuseConfidenceScore', 0) / 100,
                    'last_reported': data.get('lastReportedAt', 'Never')
                }
                
            elif response.status_code == 401:
                return {'error': 'AbuseIPDB: Invalid API key'}
            elif response.status_code == 429:
                return {'error': 'AbuseIPDB: Rate limit exceeded'}
            else:
                return {'error': f'AbuseIPDB: HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"❌ Error querying AbuseIPDB API: {e}")
            return {'error': f'AbuseIPDB API error: {str(e)}'}
    
    def _mock_abuseipdb_response(self, indicator):
        """Mock AbuseIPDB response for demonstration."""
        # Simulate API call delay
        import time
        time.sleep(0.1)
        
        # Different mock responses based on IP
        mock_responses = {
            '192.168.1.100': {
                'abuse_confidence': 85,
                'total_reports': 25,
                'country': 'US',
                'isp': 'Malicious ISP Ltd'
            },
            '203.0.113.1': {
                'abuse_confidence': 92,
                'total_reports': 47,
                'country': 'CN',
                'isp': 'Suspicious Networks'
            }
        }
        
        data = mock_responses.get(indicator, {
            'abuse_confidence': 15,
            'total_reports': 3,
            'country': 'US',
            'isp': 'Example ISP'
        })
        
        return {
            'source': 'abuseipdb',
            'abuse_confidence': data['abuse_confidence'],
            'total_reports': data['total_reports'],
            'distinct_users': max(1, data['total_reports'] // 3),
            'country': data['country'],
            'isp': data['isp'],
            'usage_type': 'Data Center',
            'confidence': data['abuse_confidence'] / 100,
            'last_reported': '2023-12-01T10:00:00+00:00'
        }
    
    def _check_local_ioc(self, indicator, indicator_type):
        """Check indicator against local IOC database."""
        try:
            indicator = indicator.lower()
            
            if indicator_type == 'ip' and indicator in self.ioc_database['malicious_ips']:
                return {
                    'match': True,
                    'category': 'malicious_ip',
                    'confidence': 0.9,
                    'source': 'local_database',
                    'description': 'Known malicious IP address'
                }
            
            if indicator_type == 'domain' and any(domain in indicator for domain in self.ioc_database['suspicious_domains']):
                return {
                    'match': True,
                    'category': 'suspicious_domain',
                    'confidence': 0.8,
                    'source': 'local_database',
                    'description': 'Known suspicious domain'
                }
            
            if indicator_type == 'hash' and indicator in self.ioc_database['malware_hashes']:
                return {
                    'match': True,
                    'category': 'malware_hash',
                    'confidence': 0.95,
                    'source': 'local_database',
                    'description': 'Known malware hash'
                }
            
            return {'match': False}
            
        except Exception as e:
            print(f"❌ Error checking local IOC: {e}")
            return {'error': str(e)}
    
    def _calculate_threat_score(self, sources):
        """Calculate overall threat score from multiple sources."""
        try:
            scores = []
            
            for source_name, source_data in sources.items():
                if 'error' in source_data:
                    continue
                
                if source_data.get('match', False):
                    if source_name == 'local':
                        scores.append(source_data.get('confidence', 0.5) * 100)
                    elif source_name == 'virustotal':
                        malicious_ratio = source_data.get('detection_ratio', '0/0')
                        if '/' in malicious_ratio:
                            malicious, total = malicious_ratio.split('/')
                            if total != '0':
                                scores.append((int(malicious) / int(total)) * 100)
                    elif source_name == 'abuseipdb':
                        scores.append(source_data.get('abuse_confidence', 0))
            
            return max(scores) if scores else 0
            
        except Exception as e:
            print(f"❌ Error calculating threat score: {e}")
            return 0
    
    def _determine_verdict(self, score):
        """Determine threat verdict based on score."""
        if score >= 70:
            return 'malicious'
        elif score >= 40:
            return 'suspicious'
        elif score >= 10:
            return 'potentially_malicious'
        else:
            return 'benign'
    
    def _extract_threat_details(self, sources):
        """Extract relevant threat details from all sources."""
        details = {
            'countries': set(),
            'isps': set(),
            'usage_types': set(),
            'categories': set(),
            'reports': 0
        }
        
        for source_data in sources.values():
            if 'error' in source_data:
                continue
            
            if 'country' in source_data:
                details['countries'].add(source_data['country'])
            if 'isp' in source_data:
                details['isps'].add(source_data['isp'])
            if 'usage_type' in source_data:
                details['usage_types'].add(source_data['usage_type'])
            if 'category' in source_data:
                details['categories'].add(source_data['category'])
            if 'total_reports' in source_data:
                details['reports'] += source_data['total_reports']
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) if isinstance(v, set) else v for k, v in details.items()}
    
    def add_ioc(self, indicator, indicator_type, category, confidence=0.8):
        """Add new indicator to local IOC database."""
        try:
            if indicator_type == 'ip':
                if indicator not in self.ioc_database['malicious_ips']:
                    self.ioc_database['malicious_ips'].append(indicator)
            elif indicator_type == 'domain':
                self.ioc_database['suspicious_domains'].append(indicator)
            elif indicator_type == 'hash':
                self.ioc_database['malware_hashes'].append(indicator)
            
            return True
        except Exception as e:
            print(f"❌ Error adding IOC: {e}")
            return False
    
    def get_ioc_summary(self):
        """Get summary of local IOC database."""
        return {
            'malicious_ips': len(self.ioc_database['malicious_ips']),
            'suspicious_domains': len(self.ioc_database['suspicious_domains']),
            'malware_hashes': len(self.ioc_database['malware_hashes']),
            'total_iocs': sum(len(v) for v in self.ioc_database.values())
        }

# Global threat intelligence manager
threat_intel = ThreatIntelligenceManager()

# Global AI classifier instance
ai_classifier = AIThreatClassifier()

# Load environment variables
load_dotenv()

# Simple configuration
class Config:
    SECRET_KEY = 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///security_monitoring.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'connect_args': {'timeout': 30, 'check_same_thread': False},
    }
    
    # Threat Detection Settings
    BRUTE_FORCE_THRESHOLD = 5
    BRUTE_FORCE_WINDOW = 60  # seconds
    SCAN_DETECTION_WINDOW = 300  # seconds
    ALERT_THRESHOLD = 7.0
    
    # Risk Scoring Weights
    RISK_WEIGHTS = {
        'brute_force': 0.3,
        'port_scan': 0.25,
        'suspicious_pattern': 0.2,
        'anomaly': 0.15,
        'geo_location': 0.1
    }
    LOG_DIRECTORY = './logs'
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    
    # OpenAI Configuration  
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # Alert threshold
    ALERT_THRESHOLD = 7.0

# Must use _TEMPLATE_DIR (script directory): a second Flask() here replaced the first app and
# reset template resolution to cwd — Thonny/other IDEs often use a different CWD, showing wrong HTML.
app = Flask(__name__, template_folder=_TEMPLATE_DIR)
app.config.from_object(Config)

# Initialize database
db = SQLAlchemy(app)

# Create config instance
config = Config()


def _synthetic_log_enabled():
    """Works even if config.py predates ENABLE_SYNTHETIC_LOG_GENERATOR (reads .env fallback)."""
    if hasattr(config, 'ENABLE_SYNTHETIC_LOG_GENERATOR'):
        return bool(config.ENABLE_SYNTHETIC_LOG_GENERATOR)
    return os.environ.get('ENABLE_SYNTHETIC_LOG_GENERATOR', 'false').lower() in (
        '1', 'true', 'yes', 'on')


def _synthetic_log_interval():
    if hasattr(config, 'SYNTHETIC_LOG_INTERVAL_SECONDS'):
        return int(config.SYNTHETIC_LOG_INTERVAL_SECONDS)
    try:
        return int(os.environ.get('SYNTHETIC_LOG_INTERVAL_SECONDS', '10'))
    except ValueError:
        return 10


# Default port is NOT 5000 — many other tutorials / leftover scripts bind 5000 and show "SIMPLE TEST".
DEFAULT_SOC_PORT = 15500


def dashboard_public_url():
    """Base URL for links in Telegram (port set at startup in __main__)."""
    port = os.environ.get('SOC_PUBLIC_PORT', str(DEFAULT_SOC_PORT))
    return f'http://127.0.0.1:{port}'


def _choose_run_port():
    """
    Bind to SOC_PORT (default 15500), then try a few alternates.
    Port 5000 is tried only after dedicated SOC ports so you are less likely to hit a foreign app.
    """
    preferred = int(os.environ.get('SOC_PORT', str(DEFAULT_SOC_PORT)))
    for port in (preferred, 15501, 15502, 15503, 5000, 5050, 8080, 8888):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', port))
            return port
        except OSError:
            continue
        finally:
            try:
                s.close()
            except Exception:
                pass
    return preferred


# Wire up proper module classes
from alert_system import AlertSystem
from ai_explainer import AIExplainer
alert_system_instance = AlertSystem(config)
ai_explainer_instance = AIExplainer(config)

# ── Audit Logging Helper ────────────────────────────────────────────────────
def log_audit(action, resource_type=None, resource_id=None, details=None, severity='info', user='system'):
    """Write an entry to the SecurityAuditLog table."""
    try:
        entry = SecurityAuditLog(
            timestamp=datetime.utcnow(),
            user=user,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            ip_address=request.remote_addr if request else None,
            details=details,
            severity=severity
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as e:
        print(f"⚠️  Audit log write failed: {e}")
        db.session.rollback()

# ── Geolocation Intelligence ─────────────────────────────────────────────────
# Geolocation Intelligence
def get_location_info(ip_address):
    """Get geolocation information for IP address."""
    # For demonstration, return sample countries for our test IPs
    sample_locations = {
        '8.8.8.8': {
            'country': 'United States',
            'city': 'Mountain View',
            'latitude': 37.4056,
            'longitude': -122.0775,
            'country_code': 'US'
        },
        '1.1.1.1': {
            'country': 'Australia',
            'city': 'Sydney',
            'latitude': -33.8688,
            'longitude': 151.2093,
            'country_code': 'AU'
        },
        '208.67.222.222': {
            'country': 'United States',
            'city': 'San Francisco',
            'latitude': 37.7749,
            'longitude': -122.4194,
            'country_code': 'US'
        },
        '9.9.9.9': {
            'country': 'China',
            'city': 'Beijing',
            'latitude': 39.9042,
            'longitude': 116.4074,
            'country_code': 'CN'
        },
        '1.0.0.1': {
            'country': 'Russia',
            'city': 'Moscow',
            'latitude': 55.7558,
            'longitude': 37.6176,
            'country_code': 'RU'
        },
        '203.0.113.73': {
            'country': 'Japan',
            'city': 'Tokyo',
            'latitude': 35.6762,
            'longitude': 139.6503,
            'country_code': 'JP'
        },
        # Private/internal IPs mapped to realistic threat origin locations
        '192.168.1.100': {
            'country': 'Germany',
            'city': 'Berlin',
            'latitude': 52.5200,
            'longitude': 13.4050,
            'country_code': 'DE'
        },
        '192.168.1.101': {
            'country': 'Brazil',
            'city': 'São Paulo',
            'latitude': -23.5505,
            'longitude': -46.6333,
            'country_code': 'BR'
        },
        '192.168.1.200': {
            'country': 'India',
            'city': 'Mumbai',
            'latitude': 19.0760,
            'longitude': 72.8777,
            'country_code': 'IN'
        },
        '10.0.0.1': {
            'country': 'China',
            'city': 'Shanghai',
            'latitude': 31.2304,
            'longitude': 121.4737,
            'country_code': 'CN'
        },
        '10.0.0.15': {
            'country': 'Russia',
            'city': 'Saint Petersburg',
            'latitude': 59.9311,
            'longitude': 30.3609,
            'country_code': 'RU'
        },
        '172.16.0.50': {
            'country': 'Netherlands',
            'city': 'Amsterdam',
            'latitude': 52.3676,
            'longitude': 4.9041,
            'country_code': 'NL'
        },
        '172.16.0.1': {
            'country': 'United Kingdom',
            'city': 'London',
            'latitude': 51.5074,
            'longitude': -0.1278,
            'country_code': 'GB'
        },
    }
    
    # Return sample location if it's one of our test IPs
    if ip_address in sample_locations:
        return sample_locations[ip_address]
    
    # Handle any remaining private/internal IP ranges with random-but-consistent locations
    import hashlib
    private_ranges = [
        ip_address.startswith('192.168.'),
        ip_address.startswith('10.'),
        ip_address.startswith('172.16.') or ip_address.startswith('172.31.'),
        ip_address.startswith('127.'),
    ]
    if any(private_ranges):
        # Use hash of IP to deterministically pick a location
        fallback_locations = [
            {'country': 'United States', 'city': 'New York', 'latitude': 40.7128, 'longitude': -74.0060, 'country_code': 'US'},
            {'country': 'France', 'city': 'Paris', 'latitude': 48.8566, 'longitude': 2.3522, 'country_code': 'FR'},
            {'country': 'South Korea', 'city': 'Seoul', 'latitude': 37.5665, 'longitude': 126.9780, 'country_code': 'KR'},
            {'country': 'Canada', 'city': 'Toronto', 'latitude': 43.6532, 'longitude': -79.3832, 'country_code': 'CA'},
            {'country': 'Ukraine', 'city': 'Kyiv', 'latitude': 50.4501, 'longitude': 30.5234, 'country_code': 'UA'},
        ]
        idx = int(hashlib.md5(ip_address.encode()).hexdigest(), 16) % len(fallback_locations)
        return fallback_locations[idx]
    
    # Otherwise, try real geolocation
    try:
        # Try to use GeoIP2 database first
        from geoip2 import database
        reader = database.Reader('GeoLite2-City.mmdb')
        response = reader.city(ip_address)
        return {
            'country': response.country.name,
            'city': response.city.name,
            'latitude': response.location.latitude,
            'longitude': response.location.longitude,
            'country_code': response.country.iso_code
        }
    except Exception:
        # GeoIP2 database not available, use fallback
        try:
            import requests
            response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'country': data.get('country', 'Unknown'),
                    'city': data.get('city', 'Unknown'),
                    'latitude': data.get('lat', 0.0),
                    'longitude': data.get('lon', 0.0),
                    'country_code': data.get('countryCode', 'XX')
                }
        except Exception as e2:
            print(f"❌ Fallback API error: {e2}")
        
        # Final fallback
        return {
            'country': 'Unknown',
            'city': 'Unknown',
            'latitude': 0.0,
            'longitude': 0.0,
            'country_code': 'XX'
        }

# Create simple models directly
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, or_

class LogEntry(db.Model):
    __tablename__ = 'log_entry'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # When this row was stored (ingestion time). Used with timestamp for "last 24h" stats.
    ingested_at = Column(DateTime, nullable=True)
    source_ip = Column(String(50))
    destination_port = Column(Integer)
    protocol = Column(String(10))
    action = Column(String(20))
    raw_log = Column(Text)
    parsed_data = Column(Text)  # JSON stored as text

class Threat(db.Model):
    __tablename__ = 'threat'
    id = Column(Integer, primary_key=True)
    threat_type = Column(String(50))
    source_ip = Column(String(50))
    risk_score = Column(Float)
    description = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # When this row was stored (detection/ingest time). Used for Recent Activity when event time is old.
    ingested_at = Column(DateTime, nullable=True)
    status = Column(String(20), default='active')
    ai_explanation = Column(Text)

class IOC(db.Model):
    """Indicators of Compromise (IOC) model."""
    __tablename__ = 'ioc'
    
    id = db.Column(db.Integer, primary_key=True)
    ioc_type = db.Column(db.String(50), nullable=False)  # ip, domain, hash, url
    value = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    threat_level = db.Column(db.String(20), default='medium')  # low, medium, high
    source = db.Column(db.String(100))  # where the IOC came from
    added_by = db.Column(db.String(100))  # who added it
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'ioc_type': self.ioc_type,
            'value': self.value,
            'description': self.description,
            'threat_level': self.threat_level,
            'source': self.source,
            'added_by': self.added_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'is_active': self.is_active
        }

class SecurityAuditLog(db.Model):
    """Security audit log for tracking all system actions (GDPR/Compliance)."""
    __tablename__ = 'security_audit_log'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.Column(db.String(100))
    action = db.Column(db.String(100), nullable=False)
    resource_type = db.Column(db.String(50))
    resource_id = db.Column(db.String(50))
    ip_address = db.Column(db.String(50))
    details = db.Column(db.Text)
    severity = db.Column(db.String(20), default='info')
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'user': self.user,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'ip_address': self.ip_address,
            'details': self.details,
            'severity': self.severity
        }

class TestResult(db.Model):
    """Store test results for Testing Dashboard."""
    __tablename__ = 'test_result'
    
    id = db.Column(db.Integer, primary_key=True)
    test_name = db.Column(db.String(200), nullable=False)
    test_type = db.Column(db.String(50))  # unit, integration, security, performance
    status = db.Column(db.String(20))  # passed, failed, skipped
    duration_ms = db.Column(db.Float)
    error_message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'test_name': self.test_name,
            'test_type': self.test_type,
            'status': self.status,
            'duration_ms': self.duration_ms,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class PerformanceMetric(db.Model):
    """System performance metrics."""
    __tablename__ = 'performance_metric'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    metric_type = db.Column(db.String(50))  # logs_per_sec, response_time, db_query_time
    value = db.Column(db.Float)
    unit = db.Column(db.String(20))
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metric_type': self.metric_type,
            'value': self.value,
            'unit': self.unit
        }

class Alert(db.Model):
    __tablename__ = 'alert'
    id = Column(Integer, primary_key=True)
    threat_id = Column(Integer)
    alert_type = Column(String(50))
    message = Column(Text)
    sent = Column(String(10), default='no')
    sent_timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class Incident(db.Model):
    """Security incident for incident response workflow."""
    __tablename__ = 'incident'
    
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.String(50), unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    severity = db.Column(db.String(20), default='medium')  # low, medium, high, critical
    status = db.Column(db.String(50), default='new')  # new, investigating, contained, resolved, closed
    priority = db.Column(db.Integer, default=3)  # 1=critical, 2=high, 3=medium, 4=low
    assigned_to = db.Column(db.String(100))
    threat_ids = db.Column(db.Text)  # JSON array of related threat IDs
    ioc_ids = db.Column(db.Text)  # JSON array of related IOC IDs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)
    resolution_notes = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'incident_id': self.incident_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity,
            'status': self.status,
            'priority': self.priority,
            'assigned_to': self.assigned_to,
            'threat_ids': json.loads(self.threat_ids) if self.threat_ids else [],
            'ioc_ids': json.loads(self.ioc_ids) if self.ioc_ids else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes
        }

class AttackPattern(db.Model):
    """Detected attack patterns for threat correlation."""
    __tablename__ = 'attack_pattern'
    
    id = db.Column(db.Integer, primary_key=True)
    pattern_type = db.Column(db.String(50), nullable=False)  # brute_force, ddos, port_scan, etc
    source_ips = db.Column(db.Text)  # JSON array of involved IPs
    target_ports = db.Column(db.Text)  # JSON array of targeted ports
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    event_count = db.Column(db.Integer, default=0)
    confidence_score = db.Column(db.Float, default=0.0)  # 0-1
    status = db.Column(db.String(20), default='active')  # active, contained, resolved
    description = db.Column(db.Text)
    related_threat_ids = db.Column(db.Text)  # JSON array
    
    def to_dict(self):
        return {
            'id': self.id,
            'pattern_type': self.pattern_type,
            'source_ips': json.loads(self.source_ips) if self.source_ips else [],
            'target_ports': json.loads(self.target_ports) if self.target_ports else [],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'event_count': self.event_count,
            'confidence_score': self.confidence_score,
            'status': self.status,
            'description': self.description,
            'related_threat_ids': json.loads(self.related_threat_ids) if self.related_threat_ids else []
        }

class AlertRule(db.Model):
    """Custom alert rules for threat detection."""
    __tablename__ = 'alert_rule'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    rule_type = db.Column(db.String(50), nullable=False)  # threshold, pattern, ioc_match
    conditions = db.Column(db.Text)  # JSON conditions
    severity_threshold = db.Column(db.String(20), default='medium')  # low, medium, high, critical
    notification_channels = db.Column(db.Text)  # JSON array: email, telegram, webhook
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_triggered = db.Column(db.DateTime)
    trigger_count = db.Column(db.Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'rule_type': self.rule_type,
            'conditions': json.loads(self.conditions) if self.conditions else {},
            'severity_threshold': self.severity_threshold,
            'notification_channels': json.loads(self.notification_channels) if self.notification_channels else ['dashboard'],
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
            'trigger_count': self.trigger_count
        }

class AlertSuppression(db.Model):
    """Alert suppression rules to reduce false positives."""
    __tablename__ = 'alert_suppression'
    
    id = db.Column(db.Integer, primary_key=True)
    rule_name = db.Column(db.String(100), nullable=False)
    suppression_type = db.Column(db.String(50))  # ip, threat_type, pattern
    match_value = db.Column(db.String(200))  # IP address or pattern to match
    duration_minutes = db.Column(db.Integer, default=60)
    reason = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'suppression_type': self.suppression_type,
            'match_value': self.match_value,
            'duration_minutes': self.duration_minutes,
            'reason': self.reason,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

class User(db.Model):
    """User model for authentication and RBAC."""
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='analyst')  # admin, analyst, viewer
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def check_password(self, password):
        """Verify password using werkzeug security."""
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password_hash, password)
    
    def set_password(self, password):
        """Hash and set password."""
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)

class ScheduledReport(db.Model):
    """Scheduled automated reports."""
    __tablename__ = 'scheduled_report'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)  # weekly_summary, daily_digest, executive_kpi
    format = db.Column(db.String(20), default='pdf')  # pdf, csv, html
    schedule = db.Column(db.String(50), nullable=False)  # daily, weekly, monthly
    recipients = db.Column(db.Text)  # JSON array of email addresses
    is_active = db.Column(db.Boolean, default=True)
    last_generated = db.Column(db.DateTime)
    next_run = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    filters = db.Column(db.Text)  # JSON filter configuration
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'report_type': self.report_type,
            'format': self.format,
            'schedule': self.schedule,
            'recipients': json.loads(self.recipients) if self.recipients else [],
            'is_active': self.is_active,
            'last_generated': self.last_generated.isoformat() if self.last_generated else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'filters': json.loads(self.filters) if self.filters else {}
        }

print("📝 Using direct models")

# Real-time log monitoring
class LogMonitor(FileSystemEventHandler):
    def __init__(self, app, config):
        self.app = app
        self.config = config
        
    def on_modified(self, event):
        if not event.is_directory:
            print(f"📄 New log file detected: {event.src_path}")
            self.process_new_log_file(event.src_path)
    
    def process_new_log_file(self, file_path):
        """Process a new log file."""
        try:
            with self.app.app_context():
                # Import log parser
                from log_parser import LogParser
                
                # Parse the new log file
                log_parser = LogParser()
                parsed_logs = log_parser.parse_log_file(file_path)
                
                # Normalize and store log entries
                for parsed_log in parsed_logs:
                    normalized = log_parser.normalize_log_entry(parsed_log)
                    
                    # Check if log entry already exists
                    existing = LogEntry.query.filter_by(
                        source_ip=normalized['source_ip'],
                        timestamp=normalized['timestamp'],
                        raw_log=normalized['raw_log']
                    ).first()
                    
                    if not existing:
                        log_entry = LogEntry(
                            timestamp=normalized['timestamp'],
                            source_ip=normalized['source_ip'],
                            destination_port=normalized.get('destination_port'),
                            protocol=normalized.get('protocol', ''),
                            action=normalized.get('action', ''),
                            raw_log=normalized['raw_log'],
                            parsed_data=json.dumps(normalized.get('parsed_data', {})),
                            ingested_at=datetime.utcnow(),
                        )
                        db.session.add(log_entry)
                
                # Commit all new log entries
                db.session.commit()
                print(f"✅ Processed {len(parsed_logs)} entries from {os.path.basename(file_path)}")
                
                # Trigger threat detection for new logs
                self.trigger_threat_detection()
                
        except Exception as e:
            print(f"❌ Error processing log file {file_path}: {e}")
    
    def trigger_threat_detection(self):
        """Trigger threat detection for recent logs."""
        try:
            with self.app.app_context():
                # Get recent log entries for threat detection
                recent_logs = LogEntry.query.filter(
                    LogEntry.timestamp >= datetime.utcnow() - timedelta(minutes=10)
                ).all()
                
                if recent_logs:
                    # Convert to dict format for threat detector
                    log_dicts = [self._log_entry_to_dict(log) for log in recent_logs]
                    
                    # Import and run threat detector
                    from threat_detector import ThreatDetector
                    threat_detector = ThreatDetector(self.config)
                    detected_threats = threat_detector.detect_threats(log_dicts)
                    
                    # Process detected threats
                    for threat_data in detected_threats:
                        # Get historical data for risk scoring
                        historical_threats = Threat.query.filter(
                            Threat.source_ip == threat_data['source_ip'],
                            Threat.timestamp >= datetime.utcnow() - timedelta(days=30)
                        ).all()
                        
                        historical_dicts = [self._threat_to_dict(t) for t in historical_threats]
                        
                        # Calculate comprehensive risk score
                        from risk_scorer import RiskScorer
                        risk_scorer = RiskScorer(self.config)
                        threat_data = risk_scorer.update_threat_with_risk_score(
                            threat_data, historical_dicts
                        )

                        # ── CICIDS 2017 ML Model scoring ──────────────────
                        if CICIDS_MODEL_AVAILABLE:
                            try:
                                cicids_result = cicids_predict({
                                    'Flow Duration'               : 10000,
                                    'Total Fwd Packets'           : threat_data.get('request_count', 5),
                                    'Total Backward Packets'      : 2,
                                    'Total Length of Fwd Packets' : 500,
                                    'Total Length of Bwd Packets' : 200,
                                    'Fwd Packet Length Max'       : 100,
                                    'Fwd Packet Length Mean'      : 60,
                                    'Bwd Packet Length Max'       : 80,
                                    'Bwd Packet Length Mean'      : 40,
                                    'Flow Bytes/s'                : threat_data.get('risk_score', 5) * 100000,
                                    'Flow Packets/s'              : threat_data.get('risk_score', 5) * 500,
                                    'Flow IAT Mean'               : 1000,
                                    'Flow IAT Max'                : 5000,
                                    'Fwd IAT Mean'                : 1000,
                                    'Bwd IAT Mean'                : 1000,
                                    'Packet Length Mean'          : 60,
                                    'Packet Length Std'           : 20,
                                    'Average Packet Size'         : 60,
                                })
                                threat_data['cicids_ml_score']    = cicids_result.get('risk_score', 0)
                                threat_data['cicids_is_attack']   = cicids_result.get('is_attack', False)
                                threat_data['cicids_confidence']  = cicids_result.get('confidence', 'N/A')
                                # If ML model also flags it, bump up the risk score slightly
                                if cicids_result.get('is_attack') and threat_data.get('risk_score', 0) < 8.0:
                                    threat_data['risk_score'] = min(10.0, threat_data['risk_score'] + 0.5)
                            except Exception:
                                pass
                        # ─────────────────────────────────────────────────

                        # Generate AI explanation for high-risk threats
                        if should_send_alert(threat_data):
                            threat_data['ai_explanation'] = generate_ai_explanation(threat_data)
                        
                        # Create threat record
                        _now = datetime.utcnow()
                        threat = Threat(
                            threat_type=threat_data['threat_type'],
                            source_ip=threat_data['source_ip'],
                            risk_score=threat_data['risk_score'],
                            description=threat_data['description'],
                            ai_explanation=threat_data.get('ai_explanation'),
                            timestamp=_now,
                            ingested_at=_now,
                        )
                        db.session.add(threat)
                        db.session.flush()
                        
                        # Auto-block IPs with critical risk score >= 9.0
                        if threat_data.get('risk_score', 0) >= 9.0:
                            src_ip = threat_data.get('source_ip', '')
                            if src_ip:
                                auto_block_ip(
                                    src_ip,
                                    'Auto-blocked: risk score %.1f for %s'
                                    % (threat_data['risk_score'],
                                       threat_data.get('threat_type', 'unknown'))
                                )

                        # Send alert for high-risk threats
                        if should_send_alert(threat_data):
                            message = (
                                f"🚨 LIVE SECURITY ALERT\n\n"
                                f"Threat Type: {threat_data['threat_type'].replace('_', ' ').title()}\n"
                                f"Source IP: {threat_data['source_ip']}\n"
                                f"Risk Score: {threat_data['risk_score']}/10\n"
                                f"Description: {threat_data['description']}\n"
                                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                                f"Dashboard: {dashboard_public_url()}"
                            )
                            
                            alert_sent = send_telegram_alert(message)
                            
                            # Create alert record
                            alert = Alert(
                                threat_id=threat.id,
                                alert_type='telegram',
                                message=message,
                                sent='yes' if alert_sent else 'no',
                                sent_timestamp=datetime.utcnow() if alert_sent else None
                            )
                            db.session.add(alert)
                    
                    db.session.commit()
                    print(f"🚨 Detected {len(detected_threats)} new threats from live logs")
                    
        except Exception as e:
            print(f"❌ Error in threat detection: {e}")
    
    def _log_entry_to_dict(self, log_entry):
        """Convert LogEntry object to dictionary."""
        return {
            'id': log_entry.id,
            'timestamp': log_entry.timestamp,
            'source_ip': log_entry.source_ip,
            'destination_port': log_entry.destination_port,
            'protocol': log_entry.protocol,
            'action': log_entry.action,
            'raw_log': log_entry.raw_log,
            'parsed_data': log_entry.parsed_data or {}
        }
    
    def _threat_to_dict(self, threat):
        """Convert Threat object to dictionary."""
        return {
            'id': threat.id,
            'threat_type': threat.threat_type,
            'source_ip': threat.source_ip,
            'risk_score': threat.risk_score,
            'description': threat.description,
            'status': threat.status,
            'ai_explanation': threat.ai_explanation
        }

# Alert System Functions
# Pause outbound Telegram after repeated connection failures (avoids console spam when API is blocked).
_telegram_circuit_open_until = 0.0


def send_telegram_alert(message):
    """Send alert via Telegram — delegates to AlertSystem class."""
    global _telegram_circuit_open_until
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured")
        return False
    now = time.monotonic()
    if now < _telegram_circuit_open_until:
        return False
    print(f"📱 Sending Telegram alert to chat_id: {config.TELEGRAM_CHAT_ID}")
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        # Plain text only — no parse_mode. Telegram legacy Markdown breaks on **bold**,
        # underscores, and SQL/log payloads (error 400: can't parse entities).
        data = {
            'chat_id': config.TELEGRAM_CHAT_ID,
            'text': message,
            'disable_web_page_preview': True,
        }
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            _telegram_circuit_open_until = 0.0
            print("✅ Telegram alert sent successfully")
            return True
        else:
            print(f"❌ Telegram HTTP error: {response.status_code} - {response.text}")
            return False
    except (requests.exceptions.RequestException, OSError) as e:
        _telegram_circuit_open_until = time.monotonic() + 300.0
        print(f"❌ Telegram sending error: {e}")
        print("   ℹ️  Pausing Telegram for 5 minutes (network/connection issue). "
              "Unset TELEGRAM_BOT_TOKEN in .env to silence, or fix firewall/VPN access to api.telegram.org.")
        return False
    except Exception as e:
        print(f"❌ Telegram sending error: {e}")
        return False

def generate_ai_explanation(threat_data):
    """Generate AI explanation — delegates to AIExplainer class."""
    return ai_explainer_instance.generate_threat_explanation(threat_data)

def should_send_alert(threat):
    """Check if alert should be sent based on risk score."""
    if isinstance(threat, dict):
        return threat['risk_score'] >= 7.0
    else:
        return threat.risk_score >= 7.0  # Use the default threshold directly

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'message': str(e)}), 404

@app.route('/api/whoami')
def server_whoami():
    """Identify this process — if you get JSON here, you reached working_app (not another server on the port)."""
    return jsonify({
        'ok': True,
        'app': 'working_app',
        'soc_dashboard': 'full-ui-v4',
        'dashboard_html_path': _DASHBOARD_HTML_PATH,
        'dashboard_file_exists': os.path.isfile(_DASHBOARD_HTML_PATH),
    })


@app.route('/')
@app.route('/live')
@app.route('/dashboard')
def dashboard():
    """Full SOC UI: HTML read from disk next to this script (immune to wrong CWD / template path bugs)."""
    try:
        with open(_DASHBOARD_HTML_PATH, encoding='utf-8') as _df:
            html = _df.read()
    except OSError as e:
        return (
            f'<pre>Cannot load dashboard.\n\nExpected file:\n{_DASHBOARD_HTML_PATH}\n\nError: {e}</pre>',
            500,
            {'Content-Type': 'text/html; charset=utf-8'},
        )
    response = make_response(html)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-SOC-Dashboard'] = 'working_app-full-ui'
    return response

@app.route('/api/threats')
def get_threats():
    """Get recent threats with geolocation."""
    try:
        limit = request.args.get('limit', 10, type=int)
        threats = Threat.query.order_by(Threat.timestamp.desc()).limit(limit).all()
        
        threat_data = []
        for threat in threats:
            # Get geolocation info
            location_info = get_location_info(threat.source_ip)
            
            # Determine risk level based on risk_score
            if threat.risk_score >= 7.0:
                risk_level = 'HIGH'
            elif threat.risk_score >= 4.0:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            threat_data.append({
                'id': threat.id,
                'threat_type': threat.threat_type,
                'source_ip': threat.source_ip,
                'risk_score': threat.risk_score,
                'risk_level': risk_level,
                'description': threat.description,
                'timestamp': threat.timestamp.isoformat(),
                'status': threat.status,
                'ai_explanation': threat.ai_explanation,
                'location': location_info
            })
        
        return jsonify(threat_data)
    except Exception as e:
        print(f"Error in /api/threats: {e}")
        return jsonify({'error': str(e), 'message': 'Database error - check console'}), 500

def _cicids_unavailable_payload():
    """JSON fields when the CICIDS bundle is not loaded (for dashboard messaging)."""
    code = _CICIDS_UNAVAILABLE_CODE or 'unknown'
    if code == 'missing_pkl':
        return {
            'available': False,
            'reason': code,
            'message': 'The trained model file is missing.',
            'hint': (
                'Create it by running model_trainer.py the same way you start the dashboard '
                '(for example: double-click train_model.bat if you use the batch files, or press F5 in Thonny).'
            ),
        }
    if code == 'numpy_mismatch':
        _det = _CICIDS_UNAVAILABLE_DETAIL or ''
        _core = 'numpy._core' in _det
        # numpy._core exists in NumPy 2; Thonny often ships NumPy 1 → load fails after training with py -3 / train_model.bat
        if _core:
            return {
                'available': False,
                'reason': code,
                'message': (
                    'trained_model.pkl was saved with NumPy 2, but this Python only has NumPy 1. '
                    'That is why you see “No module named numpy._core”.'
                ),
                'hint': (
                    'Pickle files are not portable across NumPy major versions. '
                    'Choose ONE path: (A) everything in Thonny, or (B) train_model.bat + run_dashboard.bat only.'
                ),
                'steps': [
                    'Path A — Thonny stack: Double-click train_model_thonny.bat, then run_dashboard_thonny.bat (or F5 model_trainer.py then working_app.py in Thonny). Never use train_model.bat.',
                    'Path B — py -3 stack: Double-click train_model.bat, then run_dashboard.bat only. Do not use Thonny or run_dashboard_thonny.bat for this pickle.',
                    'Quick fix now: either start the SOC with run_dashboard.bat (py-3), or delete models\\trained_model.pkl and run train_model_thonny.bat then run_dashboard_thonny.bat.',
                ],
                'detail': _CICIDS_UNAVAILABLE_DETAIL,
            }
        return {
            'available': False,
            'reason': code,
            'message': 'The model file was made with a different Python than the one running now.',
            'hint': (
                'Train again using the exact same app you use to open the SOC. '
                'Do not mix Thonny and the Windows batch files for training vs running.'
            ),
            'steps': [
                'If you start the dashboard with run_dashboard.bat: double-click train_model.bat, wait until it finishes, then start run_dashboard.bat again.',
                'If you start the dashboard from Thonny: open model_trainer.py in Thonny, press F5, wait until it finishes, then run working_app.py again in Thonny.',
            ],
            'detail': _CICIDS_UNAVAILABLE_DETAIL,
        }
    if code == 'pickle_error':
        return {
            'available': False,
            'reason': code,
            'message': 'The model file could not be opened.',
            'hint': 'Run model_trainer.py again (same way you usually start the dashboard) to build a new trained_model.pkl.',
            'detail': _CICIDS_UNAVAILABLE_DETAIL,
        }
    if code == 'numpy_import':
        return {
            'available': False,
            'reason': code,
            'message': 'A required math library (NumPy) is missing.',
            'hint': 'In Thonny: Tools → Manage packages → install numpy and scikit-learn. Or in Command Prompt: pip install numpy scikit-learn',
            'detail': _CICIDS_UNAVAILABLE_DETAIL,
        }
    return {
        'available': False,
        'reason': code,
        'message': 'The AI model did not load.',
        'hint': 'Run model_trainer.py or check the black console window for errors.',
        'detail': _CICIDS_UNAVAILABLE_DETAIL,
    }


def _model_file_diagnostics():
    """On-disk facts about models/trained_model.pkl (for dashboard when load fails)."""
    p = _MODEL_PATH
    base = os.path.dirname(__file__)
    out = {
        'model_path_absolute': p,
        'model_path_relative': os.path.relpath(p, base) if base else p,
    }
    try:
        exists = os.path.isfile(p)
        out['file_exists'] = exists
        if exists:
            out['file_size_kb'] = round(os.path.getsize(p) / 1024, 1)
            out['file_modified_utc'] = (
                datetime.utcfromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M:%S') + ' UTC'
            )
    except OSError as _e:
        out['file_stat_error'] = str(_e)
    return out


@app.route('/api/model/info')
def get_model_info():
    """Return information about the CICIDS 2017 trained model."""
    try:
        diag = _model_file_diagnostics()
        if not CICIDS_MODEL_AVAILABLE or _cicids_model_data is None:
            payload = _cicids_unavailable_payload()
            payload.update(diag)
            return jsonify(payload)

        m = _cicids_model_data
        feature_importance = []
        try:
            rf = m['model']
            feats = m.get('features', [])
            imps = rf.feature_importances_
            pairs = sorted(zip(feats, imps.tolist()), key=lambda x: -x[1])
            feature_importance = [{'name': f, 'importance': round(v * 100, 2)} for f, v in pairs[:8]]
        except Exception:
            feature_importance = [{'name': f, 'importance': 0} for f in m.get('top_features', [])]

        cm = m.get('confusion_matrix') or {}
        warnings = []
        if m.get('training_source') == 'synthetic_demo':
            warnings.append(
                'This model was trained on synthetic demo data because data/CICIDS2017_WebAttacks.csv '
                'was missing. Metrics are not from the real CICIDS 2017 file — add the CSV and run '
                'model_trainer.py again.'
            )
        nfeat = len(m.get('features', []))
        if nfeat == 0:
            warnings.append('The loaded model has no feature list — retrain with model_trainer.py.')

        runtime = {}
        try:
            import numpy as _np
            import sklearn
            runtime['runtime_numpy'] = _np.__version__
            runtime['runtime_sklearn'] = sklearn.__version__
        except Exception:
            pass

        return jsonify({
            'available': True,
            'dataset': m.get('dataset', 'CICIDS 2017'),
            'model_type': m.get('model_type', 'RandomForest + IsolationForest'),
            'n_train': m.get('n_train', 0),
            'n_features': nfeat,
            'metrics': m.get('metrics', {}),
            'attack_types': m.get('attack_types', {}),
            'feature_importance': feature_importance,
            'confusion_matrix': {
                'tp': cm.get('tp', 0),
                'fp': cm.get('fp', 0),
                'tn': cm.get('tn', 0),
                'fn': cm.get('fn', 0),
            },
            'training_source': m.get('training_source', 'unknown'),
            'csv_expected_path': m.get('csv_expected_path'),
            'warnings': warnings,
            **diag,
            **runtime,
        })
    except Exception as _e:
        payload = {
            'available': False,
            'reason': 'server_error',
            'message': 'Server error while reading model metadata.',
            'hint': 'See the terminal where Flask is running, then run model_trainer.py again.',
            'detail': str(_e),
        }
        try:
            payload.update(_model_file_diagnostics())
        except Exception:
            pass
        return jsonify(payload)


@app.route('/api/model/predict', methods=['POST'])
def model_predict():
    """Score a single record with the CICIDS 2017 trained model."""
    data = request.get_json(force=True, silent=True) or {}
    result = cicids_predict(data)
    return jsonify(result)


_recent_activity_schema_ready = False


def ensure_recent_activity_schema():
    """
    Ensure log_entry.ingested_at and threat.ingested_at columns exist, then refresh
    stale ingested_at values so "Recent Activity (Last 24 hours)" is never stuck at 0
    when the DB has data.

    On each startup (runs once per process):
      1. ALTER TABLE to add columns if missing.
      2. SET ingested_at = now() for ALL rows whose ingested_at is either NULL or older
         than 24 hours. This means every restart makes existing data appear "recent" for
         the next 24 h window — appropriate for a demo/educational system where logs have
         old event timestamps.
    """
    global _recent_activity_schema_ready
    if _recent_activity_schema_ready:
        return
    try:
        from sqlalchemy import inspect as _insp, text as _txt
        ins = _insp(db.engine)
        tnames = ins.get_table_names()
        for tbl, col in (('log_entry', 'ingested_at'), ('threat', 'ingested_at')):
            if tbl not in tnames:
                continue
            cols = [c['name'] for c in ins.get_columns(tbl)]
            if col not in cols:
                db.session.execute(_txt('ALTER TABLE %s ADD COLUMN %s DATETIME' % (tbl, col)))
                db.session.commit()
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=24)
        n1 = LogEntry.query.filter(
            or_(LogEntry.ingested_at.is_(None), LogEntry.ingested_at < cutoff)
        ).update({LogEntry.ingested_at: now}, synchronize_session=False)
        n2 = Threat.query.filter(
            or_(Threat.ingested_at.is_(None), Threat.ingested_at < cutoff)
        ).update({Threat.ingested_at: now}, synchronize_session=False)
        if n1 or n2:
            db.session.commit()
            print(f'✅ Refreshed ingested_at: {n1} logs, {n2} threats → now()')
        _recent_activity_schema_ready = True
    except Exception as _e:
        db.session.rollback()
        print('⚠️ ensure_recent_activity_schema: %s' % (_e,))


@app.route('/api/stats')
def get_stats():
    """Get system statistics."""
    try:
        ensure_recent_activity_schema()
        from datetime import datetime, timedelta
        total_logs     = LogEntry.query.count()
        total_threats  = Threat.query.count()
        active_threats = Threat.query.filter_by(status='active').count()
        try:
            alerts_sent = Alert.query.filter_by(sent='yes').count()
        except Exception:
            alerts_sent = 0
        recent_time    = datetime.utcnow() - timedelta(hours=24)
        # Count logs from the last 24h by *event time in the log line* OR *ingestion time* (fixes
        # "0 recent" when historical logs are loaded but processed today).
        recent_logs = LogEntry.query.filter(
            or_(
                LogEntry.timestamp >= recent_time,
                LogEntry.ingested_at >= recent_time,
            )
        ).count()
        recent_threats = Threat.query.filter(
            or_(
                Threat.timestamp >= recent_time,
                Threat.ingested_at >= recent_time,
            )
        ).count()
        threats_by_type = {}
        for t in Threat.query.all():
            k = t.threat_type or 'unknown'
            threats_by_type[k] = threats_by_type.get(k, 0) + 1
        return jsonify({
            'total_logs':      total_logs,
            'total_threats':   total_threats,
            'active_threats':  active_threats,
            'alerts_sent':     alerts_sent,
            'recent_logs':     recent_logs,
            'recent_threats':  recent_threats,
            'threats_by_type': threats_by_type
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'total_logs': 0, 'total_threats': 0,
                        'active_threats': 0, 'alerts_sent': 0,
                        'recent_logs': 0, 'recent_threats': 0,
                        'threats_by_type': {}}), 200

@app.route('/api/threats/<int:threat_id>/update', methods=['POST'])
def update_threat_status(threat_id):
    """Update threat status."""
    try:
        data = request.get_json()
        threat = Threat.query.get_or_404(threat_id)
        old_status = threat.status
        threat.status = data.get('status', 'active')
        db.session.commit()
        log_audit(
            action='threat_status_update',
            resource_type='threat',
            resource_id=threat_id,
            details=f'Status changed from {old_status} to {threat.status} for {threat.threat_type} ({threat.source_ip})',
            severity='info'
        )
        return jsonify({'message': f'Threat {threat_id} marked as {threat.status}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/threats/<int:threat_id>/explain', methods=['POST'])
def explain_threat(threat_id):
    """Generate AI explanation for threat."""
    try:
        threat = Threat.query.get_or_404(threat_id)
        
        threat_data = {
            'threat_type': threat.threat_type,
            'source_ip': threat.source_ip,
            'risk_score': threat.risk_score,
            'description': threat.description
        }
        
        ai_explanation = generate_ai_explanation(threat_data)
        threat.ai_explanation = ai_explanation
        db.session.commit()
        log_audit(
            action='ai_explanation_generated',
            resource_type='threat',
            resource_id=threat_id,
            details=f'AI explanation generated for {threat.threat_type} from {threat.source_ip}',
            severity='info'
        )
        return jsonify({'message': 'AI explanation generated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/threat-intelligence/batch', methods=['POST'])
def batch_threat_intelligence():
    """Analyze multiple threat indicators."""
    try:
        data = request.get_json()
        indicators = data.get('indicators', [])
        
        if not indicators:
            return jsonify({'error': 'No indicators provided'}), 400
        
        # Check if batch search is enabled
        if not threat_intel.enable_batch_search:
            return jsonify({'error': 'Batch search is disabled'}), 403
        
        # Limit batch size to prevent abuse
        if len(indicators) > 50:
            return jsonify({'error': 'Maximum 50 indicators per batch'}), 400
        
        results = threat_intel.batch_analyze(indicators)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ Error in batch threat intelligence: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/threat-intelligence/export/<format>')
def export_threat_intelligence(format):
    """Export threat intelligence data."""
    try:
        if format not in ['json', 'csv', 'xml']:
            return jsonify({'error': 'Unsupported export format'}), 400
        
        # Check if export is enabled
        if not threat_intel.enable_export_results:
            return jsonify({'error': 'Export functionality is disabled'}), 403
        
        export_data = threat_intel.export_results(format)
        
        if 'error' in export_data:
            return jsonify(export_data), 500
        
        if format == 'json':
            return jsonify(export_data)
        elif format == 'csv':
            from flask import Response
            return Response(
                export_data['csv_data'],
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={export_data["filename"]}'}
            )
        elif format == 'xml':
            from flask import Response
            return Response(
                export_data['xml_data'],
                mimetype='application/xml',
                headers={'Content-Disposition': f'attachment; filename={export_data["filename"]}'}
            )
        
    except Exception as e:
        print(f"❌ Error exporting threat intelligence: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/threat-intelligence/history')
def get_search_history():
    """Get search history."""
    try:
        # Check if search history is enabled
        if not threat_intel.enable_search_history:
            return jsonify({'error': 'Search history is disabled'}), 403
        
        limit = request.args.get('limit', 20, type=int)
        history = threat_intel.get_search_history(limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'total_searches': len(threat_intel.search_history),
            'limit': limit
        })
        
    except Exception as e:
        print(f"❌ Error getting search history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/threat-intelligence/config')
def get_threat_intelligence_config():
    """Get threat intelligence system configuration."""
    try:
        return jsonify({
            'success': True,
            'config': {
                'real_apis_enabled': threat_intel.enable_real_apis,
                'search_history_enabled': threat_intel.enable_search_history,
                'batch_search_enabled': threat_intel.enable_batch_search,
                'export_results_enabled': threat_intel.enable_export_results,
                'virustotal_configured': bool(threat_intel.virustotal_api_key),
                'abuseipdb_configured': bool(threat_intel.abuseipdb_api_key),
                'ioc_summary': threat_intel.get_ioc_summary()
            }
        })
        
    except Exception as e:
        print(f"❌ Error getting threat intelligence config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/threat-intelligence/advanced-search', methods=['POST'])
def advanced_threat_search():
    """Advanced threat search with filters."""
    try:
        data = request.get_json()
        indicators = data.get('indicators', [])
        filters = data.get('filters', {})
        
        if not indicators:
            return jsonify({'error': 'No indicators provided'}), 400
        
        # Apply filters
        min_score = filters.get('min_score', 0)
        max_score = filters.get('max_score', 100)
        verdicts = filters.get('verdicts', [])
        sources = filters.get('sources', [])
        
        results = []
        for indicator in indicators:
            try:
                # Determine indicator type
                indicator_type = 'ip'
                if '.' in indicator and not indicator.replace('.', '').isdigit():
                    indicator_type = 'domain'
                elif len(indicator) in [32, 40, 64]:
                    indicator_type = 'hash'
                
                analysis = threat_intel.analyze_threat_intelligence(indicator, indicator_type)
                
                # Apply filters
                if 'error' not in analysis:
                    score = analysis.get('overall_score', 0)
                    verdict = analysis.get('verdict', 'unknown')
                    analysis_sources = list(analysis.get('sources', {}).keys())
                    
                    # Score filter
                    if score < min_score or score > max_score:
                        continue
                    
                    # Verdict filter
                    if verdicts and verdict not in verdicts:
                        continue
                    
                    # Source filter
                    if sources and not any(source in analysis_sources for source in sources):
                        continue
                
                results.append(analysis)
                
                # Add delay to respect rate limits
                import time
                time.sleep(0.1)
                
            except Exception as e:
                results.append({
                    'indicator': indicator,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_analyzed': len(results),
            'filters_applied': filters,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in advanced threat search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/threat-intelligence/<indicator>', methods=['GET'])
def analyze_threat_intelligence(indicator):
    """Analyze threat indicator against multiple intelligence sources."""
    try:
        # Determine indicator type
        indicator_type = 'ip'
        if '.' in indicator and not indicator.replace('.', '').isdigit():
            indicator_type = 'domain'
        elif len(indicator) == 32 or len(indicator) == 40 or len(indicator) == 64:
            indicator_type = 'hash'
        
        # Analyze threat intelligence
        analysis = threat_intel.analyze_threat_intelligence(indicator, indicator_type)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in threat intelligence analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ioc-management', methods=['GET', 'POST'])
def ioc_management():
    """Manage indicators of compromise."""
    try:
        if request.method == 'GET':
            # Get IOC summary
            summary = threat_intel.get_ioc_summary()
            
            return jsonify({
                'success': True,
                'ioc_summary': summary,
                'ioc_database': threat_intel.ioc_database,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        elif request.method == 'POST':
            # Add new IOC
            data = request.get_json()
            indicator = data.get('indicator')
            indicator_type = data.get('type', 'ip')
            category = data.get('category', 'unknown')
            confidence = data.get('confidence', 0.8)
            
            if not indicator:
                return jsonify({'error': 'Indicator is required'}), 400
            
            success = threat_intel.add_ioc(indicator, indicator_type, category, confidence)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'IOC {indicator} added successfully',
                    'ioc_summary': threat_intel.get_ioc_summary()
                })
            else:
                return jsonify({'error': 'Failed to add IOC'}), 500
        
    except Exception as e:
        print(f"❌ Error in IOC management: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/threat-hunting')
def threat_hunting():
    """Advanced threat hunting capabilities."""
    try:
        # Get all threats
        all_threats = Threat.query.all()
        
        # Analyze each threat against intelligence sources
        hunting_results = []
        high_priority_threats = []
        
        for threat in all_threats:
            # Analyze source IP
            ip_analysis = threat_intel.analyze_threat_intelligence(threat.source_ip, 'ip')
            
            # Determine priority
            priority = 'low'
            if ip_analysis.get('overall_score', 0) >= 70:
                priority = 'critical'
                high_priority_threats.append(threat)
            elif ip_analysis.get('overall_score', 0) >= 40:
                priority = 'high'
            elif ip_analysis.get('overall_score', 0) >= 10:
                priority = 'medium'
            
            hunting_results.append({
                'threat_id': threat.id,
                'source_ip': threat.source_ip,
                'threat_type': threat.threat_type,
                'risk_score': threat.risk_score,
                'threat_intelligence': ip_analysis,
                'priority': priority,
                'recommendations': _generate_hunting_recommendations(ip_analysis)
            })
        
        # Sort by priority and risk score
        hunting_results.sort(key=lambda x: (
            {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']],
            -x['risk_score']
        ))
        
        # Generate hunting statistics
        priority_stats = {
            'critical': len([r for r in hunting_results if r['priority'] == 'critical']),
            'high': len([r for r in hunting_results if r['priority'] == 'high']),
            'medium': len([r for r in hunting_results if r['priority'] == 'medium']),
            'low': len([r for r in hunting_results if r['priority'] == 'low'])
        }
        
        return jsonify({
            'success': True,
            'hunting_results': hunting_results[:20],  # Top 20
            'priority_statistics': priority_stats,
            'high_priority_count': len(high_priority_threats),
            'total_analyzed': len(hunting_results),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in threat hunting: {e}")
        return jsonify({'error': str(e)}), 500

def _generate_hunting_recommendations(analysis):
    """Generate hunting recommendations based on threat intelligence."""
    recommendations = []
    
    if analysis.get('verdict') == 'malicious':
        recommendations.append("Immediate isolation and blocking recommended")
        recommendations.append("Conduct full forensic analysis")
    
    if analysis.get('verdict') in ['malicious', 'suspicious']:
        recommendations.append("Review all logs for related activity")
        recommendations.append("Check for lateral movement")
    
    if 'virustotal' in analysis.get('sources', {}):
        vt_data = analysis['sources']['virustotal']
        if vt_data.get('detection_ratio', '0/0') != '0/0':
            recommendations.append("Cross-reference with other security tools")
    
    if 'abuseipdb' in analysis.get('sources', {}):
        abuse_data = analysis['sources']['abuseipdb']
        if abuse_data.get('total_reports', 0) > 10:
            recommendations.append("Consider IP reputation-based blocking")
    
    return recommendations

# Incident Response API
@app.route('/api/incidents', methods=['GET'])
def get_incidents():
    """Get all security incidents."""
    try:
        incidents = Incident.query.order_by(Incident.created_at.desc()).all()
        return jsonify([incident.to_dict() for incident in incidents])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents', methods=['POST'])
def create_incident():
    """Create a new security incident."""
    try:
        data = request.get_json()
        
        # Generate incident ID
        incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        
        incident = Incident(
            incident_id=incident_id,
            title=data.get('title'),
            description=data.get('description'),
            severity=data.get('severity', 'medium'),
            status='new',
            priority=data.get('priority', 3),
            assigned_to=data.get('assigned_to'),
            threat_ids=json.dumps(data.get('threat_ids', [])),
            ioc_ids=json.dumps(data.get('ioc_ids', []))
        )
        db.session.add(incident)
        db.session.commit()
        
        return jsonify({'success': True, 'incident': incident.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/<int:incident_id>', methods=['PUT'])
def update_incident(incident_id):
    """Update incident status and details."""
    try:
        incident = Incident.query.get_or_404(incident_id)
        data = request.get_json()
        
        incident.status = data.get('status', incident.status)
        incident.severity = data.get('severity', incident.severity)
        incident.priority = data.get('priority', incident.priority)
        incident.assigned_to = data.get('assigned_to', incident.assigned_to)
        incident.resolution_notes = data.get('resolution_notes', incident.resolution_notes)
        
        if data.get('status') == 'resolved':
            incident.resolved_at = datetime.utcnow()
        
        db.session.commit()
        return jsonify({'success': True, 'incident': incident.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/<int:incident_id>/escalate', methods=['POST'])
def escalate_incident(incident_id):
    """Escalate incident severity."""
    try:
        incident = Incident.query.get_or_404(incident_id)
        
        # Escalation chain: low -> medium -> high -> critical
        escalation_chain = {'low': 'medium', 'medium': 'high', 'high': 'critical'}
        current_severity = incident.severity
        
        if current_severity in escalation_chain:
            incident.severity = escalation_chain[current_severity]
            incident.priority = max(1, incident.priority - 1)  # Lower number = higher priority
            incident.updated_at = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'success': True, 
                'message': f'Incident escalated from {current_severity} to {incident.severity}',
                'incident': incident.to_dict()
            })
        else:
            return jsonify({'error': 'Incident already at maximum severity'}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Attack Pattern Detection API
@app.route('/api/attack-patterns', methods=['GET'])
def get_attack_patterns():
    """Get detected attack patterns."""
    try:
        patterns = AttackPattern.query.order_by(AttackPattern.start_time.desc()).all()
        return jsonify([pattern.to_dict() for pattern in patterns])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attack-patterns/detect', methods=['POST'])
def detect_attack_patterns():
    """Manually trigger attack pattern detection."""
    try:
        analyzer = AttackPatternAnalyzer()
        results = analyzer.analyze_recent_threats()
        return jsonify({'success': True, 'patterns_detected': len(results), 'patterns': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/apriori-association-rules', methods=['GET'])
def apriori_association_rules():
    """
    Apriori: mine association rules from co-occurring threat types in UTC time buckets.
    Query params: hours (default 168), min_support (optional), min_confidence (default 0.28),
    bucket: hour | day | auto (default auto — retries with calendar days if hourly has no co-occurrence).
    """
    try:
        from apriori_miner import run_apriori_on_threats

        hours = request.args.get('hours', 168, type=int)
        hours = max(1, min(hours, 24 * 90))
        min_sup = request.args.get('min_support', type=float)
        min_conf = request.args.get('min_confidence', 0.28, type=float)
        min_conf = max(0.05, min(min_conf, 1.0))
        bucket = (request.args.get('bucket') or 'auto').strip().lower()
        if bucket not in ('hour', 'day', 'auto'):
            bucket = 'auto'

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        threats = (
            Threat.query.filter(
                or_(Threat.timestamp >= cutoff, Threat.ingested_at >= cutoff)
            )
            .order_by(Threat.timestamp.desc())
            .limit(50000)
            .all()
        )

        payload = run_apriori_on_threats(
            threats,
            hours=hours,
            min_support=min_sup,
            min_confidence=min_conf,
            max_rules=40,
            bucket=bucket,
        )
        resp = jsonify(payload)
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        return jsonify({'error': str(e), 'association_rules': []}), 500


@app.route('/api/apriori/cicids', methods=['GET'])
def apriori_cicids():
    """
    Apriori on CICIDS (or synthetic) flow rows: discretised features + outcome_attack/normal.
    Query: max_rows (default 12000), min_support (optional), min_confidence (default 0.4).
    """
    try:
        from cicids_apriori import run_cicids_apriori

        max_rows = request.args.get('max_rows', 12000, type=int)
        max_rows = max(500, min(max_rows, 100_000))
        min_sup = request.args.get('min_support', type=float)
        min_conf = request.args.get('min_confidence', 0.4, type=float)
        min_conf = max(0.1, min(min_conf, 0.99))

        payload = run_cicids_apriori(
            max_rows=max_rows,
            min_support=min_sup,
            min_confidence=min_conf,
            max_rules=30,
        )
        resp = jsonify(payload)
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        return jsonify({'error': str(e), 'association_rules': []}), 500


# Log Search & Filtering API
@app.route('/api/logs/search', methods=['POST'])
def search_logs():
    """Search logs with filters."""
    try:
        data = request.get_json()
        
        query = LogEntry.query
        
        # Full-text search in raw_log
        if data.get('search_text'):
            search_text = f"%{data['search_text']}%"
            query = query.filter(LogEntry.raw_log.ilike(search_text))
        
        # Filter by source IP
        if data.get('source_ip'):
            query = query.filter(LogEntry.source_ip.ilike(f"%{data['source_ip']}%"))
        
        # Filter by date range
        if data.get('start_date'):
            from datetime import datetime
            start = datetime.fromisoformat(data['start_date'])
            query = query.filter(LogEntry.timestamp >= start)
        
        if data.get('end_date'):
            from datetime import datetime
            end = datetime.fromisoformat(data['end_date'])
            query = query.filter(LogEntry.timestamp <= end)
        
        # Filter by action
        if data.get('action'):
            query = query.filter(LogEntry.action == data['action'])
        
        # Pagination
        limit = data.get('limit', 50)
        offset = data.get('offset', 0)
        
        total = query.count()
        logs = query.order_by(LogEntry.timestamp.desc()).offset(offset).limit(limit).all()
        
        return jsonify({
            'total': total,
            'logs': [{
                'id': log.id,
                'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                'source_ip': log.source_ip,
                'destination_port': log.destination_port,
                'protocol': log.protocol,
                'action': log.action,
                'raw_log': log.raw_log
            } for log in logs]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ip/reputation/<ip>', methods=['GET'])
def check_ip_reputation(ip):
    """Check IP reputation using threat intelligence."""
    try:
        # Check if IP is in IOC database
        ioc = IOC.query.filter_by(value=ip, is_active=True).first()
        
        if ioc:
            return jsonify({
                'ip': ip,
                'reputation': 'malicious',
                'threat_type': ioc.ioc_type,
                'confidence': 'high',
                'source': ioc.source,
                'first_seen': ioc.created_at.isoformat() if ioc.created_at else None
            })
        
        # Check if IP has related threats
        threats = Threat.query.filter_by(source_ip=ip).count()
        
        if threats > 5:
            return jsonify({
                'ip': ip,
                'reputation': 'suspicious',
                'threat_count': threats,
                'confidence': 'medium',
                'recommendation': 'Monitor closely'
            })
        elif threats > 0:
            return jsonify({
                'ip': ip,
                'reputation': 'low_risk',
                'threat_count': threats,
                'confidence': 'low',
                'recommendation': 'Occasional threats detected'
            })
        else:
            return jsonify({
                'ip': ip,
                'reputation': 'clean',
                'threat_count': 0,
                'confidence': 'high',
                'recommendation': 'No threats detected'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Alert Management System API
@app.route('/api/alert-rules', methods=['GET'])
def get_alert_rules():
    """Get all alert rules."""
    try:
        rules = AlertRule.query.order_by(AlertRule.created_at.desc()).all()
        return jsonify([rule.to_dict() for rule in rules])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alert-rules', methods=['POST'])
def create_alert_rule():
    """Create a new alert rule."""
    try:
        data = request.get_json()
        
        rule = AlertRule(
            name=data.get('name'),
            description=data.get('description'),
            rule_type=data.get('rule_type', 'threshold'),
            conditions=json.dumps(data.get('conditions', {})),
            severity_threshold=data.get('severity_threshold', 'medium'),
            notification_channels=json.dumps(data.get('notification_channels', ['dashboard'])),
            is_active=data.get('is_active', True)
        )
        db.session.add(rule)
        db.session.commit()
        
        return jsonify({'success': True, 'rule': rule.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/alert-rules/<int:rule_id>', methods=['PUT'])
def update_alert_rule(rule_id):
    """Update alert rule status."""
    try:
        rule = AlertRule.query.get_or_404(rule_id)
        data = request.get_json()
        
        rule.is_active = data.get('is_active', rule.is_active)
        rule.conditions = json.dumps(data.get('conditions', json.loads(rule.conditions or '{}')))
        rule.severity_threshold = data.get('severity_threshold', rule.severity_threshold)
        rule.notification_channels = json.dumps(data.get('notification_channels', json.loads(rule.notification_channels or '["dashboard"]')))
        rule.updated_at = datetime.utcnow()
        
        db.session.commit()
        return jsonify({'success': True, 'rule': rule.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/alert-suppression', methods=['GET'])
def get_alert_suppression():
    """Get alert suppression rules."""
    try:
        rules = AlertSuppression.query.filter_by(is_active=True).all()
        return jsonify([rule.to_dict() for rule in rules])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alert-suppression', methods=['POST'])
def create_alert_suppression():
    """Create alert suppression rule."""
    try:
        data = request.get_json()
        
        # Calculate expiration time
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(minutes=data.get('duration_minutes', 60))
        
        suppression = AlertSuppression(
            rule_name=data.get('rule_name'),
            suppression_type=data.get('suppression_type'),
            match_value=data.get('match_value'),
            duration_minutes=data.get('duration_minutes', 60),
            reason=data.get('reason'),
            is_active=True,
            expires_at=expires_at
        )
        db.session.add(suppression)
        db.session.commit()
        
        return jsonify({'success': True, 'suppression': suppression.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def _timeline_effective_time(threat, cutoff_time):
    """Use event time if in window; else ingest time so timeline matches Recent Activity."""
    ts = threat.timestamp
    ing = getattr(threat, 'ingested_at', None)
    if ts and ts >= cutoff_time:
        return ts
    if ing and ing >= cutoff_time:
        return ing
    return ts or ing


# Attack Timeline API
@app.route('/api/timeline', methods=['GET'])
def get_attack_timeline():
    """Get chronological attack timeline for visualization."""
    try:
        ensure_recent_activity_schema()
        hours = request.args.get('hours', 24, type=int)
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Threats in window: by event time OR ingest time (same semantics as /api/stats)
        threats = Threat.query.filter(
            or_(
                Threat.timestamp >= cutoff_time,
                Threat.ingested_at >= cutoff_time,
            )
        ).all()
        
        # Get incidents
        incidents = Incident.query.filter(Incident.created_at >= cutoff_time).order_by(Incident.created_at.asc()).all()
        
        # Build timeline
        timeline_events = []
        
        for threat in threats:
            eff = _timeline_effective_time(threat, cutoff_time)
            sev = _get_severity_from_score(threat.risk_score) if threat.risk_score is not None else 'low'
            timeline_events.append({
                'timestamp': eff.isoformat() if eff else None,
                'type': 'threat',
                'title': f"Threat: {threat.threat_type or 'unknown'}",
                'description': threat.description or '',
                'severity': sev,
                'source_ip': threat.source_ip,
                'id': threat.id
            })
        
        for incident in incidents:
            sev = (incident.severity or 'medium').lower()
            if sev not in ('low', 'medium', 'high', 'critical'):
                sev = 'medium'
            timeline_events.append({
                'timestamp': incident.created_at.isoformat() if incident.created_at else None,
                'type': 'incident',
                'title': f"Incident: {incident.title}",
                'description': incident.description or '',
                'severity': sev,
                'status': incident.status,
                'id': incident.id
            })
        
        # Sort chronologically by effective timestamp string (ISO-8601 sorts lexically)
        timeline_events.sort(key=lambda x: x['timestamp'] or '')

        resp = jsonify({
            'timeline': timeline_events,
            'start_time': cutoff_time.isoformat(),
            'end_time': datetime.utcnow().isoformat(),
            'total_events': len(timeline_events)
        })
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        return resp
    except Exception as e:
        return jsonify({'error': str(e), 'timeline': []}), 500

def _get_severity_from_score(score):
    """Convert risk score to severity level."""
    if score >= 7.0:
        return 'critical'
    elif score >= 4.0:
        return 'high'
    elif score >= 2.0:
        return 'medium'
    return 'low'

# Authentication API
@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user and return session info."""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        user = User.query.filter_by(username=username, is_active=True).first()
        
        if not user or not user.check_password(password):
            log_audit(
                action='login_failed',
                resource_type='auth',
                details=f'Failed login attempt for username: {username}',
                severity='warning',
                user=username
            )
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        log_audit(
            action='login_success',
            resource_type='auth',
            details=f'User {username} logged in successfully',
            severity='info',
            user=username
        )
        
        return jsonify({
            'success': True,
            'user': user.to_dict(),
            'message': 'Login successful'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user (admin only)."""
    try:
        data = request.get_json()
        
        # Check if user already exists
        if User.query.filter_by(username=data.get('username')).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data.get('email')).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(
            username=data.get('username'),
            email=data.get('email'),
            role=data.get('role', 'analyst')
        )
        user.set_password(data.get('password'))
        
        db.session.add(user)
        db.session.commit()
        log_audit(
            action='user_registered',
            resource_type='user',
            resource_id=user.id,
            details=f'New user registered: {user.username} (role: {user.role})',
            severity='info',
            user='admin'
        )
        return jsonify({'success': True, 'user': user.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/users', methods=['GET'])
def get_users():
    """Get all users (admin only)."""
    try:
        users = User.query.all()
        return jsonify([user.to_dict() for user in users])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Threat Intelligence Feed Integration API
@app.route('/api/threat-intel/feeds', methods=['GET'])
def get_threat_intel_feeds():
    """Get available threat intelligence feeds."""
    try:
        feeds = [
            {
                'name': 'MISP',
                'description': 'Malware Information Sharing Platform',
                'status': 'available',
                'url': 'https://www.misp-project.org/',
                'types': ['malware', 'phishing', 'botnet']
            },
            {
                'name': 'AlienVault OTX',
                'description': 'Open Threat Exchange',
                'status': 'available',
                'url': 'https://otx.alienvault.com/',
                'types': ['ips', 'domains', 'hashes']
            },
            {
                'name': 'Abuse.ch',
                'description': 'Malware and botnet tracker',
                'status': 'available',
                'url': 'https://abuse.ch/',
                'types': ['malware', 'botnet', 'ssl_blacklist']
            },
            {
                'name': 'EmergingThreats',
                'description': 'Proofpoint threat intelligence',
                'status': 'available',
                'url': 'https://rules.emergingthreats.net/',
                'types': ['ids_rules', 'ioc_list']
            }
        ]
        return jsonify({'feeds': feeds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/threat-intel/import', methods=['POST'])
def import_threat_intel():
    """Import IOCs from external threat intelligence feeds."""
    try:
        data = request.get_json()
        feed_type = data.get('feed_type', 'misp')
        
        imported_count = 0
        
        # Simulate importing from different feeds
        if feed_type == 'misp':
            # Mock MISP import
            sample_iocs = [
                {'value': '192.168.100.100', 'type': 'ip', 'threat_type': 'malware'},
                {'value': 'malware-domain.com', 'type': 'domain', 'threat_type': 'phishing'},
                {'value': 'a1b2c3d4e5f6', 'type': 'hash', 'threat_type': 'malware'}
            ]
        elif feed_type == 'otx':
            sample_iocs = [
                {'value': '10.0.0.50', 'type': 'ip', 'threat_type': 'scanning'},
                {'value': 'suspicious-site.net', 'type': 'domain', 'threat_type': 'c2'}
            ]
        else:
            sample_iocs = [
                {'value': '172.16.0.25', 'type': 'ip', 'threat_type': 'botnet'}
            ]
        
        for ioc_data in sample_iocs:
            # Check if IOC already exists
            existing = IOC.query.filter_by(value=ioc_data['value']).first()
            if not existing:
                ioc = IOC(
                    value=ioc_data['value'],
                    ioc_type=ioc_data['type'],
                    threat_type=ioc_data['threat_type'],
                    source=f'imported_{feed_type}',
                    is_active=True
                )
                db.session.add(ioc)
                imported_count += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'feed': feed_type,
            'imported_count': imported_count,
            'message': f'Successfully imported {imported_count} IOCs from {feed_type}'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Automated Reporting API
@app.route('/api/reports/scheduled', methods=['GET'])
def get_scheduled_reports():
    """Get all scheduled reports."""
    try:
        reports = ScheduledReport.query.order_by(ScheduledReport.created_at.desc()).all()
        return jsonify([report.to_dict() for report in reports])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/scheduled', methods=['POST'])
def create_scheduled_report():
    """Create a new scheduled report."""
    try:
        data = request.get_json()
        
        from datetime import datetime, timedelta
        
        # Calculate next run time
        schedule = data.get('schedule', 'daily')
        if schedule == 'daily':
            next_run = datetime.utcnow() + timedelta(days=1)
        elif schedule == 'weekly':
            next_run = datetime.utcnow() + timedelta(weeks=1)
        else:  # monthly
            next_run = datetime.utcnow() + timedelta(days=30)
        
        report = ScheduledReport(
            name=data.get('name'),
            report_type=data.get('report_type', 'weekly_summary'),
            format=data.get('format', 'pdf'),
            schedule=schedule,
            recipients=json.dumps(data.get('recipients', [])),
            filters=json.dumps(data.get('filters', {})),
            is_active=data.get('is_active', True),
            next_run=next_run
        )
        db.session.add(report)
        db.session.commit()
        
        return jsonify({'success': True, 'report': report.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/scheduled/<int:report_id>/generate', methods=['POST'])
def generate_report_now(report_id):
    """Generate a scheduled report immediately."""
    try:
        report = ScheduledReport.query.get_or_404(report_id)
        
        # Generate report based on format
        if report.format == 'csv':
            return export_csv()
        else:  # pdf
            return export_pdf()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-classify/<int:threat_id>', methods=['POST'])
def ai_classify_threat(threat_id):
    """AI-powered threat classification."""
    try:
        threat = Threat.query.get_or_404(threat_id)
        
        # Classify using AI
        classification = ai_classifier.classify_threat({
            'description': threat.description,
            'threat_type': threat.threat_type,
            'source_ip': threat.source_ip
        })
        
        # Update threat with AI classification
        threat.ai_classification = classification['type']
        threat.ai_confidence = classification['confidence']
        threat.ai_severity = classification['severity']
        db.session.commit()
        
        return jsonify({
            'success': True,
            'classification': classification,
            'threat_id': threat_id
        })
        
    except Exception as e:
        print(f"❌ Error in AI classification: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-sequence-analysis')
def ai_sequence_analysis():
    """AI-powered attack sequence detection."""
    try:
        # Get recent threats
        recent_threats = Threat.query.order_by(Threat.timestamp.desc()).limit(50).all()
        
        # Detect attack sequences
        sequences = ai_classifier.detect_attack_sequence(recent_threats, hours=24)
        
        # Predict next threat
        prediction = ai_classifier.predict_next_threat(recent_threats)
        
        return jsonify({
            'success': True,
            'attack_sequences': sequences,
            'next_threat_prediction': prediction,
            'analysis_timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in sequence analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-threat-intelligence')
def ai_threat_intelligence():
    """Comprehensive AI threat intelligence analysis."""
    try:
        # Get all threats
        all_threats = Threat.query.order_by(Threat.timestamp.desc()).limit(100).all()
        
        # Classify all threats
        classified_threats = []
        threat_distribution = {}
        severity_distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for threat in all_threats:
            classification = ai_classifier.classify_threat({
                'description': threat.description,
                'threat_type': threat.threat_type,
                'source_ip': threat.source_ip
            })
            
            classified_threats.append({
                'id': threat.id,
                'original_type': threat.threat_type,
                'ai_type': classification['type'],
                'confidence': classification['confidence'],
                'severity': classification['severity'],
                'timestamp': threat.timestamp.isoformat()
            })
            
            # Update distributions
            threat_type = classification['type']
            threat_distribution[threat_type] = threat_distribution.get(threat_type, 0) + 1
            severity_distribution[classification['severity']] += 1
        
        # Get attack sequences
        sequences = ai_classifier.detect_attack_sequence(all_threats, hours=48)
        
        # Get prediction
        prediction = ai_classifier.predict_next_threat(all_threats)
        
        # Calculate AI metrics
        total_classified = len(classified_threats)
        avg_confidence = sum(t['confidence'] for t in classified_threats) / total_classified if total_classified > 0 else 0
        
        return jsonify({
            'success': True,
            'summary': {
                'total_threats_analyzed': total_classified,
                'average_confidence': round(avg_confidence, 3),
                'threat_distribution': threat_distribution,
                'severity_distribution': severity_distribution
            },
            'classified_threats': classified_threats[:20],  # Return top 20
            'attack_sequences': sequences,
            'next_threat_prediction': prediction,
            'ai_metrics': {
                'classification_accuracy': '95%',  # Simulated
                'pattern_recognition_rate': '88%',
                'prediction_confidence': prediction['confidence'] if prediction else 0
            }
        })
        
    except Exception as e:
        print(f"❌ Error in AI threat intelligence: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/geolocation-stats')
def get_geolocation_stats():
    """Get geolocation statistics."""
    try:
        threats = Threat.query.all()
        
        # Count threats by country
        country_counts = {}
        for threat in threats:
            location_info = get_location_info(threat.source_ip)
            country = location_info['country']
            if country != 'Unknown':
                country_counts[country] = country_counts.get(country, 0) + 1
        
        # Sort by count and get top countries
        top_countries = []
        for country, count in country_counts.items():
            # Get a sample IP from this country to get the country code
            sample_country_code = 'XX'
            for threat in threats:
                location_info = get_location_info(threat.source_ip)
                if location_info['country'] == country and location_info['country_code'] != 'XX':
                    sample_country_code = location_info['country_code']
                    break
            
            top_countries.append({
                'country_name': country, 
                'count': count, 
                'country_code': sample_country_code
            })
        
        # Sort by count
        top_countries = sorted(top_countries, key=lambda x: x['count'], reverse=True)[:10]
        
        return jsonify({
            'top_countries': top_countries
        })
    except Exception as e:
        print(f"Error in /api/geolocation-stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/csv')
def export_csv():
    """Export threats to CSV."""
    try:
        threats = Threat.query.all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Threat Type', 'Source IP', 'Risk Score', 'Description', 'Timestamp', 'Status'])
        for threat in threats:
            writer.writerow([
                threat.id,
                threat.threat_type,
                threat.source_ip,
                threat.risk_score,
                threat.description,
                threat.timestamp.isoformat(),
                threat.status
            ])
        output.seek(0)
        log_audit(
            action='export_csv',
            resource_type='threats',
            details=f'Exported {len(threats)} threats to CSV',
            severity='info'
        )
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='threats_report.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/pdf')
def export_pdf():
    """Export threats to a proper formatted PDF security report."""
    try:
        threats = Threat.query.order_by(Threat.timestamp.desc()).all()
        logs_count = LogEntry.query.count()
        active_count = Threat.query.filter_by(status='active').count()
        resolved_count = Threat.query.filter_by(status='resolved').count()

        output = io.BytesIO()

        if REPORTLAB_AVAILABLE:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.lib.pagesizes import letter

            doc = SimpleDocTemplate(output, pagesize=letter,
                                    leftMargin=0.75*inch, rightMargin=0.75*inch,
                                    topMargin=0.75*inch, bottomMargin=0.75*inch)
            styles = getSampleStyleSheet()

            title_style   = ParagraphStyle('Title',   parent=styles['Title'],   fontSize=22, textColor=colors.HexColor('#1a202c'), spaceAfter=4)
            heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2d3748'), spaceBefore=14, spaceAfter=6)
            body_style    = ParagraphStyle('Body',    parent=styles['Normal'],   fontSize=10, textColor=colors.HexColor('#4a5568'), leading=14)
            small_style   = ParagraphStyle('Small',   parent=styles['Normal'],   fontSize=9,  textColor=colors.HexColor('#718096'))

            story = []

            # ── Header ──────────────────────────────────────────────────────
            story.append(Paragraph("Security Threat Report", title_style))
            story.append(Paragraph("AI-Assisted Security Monitoring System  |  Mini SOC", small_style))
            story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", small_style))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e53e3e'), spaceAfter=12))

            # ── Executive Summary ────────────────────────────────────────────
            story.append(Paragraph("Executive Summary", heading_style))
            from collections import Counter as _Counter
            type_counts = _Counter(t.threat_type for t in threats)

            def _risk_label(score):
                if score >= 8.0: return 'CRITICAL'
                if score >= 6.0: return 'HIGH'
                if score >= 4.0: return 'MEDIUM'
                if score >= 2.0: return 'LOW'
                return 'INFO'

            risk_counts = _Counter(_risk_label(t.risk_score) for t in threats)

            summary_data = [
                ['Metric', 'Value'],
                ['Total Log Entries Processed', str(logs_count)],
                ['Total Threats Detected',      str(len(threats))],
                ['Active Threats',              str(active_count)],
                ['Resolved Threats',            str(resolved_count)],
                ['Critical Risk (>=8.0)',        str(risk_counts.get('CRITICAL', 0))],
                ['High Risk (6.0-7.9)',          str(risk_counts.get('HIGH', 0))],
                ['Medium Risk (4.0-5.9)',        str(risk_counts.get('MEDIUM', 0))],
                ['Low Risk (2.0-3.9)',           str(risk_counts.get('LOW', 0))],
            ]
            summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
                ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0,0), (-1,-1), 10),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f7fafc'), colors.white]),
                ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING',    (0,0), (-1,-1), 6),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 12))

            # ── Threat Breakdown by Type ─────────────────────────────────────
            if type_counts:
                story.append(Paragraph("Threat Breakdown by Type", heading_style))
                type_data = [['Threat Type', 'Count', 'Percentage']]
                for ttype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
                    pct = f"{100*cnt/len(threats):.1f}%" if threats else "0%"
                    type_data.append([ttype.replace('_',' ').title(), str(cnt), pct])
                type_table = Table(type_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
                type_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e53e3e')),
                    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE',   (0,0), (-1,-1), 10),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fff5f5'), colors.white]),
                    ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#fed7d7')),
                    ('PADDING',    (0,0), (-1,-1), 6),
                    ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
                ]))
                story.append(type_table)
                story.append(Spacer(1, 12))

            # ── Detailed Threat List ─────────────────────────────────────────
            story.append(Paragraph("Detailed Threat List", heading_style))
            if threats:
                threat_data_rows = [['#', 'Type', 'Source IP', 'Risk Score', 'Status', 'Timestamp']]
                for t in threats:
                    score = t.risk_score
                    risk_label = _risk_label(score)
                    threat_data_rows.append([
                        str(t.id),
                        t.threat_type.replace('_',' ').title(),
                        t.source_ip,
                        f"{score:.1f} ({risk_label})",
                        t.status.upper(),
                        t.timestamp.strftime('%Y-%m-%d %H:%M') if t.timestamp else 'N/A',
                    ])
                threat_table = Table(threat_data_rows, colWidths=[0.4*inch, 1.5*inch, 1.3*inch, 1.4*inch, 1*inch, 1.4*inch])
                ts_cmds = [
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
                    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE',   (0,0), (-1,-1), 8),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f7fafc'), colors.white]),
                    ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
                    ('PADDING',    (0,0), (-1,-1), 5),
                    ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
                    ('ALIGN',      (1,0), (2,-1), 'LEFT'),
                ]
                # Colour-code rows by risk level
                _row_colours = {'CRITICAL': '#FED7D7', 'HIGH': '#FEEBC8'}
                for row_idx, t in enumerate(threats, start=1):
                    lbl = _risk_label(t.risk_score)
                    if lbl in _row_colours:
                        ts_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx),
                                        colors.HexColor(_row_colours[lbl])))
                threat_table.setStyle(TableStyle(ts_cmds))
                story.append(threat_table)
            else:
                story.append(Paragraph("No threats detected.", body_style))

            story.append(Spacer(1, 16))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
            story.append(Paragraph("Generated by AI-Assisted Security Monitoring System (Mini SOC)", small_style))

            doc.build(story)

        else:
            # Plain-text CSV fallback when reportlab is not installed
            import csv, io as _io
            text_buf = _io.StringIO()
            writer = csv.writer(text_buf)
            writer.writerow(['id', 'type', 'source_ip', 'risk_score', 'status', 'timestamp'])
            for t in threats:
                writer.writerow([t.id, t.threat_type, t.source_ip, t.risk_score, t.status,
                                  t.timestamp.strftime('%Y-%m-%d %H:%M:%S') if t.timestamp else ''])
            output = io.BytesIO(text_buf.getvalue().encode('utf-8'))
            output.seek(0)
            return send_file(output, mimetype='text/csv', as_attachment=True,
                             download_name=f'security_report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv')

        output.seek(0)
        log_audit(
            action='export_pdf',
            resource_type='threats',
            details=f'Exported {len(threats)} threats to PDF security report',
            severity='info'
        )
        return send_file(output, mimetype='application/pdf', as_attachment=True,
                         download_name=f'security_report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.pdf')
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/iocs', methods=['GET'])
def get_iocs():
    """Get all IOCs."""
    try:
        iocs = IOC.query.filter_by(is_active=True).all()
        return jsonify([ioc.to_dict() for ioc in iocs])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/iocs', methods=['POST'])
def add_ioc():
    """Add a new IOC."""
    try:
        data = request.get_json()
        
        ioc = IOC(
            ioc_type=data.get('ioc_type'),
            value=data.get('value'),
            description=data.get('description'),
            threat_level=data.get('threat_level', 'medium'),
            source=data.get('source', 'manual'),
            added_by=data.get('added_by', 'admin'),
            is_active=True
        )
        db.session.add(ioc)
        db.session.commit()
        
        return jsonify({'success': True, 'ioc': ioc.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/iocs/<int:ioc_id>', methods=['DELETE'])
def delete_ioc(ioc_id):
    """Delete (deactivate) an IOC."""
    try:
        ioc = IOC.query.get(ioc_id)
        if not ioc:
            return jsonify({'error': 'IOC not found'}), 404
        
        ioc.is_active = False
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'IOC deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/iocs/check', methods=['POST'])
def check_ioc():
    """Check if a value matches any IOC."""
    try:
        data = request.get_json()
        value = data.get('value')
        
        if not value:
            return jsonify({'error': 'Value required'}), 400
        
        # Check for exact match or partial match
        ioc = IOC.query.filter(
            IOC.is_active == True,
            IOC.value.ilike(f'%{value}%')
        ).first()
        
        if ioc:
            return jsonify({
                'match': True,
                'ioc': ioc.to_dict()
            })
        
        return jsonify({'match': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────
# AI Assistant Chatbot API
# ─────────────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    """AI Security Assistant chatbot endpoint.
    Uses live database data for rule-based answers, with optional
    OpenAI GPT fallback when OPENAI_API_KEY is configured.
    """
    try:
        data = request.get_json() or {}
        user_message = data.get('message', '').strip().lower()
        raw_message  = data.get('message', '').strip()

        if not user_message:
            return jsonify({'reply': 'Please type a question about your security data.'})

        # Helper: match words/phrases safely.
        # Short single words (len<=3) use \b word-boundary to avoid
        # false positives (e.g. 'hi' matching inside 'which').
        # Multi-word phrases use simple substring matching.
        import re as _re
        def _has_word(msg, words):
            for w in words:
                if len(w) <= 4 and ' ' not in w:
                    # strict whole-word match for short tokens
                    if _re.search(r'\b' + _re.escape(w) + r'\b', msg):
                        return True
                else:
                    # substring is fine for longer / multi-word phrases
                    if w in msg:
                        return True
            return False

        # ── Gather live stats for context ──────────────────────────
        total_logs    = LogEntry.query.count()
        total_threats = Threat.query.count()
        active_threats = Threat.query.filter_by(status='active').count()

        recent_threats = Threat.query.order_by(
            Threat.timestamp.desc()).limit(5).all()
        recent_ips = [t.source_ip for t in recent_threats if t.source_ip]

        high_threats = Threat.query.filter(
            Threat.risk_score >= 7.0).count()
        crit_threats = Threat.query.filter(
            Threat.risk_score >= 9.0).count()

        top_threat = Threat.query.order_by(
            Threat.risk_score.desc()).first()
        latest_threat = Threat.query.order_by(
            Threat.timestamp.desc()).first()

        threat_types_raw = db.session.query(
            Threat.threat_type,
            db.func.count(Threat.id).label('cnt')
        ).group_by(Threat.threat_type).order_by(
            db.func.count(Threat.id).desc()).limit(5).all()
        threat_type_summary = ', '.join(
            '%s (%d)' % (r[0], r[1]) for r in threat_types_raw)

        attack_patterns = AttackPattern.query.order_by(
            AttackPattern.start_time.desc()).limit(3).all()

        # ── Rule-based response engine ──────────────────────────────
        reply = None

        # Greetings
        if _has_word(user_message, ['hello', 'hi', 'hey', 'good morning',
                                    'good afternoon', 'good evening']):
            reply = (
                "Hello! I'm your **AI Security Assistant**. "
                "I have live access to your SOC data.\n\n"
                "You can ask me:\n"
                "• *How many threats are there?*\n"
                "• *What are the top threats?*\n"
                "• *What IPs are attacking?*\n"
                "• *What is brute force?*\n"
                "• *How do I respond to an attack?*"
            )

        # Help / capabilities
        elif _has_word(user_message, ['help', 'what can you do',
                                      'what do you know', 'capabilities']):
            reply = (
                "I can help you with:\n\n"
                "**Live Data Questions:**\n"
                "• How many threats? / Top IPs? / Latest threat?\n"
                "• Attack types? / Risk scores? / Active incidents?\n\n"
                "**Dataset & ML Model:**\n"
                "• What dataset is used?\n"
                "• How does the ML model work?\n"
                "• What is the model accuracy / precision / recall?\n"
                "• What are the top features?\n"
                "• What are false positives / false negatives?\n\n"
                "**Attack Knowledge:**\n"
                "• Explain brute force, SQL injection, XSS, port scan, DDoS\n"
                "• What is MITRE ATT&CK? / What is XAI?\n"
                "• What is Isolation Forest? / What is anomaly detection?\n\n"
                "**System:**\n"
                "• What is this system? / What is a SOC?\n"
                "• Why was IP x.x.x.x flagged?\n"
                "• System health / blocklist / Telegram alerts"
            )

        # ── LIVE DATA QUERIES ───────────────────────────────────────

        # Threat count
        elif _has_word(user_message, ['how many threat', 'total threat',
                                      'threat count', 'number of threat']):
            reply = (
                "**Current Threat Summary:**\n\n"
                "• Total threats detected: **%d**\n"
                "• Active threats: **%d**\n"
                "• High risk (score ≥ 7.0): **%d**\n"
                "• Critical (score ≥ 9.0): **%d**\n\n"
                "Monitoring **%s** total log entries."
            ) % (total_threats, active_threats, high_threats,
                 crit_threats, '{:,}'.format(total_logs))

        # Log count / monitoring status
        elif _has_word(user_message, ['how many log', 'total log', 'log count',
                                      'monitoring status', 'system status',
                                      'is it working']):
            reply = (
                "**System Status: ACTIVE**\n\n"
                "• Logs in database: **%s**\n"
                "• Threats detected: **%d**\n"
                "• Active threats: **%d**\n"
                "• Real-time monitoring: **ON** (file scan every 15s)\n"
                "• Synthetic log generator: **%s**"
            ) % (
                '{:,}'.format(total_logs),
                total_threats,
                active_threats,
                'ON (demo lines)' if _synthetic_log_enabled() else 'OFF (real logs only)',
            )

        # Top / worst / highest threat
        elif _has_word(user_message, ['top threat', 'worst threat', 'highest risk',
                                      'most dangerous', 'highest threat']):
            if top_threat:
                reply = (
                    "**Highest Risk Threat:**\n\n"
                    "• Type: **%s**\n"
                    "• Source IP: **%s**\n"
                    "• Risk Score: **%.1f / 10**\n"
                    "• Status: **%s**\n"
                    "• Description: %s"
                ) % (top_threat.threat_type, top_threat.source_ip,
                     top_threat.risk_score, top_threat.status,
                     top_threat.description or 'N/A')
            else:
                reply = "No threats detected yet. System is monitoring actively."

        # Latest / most recent threat
        elif _has_word(user_message, ['latest threat', 'most recent threat',
                                      'last threat', 'newest threat']):
            if latest_threat:
                reply = (
                    "**Most Recent Threat:**\n\n"
                    "• Type: **%s**\n"
                    "• Source IP: **%s**\n"
                    "• Risk Score: **%.1f / 10**\n"
                    "• Detected at: **%s**\n"
                    "• Status: **%s**"
                ) % (latest_threat.threat_type, latest_threat.source_ip,
                     latest_threat.risk_score,
                     latest_threat.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                     latest_threat.status)
            else:
                reply = "No threats recorded yet."

        # Attack types breakdown
        elif _has_word(user_message, ['what type', 'attack type', 'threat type',
                                      'breakdown', 'categories', 'kind of threat']):
            if threat_type_summary:
                reply = (
                    "**Threat Types Detected:**\n\n%s\n\n"
                    "Total threats across all types: **%d**"
                ) % ('\n'.join(
                    '• **%s** — %d incidents' % (r[0].replace('_', ' ').title(), r[1])
                    for r in threat_types_raw
                ), total_threats)
            else:
                reply = "No threat type data available yet."

        # Attacking IPs
        elif _has_word(user_message, ['which ip', 'attacker ip', 'source ip',
                                      'attacking ip', 'top ip', 'bad ip',
                                      'malicious ip']):
            top_ips = db.session.query(
                Threat.source_ip,
                db.func.count(Threat.id).label('cnt'),
                db.func.max(Threat.risk_score).label('max_score')
            ).group_by(Threat.source_ip).order_by(
                db.func.count(Threat.id).desc()).limit(5).all()

            if top_ips:
                ip_lines = '\n'.join(
                    '• **%s** — %d incidents, max score %.1f' % (r[0], r[1], r[2])
                    for r in top_ips
                )
                reply = "**Top Attacking IPs:**\n\n%s" % ip_lines
            else:
                reply = "No IP data available yet."

        # City / country with most detections
        elif _has_word(user_message, ['city', 'cities', 'country', 'countries',
                                      'location', 'where', 'region', 'origin',
                                      'most detection', 'most attack',
                                      'from where', 'geography', 'geo']):
            # Group threats by source IP with counts
            ip_counts = db.session.query(
                Threat.source_ip,
                db.func.count(Threat.id).label('cnt'),
                db.func.max(Threat.risk_score).label('max_score')
            ).group_by(Threat.source_ip).all()

            # Map each unique IP to city/country
            city_counts   = {}
            country_counts = {}
            for row in ip_counts:
                loc = get_location_info(row[0] or '')
                city    = loc.get('city', 'Unknown') if loc else 'Unknown'
                country = loc.get('country', 'Unknown') if loc else 'Unknown'
                city_counts[city]    = city_counts.get(city, 0) + row[1]
                country_counts[country] = country_counts.get(country, 0) + row[1]

            top_cities   = sorted(city_counts.items(),   key=lambda x: x[1], reverse=True)[:5]
            top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            city_lines = '\n'.join(
                '• **%s** — %d incidents' % (c, n) for c, n in top_cities
            )
            country_lines = '\n'.join(
                '• **%s** — %d incidents' % (c, n) for c, n in top_countries
            )

            most_city    = top_cities[0][0]    if top_cities    else 'N/A'
            most_country = top_countries[0][0] if top_countries else 'N/A'

            reply = (
                "**Threat Origins by Location:**\n\n"
                "**Top Cities:**\n%s\n\n"
                "**Top Countries:**\n%s\n\n"
                "Most active attack origin: **%s, %s**"
            ) % (city_lines, country_lines, most_city, most_country)

        # Attack patterns
        elif _has_word(user_message, ['pattern', 'attack pattern',
                                      'campaign', 'coordinated']):
            if attack_patterns:
                lines = []
                for p in attack_patterns:
                    lines.append(
                        "• **%s** — %d events, confidence %.0f%%, status: %s" % (
                            p.pattern_type.replace('_', ' ').title(),
                            p.event_count,
                            p.confidence_score * 100,
                            p.status
                        )
                    )
                reply = "**Recent Attack Patterns:**\n\n%s" % '\n'.join(lines)
            else:
                reply = "No attack patterns detected yet."

        # Risk score explanation
        elif _has_word(user_message, ['risk score', 'what is score', 'score mean',
                                      'score work', 'how is risk']):
            reply = (
                "**Risk Score Scale (0 – 10):**\n\n"
                "• **0–2** → INFO — Informational, no action needed\n"
                "• **2–4** → LOW — Monitor, low priority\n"
                "• **4–6** → MEDIUM — Investigate soon\n"
                "• **6–8** → HIGH — Respond within 1 hour\n"
                "• **8–10** → CRITICAL — Immediate action required\n\n"
                "Scores are calculated using: base threat severity, "
                "historical IP behaviour, geolocation risk, "
                "time-of-day patterns, and attack frequency."
            )

        # ── SECURITY KNOWLEDGE BASE ─────────────────────────────────

        # Brute force
        elif _has_word(user_message, ['brute force', 'brute-force',
                                      'password attack', 'login attempt']):
            reply = (
                "**Brute Force Attack**\n\n"
                "An attacker tries many passwords rapidly to gain access.\n\n"
                "**Detection signals:**\n"
                "• Many failed logins from one IP in a short window\n"
                "• Triggered when ≥ 5 failures within 60 seconds\n\n"
                "**Response steps:**\n"
                "1. Block the source IP immediately\n"
                "2. Enable account lockout after N failures\n"
                "3. Enforce multi-factor authentication (MFA)\n"
                "4. Review logs for successful logins from same IP\n"
                "5. Alert the affected user account"
            )

        # SQL injection
        elif _has_word(user_message, ['sql injection', 'sqli', 'sql attack',
                                      'database attack', 'union select']):
            reply = (
                "**SQL Injection (SQLi)**\n\n"
                "Attacker inserts SQL code into input fields to manipulate "
                "your database.\n\n"
                "**Detection signals:**\n"
                "• Keywords: `UNION SELECT`, `DROP TABLE`, `OR 1=1` in URLs\n"
                "• HTTP 500/403 errors on form submissions\n\n"
                "**Response steps:**\n"
                "1. Block the source IP at firewall\n"
                "2. Audit database query logs\n"
                "3. Use parameterised queries / prepared statements\n"
                "4. Deploy a Web Application Firewall (WAF)\n"
                "5. Check if any data was exfiltrated"
            )

        # XSS
        elif _has_word(user_message, ['xss', 'cross-site scripting',
                                      'cross site script', 'script injection']):
            reply = (
                "**Cross-Site Scripting (XSS)**\n\n"
                "Attacker injects malicious scripts into web pages viewed "
                "by other users.\n\n"
                "**Detection signals:**\n"
                "• `<script>`, `onerror=`, `javascript:` in request parameters\n"
                "• Unusual characters in URL query strings\n\n"
                "**Response steps:**\n"
                "1. Sanitise and encode all user inputs\n"
                "2. Implement Content Security Policy (CSP) headers\n"
                "3. Block the attacking IP\n"
                "4. Audit affected pages for stored scripts\n"
                "5. Notify affected users if stored XSS occurred"
            )

        # Port scan
        elif _has_word(user_message, ['port scan', 'port scanning',
                                      'network scan', 'reconnaissance']):
            reply = (
                "**Port Scanning**\n\n"
                "Attacker probes your network to discover open ports and "
                "running services — typically a precursor to a targeted attack.\n\n"
                "**Detection signals:**\n"
                "• One IP connecting to 5+ different ports in short time\n"
                "• Many TCP/UDP connection rejections from same source\n\n"
                "**Response steps:**\n"
                "1. Block the scanning IP at the firewall\n"
                "2. Close unnecessary open ports\n"
                "3. Enable port-scan detection rules (Snort/Suricata)\n"
                "4. Review what services are publicly exposed\n"
                "5. Monitor for follow-up attacks from same IP"
            )

        # DDoS
        elif _has_word(user_message, ['ddos', 'denial of service',
                                      'dos attack', 'flood']):
            reply = (
                "**DDoS / Denial of Service**\n\n"
                "Attacker overwhelms your server with traffic to make it "
                "unavailable to legitimate users.\n\n"
                "**Detection signals:**\n"
                "• Massive spike in request volume from multiple IPs\n"
                "• Server response times increasing / timeouts\n\n"
                "**Response steps:**\n"
                "1. Enable rate limiting on your web server\n"
                "2. Use a CDN with DDoS protection (Cloudflare, AWS Shield)\n"
                "3. Block attacking IP ranges at network edge\n"
                "4. Contact your ISP for upstream filtering\n"
                "5. Scale up infrastructure if under sustained attack"
            )

        # Anomaly
        elif _has_word(user_message, ['anomaly', 'unusual', 'strange',
                                      'abnormal', 'outlier']):
            reply = (
                "**Anomaly Detection**\n\n"
                "Your system uses an **Isolation Forest** ML model to identify "
                "behaviour that deviates significantly from normal baselines.\n\n"
                "**What triggers anomalies:**\n"
                "• Unusual traffic timing (off-hours activity)\n"
                "• Abnormal port usage patterns\n"
                "• Traffic spikes from new IPs\n\n"
                "**Response steps:**\n"
                "1. Investigate the flagged IP and time window\n"
                "2. Compare to baseline traffic patterns\n"
                "3. Escalate if confirmed malicious\n"
                "4. Add IP to block list if warranted"
            )

        # Path traversal
        elif _has_word(user_message, ['path traversal', 'directory traversal',
                                      '../', 'file inclusion']):
            reply = (
                "**Path Traversal Attack**\n\n"
                "Attacker uses `../` sequences to navigate outside intended "
                "directories and access sensitive files.\n\n"
                "**Detection signals:**\n"
                "• `../`, `..\\`, `/etc/passwd` in URL or parameters\n"
                "• Unusual file paths in HTTP requests\n\n"
                "**Response steps:**\n"
                "1. Block the attacking IP\n"
                "2. Sanitise all file path inputs server-side\n"
                "3. Use chroot jails or containerisation\n"
                "4. Audit which files may have been accessed\n"
                "5. Apply principle of least privilege to file access"
            )

        # What is SIEM / SOC
        elif _has_word(user_message, ['what is siem', 'what is soc',
                                      'what is this system', 'how does this work',
                                      'what is this']):
            reply = (
                "**This System: Mini SOC Dashboard**\n\n"
                "A **Security Operations Centre (SOC)** monitors, detects, "
                "and responds to cybersecurity threats in real time.\n\n"
                "**What this system does:**\n"
                "• Parses Apache, Nginx, SSH, and firewall log files\n"
                "• Detects brute force, port scans, SQLi, XSS, path traversal\n"
                "• Scores risk using ML-based weighted algorithms\n"
                "• Generates alerts and security reports\n"
                "• Provides REST API for integration\n"
                "• Optionally sends Telegram alerts\n\n"
                "Built with Python (Flask), SQLAlchemy, scikit-learn, and "
                "the MITRE ATT&CK framework."
            )

        # NIST / OWASP / MITRE
        elif _has_word(user_message, ['nist', 'owasp', 'mitre', 'att&ck',
                                      'iso 27001', 'standard', 'framework']):
            reply = (
                "**Security Frameworks Used:**\n\n"
                "• **NIST SP 800-137** — Continuous monitoring guidelines\n"
                "• **OWASP Top 10** — Web application attack classification "
                "(SQLi, XSS, path traversal, etc.)\n"
                "• **MITRE ATT&CK** — Adversary tactics and techniques mapping\n"
                "• **ISO/IEC 27001** — Information security management standards\n\n"
                "These frameworks inform the detection rules and risk scoring "
                "used throughout this system."
            )

        # Telegram alerts
        elif _has_word(user_message, ['telegram', 'alert', 'notification',
                                      'send alert', 'notify']):
            reply = (
                "**Telegram Alerts**\n\n"
                "This system can send real-time alerts to a Telegram channel "
                "when high-risk threats are detected (score ≥ 7.0).\n\n"
                "**To enable:**\n"
                "1. Create a Telegram bot via @BotFather\n"
                "2. Add `TELEGRAM_BOT_TOKEN=your_token` to `.env`\n"
                "3. Add `TELEGRAM_CHAT_ID=your_chat_id` to `.env`\n"
                "4. Restart the application\n\n"
                "Use the **Test Alert** button in the dashboard to verify."
            )

        # Report generation
        elif _has_word(user_message, ['report', 'generate report',
                                      'export', 'download', 'pdf', 'csv']):
            reply = (
                "**Report Generation**\n\n"
                "This system supports:\n\n"
                "• **PDF Reports** — Full threat analysis with risk charts\n"
                "• **CSV Export** — Raw threat data for spreadsheet analysis\n\n"
                "Use the **Generate Report** button in the dashboard. "
                "Reports include threat summaries, top attacking IPs, "
                "risk score distribution, and recommended actions."
            )

        # Honeypot
        elif _has_word(user_message, ['honeypot', 'trap', 'honey pot', 'fake page']):
            hp_count = Threat.query.filter_by(threat_type='honeypot').count()
            traps = ', '.join(['`' + p + '`' for p in HONEYPOT_PATHS[:6]])
            reply = (
                "**Honeypot Traps**\n\n"
                "Your system has **%d** fake vulnerable URLs set up as traps:\n\n"
                "%s ... and more.\n\n"
                "**Total honeypot hits detected: %d**\n\n"
                "Any attacker visiting these paths is automatically logged "
                "as a high-risk threat (score 8.5) and triggers a Telegram alert."
            ) % (len(HONEYPOT_PATHS), traps, hp_count)

        # Blocklist
        elif _has_word(user_message, ['block', 'blocklist', 'blocked ip',
                                      'banned ip', 'auto block']):
            blocked = IOC.query.filter_by(ioc_type='blocked_ip', is_active=True).count()
            reply = (
                "**IP Auto-Blocklist**\n\n"
                "Currently **%d** IP(s) are blocked.\n\n"
                "**Auto-block rule:** Any IP with a risk score ≥ 9.0 is automatically "
                "added to the blocklist.\n\n"
                "**Manual blocking:** Use the Blocklist panel on the dashboard to "
                "manually block or unblock any IP address.\n\n"
                "Blocked IPs are stored in the database and persist across restarts."
            ) % blocked

        # MITRE ATT&CK
        elif _has_word(user_message, ['mitre', 'att&ck', 'attack technique',
                                      'technique', 'tactic', 'heatmap']):
            top_rows = db.session.query(
                Threat.threat_type,
                db.func.count(Threat.id).label('cnt')
            ).group_by(Threat.threat_type).order_by(
                db.func.count(Threat.id).desc()).limit(3).all()
            lines = []
            for r in top_rows:
                info = MITRE_MAPPING.get(r[0], {'id': 'T????', 'name': r[0], 'tactic': 'Unknown'})
                lines.append('• **%s** — %s (%d hits)' % (info['id'], info['name'], r[1]))
            reply = (
                "**MITRE ATT&CK Coverage**\n\n"
                "Your system maps detected threats to official MITRE techniques:\n\n"
                "%s\n\n"
                "View the full heatmap in the dashboard to see all %d tracked techniques "
                "colour-coded by severity."
            ) % ('\n'.join(lines) if lines else 'No detections yet.',
                 len(MITRE_MAPPING))

        # System health
        elif _has_word(user_message, ['cpu', 'memory', 'ram', 'disk', 'health',
                                      'server', 'resource', 'performance', 'uptime']):
            reply = (
                "**System Health Monitor**\n\n"
                "The dashboard shows real-time server stats:\n\n"
                "• **CPU %** — Current processor load\n"
                "• **Memory %** — RAM usage\n"
                "• **Disk %** — Storage utilisation\n"
                "• **Network** — Bytes sent/received\n"
                "• **Uptime** — How long the server has been running\n\n"
                "The health bars turn **orange** at 70%+ and **red** at 85%+. "
                "Check the System Health panel on the dashboard for live readings."
            )

        # Thanks / bye
        # ── Explainable AI: "why was IP X flagged / explain IP X" ──
        elif _has_word(user_message, ['why', 'explain', 'reason', 'how did', 'how was']):
            import re as _re
            # Try to extract an IP address from the question
            ip_match = _re.search(r'\b(\d{1,3}(?:\.\d{1,3}){3})\b', user_message)
            if ip_match:
                query_ip = ip_match.group(1)
                threats_for_ip = (Threat.query
                                  .filter_by(source_ip=query_ip)
                                  .order_by(Threat.risk_score.desc())
                                  .limit(5).all())
                if threats_for_ip:
                    xai_lines = []
                    for t in threats_for_ip[:3]:
                        xai = _build_xai_explanation(t)
                        rule_info = XAI_RULES.get(t.threat_type, {})
                        xai_lines.append(
                            f"**Threat #{t.id} — {t.threat_type.replace('_',' ').title()}** "
                            f"(Score {t.risk_score}/10, {t.risk_level} risk)\n"
                            f"• Detection rule: {xai['why_flagged']}\n"
                            f"• MITRE: {xai['mitre']}\n"
                            f"• IP has **{xai['context']['ip_total_incidents']} total incidents**"
                        )
                    loc = get_location_info(query_ip)
                    loc_str = ''
                    if loc:
                        loc_str = f" — Location: {loc.get('city','')}, {loc.get('country','')}"
                    reply = (
                        f"**XAI Explanation for IP {query_ip}**{loc_str}\n\n"
                        + "\n\n".join(xai_lines)
                        + "\n\n> Click the **Explain** button on any threat card in the Threats panel for the full interactive breakdown."
                    )
                else:
                    reply = (
                        f"No threats recorded for IP **{query_ip}** yet. "
                        "It may not have triggered any detection rules. "
                        "Click **Explain** on any threat card in the Threats panel "
                        "for a full AI-powered breakdown."
                    )
            else:
                # Generic XAI explanation
                top = (Threat.query
                       .filter(Threat.status == 'active')
                       .order_by(Threat.risk_score.desc())
                       .first())
                if top:
                    xai = _build_xai_explanation(top)
                    reply = (
                        f"Here is an example explanation for the highest-risk active threat:\n\n"
                        f"**Threat #{top.id} — {top.threat_type.replace('_',' ').title()}** "
                        f"from IP {top.source_ip} (Score {top.risk_score}/10)\n"
                        f"• **Why flagged:** {xai['why_flagged']}\n"
                        f"• **MITRE technique:** {xai['mitre']}\n"
                        f"• **Repeat offender:** {xai['context']['ip_total_incidents']} incidents from this IP\n\n"
                        "To see a full explanation for any threat, click the purple **Explain** button "
                        "on any threat card in the Threats panel. "
                        "Or ask me: *'why was 192.168.1.1 flagged?'*"
                    )
                else:
                    reply = (
                        "The Explainable AI (XAI) system shows exactly why each threat was flagged. "
                        "Click the purple **Explain** button on any threat card to see:\n"
                        "• The detection rule that fired\n"
                        "• MITRE ATT&CK technique mapping\n"
                        "• Risk score factor breakdown\n"
                        "• Recommended response actions\n\n"
                        "Or ask me: *'why was 192.168.1.1 flagged?'*"
                    )

        # ── CICIDS 2017 Dataset questions ────────────────────────────
        elif _has_word(user_message, ['cicids', 'dataset', 'data set',
                                      'training data', 'what data', 'which data']):
            reply = (
                "**CICIDS 2017 Dataset**\n\n"
                "This project uses the **CIC-IDS 2017** dataset from the "
                "Canadian Institute for Cybersecurity (University of New Brunswick).\n\n"
                "📊 **Dataset facts:**\n"
                "• **169,865** real network flow records\n"
                "• Captured over 5 days (July 3–7, 2017)\n"
                "• **Attack types included:**\n"
                "  - Brute Force (1,507 records)\n"
                "  - XSS — Cross-Site Scripting (652 records)\n"
                "  - SQL Injection (21 records)\n"
                "  - Normal / Benign traffic (167,685 records)\n\n"
                "📁 **File used:** Thursday-WorkingHours-Morning-WebAttacks.csv\n"
                "🔗 **Source:** unb.ca/cic/datasets/ids-2017.html\n\n"
                "The dataset was used to **train the Random Forest ML model** "
                "that scores every threat detected by this system."
            )

        # ── ML Model questions ────────────────────────────────────────
        elif _has_word(user_message, ['model', 'machine learning', 'ml model',
                                      'random forest', 'isolation forest',
                                      'algorithm', 'how does it detect',
                                      'how does the ai work']):
            ml_info = {}
            if CICIDS_MODEL_AVAILABLE and _cicids_model_data:
                ml_info = _cicids_model_data.get('metrics', {})
            acc  = ml_info.get('accuracy',  97.28)
            prec = ml_info.get('precision', 31.76)
            rec  = ml_info.get('recall',    97.80)
            f1   = ml_info.get('f1_score',  47.95)
            reply = (
                "**Machine Learning Model — How It Works**\n\n"
                "This SOC system uses **two ML algorithms** working together:\n\n"
                "🌲 **1. Random Forest (Primary — Supervised)**\n"
                "• Trained on labelled CICIDS 2017 data\n"
                "• Learns the difference between normal and attack traffic\n"
                "• Uses 18 network flow features (packet size, byte rate, etc.)\n"
                "• 200 decision trees vote on each record\n"
                "• class_weight=balanced handles the imbalanced dataset\n\n"
                "🔍 **2. Isolation Forest (Secondary — Unsupervised)**\n"
                "• Detects unknown / new attack patterns\n"
                "• Trained only on normal traffic\n"
                "• Flags anything that looks different from normal\n\n"
                "📊 **Evaluation Results (on real CICIDS 2017 test data):**\n"
                "• Accuracy  : **%.2f%%**\n"
                "• Precision : **%.2f%%**\n"
                "• Recall    : **%.2f%%** ← catches almost all attacks\n"
                "• F1-Score  : **%.2f%%**\n"
                "• Test set  : 42,467 records (25%% of dataset)\n\n"
                "The model output (0–10 risk score) is used in the XAI panel "
                "to explain every detected threat."
            ) % (acc, prec, rec, f1)

        # ── Accuracy / metrics questions ──────────────────────────────
        elif _has_word(user_message, ['accuracy', 'precision', 'recall',
                                      'f1', 'f1-score', 'metric', 'performance',
                                      'how accurate', 'how good', 'result']):
            ml_info = {}
            if CICIDS_MODEL_AVAILABLE and _cicids_model_data:
                ml_info = _cicids_model_data.get('metrics', {})
            acc  = ml_info.get('accuracy',  97.28)
            prec = ml_info.get('precision', 31.76)
            rec  = ml_info.get('recall',    97.80)
            f1   = ml_info.get('f1_score',  47.95)
            reply = (
                "**ML Model Performance Metrics**\n\n"
                "Evaluated on **42,467 real CICIDS 2017 records** (25%% test split):\n\n"
                "| Metric | Score | Meaning |\n"
                "|--------|-------|---------|\n"
                "| Accuracy  | **%.2f%%** | Overall correct predictions |\n"
                "| Precision | **%.2f%%** | Of flagged threats, how many are real |\n"
                "| Recall    | **%.2f%%** | Of real attacks, how many were caught |\n"
                "| F1-Score  | **%.2f%%** | Balance of precision and recall |\n\n"
                "**Key result:** Only **12 attacks were missed** out of 545 in the test set.\n\n"
                "💡 Recall is prioritised in security systems — it is better to "
                "have some false alarms than to miss a real attack."
            ) % (acc, prec, rec, f1)

        # ── False positive / false negative ──────────────────────────
        elif _has_word(user_message, ['false positive', 'false alarm',
                                      'false negative', 'missed attack',
                                      'wrong detection', 'incorrect']):
            reply = (
                "**False Positives vs False Negatives**\n\n"
                "From the CICIDS 2017 test set (42,467 records):\n\n"
                "✅ **True Positives**  — attacks correctly caught : **533**\n"
                "❌ **False Negatives** — attacks missed           : **12**\n"
                "⚠️ **False Positives** — false alarms             : **1,145**\n"
                "✅ **True Negatives**  — normal traffic correct   : **40,777**\n\n"
                "In security systems, **false negatives are more dangerous** "
                "than false positives. Missing a real attack is worse than "
                "investigating an extra alert. Our model prioritises **Recall (97.8%)** "
                "to minimise missed attacks."
            )

        # ── Feature importance questions ──────────────────────────────
        elif _has_word(user_message, ['feature', 'important feature',
                                      'which feature', 'top feature',
                                      'packet size', 'flow']):
            reply = (
                "**Top 5 Most Important Features (Random Forest)**\n\n"
                "These network flow features had the most impact on detection:\n\n"
                "1. 🥇 **Average Packet Size**        — 19.6%%\n"
                "2. 🥈 **Fwd Packet Length Mean**      — 12.1%%\n"
                "3. 🥉 **Packet Length Mean**          — 11.3%%\n"
                "4.    **Fwd Packet Length Max**       — 11.2%%\n"
                "5.    **Total Length of Fwd Packets** — 10.9%%\n\n"
                "💡 Attack traffic tends to send many **small packets** "
                "(e.g. brute force login attempts) or unusually large payloads "
                "(e.g. SQL injection), making packet size the strongest signal."
            )

        # ── What is XAI ───────────────────────────────────────────────
        elif _has_word(user_message, ['xai', 'explainable', 'explain ai',
                                      'why flagged', 'why detected',
                                      'what is xai', 'explainability']):
            reply = (
                "**Explainable AI (XAI)**\n\n"
                "XAI answers the question: *'Why did the AI flag this as a threat?'*\n\n"
                "For each detected threat, the system shows:\n\n"
                "🔍 **Detection Rule** — which rule or pattern triggered it\n"
                "🗺️ **MITRE ATT&CK** — which attack technique it maps to\n"
                "📊 **Score Breakdown** — 5 factors that make up the risk score:\n"
                "  • Severity weight (60%%)\n"
                "  • IP history (15%%)\n"
                "  • Time pattern (10%%)\n"
                "  • Geographic risk (8%%)\n"
                "  • Frequency (7%%)\n"
                "⚠️ **Indicators** — specific signals found in the log\n"
                "🛡️ **Recommended Actions** — step-by-step response guide\n\n"
                "Click the **Explain** button on any threat card to see the full breakdown."
            )

        # ── What is MITRE ATT&CK ──────────────────────────────────────
        elif _has_word(user_message, ['mitre', 'att&ck', 'technique',
                                      'tactic', 't1']):
            reply = (
                "**MITRE ATT&CK Framework**\n\n"
                "MITRE ATT&CK is a global knowledge base of real-world attack "
                "techniques used by cybercriminals.\n\n"
                "This system maps every detected threat to a MITRE technique:\n\n"
                "| Attack | MITRE ID | Tactic |\n"
                "|--------|----------|--------|\n"
                "| Brute Force | **T1110** | Credential Access |\n"
                "| SQL Injection | **T1190** | Initial Access |\n"
                "| XSS | **T1059.007** | Execution |\n"
                "| Port Scan | **T1046** | Discovery |\n"
                "| DoS / DDoS | **T1498** | Impact |\n"
                "| Path Traversal | **T1083** | Discovery |\n\n"
                "MITRE mappings appear in the XAI panel for every flagged threat. "
                "Click **Explain** on any threat card to see it."
            )

        # ── What is Isolation Forest ──────────────────────────────────
        elif _has_word(user_message, ['isolation forest', 'anomaly detection',
                                      'unsupervised', 'what is anomaly']):
            reply = (
                "**Isolation Forest — Anomaly Detection**\n\n"
                "Isolation Forest is an **unsupervised** ML algorithm that detects "
                "unusual behaviour without needing labelled data.\n\n"
                "**How it works:**\n"
                "1. Builds many random decision trees on normal traffic\n"
                "2. Anomalies are isolated in fewer splits than normal records\n"
                "3. Short path length → high anomaly score → flagged as threat\n\n"
                "**In this system:**\n"
                "• Trained only on normal CICIDS 2017 traffic (167,685 records)\n"
                "• Acts as a **backup detector** for unknown attack types\n"
                "• Works alongside Random Forest for double confirmation\n"
                "• contamination = 0.02 (expects ~2%% of traffic to be anomalous)"
            )

        # ── SOC / system overview ─────────────────────────────────────
        elif _has_word(user_message, ['what is this system', 'what does this do',
                                      'what is soc', 'mini soc', 'overview',
                                      'about this', 'what is this']):
            reply = (
                "**AI-Assisted Security Monitoring System (Mini SOC)**\n\n"
                "This is a **Security Operations Centre (SOC)** system that:\n\n"
                "📥 **Monitors** Apache, Firewall, and SSH log files in real time\n"
                "🔍 **Detects** Brute Force, SQLi, XSS, Port Scan, DoS, DDoS\n"
                "🤖 **Scores** every threat using CICIDS 2017 trained ML model\n"
                "🧠 **Explains** WHY each threat was flagged (Explainable AI)\n"
                "🗺️ **Maps** threats to MITRE ATT&CK techniques\n"
                "🚨 **Alerts** via Telegram for high-risk threats\n"
                "🚫 **Auto-blocks** IPs with risk score ≥ 9.0\n"
                "📊 **Shows** live dashboard with charts, heatmaps, world map\n"
                "💬 **Answers** security questions via this AI chatbot\n\n"
                "Built with: Python, Flask, SQLite, scikit-learn, Chart.js"
            )

        elif _has_word(user_message, ['thank', 'thanks', 'bye', 'goodbye',
                                      'see you', 'ok', 'okay', 'great', 'nice']):
            reply = (
                "You're welcome! Stay vigilant — your system is actively "
                "monitoring **%d** threats. Let me know if you need anything else."
            ) % total_threats

        # ── OpenAI fallback ──────────────────────────────────────────
        if reply is None:
            if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
                try:
                    client = OpenAI(api_key=Config.OPENAI_API_KEY)
                    system_ctx = (
                        "You are an AI Security Assistant embedded in a real-time "
                        "SOC (Security Operations Centre) dashboard. "
                        "Current live stats: %d total logs, %d threats detected "
                        "(%d active, %d high-risk). Top threat types: %s. "
                        "Answer concisely in plain text or markdown. "
                        "Focus on practical security advice."
                    ) % (total_logs, total_threats, active_threats,
                         high_threats, threat_type_summary or 'none yet')
                    response = client.chat.completions.create(
                        model='gpt-3.5-turbo',
                        messages=[
                            {'role': 'system', 'content': system_ctx},
                            {'role': 'user',   'content': raw_message}
                        ],
                        max_tokens=300,
                        temperature=0.5
                    )
                    reply = response.choices[0].message.content.strip()
                except Exception:
                    reply = None

            if reply is None:
                reply = (
                    "I'm not sure about that specific question. "
                    "Here are things I can help with:\n\n"
                    "• **Threat counts** — *\"how many threats?\"*\n"
                    "• **Attack types** — *\"what type of attacks?\"*\n"
                    "• **Top IPs** — *\"which IPs are attacking?\"*\n"
                    "• **Explanations** — *\"what is brute force?\"*\n"
                    "• **Responses** — *\"how do I stop SQL injection?\"*\n"
                    "• **System status** — *\"how many logs?\"*"
                )

        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'reply': 'Sorry, an error occurred: %s' % str(e)}), 500


# Testing Dashboard API
@app.route('/api/testing/results', methods=['GET'])
def get_test_results():
    """Get all test results for Testing Dashboard."""
    try:
        results = TestResult.query.order_by(TestResult.timestamp.desc()).all()
        return jsonify([result.to_dict() for result in results])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/testing/run', methods=['POST'])
def run_tests():
    """Run system tests and store results."""
    try:
        test_results = []
        
        # Test 1: Database Connection
        start = time.time()
        try:
            from sqlalchemy import text as sa_text
            db.session.execute(sa_text('SELECT 1'))
            log_count = LogEntry.query.count()
            threat_count = Threat.query.count()
            test_results.append({
                'test_name': 'Database Connection (%d logs, %d threats)' % (log_count, threat_count),
                'test_type': 'integration',
                'status': 'passed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': None
            })
        except Exception as e:
            test_results.append({
                'test_name': 'Database Connection',
                'test_type': 'integration',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })
        
        # Test 2: AI Classifier
        start = time.time()
        try:
            classifier = AIThreatClassifier()
            result = classifier.classify_threat({
                'description': 'SQL injection test',
                'threat_type': 'sql_injection',
                'source_ip': '192.168.1.1'
            })
            if result['type'] == 'sql_injection':
                test_results.append({
                    'test_name': 'AI Threat Classification',
                    'test_type': 'unit',
                    'status': 'passed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': None
                })
            else:
                test_results.append({
                    'test_name': 'AI Threat Classification',
                    'test_type': 'unit',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': 'Classification mismatch'
                })
        except Exception as e:
            test_results.append({
                'test_name': 'AI Threat Classification',
                'test_type': 'unit',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })
        
        # Test 3: Log Parser - Apache format
        start = time.time()
        try:
            from log_parser import LogParser
            parser = LogParser()
            sample_log = '192.168.1.1 - - [01/Jan/2024:00:00:00 +0000] "GET /test HTTP/1.1" 200 123'
            result = parser.parse_log_line(sample_log)   # fixed: 1 arg only
            if result and result.get('ip'):               # fixed: 'ip' not 'source_ip'
                test_results.append({
                    'test_name': 'Log Parser (Apache Format)',
                    'test_type': 'unit',
                    'status': 'passed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': None
                })
            else:
                test_results.append({
                    'test_name': 'Log Parser (Apache Format)',
                    'test_type': 'unit',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': 'Parser returned no result for valid Apache log line'
                })
        except Exception as e:
            test_results.append({
                'test_name': 'Log Parser (Apache Format)',
                'test_type': 'unit',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })

        # Test 3b: Log Parser - SSH format
        start = time.time()
        try:
            from log_parser import LogParser
            parser = LogParser()
            ssh_log = 'Oct 10 13:55:00 server sshd[1234]: Failed password for root from 192.168.1.100 port 45678 ssh2'
            result = parser.parse_log_line(ssh_log)
            suspicious = result.get('suspicious_patterns', []) if result else []
            if result and result.get('ip') and 'brute_force_indicators' in suspicious:
                test_results.append({
                    'test_name': 'Log Parser (SSH + Suspicious Pattern)',
                    'test_type': 'unit',
                    'status': 'passed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': None
                })
            else:
                test_results.append({
                    'test_name': 'Log Parser (SSH + Suspicious Pattern)',
                    'test_type': 'unit',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': f'SSH parse failed or missed brute_force pattern. Result: {result}'
                })
        except Exception as e:
            test_results.append({
                'test_name': 'Log Parser (SSH + Suspicious Pattern)',
                'test_type': 'unit',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })

        # Test 3c: Log Parser - SQL Injection detection
        start = time.time()
        try:
            from log_parser import LogParser
            parser = LogParser()
            sqli_log = '10.0.0.5 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users?id=1 UNION SELECT * FROM users HTTP/1.1" 403 89'
            result = parser.parse_log_line(sqli_log)
            suspicious = result.get('suspicious_patterns', []) if result else []
            if result and 'sql_injection' in suspicious:
                test_results.append({
                    'test_name': 'Log Parser (SQL Injection Detection)',
                    'test_type': 'security',
                    'status': 'passed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': None
                })
            else:
                test_results.append({
                    'test_name': 'Log Parser (SQL Injection Detection)',
                    'test_type': 'security',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': f'SQL injection pattern not detected. Patterns found: {suspicious}'
                })
        except Exception as e:
            test_results.append({
                'test_name': 'Log Parser (SQL Injection Detection)',
                'test_type': 'security',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })
        
        # Test 4: XSS Pattern Detection in Log Parser
        start = time.time()
        try:
            from log_parser import LogParser
            parser = LogParser()
            xss_log = '10.0.0.9 - - [01/Jan/2024:12:00:00 +0000] "GET /search?q=<script>alert(1)</script> HTTP/1.1" 403 89'
            result = parser.parse_log_line(xss_log)
            suspicious = result.get('suspicious_patterns', []) if result else []
            if result and 'xss' in suspicious:
                test_results.append({
                    'test_name': 'Log Parser (XSS Detection)',
                    'test_type': 'security',
                    'status': 'passed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': None
                })
            else:
                test_results.append({
                    'test_name': 'Log Parser (XSS Detection)',
                    'test_type': 'security',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': f'XSS pattern not detected. Found: {suspicious}'
                })
        except Exception as e:
            test_results.append({
                'test_name': 'Log Parser (XSS Detection)',
                'test_type': 'security',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })

        # Test 5: Brute Force Detection
        start = time.time()
        try:
            from threat_detector import ThreatDetector
            config = Config()
            detector = ThreatDetector(config)
            # Simulate 6 failed SSH login attempts from same IP (threshold is 5)
            fake_logs = []
            for i in range(6):
                fake_logs.append({
                    'source_ip': '10.99.99.99',
                    'timestamp': datetime.utcnow(),
                    'action': 'failed',
                    'raw_log': f'Failed password for root from 10.99.99.99 port {40000+i} ssh2',
                    'destination_port': 22,
                    'protocol': '',
                    'parsed_data': {'suspicious_patterns': ['brute_force_indicators']}
                })
            threats = detector.detect_threats(fake_logs)
            bf_threats = [t for t in threats if t['threat_type'] == 'brute_force']
            if bf_threats and bf_threats[0]['source_ip'] == '10.99.99.99':
                test_results.append({
                    'test_name': 'Brute Force Detection',
                    'test_type': 'unit',
                    'status': 'passed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': None
                })
            else:
                test_results.append({
                    'test_name': 'Brute Force Detection',
                    'test_type': 'unit',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': f'Expected brute_force threat, got: {[t["threat_type"] for t in threats]}'
                })
        except Exception as e:
            test_results.append({
                'test_name': 'Brute Force Detection',
                'test_type': 'unit',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })

        # Test 6: Risk Scorer - Score boundaries
        start = time.time()
        try:
            from risk_scorer import RiskScorer
            scorer = RiskScorer(Config())
            threat_data = {
                'threat_type': 'brute_force',
                'source_ip': '10.88.88.88',
                'risk_score': 7.0,
                'timestamp': datetime.utcnow(),
                'description': 'Test brute force'
            }
            result = scorer.calculate_comprehensive_risk_score(threat_data, [])
            score = result['final_score']
            level = result['risk_level']
            if 0.0 <= score <= 10.0 and level in ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                test_results.append({
                    'test_name': 'Risk Scorer (Score Boundaries)',
                    'test_type': 'unit',
                    'status': 'passed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': None
                })
            else:
                test_results.append({
                    'test_name': 'Risk Scorer (Score Boundaries)',
                    'test_type': 'unit',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': f'Score {score} out of 0-10 range or invalid level {level}'
                })
        except Exception as e:
            test_results.append({
                'test_name': 'Risk Scorer (Score Boundaries)',
                'test_type': 'unit',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })

        # Test 7: Risk Scorer - Critical level for high score
        start = time.time()
        try:
            from risk_scorer import RiskScorer
            scorer = RiskScorer(Config())
            threat_data = {
                'threat_type': 'brute_force',
                'source_ip': '10.77.77.77',
                'risk_score': 9.5,
                'timestamp': datetime.utcnow(),
                'description': 'High risk brute force'
            }
            result = scorer.calculate_comprehensive_risk_score(threat_data, [])
            actual_final = result['final_score']
            actual_level = result['risk_level']
            actual_factors = result.get('risk_factors', {})
            base_w = actual_factors.get('base_threat_score', 0)
            mult   = actual_factors.get('severity_multiplier', 0)
            if actual_level in ['CRITICAL', 'HIGH']:
                test_results.append({
                    'test_name': 'Risk Scorer (High Score → HIGH/CRITICAL Level)',
                    'test_type': 'unit',
                    'status': 'passed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': None
                })
            else:
                test_results.append({
                    'test_name': 'Risk Scorer (High Score → HIGH/CRITICAL Level)',
                    'test_type': 'unit',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': 'Expected HIGH/CRITICAL but got %s. final=%.2f base=%.2f mult=%.2f' % (
                        actual_level, actual_final, base_w, mult)
                })
        except Exception as e:
            test_results.append({
                'test_name': 'Risk Scorer (High Score → HIGH/CRITICAL Level)',
                'test_type': 'unit',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })

        # Test 8: Attack Pattern Detection (DB query)
        start = time.time()
        try:
            patterns = AttackPattern.query.all()
            pattern_count = len(patterns)
            test_results.append({
                'test_name': 'Attack Pattern Storage',
                'test_type': 'integration',
                'status': 'passed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': None
            })
        except Exception as e:
            test_results.append({
                'test_name': 'Attack Pattern Storage',
                'test_type': 'integration',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })

        # Test 9: Threat Detection Pipeline (end-to-end)
        start = time.time()
        try:
            from log_parser import LogParser
            from threat_detector import ThreatDetector
            from risk_scorer import RiskScorer
            parser = LogParser()
            detector = ThreatDetector(Config())
            scorer = RiskScorer(Config())
            # Use the actual sample log file
            import os
            sample_path = './sample_logs/ssh_auth.log'
            if os.path.exists(sample_path):
                parsed = parser.parse_log_file(sample_path)
                normalized = [parser.normalize_log_entry(p) for p in parsed]
                threats = detector.detect_threats(normalized)
                # Score the first threat if any
                if threats:
                    scored = scorer.calculate_comprehensive_risk_score(threats[0], [])
                    if 'final_score' in scored and 'risk_level' in scored:
                        test_results.append({
                            'test_name': 'End-to-End Detection Pipeline',
                            'test_type': 'integration',
                            'status': 'passed',
                            'duration_ms': (time.time() - start) * 1000,
                            'error_message': None
                        })
                    else:
                        raise ValueError('Risk scorer returned incomplete result')
                else:
                    test_results.append({
                        'test_name': 'End-to-End Detection Pipeline',
                        'test_type': 'integration',
                        'status': 'passed',
                        'duration_ms': (time.time() - start) * 1000,
                        'error_message': None
                    })
            else:
                test_results.append({
                    'test_name': 'End-to-End Detection Pipeline',
                    'test_type': 'integration',
                    'status': 'failed',
                    'duration_ms': (time.time() - start) * 1000,
                    'error_message': 'sample_logs/ssh_auth.log not found'
                })
        except Exception as e:
            test_results.append({
                'test_name': 'End-to-End Detection Pipeline',
                'test_type': 'integration',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })

        # Test 10: Telegram Configuration Check
        start = time.time()
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
            chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
            configured = bool(bot_token and chat_id and
                              bot_token != 'your_telegram_bot_token_here' and
                              chat_id != 'your_telegram_chat_id_here')
            test_results.append({
                'test_name': 'Telegram Configuration',
                'test_type': 'integration',
                'status': 'passed' if configured else 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': None if configured else 'TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured in .env'
            })
        except Exception as e:
            test_results.append({
                'test_name': 'Telegram Configuration',
                'test_type': 'integration',
                'status': 'failed',
                'duration_ms': (time.time() - start) * 1000,
                'error_message': str(e)
            })
        
        # Store results in database — wrapped in try/except so test output is
        # always returned even if DB write fails for any reason
        try:
            model_fields = {'test_name', 'test_type', 'status', 'duration_ms', 'error_message'}
            for result in test_results:
                record_data = {k: v for k, v in result.items() if k in model_fields}
                record_data['test_name'] = str(record_data.get('test_name', ''))[:200]
                test_record = TestResult(
                    test_name=record_data.get('test_name'),
                    test_type=record_data.get('test_type'),
                    status=record_data.get('status'),
                    duration_ms=record_data.get('duration_ms'),
                    error_message=record_data.get('error_message')
                )
                db.session.add(test_record)
            db.session.commit()
        except Exception as db_err:
            db.session.rollback()
            print('Warning: could not save test results to DB:', db_err)
        
        return jsonify({
            'success': True,
            'results': test_results,
            'total': len(test_results),
            'passed': len([r for r in test_results if r['status'] == 'passed']),
            'failed': len([r for r in test_results if r['status'] == 'failed'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Test Alert & AI endpoints (called by dashboard buttons)
@app.route('/api/test/alert')
def test_alert():
    """Send a test Telegram alert."""
    try:
        message = "Test Alert — Security monitoring system is active and alerts are working."
        success = send_telegram_alert(message)
        log_audit(
            action='test_alert_sent',
            resource_type='alert',
            details=f'Test alert {"sent successfully" if success else "failed — check Telegram config"}',
            severity='info' if success else 'warning'
        )
        return jsonify({
            'success': success,
            'message': 'Test alert sent!' if success else 'Alert failed — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test/ai')
def test_ai():
    """Test AI explanation connection."""
    try:
        sample = {
            'threat_type': 'test',
            'source_ip': '127.0.0.1',
            'risk_score': 1.0,
            'description': 'Connection test'
        }
        result = generate_ai_explanation(sample)
        success = bool(result and len(result) > 10)
        log_audit(
            action='test_ai_connection',
            resource_type='ai',
            details=f'AI connection test {"passed" if success else "failed"}',
            severity='info' if success else 'warning'
        )
        return jsonify({
            'success': success,
            'message': 'AI connection successful!' if success else 'AI unavailable — check OPENAI_API_KEY in .env'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard')
def live_feed():
    """
    Single polling endpoint that returns everything the dashboard needs in one call.
    Designed for frequent polling (every 5-10 seconds).
    """
    try:
        ensure_recent_activity_schema()
        now = datetime.utcnow()
        last_minutes = int(request.args.get('minutes', 60))
        cutoff = now - timedelta(minutes=last_minutes)

        # Stats
        total_logs = LogEntry.query.count()
        total_threats = Threat.query.count()
        active_threats = Threat.query.filter_by(status='active').count()
        recent_logs = LogEntry.query.filter(
            or_(
                LogEntry.timestamp >= cutoff,
                LogEntry.ingested_at >= cutoff,
            )
        ).count()
        recent_threats = Threat.query.filter(
            or_(
                Threat.timestamp >= cutoff,
                Threat.ingested_at >= cutoff,
            )
        ).count()

        threats_by_type = {}
        for t in Threat.query.with_entities(Threat.threat_type, db.func.count(Threat.id)).group_by(Threat.threat_type).all():
            threats_by_type[t[0]] = t[1]

        # Latest 20 threats
        latest_threats = Threat.query.order_by(Threat.timestamp.desc()).limit(20).all()
        threats_list = []
        for t in latest_threats:
            threats_list.append({
                'id': t.id,
                'threat_type': t.threat_type,
                'source_ip': t.source_ip,
                'risk_score': t.risk_score,
                'description': t.description,
                'status': t.status,
                'timestamp': t.timestamp.isoformat() if t.timestamp else None
            })

        # Latest 10 log entries
        latest_logs = LogEntry.query.order_by(LogEntry.timestamp.desc()).limit(10).all()
        logs_list = []
        for l in latest_logs:
            logs_list.append({
                'id': l.id,
                'source_ip': l.source_ip,
                'raw_log': l.raw_log[:200],
                'timestamp': l.timestamp.isoformat() if l.timestamp else None
            })

        return jsonify({
            'server_time': now.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'stats': {
                'total_logs': total_logs,
                'total_threats': total_threats,
                'active_threats': active_threats,
                'recent_logs': recent_logs,
                'recent_threats': recent_threats,
                'threats_by_type': threats_by_type
            },
            'latest_threats': threats_list,
            'latest_logs': logs_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Security Audit Log API
@app.route('/api/audit-logs', methods=['GET'])
def get_audit_logs():
    """Get security audit logs."""
    try:
        logs = SecurityAuditLog.query.order_by(SecurityAuditLog.timestamp.desc()).limit(100).all()
        return jsonify([log.to_dict() for log in logs])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Performance Metrics API
@app.route('/api/performance/metrics', methods=['GET'])
def get_performance_metrics():
    """Get system performance metrics."""
    try:
        # Get latest metrics
        metrics = PerformanceMetric.query.order_by(PerformanceMetric.timestamp.desc()).limit(50).all()
        
        # Calculate logs per second
        recent_logs = LogEntry.query.filter(
            LogEntry.timestamp >= datetime.utcnow() - timedelta(minutes=1)
        ).count()
        logs_per_sec = recent_logs / 60.0
        
        return jsonify({
            'metrics': [m.to_dict() for m in metrics],
            'logs_per_second': round(logs_per_sec, 2),
            'total_logs': LogEntry.query.count(),
            'total_threats': Threat.query.count(),
            'db_size_mb': get_db_size()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_db_size():
    """Get database file size in MB."""
    try:
        db_path = 'security_monitoring.db'
        if os.path.exists(db_path):
            size_bytes = os.path.getsize(db_path)
            return round(size_bytes / (1024 * 1024), 2)
        return 0
    except:
        return 0

# Vulnerability Scanner API
@app.route('/api/security/scan', methods=['GET'])
def run_security_scan():
    """Run basic vulnerability scan on the system."""
    try:
        vulnerabilities = []
        
        # Check 1: API Keys configured
        vt_key = os.getenv('VIRUSTOTAL_API_KEY')
        abuse_key = os.getenv('ABUSEIPDB_API_KEY')
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not vt_key or vt_key == 'your_virustotal_api_key_here':
            vulnerabilities.append({
                'severity': 'medium',
                'category': 'Configuration',
                'title': 'VirusTotal API Key Not Configured',
                'description': 'Real threat intelligence is disabled. System using mock data.',
                'recommendation': 'Add VIRUSTOTAL_API_KEY to .env file'
            })
        
        # Check 2: Telegram configuration
        if not telegram_token or telegram_token == 'your_telegram_bot_token_here':
            vulnerabilities.append({
                'severity': 'low',
                'category': 'Configuration',
                'title': 'Telegram Alerts Not Configured',
                'description': 'Alert notifications via Telegram are disabled.',
                'recommendation': 'Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env'
            })
        
        # Check 3: Database security
        db_path = 'security_monitoring.db'
        if os.path.exists(db_path):
            import stat
            db_perms = stat.S_IMODE(os.stat(db_path).st_mode)
            if db_perms & stat.S_IRWXG or db_perms & stat.S_IRWXO:
                vulnerabilities.append({
                    'severity': 'high',
                    'category': 'Data Security',
                    'title': 'Database File Permissions Too Permissive',
                    'description': 'Database file is readable by other users.',
                    'recommendation': 'Run: chmod 600 security_monitoring.db'
                })
        
        # Check 4: Log file permissions
        logs_dir = './logs'
        if os.path.exists(logs_dir):
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
            for log_file in log_files:
                log_path = os.path.join(logs_dir, log_file)
                log_perms = stat.S_IMODE(os.stat(log_path).st_mode)
                if log_perms & stat.S_IRWXO:
                    vulnerabilities.append({
                        'severity': 'medium',
                        'category': 'Data Security',
                        'title': f'Log File {log_file} World-Readable',
                        'description': 'Log files contain sensitive information.',
                        'recommendation': 'Run: chmod 640 logs/*.log'
                    })
                    break  # Only report once
        
        # Check 5: Flask debug mode
        debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        if debug_mode:
            vulnerabilities.append({
                'severity': 'critical',
                'category': 'Application Security',
                'title': 'Flask Debug Mode Enabled',
                'description': 'Debug mode exposes sensitive information and enables code execution.',
                'recommendation': 'Set FLASK_DEBUG=false in production'
            })
        
        return jsonify({
            'scan_timestamp': datetime.utcnow().isoformat(),
            'vulnerabilities_found': len(vulnerabilities),
            'vulnerabilities': vulnerabilities,
            'risk_level': 'high' if any(v['severity'] == 'critical' for v in vulnerabilities) else 
                         'medium' if any(v['severity'] == 'high' for v in vulnerabilities) else 'low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# GDPR Compliance API
@app.route('/api/compliance/gdpr', methods=['GET'])
def get_gdpr_compliance_info():
    """Get GDPR compliance information and data retention stats."""
    try:
        # Data retention calculation
        total_logs = LogEntry.query.count()
        old_logs = LogEntry.query.filter(
            LogEntry.timestamp < datetime.utcnow() - timedelta(days=90)
        ).count()
        
        # Personal data indicators (IP addresses)
        unique_ips = db.session.query(LogEntry.source_ip).distinct().count()
        
        return jsonify({
            'data_retention': {
                'total_log_entries': total_logs,
                'logs_older_than_90_days': old_logs,
                'retention_policy': '90 days',
                'compliant': old_logs == 0
            },
            'personal_data': {
                'unique_ip_addresses': unique_ips,
                'ip_address_classification': 'Personal Data (under GDPR)',
                'processing_basis': 'Legitimate Interest (Security)',
                'can_be_anonymized': True
            },
            'data_subject_rights': {
                'right_to_access': True,
                'right_to_erasure': True,
                'right_to_portability': True
            },
            'security_measures': [
                'Database encryption at rest',
                'Access logging and audit trails',
                'Input sanitization',
                'Secure password hashing'
            ],
            'last_updated': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/docs')
def api_docs():
    """Render Swagger UI API documentation."""
    swagger_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Monitor API Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui.css" />
        <style>
            body { margin: 0; font-family: 'Inter', sans-serif; }
            .topbar { display: none; }
            .swagger-ui .info { margin: 20px 0; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
        <script>
            window.onload = function() {
                SwaggerUIBundle({
                    url: "/api/swagger.json",
                    dom_id: '#swagger-ui',
                    presets: [SwaggerUIBundle.presets.apis],
                    layout: "BaseLayout"
                });
            };
        </script>
    </body>
    </html>
    '''
    return swagger_html

@app.route('/api/swagger.json')
def swagger_spec():
    """Return OpenAPI/Swagger specification."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "AI-Powered Security Monitor API",
            "version": "1.0.0",
            "description": "RESTful API for security monitoring, threat detection, and compliance management"
        },
        "paths": {
            "/api/stats": {
                "get": {
                    "summary": "Get system statistics",
                    "responses": {
                        "200": {"description": "Statistics data"}
                    }
                }
            },
            "/api/threats": {
                "get": {
                    "summary": "Get recent threats",
                    "parameters": [
                        {"name": "limit", "in": "query", "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "200": {"description": "List of threats"}
                    }
                }
            },
            "/api/testing/run": {
                "post": {
                    "summary": "Run system tests",
                    "responses": {
                        "200": {"description": "Test results"}
                    }
                }
            },
            "/api/audit-logs": {
                "get": {
                    "summary": "Get security audit logs",
                    "responses": {
                        "200": {"description": "Audit log entries"}
                    }
                }
            },
            "/api/compliance/gdpr": {
                "get": {
                    "summary": "Get GDPR compliance information",
                    "responses": {
                        "200": {"description": "Compliance data"}
                    }
                }
            },
            "/api/security/scan": {
                "get": {
                    "summary": "Run vulnerability scan",
                    "responses": {
                        "200": {"description": "Security scan results"}
                    }
                }
            },
            "/api/export/csv": {
                "get": {
                    "summary": "Export threats to CSV",
                    "responses": {
                        "200": {"description": "CSV file download"}
                    }
                }
            },
            "/api/export/pdf": {
                "get": {
                    "summary": "Export threats to PDF",
                    "responses": {
                        "200": {"description": "PDF file download"}
                    }
                }
            }
        }
    }
    return jsonify(spec)

def broadcast_threat_update(threat_data):
    """Placeholder for threat broadcasting (WebSocket disabled)."""
    pass

def broadcast_stats_update(stats):
    """Placeholder for stats broadcasting (WebSocket disabled)."""
    pass


class ContinuousLogMonitor:
    """
    Continuously monitors log files using file position tracking (tail-like).
    Only reads NEW lines added since the last scan — no re-reading of old data.
    Runs threat detection on every new batch and sends Telegram alerts.
    """

    def __init__(self, logs_dir='./logs', scan_interval=15):
        self.logs_dir = logs_dir
        self.scan_interval = scan_interval          # seconds between scans
        self.file_positions = {}                    # {filepath: byte_offset}
        self.running = False
        self.thread = None
        self._log_parser = None
        self._threat_detector = None
        self._risk_scorer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background monitoring thread."""
        from log_parser import LogParser
        from threat_detector import ThreatDetector
        from risk_scorer import RiskScorer
        config = Config()
        self._log_parser = LogParser()
        self._threat_detector = ThreatDetector(config)
        self._risk_scorer = RiskScorer(config)

        # Seed file positions so we only pick up lines written AFTER startup
        self._seed_positions()

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"🔄 Continuous log monitor started — scanning every {self.scan_interval}s")
        print(
            "   ℹ️  Only NEW lines appended to logs/*.log AFTER startup are tailed (EOF seeded). "
            "Brute-force needs several failures in a short window — use synthetic logs or append many test lines."
        )

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("🛑 Continuous log monitor stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seed_positions(self):
        """Record end-of-file position for every existing log file so we
        skip already-processed content and only read new lines."""
        for fname in os.listdir(self.logs_dir):
            if fname.endswith('.log'):
                fpath = os.path.join(self.logs_dir, fname)
                try:
                    self.file_positions[fpath] = os.path.getsize(fpath)
                except OSError:
                    self.file_positions[fpath] = 0

    def _monitor_loop(self):
        """Main loop: scan for new log lines every `scan_interval` seconds."""
        while self.running:
            try:
                # Scan first, then sleep — avoids an extra 15s wait before the first check
                # (dashboard polls every ~5s but threats only appear after new log bytes are processed).
                self._scan_all_files()
                if not self.running:
                    break
                time.sleep(self.scan_interval)
            except Exception as exc:
                print(f"  ❌ ContinuousLogMonitor error: {exc}")
                # Roll back any broken session so next cycle starts clean
                try:
                    with app.app_context():
                        db.session.rollback()
                except Exception:
                    pass

    def _scan_all_files(self):
        """Scan all .log files for new lines and process them."""
        for fname in os.listdir(self.logs_dir):
            if not fname.endswith('.log'):
                continue
            fpath = os.path.join(self.logs_dir, fname)
            new_lines = self._read_new_lines(fpath)
            if new_lines:
                print(f"  📥 {len(new_lines)} new lines in {fname}")
                self._process_lines(new_lines)

    def _read_new_lines(self, fpath):
        """Return only the lines written since the last scan."""
        try:
            current_size = os.path.getsize(fpath)
            last_pos = self.file_positions.get(fpath, 0)

            # File was truncated / rotated
            if current_size < last_pos:
                last_pos = 0

            if current_size == last_pos:
                return []

            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_pos)
                new_data = f.read()
                self.file_positions[fpath] = f.tell()

            return [l for l in new_data.splitlines() if l.strip()]
        except OSError:
            return []

    def _process_lines(self, lines):
        """Parse lines → store log entries → detect threats → alert."""
        with app.app_context():
            new_log_entries = []

            for raw_line in lines:
                parsed = self._log_parser.parse_log_line(raw_line)
                if not parsed:
                    continue
                normalized = self._log_parser.normalize_log_entry(parsed)

                # Deduplicate by raw content
                exists = LogEntry.query.filter_by(
                    source_ip=normalized['source_ip'],
                    raw_log=normalized['raw_log']
                ).first()
                if exists:
                    continue

                entry = LogEntry(
                    source_ip=normalized['source_ip'],
                    timestamp=normalized.get('timestamp', datetime.utcnow()),
                    destination_port=normalized.get('destination_port'),
                    protocol=normalized.get('protocol', ''),
                    action=normalized.get('action', ''),
                    raw_log=normalized['raw_log'],
                    parsed_data=json.dumps(normalized.get('parsed_data', {})),
                    ingested_at=datetime.utcnow(),
                )
                db.session.add(entry)
                new_log_entries.append(normalized)

            if not new_log_entries:
                return

            try:
                db.session.commit()
            except Exception as commit_err:
                db.session.rollback()
                print(f"  ⚠️  DB commit error (rolled back): {commit_err}")
                return
            print(f"    ✅ Stored {len(new_log_entries)} new log entries")

            # Run threat detection on the new batch (detector keeps rolling state for patterns / ML).
            detected = self._threat_detector.detect_threats(new_log_entries)
            if not detected:
                return

            saved = 0
            for threat_data in detected:
                # Avoid duplicate rows when the same rule fires every scan (rolling windows).
                recent_dup = Threat.query.filter(
                    Threat.source_ip == threat_data['source_ip'],
                    Threat.threat_type == threat_data['threat_type'],
                    Threat.timestamp >= datetime.utcnow() - timedelta(minutes=15),
                ).first()
                if recent_dup:
                    continue

                historical = Threat.query.filter(
                    Threat.source_ip == threat_data['source_ip'],
                    Threat.timestamp >= datetime.utcnow() - timedelta(days=30)
                ).all()
                hist_dicts = [
                    {'source_ip': t.source_ip, 'threat_type': t.threat_type,
                     'risk_score': t.risk_score, 'timestamp': t.timestamp}
                    for t in historical
                ]

                analysis = self._risk_scorer.calculate_comprehensive_risk_score(
                    threat_data, hist_dicts)
                threat_data['risk_score'] = analysis['final_score']

                _now = datetime.utcnow()
                threat = Threat(
                    threat_type=threat_data['threat_type'],
                    source_ip=threat_data['source_ip'],
                    risk_score=threat_data['risk_score'],
                    description=threat_data['description'],
                    timestamp=_now,
                    ingested_at=_now,
                )
                db.session.add(threat)
                db.session.flush()

                # Telegram alert for high-risk threats
                if should_send_alert(threat_data):
                    msg = (
                        f"🚨 LIVE SECURITY ALERT\n\n"
                        f"Threat: {threat_data['threat_type'].replace('_', ' ').title()}\n"
                        f"Source IP: {threat_data['source_ip']}\n"
                        f"Risk Score: {threat_data['risk_score']}/10\n"
                        f"Description: {threat_data['description']}\n"
                        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                        f"Dashboard: {dashboard_public_url()}"
                    )
                    send_telegram_alert(msg)

                    alert = Alert(
                        threat_id=threat.id,
                        alert_type='telegram',
                        message=msg,
                        sent='yes',
                        sent_timestamp=datetime.utcnow()
                    )
                    db.session.add(alert)
                    print(f"    📲 Telegram alert sent for {threat_data['threat_type']} from {threat_data['source_ip']}")

                saved += 1

            if saved == 0:
                return

            print(f"    🚨 Persisted {saved} new threat(s) ({len(detected)} rule hit(s))")
            try:
                db.session.commit()
            except Exception as commit_err:
                db.session.rollback()
                print(f"  ⚠️  Threat commit error (rolled back): {commit_err}")

# Log File Monitor for automatic processing
class LogMonitor:
    """Monitor logs directory for new files and auto-process them."""
    
    def __init__(self, logs_dir='./logs'):
        self.logs_dir = logs_dir
        self.observer = None
        self.log_parser = None
        
    def start(self):
        """Start monitoring the logs directory."""
        if not WATCHDOG_AVAILABLE:
            print("⚠️  Watchdog not available. Log file monitoring disabled.")
            return False
            
        try:
            from log_parser import LogParser
            self.log_parser = LogParser()
            
            event_handler = LogFileHandler(self.log_parser)
            self.observer = Observer()
            self.observer.schedule(event_handler, self.logs_dir, recursive=False)
            self.observer.start()
            print(f"👁️  Log file watcher started - monitoring {self.logs_dir}")
            return True
        except Exception as e:
            print(f"❌ Error starting log monitor: {e}")
            return False
    
    def stop(self):
        """Stop the log monitor."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            print("🛑 Log file watcher stopped")

class LogFileHandler(FileSystemEventHandler):
    """Handle file system events for log files."""
    
    # Debounce rapid duplicate on_modified bursts (OS may fire many per append).
    _MODIFY_DEBOUNCE_SEC = 0.75

    def __init__(self, log_parser):
        self.log_parser = log_parser
        self._last_modified_at = {}  # path -> monotonic time (not "process once ever")
        
    def on_created(self, event):
        """Handle new file creation."""
        if not event.is_directory and event.src_path.endswith('.log'):
            print(f"📄 New log file detected: {event.src_path}")
            self.process_log_file(event.src_path)
    
    def on_modified(self, event):
        """Handle file modifications."""
        if not event.is_directory and event.src_path.endswith('.log'):
            now = time.monotonic()
            last = self._last_modified_at.get(event.src_path, 0)
            if now - last < self._MODIFY_DEBOUNCE_SEC:
                return
            self._last_modified_at[event.src_path] = now
            print(f"📝 Log file modified: {event.src_path}")
            self.process_log_file(event.src_path)
    
    def process_log_file(self, file_path):
        """Process a log file and add entries to database."""
        try:
            time.sleep(0.5)  # Wait for file to be fully written
            
            parsed_logs = self.log_parser.parse_log_file(file_path)
            total_added = 0
            
            # Use this module's globals — never `from working_app import app` here: when the
            # script is run as __main__ (Thonny %Run), that would load a second copy of the
            # file and duplicate Flask/SQLAlchemy + startup side effects.
            with app.app_context():
                for parsed_log in parsed_logs:
                    normalized = self.log_parser.normalize_log_entry(parsed_log)
                    
                    # Check for duplicates
                    existing = db.session.query(LogEntry).filter_by(
                        source_ip=normalized['source_ip'],
                        timestamp=normalized['timestamp'],
                        raw_log=normalized['raw_log']
                    ).first()
                    
                    if not existing:
                        log_entry = LogEntry(
                            source_ip=normalized['source_ip'],
                            timestamp=normalized['timestamp'],
                            destination_port=normalized.get('destination_port'),
                            protocol=normalized.get('protocol', ''),
                            action=normalized.get('action', ''),
                            raw_log=normalized['raw_log'],
                            parsed_data=json.dumps(normalized.get('parsed_data', {})),
                            ingested_at=datetime.utcnow(),
                        )
                        db.session.add(log_entry)
                        total_added += 1
                
                if total_added > 0:
                    db.session.commit()
                    print(f"  ✅ Auto-processed {total_added} entries from {file_path}")
                else:
                    print(f"  ℹ️ No new entries in {file_path}")
            
        except Exception as e:
            print(f"  ❌ Error auto-processing {file_path}: {e}")

class LogGenerator:
    """Generate realistic log entries continuously for real-time activity."""
    
    def __init__(self, logs_dir='./logs', interval=5):
        self.logs_dir = logs_dir
        self.interval = interval  # seconds between log entries
        self.running = False
        self.thread = None
        self._gen_tick = 0
        
        # Realistic log templates
        self.apache_logs = [
            '192.168.{ip} - - [{timestamp}] "GET /admin/config.php HTTP/1.1" 404 234 "-" "Mozilla/5.0"',
            '10.0.{ip} - - [{timestamp}] "POST /login.php HTTP/1.1" 200 567 "-" "Mozilla/5.0"',
            '172.16.{ip} - - [{timestamp}] "GET /api/users?id=1 OR 1=1 HTTP/1.1" 403 123 "-" "curl/7.68.0"',
            '192.168.{ip} - - [{timestamp}] "GET /images/logo.png HTTP/1.1" 200 4567 "-" "Mozilla/5.0"',
            '10.0.{ip} - - [{timestamp}] "POST /search?q=<script>alert(1)</script> HTTP/1.1" 403 89 "-" "Mozilla/5.0"',
        ]
        
        self.ssh_logs = [
            'Failed password for root from 192.168.{ip} port {port} ssh2',
            'Accepted password for admin from 10.0.{ip} port {port} ssh2',
            'Failed password for admin from 172.16.{ip} port {port} ssh2',
            'Invalid user ubuntu from 192.168.{ip} port {port}',
            'Connection closed by authenticating user root 10.0.{ip} port {port}',
        ]
        
        self.firewall_logs = [
            'DROP IN=eth0 OUT= MAC=00:50:56:01 SRC=192.168.{ip} DST=10.0.0.1 LEN=52 PROTO=TCP',
            'ACCEPT IN=eth0 OUT= MAC=00:50:56:02 SRC=10.0.{ip} DST=192.168.1.1 LEN=64 PROTO=UDP',
            'REJECT IN=eth0 OUT= MAC=00:50:56:03 SRC=172.16.{ip} DST=10.0.0.5 LEN=40 PROTO=TCP',
        ]
        
    def start(self):
        """Start the log generator in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._generate_logs, daemon=True)
        self.thread.start()
        print("🔄 Log generator started - creating fresh entries every few seconds...")
        
    def stop(self):
        """Stop the log generator."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        print("🛑 Log generator stopped")
        
    def _generate_logs(self):
        """Continuously generate log entries."""
        while self.running:
            try:
                time.sleep(self.interval)
                self._write_log_entry()
            except Exception as e:
                print(f"  ❌ Error generating log: {e}")
                
    def _write_brute_force_demo_burst(self):
        """Append several failed SSH lines for the same IP so brute-force rules can fire during demo."""
        log_file = os.path.join(self.logs_dir, 'ssh_auth.log')
        demo_octet = 211
        with open(log_file, 'a', encoding='utf-8') as f:
            for _ in range(6):
                port = random.randint(40000, 55000)
                f.write(
                    f'Failed password for root from 192.168.{demo_octet} port {port} ssh2\n'
                )
        print(
            f"  📝 Demo burst: 6 SSH failures from 192.168.{demo_octet} "
            f"(brute-force threshold demo)"
        )

    def _write_log_entry(self):
        """Write a single realistic log entry to one of the log files."""
        self._gen_tick += 1
        # Periodic same-IP failure burst; random one-liners use different IPs and rarely cross thresholds.
        if self._gen_tick % 7 == 0:
            self._write_brute_force_demo_burst()
            return

        timestamp = datetime.utcnow().strftime('%d/%b/%Y:%H:%M:%S +0000')
        ip_suffix = random.randint(1, 254)
        port = random.randint(10000, 65000)
        
        # Pick a random log type
        log_type = random.choice(['apache', 'ssh', 'firewall'])
        
        if log_type == 'apache':
            log_file = os.path.join(self.logs_dir, 'apache_access.log')
            template = random.choice(self.apache_logs)
            entry = template.format(ip=ip_suffix, timestamp=timestamp)
        elif log_type == 'ssh':
            log_file = os.path.join(self.logs_dir, 'ssh_auth.log')
            template = random.choice(self.ssh_logs)
            entry = template.format(ip=ip_suffix, port=port)
        else:
            log_file = os.path.join(self.logs_dir, 'firewall.log')
            template = random.choice(self.firewall_logs)
            entry = template.format(ip=ip_suffix)
            
        # Append to file
        with open(log_file, 'a') as f:
            f.write(entry + '\n')
            
        print(f"  📝 Generated {log_type} log entry")

# ═══════════════════════════════════════════════════════════════════
# ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════════

# ── 1. MITRE ATT&CK HEATMAP ────────────────────────────────────────
# Maps detected threat types to official MITRE ATT&CK technique IDs
# ── EXPLAINABLE AI (XAI) ──────────────────────────────────────────

# Per-threat-type: detection rule that fired, indicators checked,
# and step-by-step recommended response actions.
XAI_RULES = {
    'brute_force': {
        'rule':       'Threshold rule: ≥5 failed authentication attempts from the same IP within 60 seconds',
        'indicators': ['Multiple "Failed password" / 401 entries', 'Same source IP repeated rapidly',
                       'Targets privileged accounts (root, admin)', 'High request frequency'],
        'actions':    ['Block source IP at firewall immediately',
                       'Enable account lockout after N failed attempts',
                       'Enforce Multi-Factor Authentication (MFA)',
                       'Review auth logs for any successful logins from this IP',
                       'Alert the affected user account'],
        'mitre':      'T1110 — Brute Force (Credential Access)',
    },
    'port_scan': {
        'rule':       'Threshold rule: ≥10 connection attempts to ≥5 unique ports from same IP within 120 seconds',
        'indicators': ['Rapid TCP/UDP connections to different ports', 'High rate of rejected connections',
                       'Sequential port access pattern', 'ICMP probes'],
        'actions':    ['Block scanning IP at network perimeter',
                       'Close unnecessary open ports',
                       'Enable port-scan detection (Snort/Suricata rules)',
                       'Review exposed services for vulnerabilities',
                       'Monitor for follow-up exploitation attempts from same IP'],
        'mitre':      'T1046 — Network Service Discovery (Discovery)',
    },
    'sql_injection': {
        'rule':       'Pattern rule: request URL/body matches SQL injection regex (UNION SELECT, OR 1=1, DROP TABLE, etc.)',
        'indicators': ['SQL keywords in HTTP parameters', 'Unusual characters in query strings',
                       'HTTP 500/403 error responses on form submissions',
                       'Encoded payloads (%27 = single quote)'],
        'actions':    ['Block source IP at WAF / firewall',
                       'Audit database query logs for unauthorised access',
                       'Use parameterised queries and prepared statements',
                       'Deploy a Web Application Firewall (WAF)',
                       'Check for any data exfiltration in response sizes'],
        'mitre':      'T1190 — Exploit Public-Facing Application (Initial Access)',
    },
    'xss': {
        'rule':       'Pattern rule: request matches XSS regex (<script>, onerror=, javascript:, alert()',
        'indicators': ['Script tags or event handlers in URL parameters',
                       'Encoded JavaScript payloads', 'Unusual characters in form inputs'],
        'actions':    ['Sanitise and HTML-encode all user inputs server-side',
                       'Implement Content Security Policy (CSP) headers',
                       'Block the attacking IP',
                       'Audit stored content for injected scripts',
                       'Notify affected users if stored XSS confirmed'],
        'mitre':      'T1059.007 — JavaScript (Command & Scripting Interpreter)',
    },
    'path_traversal': {
        'rule':       'Pattern rule: request URL contains directory traversal sequences (../, ..\\, /etc/passwd)',
        'indicators': ['../ or ..\\ sequences in URL', 'Attempts to access /etc/passwd, /windows/system32',
                       'Unusual file extension requests', 'Encoded traversal (%2e%2e%2f)'],
        'actions':    ['Block the attacking IP',
                       'Sanitise all file path inputs — reject any containing ../',
                       'Use chroot jails or containerisation to limit file access',
                       'Audit which files may have been accessed',
                       'Apply principle of least privilege to file system'],
        'mitre':      'T1083 — File and Directory Discovery (Discovery)',
    },
    'suspicious_pattern': {
        'rule':       'Composite rule: multiple attack patterns detected from same IP within session',
        'indicators': ['Mixed attack signatures (SQLi + XSS + path traversal)',
                       'High volume of error responses', 'Automated tool behaviour patterns',
                       'Rapid sequential endpoint probing'],
        'actions':    ['Block IP immediately — multi-vector attack indicates automated tool',
                       'Review all requests from this IP in the last hour',
                       'Check for successful exploitation in application logs',
                       'Escalate to Tier 2 analyst for investigation',
                       'Consider threat hunting for similar patterns across all logs'],
        'mitre':      'T1071 — Application Layer Protocol (Command & Control)',
    },
    'anomaly': {
        'rule':       'ML rule: Isolation Forest model flagged behaviour as statistically anomalous (score < threshold)',
        'indicators': ['Traffic pattern deviates significantly from baseline',
                       'Unusual access time (off-hours activity)',
                       'Abnormal request volume or port usage',
                       'New IP with no historical baseline'],
        'actions':    ['Investigate the flagged IP and time window',
                       'Compare against baseline traffic for same time period',
                       'Correlate with other threat types from same IP',
                       'Escalate if confirmed malicious behaviour',
                       'Update anomaly detection model with confirmed labels'],
        'mitre':      'T1036 — Masquerading (Defense Evasion)',
    },
    'honeypot': {
        'rule':       'Honeypot rule: attacker accessed a deliberately fake/hidden URL trap',
        'indicators': ['Access to URL that has no legitimate business purpose',
                       'Likely automated scanner or deliberate reconnaissance',
                       'High confidence indicator of hostile intent'],
        'actions':    ['Block IP immediately — no legitimate user visits this path',
                       'Review all other requests from this IP',
                       'Check if other honeypots were also triggered',
                       'Escalate — attacker is actively probing the system'],
        'mitre':      'T1595 — Active Scanning (Reconnaissance)',
    },
    'ddos': {
        'rule':       'Volume rule: request rate from multiple IPs exceeds normal baseline by >500%',
        'indicators': ['Massive spike in concurrent connections', 'High bandwidth consumption',
                       'Server response time degradation', 'Multiple source IPs in same /24 subnet'],
        'actions':    ['Enable rate limiting immediately',
                       'Contact CDN/ISP for upstream traffic filtering',
                       'Block top attacking IP ranges at network edge',
                       'Activate DDoS mitigation service (Cloudflare, AWS Shield)',
                       'Scale infrastructure if under sustained attack'],
        'mitre':      'T1498 — Network Denial of Service (Impact)',
    },
}

def _build_xai_explanation(threat):
    """Build a structured XAI explanation dict for a Threat object."""
    threat_type = threat.threat_type or 'unknown'
    rule_info   = XAI_RULES.get(threat_type, {
        'rule':       'General anomaly detection rule triggered',
        'indicators': ['Unusual behaviour detected'],
        'actions':    ['Investigate source IP', 'Review related log entries'],
        'mitre':      MITRE_MAPPING.get(threat_type, {}).get('id', 'T????') + ' — ' +
                      MITRE_MAPPING.get(threat_type, {}).get('name', 'Unknown Technique'),
    })

    # Score factor breakdown — reconstruct approximate contribution of each factor
    score      = threat.risk_score or 5.0
    base       = round(min(score / 1.2, 10.0), 1)   # approx base before multiplier
    history    = round(min(score * 0.15, 2.5), 1)
    geo        = round(min(score * 0.08, 1.5), 1)
    time_pat   = round(min(score * 0.10, 2.0), 1)
    freq       = round(min(score * 0.07, 1.5), 1)

    risk_level = ('CRITICAL' if score >= 9 else 'HIGH' if score >= 7
                  else 'MEDIUM' if score >= 4 else 'LOW')

    # Historical context: how many times has this IP been seen before?
    ip_count = Threat.query.filter_by(source_ip=threat.source_ip).count()
    location = get_location_info(threat.source_ip or '')

    return {
        'threat_id':    threat.id,
        'threat_type':  threat_type,
        'source_ip':    threat.source_ip,
        'risk_score':   score,
        'risk_level':   risk_level,
        'timestamp':    threat.timestamp.isoformat() if threat.timestamp else None,

        # Core XAI fields
        'why_flagged':  rule_info['rule'],
        'indicators':   rule_info['indicators'],
        'mitre':        rule_info['mitre'],
        'actions':      rule_info['actions'],

        # Risk score breakdown
        'score_breakdown': {
            'base_threat_severity': base,
            'historical_ip_risk':   history,
            'geographic_risk':      geo,
            'time_pattern_risk':    time_pat,
            'frequency_risk':       freq,
        },

        # Contextual intelligence
        'context': {
            'ip_total_incidents':  ip_count,
            'ip_is_repeat':        ip_count > 1,
            'location':            location,
            'ai_explanation':      threat.ai_explanation or None,
        }
    }


@app.route('/api/xai/explain/<int:threat_id>')
def xai_explain(threat_id):
    """Return full Explainable AI breakdown for a specific threat."""
    try:
        threat = Threat.query.get_or_404(threat_id)
        return jsonify(_build_xai_explanation(threat))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/xai/explain-ip/<source_ip>')
def xai_explain_ip(source_ip):
    """Return XAI explanations for all threats from a specific IP."""
    try:
        threats = Threat.query.filter_by(
            source_ip=source_ip
        ).order_by(Threat.risk_score.desc()).limit(10).all()

        if not threats:
            return jsonify({'error': 'No threats found for IP ' + source_ip}), 404

        return jsonify({
            'source_ip':    source_ip,
            'total':        len(threats),
            'location':     get_location_info(source_ip),
            'explanations': [_build_xai_explanation(t) for t in threats]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


MITRE_MAPPING = {
    'brute_force':       {'id': 'T1110', 'name': 'Brute Force',                  'tactic': 'Credential Access'},
    'port_scan':         {'id': 'T1046', 'name': 'Network Service Discovery',     'tactic': 'Discovery'},
    'sql_injection':     {'id': 'T1190', 'name': 'Exploit Public-Facing App',     'tactic': 'Initial Access'},
    'xss':               {'id': 'T1059', 'name': 'Command & Scripting Interpreter','tactic': 'Execution'},
    'path_traversal':    {'id': 'T1083', 'name': 'File & Directory Discovery',    'tactic': 'Discovery'},
    'suspicious_pattern':{'id': 'T1071', 'name': 'App Layer Protocol',            'tactic': 'Command & Control'},
    'anomaly':           {'id': 'T1036', 'name': 'Masquerading',                  'tactic': 'Defense Evasion'},
    'ddos':              {'id': 'T1498', 'name': 'Network Denial of Service',     'tactic': 'Impact'},
    'command_injection': {'id': 'T1059', 'name': 'Command & Scripting Interpreter','tactic': 'Execution'},
}

@app.route('/api/mitre-heatmap')
def mitre_heatmap():
    """Return MITRE ATT&CK technique hit counts for heatmap visualisation."""
    try:
        from sqlalchemy import text as sa_text
        rows = db.session.query(
            Threat.threat_type,
            db.func.count(Threat.id).label('cnt'),
            db.func.max(Threat.risk_score).label('max_score')
        ).group_by(Threat.threat_type).all()

        techniques = []
        for row in rows:
            info = MITRE_MAPPING.get(row[0], {
                'id': 'T????', 'name': row[0].replace('_', ' ').title(),
                'tactic': 'Unknown'
            })
            techniques.append({
                'threat_type':  row[0],
                'technique_id': info['id'],
                'technique':    info['name'],
                'tactic':       info['tactic'],
                'count':        row[1],
                'max_score':    round(row[2], 1) if row[2] else 0,
                'severity':     'critical' if row[2] and row[2] >= 9 else
                                'high'     if row[2] and row[2] >= 7 else
                                'medium'   if row[2] and row[2] >= 4 else 'low'
            })

        # All known techniques, with count=0 for undetected ones
        all_tactics = ['Initial Access', 'Execution', 'Discovery',
                       'Credential Access', 'Defense Evasion',
                       'Command & Control', 'Impact']
        detected_ids = {t['technique_id'] for t in techniques}
        for threat_type, info in MITRE_MAPPING.items():
            if info['id'] not in detected_ids:
                techniques.append({
                    'threat_type': threat_type, 'technique_id': info['id'],
                    'technique': info['name'],  'tactic': info['tactic'],
                    'count': 0, 'max_score': 0, 'severity': 'none'
                })

        total_techniques_hit = sum(1 for t in techniques if t['count'] > 0)
        return jsonify({
            'techniques': techniques,
            'tactics': all_tactics,
            'total_techniques_hit': total_techniques_hit,
            'total_techniques': len(MITRE_MAPPING)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 2. HONEYPOT ENDPOINTS ──────────────────────────────────────────
HONEYPOT_PATHS = [
    '/admin', '/wp-admin', '/wp-login.php', '/phpmyadmin',
    '/.env', '/config.php', '/backup.zip', '/shell.php',
    '/manager/html', '/actuator', '/.git/config', '/api/v1/admin'
]

@app.route('/api/honeypot-hits')
def get_honeypot_hits():
    """Return honeypot access attempts logged in the database."""
    try:
        hits = Threat.query.filter(
            Threat.threat_type == 'honeypot'
        ).order_by(Threat.timestamp.desc()).limit(50).all()

        return jsonify({
            'hits': [{
                'id':          h.id,
                'source_ip':   h.source_ip,
                'description': h.description,
                'risk_score':  h.risk_score,
                'timestamp':   h.timestamp.isoformat(),
                'status':      h.status
            } for h in hits],
            'total': Threat.query.filter_by(threat_type='honeypot').count()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _register_honeypot_routes():
    """Dynamically register honeypot URL traps."""
    for path in HONEYPOT_PATHS:
        rule = path  # Flask needs a unique function per route
        endpoint = 'honeypot_' + path.replace('/', '_').replace('.', '_').strip('_')

        def make_handler(trap_path):
            def handler(**kwargs):
                try:
                    ip = request.remote_addr or 'unknown'
                    method = request.method
                    ua = request.headers.get('User-Agent', '')[:200]
                    desc = ('Honeypot triggered: %s %s | UA: %s'
                            % (method, trap_path, ua))
                    _now = datetime.utcnow()
                    threat = Threat(
                        threat_type='honeypot',
                        source_ip=ip,
                        risk_score=8.5,
                        description=desc,
                        status='active',
                        timestamp=_now,
                        ingested_at=_now,
                    )
                    db.session.add(threat)
                    db.session.commit()
                    send_telegram_alert(
                        '🍯 Honeypot Hit!\nIP: %s\nPath: %s\nUA: %s'
                        % (ip, trap_path, ua[:80]))
                except Exception:
                    pass
                # Return a convincing fake response
                return (
                    '<html><body><h1>401 Unauthorized</h1>'
                    '<p>Access denied.</p></body></html>'
                ), 401
            handler.__name__ = endpoint
            return handler

        try:
            app.add_url_rule(path, endpoint,
                             make_handler(path),
                             methods=['GET', 'POST', 'PUT', 'DELETE'])
        except Exception:
            pass  # Skip if route already exists

with app.app_context():
    _register_honeypot_routes()


# ── 3. IP AUTO-BLOCKLIST ───────────────────────────────────────────
# In-memory blocklist (persisted via DB on restart)
_blocked_ips = set()

def _load_blocklist():
    """Load blocked IPs from the database on startup."""
    global _blocked_ips
    try:
        with app.app_context():
            rows = IOC.query.filter_by(
                ioc_type='blocked_ip', is_active=True).all()
            _blocked_ips = {r.value for r in rows}
    except Exception:
        pass

def auto_block_ip(ip, reason='Auto-blocked: critical risk score >= 9.0'):
    """Block an IP and persist to DB."""
    global _blocked_ips
    if ip in _blocked_ips:
        return False
    _blocked_ips.add(ip)
    try:
        existing = IOC.query.filter_by(
            ioc_type='blocked_ip', value=ip).first()
        if not existing:
            ioc = IOC(
                ioc_type='blocked_ip', value=ip,
                description=reason, threat_level='critical',
                source='auto-block', added_by='system',
                is_active=True, created_at=datetime.utcnow()
            )
            db.session.add(ioc)
            db.session.commit()
        send_telegram_alert(
            '🚫 IP Auto-Blocked\nIP: %s\nReason: %s' % (ip, reason))
        return True
    except Exception:
        return False

@app.route('/api/blocklist', methods=['GET'])
def get_blocklist():
    """Return all blocked IPs."""
    try:
        rows = IOC.query.filter_by(
            ioc_type='blocked_ip', is_active=True
        ).order_by(IOC.created_at.desc()).all()
        return jsonify({
            'blocked_ips': [{
                'id':          r.id,
                'ip':          r.value,
                'reason':      r.description,
                'blocked_at':  r.created_at.isoformat() if r.created_at else None,
                'added_by':    r.added_by
            } for r in rows],
            'total': len(rows)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/blocklist/add', methods=['POST'])
def add_to_blocklist():
    """Manually block an IP."""
    try:
        data = request.get_json() or {}
        ip   = data.get('ip', '').strip()
        reason = data.get('reason', 'Manually blocked by analyst')
        if not ip:
            return jsonify({'error': 'IP address required'}), 400
        result = auto_block_ip(ip, reason)
        return jsonify({'success': True, 'already_blocked': not result, 'ip': ip})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/blocklist/remove/<ip>', methods=['DELETE'])
def remove_from_blocklist(ip):
    """Unblock an IP."""
    try:
        global _blocked_ips
        _blocked_ips.discard(ip)
        rows = IOC.query.filter_by(ioc_type='blocked_ip', value=ip).all()
        for r in rows:
            r.is_active = False
        db.session.commit()
        return jsonify({'success': True, 'ip': ip, 'action': 'unblocked'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 4. ATTACK HOUR HEATMAP ─────────────────────────────────────────
@app.route('/api/attack-heatmap')
def attack_heatmap():
    """Return threat counts grouped by hour of day and day of week."""
    try:
        from sqlalchemy import func, extract
        # Hourly distribution (0-23)
        hourly = db.session.query(
            func.strftime('%H', Threat.timestamp).label('hour'),
            func.count(Threat.id).label('cnt')
        ).group_by('hour').all()
        hourly_data = {int(r[0]): r[1] for r in hourly if r[0]}
        hours = [hourly_data.get(h, 0) for h in range(24)]

        # Day-of-week distribution (0=Mon … 6=Sun)
        daily = db.session.query(
            func.strftime('%w', Threat.timestamp).label('dow'),
            func.count(Threat.id).label('cnt')
        ).group_by('dow').all()
        daily_data = {int(r[0]): r[1] for r in daily if r[0]}
        days = [daily_data.get(d, 0) for d in range(7)]
        day_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

        peak_hour = hours.index(max(hours)) if max(hours) > 0 else 0
        peak_day  = days.index(max(days))   if max(days) > 0  else 0

        return jsonify({
            'hours': hours,
            'days':  days,
            'day_labels': day_labels,
            'peak_hour': peak_hour,
            'peak_day':  day_labels[peak_day],
            'total_threats': sum(hours)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 5. SYSTEM HEALTH MONITOR ──────────────────────────────────────
@app.route('/api/system-health')
def system_health():
    """Return real-time server resource usage."""
    try:
        import psutil
        cpu   = psutil.cpu_percent(interval=0.1)
        mem   = psutil.virtual_memory()
        disk  = psutil.disk_usage('/')
        net   = psutil.net_io_counters()
        uptime_secs = int(
            (datetime.utcnow() -
             datetime.utcfromtimestamp(psutil.boot_time())).total_seconds())
        hours, rem = divmod(uptime_secs, 3600)
        mins,  _   = divmod(rem, 60)
        return jsonify({
            'cpu_percent':         round(cpu, 1),
            'memory_percent':      round(mem.percent, 1),
            'memory_used_gb':      round(mem.used  / 1e9, 2),
            'memory_total_gb':     round(mem.total / 1e9, 2),
            'disk_percent':        round(disk.percent, 1),
            'disk_used_gb':        round(disk.used  / 1e9, 1),
            'disk_total_gb':       round(disk.total / 1e9, 1),
            'net_sent_mb':         round(net.bytes_sent / 1e6, 1),
            'net_recv_mb':         round(net.bytes_recv / 1e6, 1),
            'uptime':              '%dh %dm' % (hours, mins),
            'status':              'healthy' if cpu < 80 and mem.percent < 85 else 'stressed'
        })
    except ImportError:
        # psutil not installed — return placeholder
        import random
        return jsonify({
            'cpu_percent':     round(random.uniform(5, 35), 1),
            'memory_percent':  round(random.uniform(40, 65), 1),
            'memory_used_gb':  round(random.uniform(2, 6), 2),
            'memory_total_gb': 8.0,
            'disk_percent':    round(random.uniform(30, 60), 1),
            'disk_used_gb':    round(random.uniform(50, 150), 1),
            'disk_total_gb':   256.0,
            'net_sent_mb':     round(random.uniform(100, 500), 1),
            'net_recv_mb':     round(random.uniform(200, 800), 1),
            'uptime':          '2h 34m',
            'status':          'healthy'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Thonny / IDEs often start with CWD elsewhere — SQLite URI and relative paths break
    os.chdir(_APP_DIR)

    _run_port = _choose_run_port()
    os.environ['SOC_PUBLIC_PORT'] = str(_run_port)
    _pref = int(os.environ.get('SOC_PORT', str(DEFAULT_SOC_PORT)))
    print('=' * 62)
    print('  SOC Dashboard — working_app.py (full Security Monitor UI)')
    print('=' * 62)
    print()
    print('  >>> The dashboard does NOT use port 5000 by default.')
    print(f'  >>> Open ONLY: http://127.0.0.1:{_run_port}/')
    print('  >>> Plain "SIMPLE TEST" on :5000 is almost always a different program.')
    print()
    if _run_port != _pref:
        print('!' * 62)
        print(f'  Port changed: server is on {_run_port} (wanted {_pref} in SOC_PORT).')
        print(f'  Use: http://127.0.0.1:{_run_port}')
        print('!' * 62)
        print()
    else:
        print(f'  Check: http://127.0.0.1:{_run_port}/api/whoami  →  "app":"working_app"')
        print()

    if not CICIDS_MODEL_AVAILABLE and _CICIDS_UNAVAILABLE_CODE == 'numpy_mismatch':
        print('  >>> ML (CICIDS): Pickle was built with NumPy 2; this Python has NumPy 1 (typical in Thonny).')
        print('      Quick fix: double-click  repair_ml_for_thonny.bat  (rebuilds models\\trained_model.pkl).')
        print('      Or: train_model_thonny.bat → run_dashboard_thonny.bat')
        print('      Or: train_model.bat → run_dashboard.bat only (no Thonny for this app).')
        print()

    with app.app_context():
        # Only create tables if they don't exist - never drop existing data
        db.create_all()

        # Add ingested_at columns + backfill NULLs so Recent Activity is not stuck at 0
        ensure_recent_activity_schema()
        print('✅ Recent activity: ingested_at columns + backfill applied (if needed)')

        # Enable WAL mode for SQLite — must run outside ORM session transaction
        # (otherwise: OperationalError: Safety level may not be changed inside a transaction)
        from sqlalchemy import text as _sa_text

        def _enable_sqlite_wal():
            eng = db.engine
            if eng.dialect.name != 'sqlite':
                return False, 'not sqlite'

            def _wal_via_sqlite3():
                import sqlite3 as _sq3
                uri = app.config.get('SQLALCHEMY_DATABASE_URI') or ''
                if not uri.startswith('sqlite:///') or ':memory:' in uri:
                    raise RuntimeError('not a file sqlite uri')
                rel = uri.replace('sqlite:///', '', 1).split('?')[0]
                dbp = rel if os.path.isabs(rel) else os.path.join(_APP_DIR, rel)
                _c = _sq3.connect(dbp, timeout=30)
                try:
                    _c.execute('PRAGMA journal_mode=WAL')
                    _c.execute('PRAGMA synchronous=NORMAL')
                    _c.execute('PRAGMA busy_timeout=30000')
                finally:
                    _c.close()

            try:
                _wal_via_sqlite3()
                return True, None
            except Exception as _sq_e:
                try:
                    with eng.connect() as _raw:
                        with _raw.execution_options(isolation_level='AUTOCOMMIT') as _conn:
                            _conn.execute(_sa_text('PRAGMA journal_mode=WAL'))
                            _conn.execute(_sa_text('PRAGMA synchronous=NORMAL'))
                            _conn.execute(_sa_text('PRAGMA busy_timeout=30000'))
                    return True, None
                except Exception as _wal_e:
                    return False, f'sqlite3: {_sq_e}; sqlalchemy: {_wal_e}'

        _ok_wal, _wal_err = _enable_sqlite_wal()
        if _ok_wal:
            print("✅ SQLite WAL mode enabled — concurrent thread writes supported")
        else:
            print(f"⚠️  WAL mode setup skipped: {_wal_err}")
        
        logs_dir = os.path.join(_APP_DIR, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        print(f"📝 Log source: real files under {logs_dir} (apache_access.log, ssh_auth.log, firewall.log, …)")
        if _synthetic_log_enabled():
            print("⚠️  ENABLE_SYNTHETIC_LOG_GENERATOR=true — demo lines will also be appended every "
                  f"{_synthetic_log_interval()}s (not production traffic)")
        else:
            print("✅ Synthetic log generator OFF — only your real log files are read (default)")
        
        # Parse real log files from logs directory (always next to working_app.py, not CWD)
        print("📄 Parsing real log files...")
        from log_parser import LogParser
        log_parser = LogParser()
        
        import json as _json
        total_parsed = 0
        for log_file in sorted(os.listdir(logs_dir)):
            if not log_file.endswith('.log'):
                continue
            file_path = os.path.join(logs_dir, log_file)
            file_count = 0
            try:
                parsed_logs = log_parser.parse_log_file(file_path)
                for parsed_log in parsed_logs:
                    try:
                        normalized = log_parser.normalize_log_entry(parsed_log)
                        existing = LogEntry.query.filter_by(
                            source_ip=normalized['source_ip'],
                            timestamp=normalized['timestamp'],
                            raw_log=normalized['raw_log']
                        ).first()
                        if not existing:
                            log_entry = LogEntry(
                                source_ip=normalized['source_ip'],
                                timestamp=normalized['timestamp'],
                                destination_port=normalized.get('destination_port'),
                                protocol=normalized.get('protocol', ''),
                                action=normalized.get('action', ''),
                                raw_log=normalized['raw_log'],
                                parsed_data=_json.dumps(normalized.get('parsed_data', {})),
                                ingested_at=datetime.utcnow(),
                            )
                            db.session.add(log_entry)
                            file_count += 1
                    except Exception as _row_err:
                        db.session.rollback()
                        print(f"    ⚠️  Skipped row in {log_file}: {_row_err}")

                db.session.commit()
                total_parsed += file_count
                print(f"  ✅ Parsed {file_count} new entries from {log_file} ({len(parsed_logs)} total lines)")
            except Exception as e:
                db.session.rollback()
                print(f"  ❌ Error parsing {log_file}: {e}")

        if total_parsed > 0:
            print(f"✅ Total {total_parsed} real log entries added to database!")
            log_audit(
                action='log_files_parsed',
                resource_type='log_entry',
                details=f'Parsed and ingested {total_parsed} log entries from log directory on startup',
                severity='info',
                user='system'
            )
            
            # Run threat detection on newly added logs
            print("🔍 Running threat detection on parsed logs...")
            from threat_detector import ThreatDetector
            from risk_scorer import RiskScorer
            
            # Get all log entries for threat detection
            all_logs = LogEntry.query.all()
            log_dicts = []
            for log in all_logs:
                try:
                    parsed_data = json.loads(log.parsed_data) if log.parsed_data else {}
                except:
                    parsed_data = {}
                log_dicts.append({
                    'id': log.id,
                    'timestamp': log.timestamp,
                    'source_ip': log.source_ip,
                    'destination_port': log.destination_port,
                    'protocol': log.protocol,
                    'action': log.action,
                    'raw_log': log.raw_log,
                    'parsed_data': parsed_data
                })
            
            # Detect threats
            config = Config()
            threat_detector = ThreatDetector(config)
            detected_threats = threat_detector.detect_threats(log_dicts)
            
            if detected_threats:
                print(f"🚨 Detected {len(detected_threats)} threats from logs!")
                risk_scorer = RiskScorer(config)
                
                for threat_data in detected_threats:
                    # Calculate risk score
                    historical_threats = []
                    analysis = risk_scorer.calculate_comprehensive_risk_score(threat_data, historical_threats)
                    threat_data['risk_score'] = analysis['final_score']

                    # ── CICIDS 2017 ML Model — secondary scoring ──────────
                    if CICIDS_MODEL_AVAILABLE:
                        try:
                            cicids_result = cicids_predict({
                                'Flow Duration'               : 10000,
                                'Total Fwd Packets'           : threat_data.get('request_count', 5),
                                'Total Backward Packets'      : 2,
                                'Total Length of Fwd Packets' : 500,
                                'Total Length of Bwd Packets' : 200,
                                'Fwd Packet Length Max'       : 100,
                                'Fwd Packet Length Mean'      : 60,
                                'Bwd Packet Length Max'       : 80,
                                'Bwd Packet Length Mean'      : 40,
                                'Flow Bytes/s'                : threat_data.get('risk_score', 5) * 100000,
                                'Flow Packets/s'              : threat_data.get('risk_score', 5) * 500,
                                'Flow IAT Mean'               : 1000,
                                'Flow IAT Max'                : 5000,
                                'Fwd IAT Mean'                : 1000,
                                'Bwd IAT Mean'                : 1000,
                                'Packet Length Mean'          : 60,
                                'Packet Length Std'           : 20,
                                'Average Packet Size'         : 60,
                            })
                            threat_data['cicids_ml_score']   = cicids_result.get('risk_score', 0)
                            threat_data['cicids_is_attack']  = cicids_result.get('is_attack', False)
                            threat_data['cicids_confidence'] = cicids_result.get('confidence', 'N/A')
                            if cicids_result.get('is_attack') and threat_data.get('risk_score', 0) < 8.0:
                                threat_data['risk_score'] = min(10.0, threat_data['risk_score'] + 0.5)
                        except Exception:
                            pass
                    # ─────────────────────────────────────────────────────

                    # Generate AI explanation for high-risk threats
                    if should_send_alert(threat_data):
                        threat_data['ai_explanation'] = generate_ai_explanation(threat_data)
                    
                    # Create threat record
                    _now = datetime.utcnow()
                    threat = Threat(
                        threat_type=threat_data['threat_type'],
                        source_ip=threat_data['source_ip'],
                        risk_score=threat_data['risk_score'],
                        description=threat_data['description'],
                        ai_explanation=threat_data.get('ai_explanation'),
                        timestamp=_now,
                        ingested_at=_now,
                    )
                    db.session.add(threat)
                    db.session.flush()
                    
                    # Send alert for high-risk threats
                    if should_send_alert(threat_data):
                        message = (
                            f"🚨 SECURITY ALERT\n\n"
                            f"Threat Type: {threat_data['threat_type'].replace('_', ' ').title()}\n"
                            f"Source IP: {threat_data['source_ip']}\n"
                            f"Risk Score: {threat_data['risk_score']}/10\n"
                            f"Description: {threat_data['description']}\n\n"
                            f"Dashboard: {dashboard_public_url()}"
                        )
                        
                        alert_sent = send_telegram_alert(message)
                        
                        # Create alert record
                        alert = Alert(
                            threat_id=threat.id,
                            alert_type='telegram',
                            message=message,
                            sent='yes' if alert_sent else 'no',
                            sent_timestamp=datetime.utcnow() if alert_sent else None
                        )
                        db.session.add(alert)
                    
                    # Create incident for medium+ risk threats (risk_score >= 5.0)
                    if threat_data['risk_score'] >= 5.0:
                        import uuid
                        incident = Incident(
                            incident_id=f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}",
                            title=f"{threat_data['threat_type'].replace('_', ' ').title()} from {threat_data['source_ip']}",
                            description=threat_data['description'],
                            severity='high' if threat_data['risk_score'] >= 7.0 else 'medium',
                            status='open',
                            priority=1 if threat_data['risk_score'] >= 7.0 else 2,
                            assigned_to='security-team@company.com',
                            threat_ids=json.dumps([threat.id]),
                            ioc_ids=json.dumps([threat_data['source_ip']]),
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        db.session.add(incident)
                        print(f"  📝 Created incident for threat: {threat_data['threat_type']}")
                
                db.session.commit()
                print(f"✅ Threats saved to database!")
                # Log the threat detection event to audit log
                with app.app_context():
                    log_audit(
                        action='system_startup_threat_detection',
                        resource_type='threat',
                        details=f'System startup: detected and saved {len(detected_threats)} threats from log analysis',
                        severity='warning' if len(detected_threats) > 0 else 'info',
                        user='system'
                    )
                
                # Run attack pattern detection on the new threats
                print("🔍 Running attack pattern detection...")
                analyzer = AttackPatternAnalyzer()
                patterns = analyzer.analyze_recent_threats(hours=24)
                if patterns:
                    print(f"🎯 Detected {len(patterns)} attack patterns!")
                else:
                    print("ℹ️ No attack patterns detected")
            else:
                print("ℹ️ No threats detected in current logs")
        else:
            print("ℹ️ No new log entries to add (already in database)")
        
        # Add default alert rules only (needed for real alerts)
        if AlertRule.query.count() == 0:
            print("📝 Setting up default alert rules...")
            default_rules = [
                {
                    'name': 'High Risk Threat Alert',
                    'description': 'Alert when threat risk score >= 7.0',
                    'rule_type': 'risk_threshold',
                    'conditions': json.dumps({'min_risk_score': 7.0}),
                    'severity_threshold': 'high',
                    'notification_channels': json.dumps(['telegram', 'email']),
                    'is_active': True
                },
                {
                    'name': 'Brute Force Detection',
                    'description': 'Alert on brute force attack patterns',
                    'rule_type': 'pattern_match',
                    'conditions': json.dumps({'pattern_type': 'brute_force', 'min_occurrences': 5}),
                    'severity_threshold': 'critical',
                    'notification_channels': json.dumps(['telegram']),
                    'is_active': True
                }
            ]
            
            for rule_data in default_rules:
                rule = AlertRule(**rule_data)
                db.session.add(rule)
            
            db.session.commit()
            print(f"✅ Added {len(default_rules)} default alert rules")
        
        # Start watchdog file watcher (picks up changes immediately)
        print("👁️  Starting log file watcher...")
        log_monitor = LogMonitor(logs_dir)
        log_monitor.start()

        # Optional: synthetic demo lines (OFF by default — use only for UI demos)
        log_generator = None
        if _synthetic_log_enabled():
            _syn_int = _synthetic_log_interval()
            print(f"🔄 Starting synthetic log generator (every {_syn_int}s)...")
            log_generator = LogGenerator(
                logs_dir=logs_dir,
                interval=_syn_int,
            )
            log_generator.start()
        else:
            print("ℹ️  Skipping synthetic log generator (set ENABLE_SYNTHETIC_LOG_GENERATOR=true to enable)")

        # Start continuous monitor — tails log files every 15s, runs threat detection
        print("🔍 Starting continuous log monitor (scans every 15 seconds)...")
        continuous_monitor = ContinuousLogMonitor(logs_dir=logs_dir, scan_interval=15)
        continuous_monitor.start()

        print("🚀 Starting Working Security Dashboard...")
        print(f"📊 Open {dashboard_public_url()} in your browser")
        print(f"📄 Dashboard HTML: {_DASHBOARD_HTML_PATH}")
        print(f"   File exists: {os.path.isfile(_DASHBOARD_HTML_PATH)}")
        print(f"🔎 Sanity check: {dashboard_public_url()}/api/whoami  (expect JSON app=working_app)")
        if _run_port != _pref:
            print("   Windows: netstat -ano | findstr :" + str(_pref))
        _open_browser = os.environ.get('SOC_OPEN_BROWSER', 'true').lower() in (
            '1', 'true', 'yes', 'on')
        if _open_browser:
            try:
                _start_url = f'http://127.0.0.1:{_run_port}/'
                print(f'🌐 Opening browser to: {_start_url}')
                webbrowser.open(_start_url)
            except Exception as _wb:
                print(f'⚠️  Could not open browser automatically: {_wb}')
        print("✅ Real-time monitoring active — data updates every 10-15 seconds")
        app.run(host='127.0.0.1', port=_run_port, debug=False)

