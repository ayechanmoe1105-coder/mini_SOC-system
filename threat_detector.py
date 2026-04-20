import re
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

# Machine learning libraries are optional
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class ThreatDetector:
    def __init__(self, config):
        self.config = config
        self.brute_force_threshold = config.BRUTE_FORCE_THRESHOLD
        self.brute_force_window = config.BRUTE_FORCE_WINDOW
        self.scan_detection_window = config.SCAN_DETECTION_WINDOW
        
        # Track failed attempts by IP
        self.failed_attempts = defaultdict(list)
        self.port_scan_attempts = defaultdict(list)
        self.traffic_patterns = defaultdict(list)
        # Rolling pattern hits per IP (live tail passes one line per batch; rules need history)
        self.suspicious_pattern_hits = defaultdict(list)  # ip -> [{'pattern', 'timestamp'}, ...]
        self._pattern_window_sec = int(getattr(config, 'SUSPICIOUS_PATTERN_WINDOW_SEC', 900))
        self._suspicious_emit_cooldown = {}  # ip -> monotonic time of last emitted threat
        self._anomaly_rolling: List[Dict[str, Any]] = []
        self._max_anomaly_buffer = int(getattr(config, 'ANOMALY_ROLLING_BUFFER', 128))
        
        # Anomaly detection model (optional)
        self.anomaly_detector = None
        self.scaler = None
        self.model_trained = False
        
        if ML_AVAILABLE:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()

    def detect_threats(self, log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main threat detection method."""
        threats = []
        
        # Update tracking data
        self._update_tracking_data(log_entries)
        
        # Detect different threat types
        threats.extend(self._detect_brute_force())
        threats.extend(self._detect_port_scans())
        threats.extend(self._detect_suspicious_patterns())
        threats.extend(self._detect_anomalies(log_entries))
        
        return threats

    def _update_tracking_data(self, log_entries: List[Dict[str, Any]]):
        """Update internal tracking data with new log entries."""
        current_time = datetime.utcnow()
        
        for entry in log_entries:
            ip = entry.get('source_ip', '')
            timestamp = entry.get('timestamp', current_time)
            
            # Track failed login attempts
            if self._is_failed_attempt(entry):
                self.failed_attempts[ip].append(timestamp)
            
            # Track port scan attempts
            if self._is_port_scan_attempt(entry):
                port = entry.get('destination_port')
                if port:
                    self.port_scan_attempts[ip].append((timestamp, port))
            
            # Track traffic patterns for anomaly detection
            self.traffic_patterns[ip].append(entry)

            # Accumulate suspicious pattern matches per IP across many small batches
            if ip:
                pd = entry.get('parsed_data', {})
                if isinstance(pd, str):
                    try:
                        pd = json.loads(pd)
                    except (json.JSONDecodeError, TypeError):
                        pd = {}
                ts = timestamp if isinstance(timestamp, datetime) else current_time
                for pattern in (pd.get('suspicious_patterns') or []):
                    self.suspicious_pattern_hits[ip].append({
                        'pattern': pattern,
                        'timestamp': ts,
                    })

    def _detect_brute_force(self) -> List[Dict[str, Any]]:
        """Detect brute force attacks."""
        threats = []
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=self.brute_force_window)
        
        for ip, attempts in self.failed_attempts.items():
            # Filter attempts within time window
            recent_attempts = [t for t in attempts if t >= window_start]
            
            if len(recent_attempts) >= self.brute_force_threshold:
                threat = {
                    'threat_type': 'brute_force',
                    'source_ip': ip,
                    'risk_score': self._calculate_brute_force_score(len(recent_attempts)),
                    'description': f'Brute force attack detected: {len(recent_attempts)} failed attempts in {self.brute_force_window} seconds',
                    'details': {
                        'attempt_count': len(recent_attempts),
                        'time_window': self.brute_force_window,
                        'attempts': [t.isoformat() for t in recent_attempts]
                    }
                }
                threats.append(threat)
        
        return threats

    def _detect_port_scans(self) -> List[Dict[str, Any]]:
        """Detect port scanning activities."""
        threats = []
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=self.scan_detection_window)
        
        for ip, scan_data in self.port_scan_attempts.items():
            # Filter scans within time window
            recent_scans = [(t, p) for t, p in scan_data if t >= window_start]
            
            if len(recent_scans) >= 10:  # Threshold for port scan detection
                unique_ports = len(set(p for _, p in recent_scans))
                
                if unique_ports >= 5:  # Minimum unique ports to consider it a scan
                    threat = {
                        'threat_type': 'port_scan',
                        'source_ip': ip,
                        'risk_score': self._calculate_port_scan_score(len(recent_scans), unique_ports),
                        'description': f'Port scan detected: {len(recent_scans)} connection attempts to {unique_ports} different ports',
                        'details': {
                            'total_attempts': len(recent_scans),
                            'unique_ports': unique_ports,
                            'time_window': self.scan_detection_window,
                            'ports_scanned': list(set(p for _, p in recent_scans))
                        }
                    }
                    threats.append(threat)
        
        return threats

    def _detect_suspicious_patterns(self) -> List[Dict[str, Any]]:
        """Detect suspicious patterns using a rolling window per IP (works with 1-line tail batches)."""
        threats = []
        cutoff = datetime.utcnow() - timedelta(seconds=self._pattern_window_sec)
        now_m = time.monotonic()

        for ip in list(self.suspicious_pattern_hits.keys()):
            matches = [m for m in self.suspicious_pattern_hits[ip] if m['timestamp'] >= cutoff]
            self.suspicious_pattern_hits[ip] = matches
            if not matches:
                del self.suspicious_pattern_hits[ip]
                continue
            if len(matches) < 3:
                continue
            last_emit = self._suspicious_emit_cooldown.get(ip, 0)
            if now_m - last_emit < 300:
                continue
            self._suspicious_emit_cooldown[ip] = now_m

            pattern_counts = Counter(m['pattern'] for m in matches)
            threat = {
                'threat_type': 'suspicious_pattern',
                'source_ip': ip,
                'risk_score': self._calculate_suspicious_pattern_score(len(matches), pattern_counts),
                'description': f'Suspicious patterns detected: {dict(pattern_counts)}',
                'details': {
                    'total_matches': len(matches),
                    'pattern_breakdown': dict(pattern_counts),
                    'recent_matches': [m['timestamp'].isoformat() for m in matches[-5:]]
                }
            }
            threats.append(threat)

        return threats

    def _detect_anomalies(self, log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalous behavior using machine learning."""
        if not ML_AVAILABLE:
            return []

        self._anomaly_rolling.extend(log_entries)
        if len(self._anomaly_rolling) > self._max_anomaly_buffer:
            self._anomaly_rolling = self._anomaly_rolling[-self._max_anomaly_buffer:]

        buf = self._anomaly_rolling
        if len(buf) < 10:
            return []

        threats = []

        # Prepare features for anomaly detection
        features = self._extract_features(buf)
        
        if len(features) < 5:
            return []
        
        # Train or update the model
        if not self.model_trained:
            try:
                scaled_features = self.scaler.fit_transform(features)
                self.anomaly_detector.fit(scaled_features)
                self.model_trained = True
            except Exception as e:
                print(f"Error training anomaly detection model: {e}")
                return []
        
        # Detect anomalies
        try:
            scaled_features = self.scaler.transform(features)
            predictions = self.anomaly_detector.predict(scaled_features)
            scores = self.anomaly_detector.decision_function(scaled_features)
            
            # Group anomalies by IP
            ip_anomalies = defaultdict(list)
            for i, (entry, prediction, score) in enumerate(zip(buf, predictions, scores)):
                if prediction == -1:  # Anomaly detected
                    ip = entry.get('source_ip', '')
                    ip_anomalies[ip].append({
                        'score': score,
                        'timestamp': entry.get('timestamp'),
                        'entry': entry
                    })
            
            # Create threats for IPs with multiple anomalies
            for ip, anomalies in ip_anomalies.items():
                if len(anomalies) >= 2:
                    avg_score = np.mean([a['score'] for a in anomalies])
                    
                    threat = {
                        'threat_type': 'anomaly',
                        'source_ip': ip,
                        'risk_score': self._calculate_anomaly_score(avg_score, len(anomalies)),
                        'description': f'Anomalous behavior detected: {len(anomalies)} anomalous events',
                        'details': {
                            'anomaly_count': len(anomalies),
                            'average_anomaly_score': float(avg_score),
                            'recent_anomalies': [a['timestamp'].isoformat() for a in anomalies[-3:]]
                        }
                    }
                    threats.append(threat)
        
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
        
        return threats

    def _extract_features(self, log_entries: List[Dict[str, Any]]) -> List[List[float]]:
        """Extract numerical features for machine learning."""
        features = []
        
        for entry in log_entries:
            feature_vector = []
            
            # Time-based features
            timestamp = entry.get('timestamp', datetime.utcnow())
            feature_vector.extend([
                timestamp.hour,
                timestamp.minute,
                timestamp.weekday()
            ])
            
            # Port-based features
            port = entry.get('destination_port', 0)
            feature_vector.append(port if port else 0)
            
            # Protocol features (one-hot encoded)
            protocol = entry.get('protocol', '').lower()
            feature_vector.extend([
                1 if protocol == 'tcp' else 0,
                1 if protocol == 'udp' else 0,
                1 if protocol == 'icmp' else 0
            ])
            
            # Status/action features
            action = entry.get('action', '').lower()
            feature_vector.append(1 if action == 'drop' or action == 'reject' else 0)
            
            # Suspicious pattern count
            parsed_data = entry.get('parsed_data', {})
            suspicious_count = len(parsed_data.get('suspicious_patterns', []))
            feature_vector.append(suspicious_count)
            
            features.append(feature_vector)
        
        return features

    def _is_failed_attempt(self, entry: Dict[str, Any]) -> bool:
        """Check if log entry represents a failed attempt."""
        action = entry.get('action', '').lower()
        parsed_data = entry.get('parsed_data', {})
        status = parsed_data.get('status', '')
        
        # Check various failure indicators
        failure_indicators = [
            action in ['drop', 'reject', 'failed'],
            'failed' in entry.get('raw_log', '').lower(),
            status.startswith('4') or status.startswith('5'),  # HTTP error codes
            'authentication failure' in entry.get('raw_log', '').lower()
        ]
        
        return any(failure_indicators)

    def _is_port_scan_attempt(self, entry: Dict[str, Any]) -> bool:
        """Check if log entry represents a port scan attempt."""
        return (
            entry.get('destination_port') is not None and
            entry.get('protocol') in ['TCP', 'UDP'] and
            entry.get('action', '').lower() in ['drop', 'reject']
        )

    def _calculate_brute_force_score(self, attempt_count: int) -> float:
        """Calculate risk score for brute force attacks."""
        base_score = min(attempt_count * 2.0, 10.0)
        return round(base_score, 1)

    def _calculate_port_scan_score(self, total_attempts: int, unique_ports: int) -> float:
        """Calculate risk score for port scans."""
        base_score = min((total_attempts * 0.3 + unique_ports * 0.5), 10.0)
        return round(base_score, 1)

    def _calculate_suspicious_pattern_score(self, match_count: int, pattern_counts: Counter) -> float:
        """Calculate risk score for suspicious patterns."""
        # Higher weight for more dangerous patterns
        pattern_weights = {
            'sql_injection': 3.0,
            'xss': 2.5,
            'command_injection': 3.0,
            'path_traversal': 2.0,
            'brute_force_indicators': 1.5
        }
        
        total_score = 0
        for pattern, count in pattern_counts.items():
            weight = pattern_weights.get(pattern, 1.0)
            total_score += count * weight
        
        return round(min(total_score, 10.0), 1)

    def _calculate_anomaly_score(self, anomaly_score: float, anomaly_count: int) -> float:
        """Calculate risk score for anomalies."""
        # Anomaly scores are negative, so we invert and scale
        base_score = min(abs(anomaly_score) * 5 + anomaly_count * 1.5, 10.0)
        return round(base_score, 1)

    def cleanup_old_data(self):
        """Clean up old tracking data to prevent memory leaks."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=24)
        
        # Clean up failed attempts
        for ip in list(self.failed_attempts.keys()):
            self.failed_attempts[ip] = [t for t in self.failed_attempts[ip] if t >= cutoff_time]
            if not self.failed_attempts[ip]:
                del self.failed_attempts[ip]
        
        # Clean up port scan attempts
        for ip in list(self.port_scan_attempts.keys()):
            self.port_scan_attempts[ip] = [(t, p) for t, p in self.port_scan_attempts[ip] if t >= cutoff_time]
            if not self.port_scan_attempts[ip]:
                del self.port_scan_attempts[ip]
        
        # Clean up traffic patterns (keep only recent data)
        for ip in list(self.traffic_patterns.keys()):
            self.traffic_patterns[ip] = [e for e in self.traffic_patterns[ip] 
                                        if e.get('timestamp', current_time) >= cutoff_time]
            if not self.traffic_patterns[ip]:
                del self.traffic_patterns[ip]

        for ip in list(self.suspicious_pattern_hits.keys()):
            self.suspicious_pattern_hits[ip] = [
                m for m in self.suspicious_pattern_hits[ip] if m['timestamp'] >= cutoff_time
            ]
            if not self.suspicious_pattern_hits[ip]:
                del self.suspicious_pattern_hits[ip]
