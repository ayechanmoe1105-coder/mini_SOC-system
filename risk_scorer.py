from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# GeoIP is optional - try to import but handle gracefully
try:
    import geoip2.database
    import geoip2.errors
    import os
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False

class RiskScorer:
    def __init__(self, config):
        self.config = config
        self.risk_weights = config.RISK_WEIGHTS
        self.geoip_reader = None
        
        # Initialize GeoIP database if available
        if GEOIP_AVAILABLE:
            geoip_path = os.path.join(os.path.dirname(__file__), 'GeoLite2-Country.mmdb')
            if os.path.exists(geoip_path):
                try:
                    self.geoip_reader = geoip2.database.Reader(geoip_path)
                except Exception as e:
                    logging.warning(f"Could not load GeoIP database: {e}")
                    self.geoip_reader = None
            else:
                self.geoip_reader = None
        else:
            self.geoip_reader = None

    def calculate_comprehensive_risk_score(self, threat: Dict[str, Any], 
                                         historical_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate comprehensive risk score with multiple factors."""
        
        base_score = threat.get('risk_score', 5.0)
        threat_type = threat.get('threat_type', '')
        source_ip = threat.get('source_ip', '')
        
        # Initialize risk factors
        risk_factors = {
            'base_threat_score': base_score,
            'threat_history': 0.0,
            'geo_location': 0.0,
            'time_pattern': 0.0,
            'frequency': 0.0,
            'severity_multiplier': 1.0
        }
        
        # Calculate each risk factor
        risk_factors['threat_history'] = self._calculate_threat_history_score(source_ip, historical_data)
        risk_factors['geo_location'] = self._calculate_geo_risk_score(source_ip)
        risk_factors['time_pattern'] = self._calculate_time_pattern_score(threat)
        risk_factors['frequency'] = self._calculate_frequency_score(threat, historical_data)
        risk_factors['severity_multiplier'] = self._get_severity_multiplier(threat_type)
        
        # Calculate final weighted score
        final_score = self._calculate_weighted_score(risk_factors)
        final_score_rounded = round(final_score, 1)
        risk_level = self._get_risk_level(final_score)
        
        # Generate recommendations using the calculated score
        recommendations = self._generate_recommendations(threat, final_score_rounded, risk_level)
        
        return {
            'final_score': final_score_rounded,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }

    def _calculate_threat_history_score(self, source_ip: str, 
                                       historical_data: List[Dict[str, Any]]) -> float:
        """Calculate risk score based on threat history for this IP."""
        if not historical_data:
            return 0.0
        
        # Count previous threats from this IP
        previous_threats = [t for t in historical_data if t.get('source_ip') == source_ip]
        
        if not previous_threats:
            return 0.0
        
        # Calculate recency and frequency scores
        current_time = datetime.utcnow()
        recent_threats = []
        old_threats = []
        
        for threat in previous_threats:
            threat_time = threat.get('timestamp', current_time)
            if isinstance(threat_time, str):
                threat_time = datetime.fromisoformat(threat_time.replace('Z', '+00:00'))
            
            days_diff = (current_time - threat_time).days
            if days_diff <= 7:
                recent_threats.append(threat)
            elif days_diff <= 30:
                old_threats.append(threat)
        
        # Score based on recency and frequency
        score = 0.0
        score += len(recent_threats) * 2.0  # Recent threats weigh more
        score += len(old_threats) * 0.5     # Older threats weigh less
        
        # Bonus for multiple threat types
        threat_types = set(t.get('threat_type', '') for t in previous_threats)
        if len(threat_types) > 1:
            score += len(threat_types) * 1.5
        
        return min(score, 10.0)

    def _calculate_geo_risk_score(self, source_ip: str) -> float:
        """Calculate risk score based on geographical location."""
        if not self.geoip_reader:
            return 0.0
        
        try:
            response = self.geoip_reader.country(source_ip)
            country = response.country.iso_code
            
            # Country risk levels (simplified)
            high_risk_countries = {'CN', 'RU', 'KP', 'IR'}
            medium_risk_countries = {'PK', 'BD', 'NG', 'UA'}
            
            if country in high_risk_countries:
                return 3.0
            elif country in medium_risk_countries:
                return 1.5
            else:
                return 0.0
                
        except (geoip2.errors.AddressNotFoundError, ValueError):
            # Unknown location - moderate risk
            return 1.0
        except Exception as e:
            print(f"Error in GeoIP lookup: {e}")
            return 0.0

    def _calculate_time_pattern_score(self, threat: Dict[str, Any]) -> float:
        """Calculate risk score based on time patterns."""
        threat_time = threat.get('timestamp', datetime.utcnow())
        if isinstance(threat_time, str):
            threat_time = datetime.fromisoformat(threat_time.replace('Z', '+00:00'))
        
        hour = threat_time.hour
        weekday = threat_time.weekday()
        
        score = 0.0
        
        # Off-hours attacks (night time in most timezones)
        if hour >= 22 or hour <= 6:
            score += 1.5
        
        # Weekend attacks
        if weekday >= 5:  # Saturday, Sunday
            score += 1.0
        
        # Business hours in suspicious timezones (simplified)
        # This would need more sophisticated logic for real implementation
        
        return score

    def _calculate_frequency_score(self, threat: Dict[str, Any], 
                                 historical_data: List[Dict[str, Any]]) -> float:
        """Calculate risk score based on attack frequency."""
        if not historical_data:
            return 0.0
        
        source_ip = threat.get('source_ip', '')
        threat_type = threat.get('threat_type', '')
        
        # Count similar threats in last 24 hours
        current_time = datetime.utcnow()
        yesterday = current_time - timedelta(hours=24)
        
        recent_similar_threats = [
            t for t in historical_data 
            if (t.get('source_ip') == source_ip and 
                t.get('threat_type') == threat_type and
                t.get('timestamp', current_time) >= yesterday)
        ]
        
        threat_count = len(recent_similar_threats)
        
        if threat_count == 0:
            return 0.0
        elif threat_count <= 2:
            return 1.0
        elif threat_count <= 5:
            return 2.5
        elif threat_count <= 10:
            return 4.0
        else:
            return 6.0

    def _get_severity_multiplier(self, threat_type: str) -> float:
        """Get severity multiplier based on threat type."""
        multipliers = {
            'brute_force': 1.2,
            'port_scan': 1.0,
            'suspicious_pattern': 1.5,
            'anomaly': 1.3,
            'malware': 2.0,
            'ddos': 2.5,
            'data_exfiltration': 3.0
        }
        
        return multipliers.get(threat_type, 1.0)

    def _calculate_weighted_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate final weighted risk score.
        Base threat score carries 60% of the final score so that a high
        base score (e.g. 9.5) always results in at least a HIGH rating
        even when contextual factors (history, geo, frequency) are absent.
        """
        weights = self.risk_weights

        weighted_score = (
            risk_factors['base_threat_score'] * 0.60 +   # Primary driver
            risk_factors['threat_history'] * weights.get('brute_force', 0.15) +
            risk_factors['geo_location'] * weights.get('geo_location', 0.08) +
            risk_factors['time_pattern'] * 0.10 +
            risk_factors['frequency'] * 0.07
        )

        # Apply severity multiplier
        final_score = weighted_score * risk_factors['severity_multiplier']

        return min(final_score, 10.0)  # Cap at 10.0

    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on score."""
        if score >= 8.0:
            return 'CRITICAL'
        elif score >= 6.0:
            return 'HIGH'
        elif score >= 4.0:
            return 'MEDIUM'
        elif score >= 2.0:
            return 'LOW'
        else:
            return 'INFO'

    def _generate_recommendations(self, threat: Dict[str, Any], 
                                final_score: float, risk_level: str) -> List[str]:
        """Generate security recommendations based on threat and risk factors."""
        recommendations = []
        threat_type = threat.get('threat_type', '')
        
        # Base recommendations by threat type
        if threat_type == 'brute_force':
            recommendations.extend([
                'Block IP address after multiple failed attempts',
                'Implement rate limiting on authentication endpoints',
                'Consider implementing CAPTCHA',
                'Review password policies'
            ])
        elif threat_type == 'port_scan':
            recommendations.extend([
                'Block scanning IP address',
                'Review firewall rules',
                'Implement port knocking for sensitive services',
                'Monitor for follow-up attacks'
            ])
        elif threat_type == 'suspicious_pattern':
            recommendations.extend([
                'Review Web Application Firewall (WAF) rules',
                'Implement input validation and sanitization',
                'Conduct security code review',
                'Monitor for data exfiltration'
            ])
        elif threat_type == 'anomaly':
            recommendations.extend([
                'Investigate unusual traffic patterns',
                'Review system logs for related activities',
                'Consider implementing behavioral analysis',
                'Monitor for lateral movement'
            ])
        
        # Additional recommendations based on risk score
        if final_score >= 8.0:
            recommendations.extend([
                'Immediate incident response required',
                'Escalate to security team',
                'Consider temporary service disruption'
            ])
        
        return recommendations

    def update_threat_with_risk_score(self, threat: Dict[str, Any], 
                                    historical_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update threat object with comprehensive risk assessment."""
        risk_assessment = self.calculate_comprehensive_risk_score(threat, historical_data)
        
        threat['risk_score'] = risk_assessment['final_score']
        threat['risk_level'] = risk_assessment['risk_level']
        threat['risk_factors'] = risk_assessment['risk_factors']
        threat['recommendations'] = risk_assessment['recommendations']
        
        return threat

    def __del__(self):
        """Cleanup GeoIP reader."""
        if self.geoip_reader:
            self.geoip_reader.close()
