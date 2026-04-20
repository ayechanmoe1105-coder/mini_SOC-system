import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# OpenAI is optional - try to import but handle gracefully
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class AIExplainer:
    def __init__(self, config):
        self.config = config
        self.openai_api_key = config.OPENAI_API_KEY
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client if API key is available
        self.client = None
        if self.openai_api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.openai_api_key)
        else:
            if not OPENAI_AVAILABLE:
                self.logger.warning("OpenAI library not installed. AI explanations will be disabled.")
            else:
                self.logger.warning("OpenAI API key not configured. AI explanations will be disabled.")

    def generate_threat_explanation(self, threat: Dict[str, Any], 
                                  context: Dict[str, Any] = None) -> str:
        """Generate AI-powered explanation for detected threat."""
        if not self.client or not OPENAI_AVAILABLE:
            return self._generate_fallback_explanation(threat)
        
        try:
            prompt = self._build_explanation_prompt(threat, context)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a cybersecurity expert AI assistant. 
                        Provide clear, concise explanations of security threats. 
                        Focus on practical insights and actionable recommendations. 
                        Use technical terminology appropriately but explain complex concepts."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            explanation = response.choices[0].message.content.strip()
            self.logger.info(f"Generated AI explanation for threat from {threat.get('source_ip')}")
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating AI explanation: {e}")
            return self._generate_fallback_explanation(threat)

    def _build_explanation_prompt(self, threat: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> str:
        """Build prompt for AI explanation generation."""
        threat_type = threat.get('threat_type', 'unknown')
        source_ip = threat.get('source_ip', 'unknown')
        risk_score = threat.get('risk_score', 0)
        risk_level = threat.get('risk_level', 'UNKNOWN')
        description = threat.get('description', 'No description available')
        details = threat.get('details', {})
        
        prompt = f"""
Analyze this security threat and provide a comprehensive explanation:

THREAT DETAILS:
- Type: {threat_type.replace('_', ' ').title()}
- Source IP: {source_ip}
- Risk Score: {risk_score}/10.0 ({risk_level})
- Description: {description}

ADDITIONAL DETAILS:
"""
        
        # Add specific details based on threat type
        if threat_type == 'brute_force':
            prompt += f"""
- Failed login attempts: {details.get('attempt_count', 'N/A')}
- Time window: {details.get('time_window', 'N/A')} seconds
- Attack pattern: Repeated authentication failures
"""
        elif threat_type == 'port_scan':
            prompt += f"""
- Total connection attempts: {details.get('total_attempts', 'N/A')}
- Unique ports scanned: {details.get('unique_ports', 'N/A')}
- Ports targeted: {', '.join(map(str, details.get('ports_scanned', [])))}
- Attack pattern: Systematic port enumeration
"""
        elif threat_type == 'suspicious_pattern':
            prompt += f"""
- Suspicious patterns detected: {details.get('pattern_breakdown', {})}
- Total matches: {details.get('total_matches', 'N/A')}
- Attack pattern: Potential injection or exploitation attempts
"""
        elif threat_type == 'anomaly':
            prompt += f"""
- Anomaly count: {details.get('anomaly_count', 'N/A')}
- Average anomaly score: {details.get('average_anomaly_score', 'N/A')}
- Attack pattern: Unusual behavior deviating from baseline
"""
        
        # Add context if available
        if context:
            prompt += f"\nCONTEXTUAL INFORMATION:\n"
            if 'historical_threats' in context:
                historical_count = len(context['historical_threats'])
                prompt += f"- Previous threats from this IP: {historical_count}\n"
            
            if 'geo_location' in context:
                prompt += f"- Geographic location: {context['geo_location']}\n"
            
            if 'recent_similar_attacks' in context:
                similar_count = context['recent_similar_attacks']
                prompt += f"- Similar attacks in last 24h: {similar_count}\n"
        
        prompt += """

Please provide:
1. A clear explanation of what this threat means
2. The potential impact on the system
3. Likely attacker motivations or methods
4. Immediate containment recommendations
5. Long-term prevention strategies

Keep the explanation technical but accessible to security professionals.
"""
        
        return prompt

    def _generate_fallback_explanation(self, threat: Dict[str, Any]) -> str:
        """Generate fallback explanation when AI is not available."""
        threat_type = threat.get('threat_type', 'unknown')
        source_ip = threat.get('source_ip', 'unknown')
        risk_score = threat.get('risk_score', 0)
        risk_level = threat.get('risk_level', 'UNKNOWN')
        
        explanations = {
            'brute_force': f"""
Brute Force Attack Detected (Risk: {risk_level})

A brute force attack from {source_ip} has been detected with a risk score of {risk_score}/10. 
This indicates repeated attempts to guess credentials or gain unauthorized access.

Impact: Potential account compromise, unauthorized access to sensitive data.

Recommendations:
- Block the source IP address temporarily
- Implement account lockout policies
- Enable multi-factor authentication
- Monitor for successful login attempts from this IP
""",
            'port_scan': f"""
Port Scanning Activity Detected (Risk: {risk_level})

Port scanning activity from {source_ip} has been detected with a risk score of {risk_score}/10. 
This indicates reconnaissance activity to identify open services and potential vulnerabilities.

Impact: Information gathering phase of potential attack, vulnerability discovery.

Recommendations:
- Block the scanning IP address
- Review firewall rules
- Close unnecessary ports
- Monitor for follow-up attacks targeting discovered services
""",
            'suspicious_pattern': f"""
Suspicious Activity Pattern Detected (Risk: {risk_level})

Suspicious patterns have been detected from {source_ip} with a risk score of {risk_score}/10. 
This may indicate injection attempts, exploitation attempts, or other malicious activities.

Impact: Potential code execution, data theft, or system compromise.

Recommendations:
- Review Web Application Firewall rules
- Analyze the specific patterns for exploitation attempts
- Check for successful breaches or data exfiltration
- Update security controls based on detected patterns
""",
            'anomaly': f"""
Anomalous Behavior Detected (Risk: {risk_level})

Unusual behavior has been detected from {source_ip} with a risk score of {risk_score}/10. 
This activity deviates from normal traffic patterns and may indicate novel attack methods.

Impact: Unknown potential impact, requires investigation.

Recommendations:
- Conduct detailed log analysis
- Correlate with other security events
- Investigate the source and nature of the anomaly
- Consider updating detection rules based on findings
"""
        }
        
        return explanations.get(threat_type, f"""
Unknown Threat Type Detected (Risk: {risk_level})

A threat of type '{threat_type}' has been detected from {source_ip} with a risk score of {risk_score}/10. 
This requires further investigation to determine the nature and impact.

Recommendations:
- Conduct manual log analysis
- Investigate the source IP and activity patterns
- Escalate to security team for detailed analysis
""")

    def generate_incident_summary(self, threats: List[Dict[str, Any]]) -> str:
        """Generate AI-powered summary of multiple threats."""
        if not self.client or not OPENAI_AVAILABLE or not threats:
            return self._generate_fallback_summary(threats)
        
        try:
            prompt = self._build_summary_prompt(threats)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a cybersecurity expert AI assistant. 
                        Provide concise summaries of security incidents. 
                        Focus on key insights, trends, and priority actions."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=600,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            self.logger.info(f"Generated AI summary for {len(threats)} threats")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating AI summary: {e}")
            return self._generate_fallback_summary(threats)

    def _build_summary_prompt(self, threats: List[Dict[str, Any]]) -> str:
        """Build prompt for incident summary generation."""
        if not threats:
            return "No threats to summarize."
        
        # Analyze threats
        threat_types = {}
        risk_levels = {}
        source_ips = {}
        
        for threat in threats:
            # Count by type
            threat_type = threat.get('threat_type', 'unknown')
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
            
            # Count by risk level
            risk_level = threat.get('risk_level', 'UNKNOWN')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            
            # Count by IP
            source_ip = threat.get('source_ip', 'unknown')
            source_ips[source_ip] = source_ips.get(source_ip, 0) + 1
        
        # Get top threats
        top_threats = sorted(threats, key=lambda x: x.get('risk_score', 0), reverse=True)[:5]
        
        prompt = f"""
Analyze this security incident summary and provide insights:

INCIDENT OVERVIEW:
- Total threats detected: {len(threats)}
- Time period: Last 24 hours

THREAT BREAKDOWN:
"""
        
        for threat_type, count in threat_types.items():
            prompt += f"- {threat_type.replace('_', ' ').title()}: {count}\n"
        
        prompt += "\nRISK LEVEL DISTRIBUTION:\n"
        for level, count in risk_levels.items():
            prompt += f"- {level}: {count}\n"
        
        prompt += f"\nTOP SOURCE IPS:\n"
        for ip, count in sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:5]:
            prompt += f"- {ip}: {count} threats\n"
        
        prompt += "\nHIGHEST RISK THREATS:\n"
        for i, threat in enumerate(top_threats, 1):
            prompt += f"{i}. {threat.get('threat_type', 'unknown')} from {threat.get('source_ip', 'unknown')} (Risk: {threat.get('risk_score', 0)})\n"
        
        prompt += """

Please provide:
1. Overall assessment of the security situation
2. Key patterns or trends observed
3. Most critical threats requiring immediate attention
4. Recommended priority actions
5. Potential attack campaign indicators

Keep the summary concise and actionable.
"""
        
        return prompt

    def _generate_fallback_summary(self, threats: List[Dict[str, Any]]) -> str:
        """Generate fallback summary when AI is not available."""
        if not threats:
            return "No threats detected in the specified time period."
        
        # Basic statistics
        threat_types = {}
        risk_levels = {}
        
        for threat in threats:
            threat_type = threat.get('threat_type', 'unknown')
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
            
            risk_level = threat.get('risk_level', 'UNKNOWN')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        summary = f"Security Incident Summary - {len(threats)} threats detected\n\n"
        summary += "Threat Types:\n"
        for threat_type, count in threat_types.items():
            summary += f"- {threat_type.replace('_', ' ').title()}: {count}\n"
        
        summary += "\nRisk Levels:\n"
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = risk_levels.get(level, 0)
            if count > 0:
                summary += f"- {level}: {count}\n"
        
        # Add priority recommendations
        critical_count = risk_levels.get('CRITICAL', 0)
        high_count = risk_levels.get('HIGH', 0)
        
        if critical_count > 0 or high_count > 0:
            summary += f"\nPriority Actions:\n"
            if critical_count > 0:
                summary += f"- Investigate {critical_count} critical threats immediately\n"
            if high_count > 0:
                summary += f"- Review {high_count} high-risk threats for escalation\n"
            summary += "- Consider blocking malicious source IPs\n"
            summary += "- Monitor for follow-up attacks\n"
        
        return summary

    def test_ai_connection(self) -> bool:
        """Test OpenAI API connection."""
        if not self.client or not OPENAI_AVAILABLE:
            return False
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            self.logger.info("OpenAI API connection successful")
            return True
        except Exception as e:
            self.logger.error(f"OpenAI API connection failed: {e}")
            return False
