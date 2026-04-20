import requests
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

class AlertSystem:
    def __init__(self, config):
        self.config = config
        self.telegram_bot_token = config.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = config.TELEGRAM_CHAT_ID
        self.alert_threshold = config.ALERT_THRESHOLD
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def send_alert(self, threat: Dict[str, Any], alert_type: str = 'telegram') -> bool:
        """Send alert for detected threat."""
        try:
            if alert_type == 'telegram' and self._is_telegram_configured():
                return self._send_telegram_alert(threat)
            else:
                self.logger.warning(f"Alert type {alert_type} not configured")
                return False
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            return False

    def _is_telegram_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def _send_telegram_alert(self, threat: Dict[str, Any]) -> bool:
        """Send alert via Telegram bot."""
        try:
            message = self._format_telegram_message(threat)
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'disable_web_page_preview': True,
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Telegram alert sent for threat from {threat.get('source_ip')}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in Telegram alert: {e}")
            return False

    def _format_telegram_message(self, threat: Dict[str, Any]) -> str:
        """Format threat information for Telegram message."""
        risk_score = threat.get('risk_score', 0)
        risk_level = threat.get('risk_level', 'UNKNOWN')
        threat_type = threat.get('threat_type', 'unknown')
        source_ip = threat.get('source_ip', 'unknown')
        description = threat.get('description', 'No description available')
        timestamp = threat.get('timestamp', datetime.utcnow())
        
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Choose emoji based on risk level
        risk_emojis = {
            'CRITICAL': '🚨',
            'HIGH': '⚠️',
            'MEDIUM': '⚡',
            'LOW': 'ℹ️',
            'INFO': '📋'
        }
        
        emoji = risk_emojis.get(risk_level, '⚠️')
        
        message = f"""
{emoji} SECURITY ALERT {emoji}

Threat Type: {threat_type.replace('_', ' ').title()}
Source IP: {source_ip}
Risk Score: {risk_score}/10.0 ({risk_level})
Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Description:
{description}

"""
        
        # Add additional details if available
        details = threat.get('details', {})
        if details:
            message += "\nAdditional Details:\n"
            
            if 'attempt_count' in details:
                message += f"• Failed attempts: {details['attempt_count']}\n"
            
            if 'unique_ports' in details:
                message += f"• Unique ports scanned: {details['unique_ports']}\n"
            
            if 'pattern_breakdown' in details:
                patterns = details['pattern_breakdown']
                message += f"• Suspicious patterns: {', '.join(patterns.keys())}\n"
            
            if 'total_matches' in details:
                message += f"• Total pattern matches: {details['total_matches']}\n"
        
        # Add recommendations if available
        recommendations = threat.get('recommendations', [])
        if recommendations:
            message += "\nRecommendations:\n"
            for i, rec in enumerate(recommendations[:3], 1):  # Limit to first 3
                message += f"{i}. {rec}\n"
        
        # Add footer
        message += f"\nThreat ID: {threat.get('id', 'N/A')}"
        
        return message.strip()

    def send_summary_alert(self, threats: List[Dict[str, Any]]) -> bool:
        """Send summary alert for multiple threats."""
        if not threats:
            return True
        
        try:
            message = self._format_summary_message(threats)
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'disable_web_page_preview': True,
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Summary alert sent for {len(threats)} threats")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send summary alert: {e}")
            return False

    def _format_summary_message(self, threats: List[Dict[str, Any]]) -> str:
        """Format summary message for multiple threats."""
        if not threats:
            return ""
        
        # Count threats by type and risk level
        threat_types = {}
        risk_levels = {}
        top_ips = {}
        
        for threat in threats:
            # Count by type
            threat_type = threat.get('threat_type', 'unknown')
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
            
            # Count by risk level
            risk_level = threat.get('risk_level', 'UNKNOWN')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            
            # Count by IP
            source_ip = threat.get('source_ip', 'unknown')
            top_ips[source_ip] = top_ips.get(source_ip, 0) + 1
        
        # Get top risk threats
        critical_threats = [t for t in threats if t.get('risk_level') == 'CRITICAL']
        high_threats = [t for t in threats if t.get('risk_level') == 'HIGH']
        
        message = f"""
SECURITY SUMMARY REPORT

Period: Last 24 hours
Total Threats: {len(threats)}

Threat Breakdown:
"""
        
        # Add threat type breakdown
        for threat_type, count in sorted(threat_types.items(), key=lambda x: x[1], reverse=True):
            message += f"• {threat_type.replace('_', ' ').title()}: {count}\n"
        
        message += "\nRisk Level Distribution:\n"
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = risk_levels.get(level, 0)
            if count > 0:
                emoji = {'CRITICAL': '🚨', 'HIGH': '⚠️', 'MEDIUM': '⚡', 'LOW': 'ℹ️'}[level]
                message += f"{emoji} {level}: {count}\n"
        
        # Top source IPs
        message += "\nTop Source IPs:\n"
        for ip, count in sorted(top_ips.items(), key=lambda x: x[1], reverse=True)[:5]:
            message += f"• {ip}: {count} threats\n"
        
        # Critical threats highlight
        if critical_threats:
            message += f"\nCRITICAL THREATS ({len(critical_threats)}):\n"
            for threat in critical_threats[:3]:  # Show top 3
                ip = threat.get('source_ip', 'unknown')
                t_type = threat.get('threat_type', 'unknown')
                message += f"• {t_type} from {ip}\n"
        
        return message.strip()

    def test_telegram_connection(self) -> bool:
        """Test Telegram bot connection."""
        if not self._is_telegram_configured():
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getMe"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            bot_info = response.json()
            if bot_info.get('ok'):
                self.logger.info(f"Telegram bot connection successful: {bot_info['result']['username']}")
                return True
            else:
                self.logger.error("Telegram bot connection failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Telegram connection test failed: {e}")
            return False

    def should_send_alert(self, threat: Dict[str, Any]) -> bool:
        """Determine if alert should be sent based on threshold and other criteria."""
        risk_score = threat.get('risk_score', 0)
        risk_level = threat.get('risk_level', '')
        
        # Always send critical and high alerts
        if risk_level in ['CRITICAL', 'HIGH']:
            return True
        
        # Send based on threshold
        if risk_score >= self.alert_threshold:
            return True
        
        # Send specific threat types even if lower score
        high_priority_types = ['brute_force', 'suspicious_pattern']
        if threat.get('threat_type') in high_priority_types and risk_score >= 5.0:
            return True
        
        return False

    def create_alert_record(self, threat: Dict[str, Any], sent: bool = False) -> Dict[str, Any]:
        """Create alert record for database storage."""
        return {
            'threat_id': threat.get('id'),
            'alert_type': 'telegram',
            'message': self._format_telegram_message(threat),
            'sent': sent,
            'sent_timestamp': datetime.utcnow() if sent else None
        }
