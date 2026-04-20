import re
import json
import ipaddress
from datetime import datetime
from typing import Dict, List, Optional, Any

class LogParser:
    def __init__(self):
        # Common log patterns
        self.patterns = {
            'apache_access': re.compile(
                r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<url>[^"]+)" (?P<status>\d+) (?P<size>\d+)'
            ),
            'nginx_access': re.compile(
                r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<url>[^"]+)" (?P<status>\d+) (?P<size>\d+)'
            ),
            'firewall': re.compile(
                r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+).*?SRC=(?P<ip>\d+\.\d+\.\d+\.\d+).*?PROTO=(?P<protocol>\w+).*?DPT=(?P<port>\d+)'
            ),
            'ssh_auth': re.compile(
                r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+).*?sshd.*?(?P<action>Accepted|Failed) password.*?from (?P<ip>\d+\.\d+\.\d+\.\d+)'
            ),
            'syslog': re.compile(
                r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+).+?(?P<ip>\d+\.\d+\.\d+\.\d+).+?(?P<message>.+)'
            )
        }
        
        # Suspicious patterns
        self.suspicious_patterns = {
            'sql_injection': re.compile(r'(?i)(union|select|insert|update|delete|drop|exec|script)'),
            'xss': re.compile(r'(?i)(<script|javascript:|onload=|onerror=)'),
            'path_traversal': re.compile(r'(\.\./|\.\.\\|%2e%2e%2f)'),
            'command_injection': re.compile(r'(?i)(;|\||&|`|\$\(|\$\{)'),
            'brute_force_indicators': re.compile(r'(?i)(admin|root|test|guest|user|login|password)')
        }

    def parse_log_line(self, log_line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line and return structured data."""
        log_line = log_line.strip()
        if not log_line:
            return None
        
        parsed = None
        
        # Try each pattern
        for log_type, pattern in self.patterns.items():
            match = pattern.search(log_line)
            if match:
                parsed = match.groupdict()
                parsed['log_type'] = log_type
                parsed['raw_log'] = log_line
                parsed['timestamp'] = self._parse_timestamp(parsed.get('timestamp', ''))
                break
        
        if not parsed:
            # Fallback: try to extract IP and basic info
            ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', log_line)
            if ip_match:
                parsed = {
                    'ip': ip_match.group(1),
                    'log_type': 'unknown',
                    'raw_log': log_line,
                    'timestamp': datetime.utcnow()
                }
        
        if parsed:
            parsed['suspicious_patterns'] = self._detect_suspicious_patterns(log_line)
            parsed['is_private_ip'] = self._is_private_ip(parsed.get('ip', ''))
        
        return parsed

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various log formats."""
        formats = [
            '%d/%b/%Y:%H:%M:%S %z',  # Apache/Nginx
            '%b %d %H:%M:%S',        # Syslog
            '%Y-%m-%d %H:%M:%S',     # Common
            '%Y-%m-%dT%H:%M:%S',     # ISO
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return datetime.utcnow()

    def _detect_suspicious_patterns(self, log_line: str) -> List[str]:
        """Detect suspicious patterns in log line."""
        detected = []
        for pattern_name, pattern in self.suspicious_patterns.items():
            if pattern.search(log_line):
                detected.append(pattern_name)
        return detected

    def _is_private_ip(self, ip_str: str) -> bool:
        """Check if IP is private/internal."""
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_private
        except ValueError:
            return False

    def parse_log_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse entire log file."""
        parsed_logs = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    parsed = self.parse_log_line(line)
                    if parsed:
                        parsed['line_number'] = line_num
                        parsed['file_path'] = file_path
                        parsed_logs.append(parsed)
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return parsed_logs

    def normalize_log_entry(self, parsed_log: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parsed log to standard format."""
        normalized = {
            'timestamp': parsed_log.get('timestamp', datetime.utcnow()),
            'source_ip': parsed_log.get('ip', ''),
            'destination_port': parsed_log.get('port'),
            'protocol': parsed_log.get('protocol', ''),
            'action': parsed_log.get('action', ''),
            'raw_log': parsed_log.get('raw_log', ''),
            'parsed_data': {
                'log_type': parsed_log.get('log_type', 'unknown'),
                'method': parsed_log.get('method', ''),
                'url': parsed_log.get('url', ''),
                'status': parsed_log.get('status', ''),
                'size': parsed_log.get('size', ''),
                'suspicious_patterns': parsed_log.get('suspicious_patterns', []),
                'is_private_ip': parsed_log.get('is_private_ip', False),
                'file_path': parsed_log.get('file_path', ''),
                'line_number': parsed_log.get('line_number', '')
            }
        }
        
        return normalized
