

import os
import sys
import shutil
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models import db, LogEntry, Threat, Alert, SystemStats
from log_parser import LogParser
from threat_detector import ThreatDetector
from risk_scorer import RiskScorer
from alert_system import AlertSystem
from ai_explainer import AIExplainer
from flask import Flask

class SecuritySystemTester:
    def __init__(self):
        self.config = Config()
        self.app = Flask(__name__)
        self.app.config.from_object(self.config)
        db.init_app(self.app)
        
        # Initialize components
        self.log_parser = LogParser()
        self.threat_detector = ThreatDetector(self.config)
        self.risk_scorer = RiskScorer(self.config)
        self.alert_system = AlertSystem(self.config)
        self.ai_explainer = AIExplainer(self.config)

    def setup_test_environment(self):
        """Set up test environment with sample data."""
        print("🔧 Setting up test environment...")
        
        with self.app.app_context():
            # Drop all tables and recreate them
            db.drop_all()
            db.create_all()
            
            # Create logs directory
            logs_dir = self.config.LOG_DIRECTORY
            if os.path.exists(logs_dir):
                shutil.rmtree(logs_dir)
            os.makedirs(logs_dir)
            
            # Copy sample logs to logs directory
            sample_logs_dir = os.path.join(os.path.dirname(__file__), 'sample_logs')
            if os.path.exists(sample_logs_dir):
                for filename in os.listdir(sample_logs_dir):
                    src_path = os.path.join(sample_logs_dir, filename)
                    dst_path = os.path.join(logs_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    print(f"  ✓ Copied {filename} to logs directory")
            
            print("✅ Test environment setup complete")

    def test_log_parsing(self):
        """Test log parsing functionality."""
        print("\n📄 Testing log parsing...")
        
        logs_dir = self.config.LOG_DIRECTORY
        total_parsed = 0
        
        with self.app.app_context():
            for filename in os.listdir(logs_dir):
                if filename.endswith(('.log', '.txt')):
                    file_path = os.path.join(logs_dir, filename)
                    print(f"  Parsing {filename}...")
                    
                    parsed_logs = self.log_parser.parse_log_file(file_path)
                    
                    for parsed_log in parsed_logs:
                        normalized = self.log_parser.normalize_log_entry(parsed_log)
                        
                        # Store in database
                        log_entry = LogEntry(**normalized)
                        db.session.add(log_entry)
                        total_parsed += 1
                    
                    print(f"    ✓ Parsed {len(parsed_logs)} entries from {filename}")
            
            db.session.commit()
            print(f"✅ Log parsing complete. Total entries stored: {total_parsed}")

    def test_threat_detection(self):
        """Test threat detection algorithms."""
        print("\n🔍 Testing threat detection...")
        
        with self.app.app_context():
            # Get recent log entries
            recent_logs = LogEntry.query.all()
            
            if not recent_logs:
                print("  ❌ No log entries found for threat detection")
                return
            
            # Convert to dict format
            log_dicts = []
            for log_entry in recent_logs:
                log_dict = {
                    'id': log_entry.id,
                    'timestamp': log_entry.timestamp,
                    'source_ip': log_entry.source_ip,
                    'destination_port': log_entry.destination_port,
                    'protocol': log_entry.protocol,
                    'action': log_entry.action,
                    'raw_log': log_entry.raw_log,
                    'parsed_data': log_entry.parsed_data
                }
                log_dicts.append(log_dict)
            
            # Detect threats
            detected_threats = self.threat_detector.detect_threats(log_dicts)
            
            print(f"  ✓ Detected {len(detected_threats)} potential threats")
            
            # Process and store threats
            for threat_data in detected_threats:
                # Get historical data for risk scoring
                historical_threats = Threat.query.filter(
                    Threat.source_ip == threat_data['source_ip']
                ).all()
                
                historical_dicts = []
                for t in historical_threats:
                    historical_dicts.append({
                        'id': t.id,
                        'timestamp': t.timestamp,
                        'threat_type': t.threat_type,
                        'source_ip': t.source_ip,
                        'risk_score': t.risk_score
                    })
                
                # Calculate comprehensive risk score
                threat_data = self.risk_scorer.update_threat_with_risk_score(
                    threat_data, historical_dicts
                )
                
                # Generate AI explanation if available
                if self.config.OPENAI_API_KEY:
                    context = {
                        'historical_threats': historical_dicts,
                        'recent_similar_attacks': len([
                            t for t in historical_dicts 
                            if t.get('threat_type') == threat_data['threat_type']
                        ])
                    }
                    threat_data['ai_explanation'] = self.ai_explainer.generate_threat_explanation(
                        threat_data, context
                    )
                
                # Create threat record
                threat = Threat(
                    threat_type=threat_data['threat_type'],
                    source_ip=threat_data['source_ip'],
                    risk_score=threat_data['risk_score'],
                    description=threat_data['description'],
                    details=threat_data['details'],
                    ai_explanation=threat_data.get('ai_explanation')
                )
                
                db.session.add(threat)
                db.session.flush()
                
                # Link to relevant log entries based on threat type
                linked_log_ids = set()
                for log_entry in recent_logs:
                    if log_entry.source_ip == threat_data['source_ip']:
                        # Check if this log entry is relevant to this threat type
                        is_relevant = False
                        
                        if threat_data['threat_type'] == 'brute_force':
                            # Link failed login attempts
                            if (log_entry.action in ['failed', 'FAILED'] or 
                                'failed' in log_entry.raw_log.lower()):
                                is_relevant = True
                        elif threat_data['threat_type'] == 'port_scan':
                            # Link port scan attempts
                            if (log_entry.destination_port and 
                                log_entry.action in ['drop', 'DROP']):
                                is_relevant = True
                        elif threat_data['threat_type'] == 'suspicious_pattern':
                            # Link entries with suspicious patterns
                            parsed_data = log_entry.parsed_data or {}
                            if parsed_data.get('suspicious_patterns'):
                                is_relevant = True
                        elif threat_data['threat_type'] == 'anomaly':
                            # Link a few recent entries for anomalies
                            is_relevant = True
                        
                        if is_relevant and log_entry.id not in linked_log_ids:
                            threat.log_entries.append(log_entry)
                            linked_log_ids.add(log_entry.id)
                            # Limit to 5 entries per threat to avoid overcrowding
                            if len(linked_log_ids) >= 5:
                                break
                
                print(f"    ✓ {threat_data['threat_type']} from {threat_data['source_ip']} - Risk: {threat_data['risk_score']}")
            
            db.session.commit()
            print(f"✅ Threat detection complete. {len(detected_threats)} threats stored")

    def test_alert_system(self):
        """Test alert system functionality."""
        print("\n📢 Testing alert system...")
        
        with self.app.app_context():
            # Get high-risk threats
            high_risk_threats = Threat.query.filter(
                Threat.risk_score >= 6.0
            ).all()
            
            if not high_risk_threats:
                print("  ℹ️  No high-risk threats found for alert testing")
                return
            
            alerts_sent = 0
            
            for threat in high_risk_threats:
                threat_data = {
                    'id': threat.id,
                    'threat_type': threat.threat_type,
                    'source_ip': threat.source_ip,
                    'risk_score': threat.risk_score,
                    'risk_level': self.risk_scorer._get_risk_level(threat.risk_score),
                    'description': threat.description,
                    'timestamp': threat.timestamp
                }
                
                if self.alert_system.should_send_alert(threat_data):
                    # Test alert (don't actually send unless configured)
                    if self.alert_system._is_telegram_configured():
                        success = self.alert_system.send_alert(threat_data)
                        if success:
                            alerts_sent += 1
                            print(f"    ✓ Alert sent for {threat.threat_type} from {threat.source_ip}")
                        else:
                            print(f"    ❌ Failed to send alert for {threat.threat_type} from {threat.source_ip}")
                    else:
                        print(f"    ℹ️  Would send alert for {threat.threat_type} from {threat.source_ip} (Telegram not configured)")
                        alerts_sent += 1
            
            print(f"✅ Alert system test complete. {alerts_sent} alerts processed")

    def test_ai_explanations(self):
        """Test AI explanation generation."""
        print("\n🤖 Testing AI explanations...")
        
        if not self.config.OPENAI_API_KEY:
            print("  ℹ️  OpenAI API key not configured. Skipping AI test.")
            return
        
        with self.app.app_context():
            # Get a sample threat
            sample_threat = Threat.query.first()
            
            if not sample_threat:
                print("  ❌ No threats found for AI explanation testing")
                return
            
            # Test AI connection
            ai_connected = self.ai_explainer.test_ai_connection()
            if ai_connected:
                print("  ✓ AI connection successful")
                
                # Test explanation generation
                threat_data = {
                    'id': sample_threat.id,
                    'threat_type': sample_threat.threat_type,
                    'source_ip': sample_threat.source_ip,
                    'risk_score': sample_threat.risk_score,
                    'description': sample_threat.description,
                    'details': sample_threat.details,
                    'timestamp': sample_threat.timestamp
                }
                
                explanation = self.ai_explainer.generate_threat_explanation(threat_data)
                
                if explanation and len(explanation) > 50:
                    print(f"  ✓ AI explanation generated for {sample_threat.threat_type}")
                    print(f"    Explanation preview: {explanation[:100]}...")
                else:
                    print("  ❌ AI explanation generation failed")
            else:
                print("  ❌ AI connection failed")

    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📊 Generating test report...")
        
        with self.app.app_context():
            total_logs = LogEntry.query.count()
            total_threats = Threat.query.count()
            active_threats = Threat.query.filter_by(status='active').count()
            
            # Threat breakdown by type
            threats_by_type = db.session.query(
                Threat.threat_type, 
                db.func.count(Threat.id)
            ).group_by(Threat.threat_type).all()
            
            # Risk level distribution
            risk_levels = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for threat in Threat.query.all():
                level = self.risk_scorer._get_risk_level(threat.risk_score)
                risk_levels[level] += 1
            
            print("\n" + "="*50)
            print("🔐 SECURITY MONITORING SYSTEM TEST REPORT")
            print("="*50)
            print(f"📄 Total Log Entries Processed: {total_logs}")
            print(f"⚠️  Total Threats Detected: {total_threats}")
            print(f"🔥 Active Threats: {active_threats}")
            
            print("\n📈 Threats by Type:")
            for threat_type, count in threats_by_type:
                print(f"  • {threat_type.replace('_', ' ').title()}: {count}")
            
            print("\n🎯 Risk Level Distribution:")
            for level, count in risk_levels.items():
                if count > 0:
                    emoji = {'CRITICAL': '🚨', 'HIGH': '⚠️', 'MEDIUM': '⚡', 'LOW': 'ℹ️'}[level]
                    print(f"  {emoji} {level}: {count}")
            
            # Top source IPs
            source_ips = db.session.query(
                Threat.source_ip,
                db.func.count(Threat.id)
            ).group_by(Threat.source_ip).order_by(
                db.func.count(Threat.id).desc()
            ).limit(5).all()
            
            print("\n🌐 Top Source IPs:")
            for ip, count in source_ips:
                print(f"  • {ip}: {count} threats")
            
            print("\n✅ System Test Complete!")
            print("="*50)

    def run_all_tests(self):
        """Run complete test suite."""
        print("🚀 Starting AI-Assisted Security Monitoring System Tests")
        print("="*60)
        
        try:
            self.setup_test_environment()
            self.test_log_parsing()
            self.test_threat_detection()
            self.test_alert_system()
            self.test_ai_explanations()
            self.generate_test_report()
            
            print("\n🎉 All tests completed successfully!")
            print("\nNext steps:")
            print("1. Configure your .env file with API keys")
            print("2. Run 'py working_app.py' to start the monitoring system")
            print("3. Open http://127.0.0.1:15500 to view the dashboard")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    tester = SecuritySystemTester()
    tester.run_all_tests()
