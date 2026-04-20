import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///security_monitoring.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    
    # Monitoring Configuration
    LOG_DIRECTORY = os.environ.get('LOG_DIRECTORY') or './logs'
    # When False (default): only real log files you place under ./logs are processed.
    # Set ENABLE_SYNTHETIC_LOG_GENERATOR=true only for demos (appends fake lines every N seconds).
    ENABLE_SYNTHETIC_LOG_GENERATOR = os.environ.get(
        'ENABLE_SYNTHETIC_LOG_GENERATOR', 'false'
    ).lower() in ('1', 'true', 'yes', 'on')
    SYNTHETIC_LOG_INTERVAL_SECONDS = int(os.environ.get('SYNTHETIC_LOG_INTERVAL_SECONDS', '10'))

    ALERT_THRESHOLD = float(os.environ.get('ALERT_THRESHOLD', 7.0))
    SCAN_DETECTION_WINDOW = int(os.environ.get('SCAN_DETECTION_WINDOW', 300))  # 5 minutes
    BRUTE_FORCE_THRESHOLD = int(os.environ.get('BRUTE_FORCE_THRESHOLD', 5))
    BRUTE_FORCE_WINDOW = int(os.environ.get('BRUTE_FORCE_WINDOW', 60))  # 1 minute
    
    # Risk Scoring Weights
    RISK_WEIGHTS = {
        'brute_force': 0.3,
        'port_scan': 0.25,
        'suspicious_pattern': 0.2,
        'anomaly': 0.15,
        'geo_location': 0.1
    }
