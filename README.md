# 🔐 AI-Assisted Security Monitoring & Threat Detection System

A comprehensive security monitoring system that analyzes server logs, detects threats, assigns risk scores, and provides AI-powered incident explanations. This system simulates a mini Security Operations Center (SOC) with real-time monitoring and alerting capabilities.

## 🎯 Project Features

### Core Functionality
- **Multi-format Log Parsing**: Supports Apache, Nginx, firewall, SSH, and syslog formats
- **Advanced Threat Detection**: 
  - Brute-force attack detection
  - Port scanning identification
  - Suspicious pattern recognition (SQL injection, XSS, path traversal)
  - Anomaly detection using machine learning
- **Intelligent Risk Scoring**: Multi-factor risk assessment with historical analysis
- **Real-time Alerting**: Telegram integration for instant notifications
- **AI-Powered Analysis**: OpenAI integration for intelligent threat explanations
- **Web Dashboard**: Interactive monitoring interface with real-time updates

### Technical Highlights
- **Machine Learning**: Isolation Forest for anomaly detection
- **Geographic Analysis**: GeoIP location-based risk assessment
- **Temporal Analysis**: Time-based threat pattern recognition
- **Database Storage**: SQLAlchemy with comprehensive threat tracking
- **RESTful API**: Full API for integration and monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   # Navigate to the project directory
   cd "d:\Users\Aye Chan\Desktop\Final"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   # Copy the example environment file
   copy .env.example .env
   
   # Edit .env with your configuration
   notepad .env
   ```

4. **Run the test suite**
   ```bash
   python test_system.py
   ```

5. **Start the monitoring system**
   ```bash
   python working_app.py
   ```

6. **Access the dashboard**
   - The server prints the listening URL when it starts. By default it uses **port 15500** (not 5000), e.g. `http://127.0.0.1:15500` or `http://localhost:15500`.
   - Override with `SOC_PORT` in `.env` if you need another port.
   - The dashboard shows real-time security monitoring data.

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Database Configuration
DATABASE_URL=sqlite:///security_monitoring.db

# OpenAI API Key (for AI explanations)
OPENAI_API_KEY=your_openai_api_key_here

# Telegram Bot Configuration (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development

# Dashboard port (default 15500 — avoids clashes with other apps on 5000)
SOC_PORT=15500

# Monitoring Configuration
LOG_DIRECTORY=./logs
ALERT_THRESHOLD=7.0
SCAN_DETECTION_WINDOW=300
BRUTE_FORCE_THRESHOLD=5
BRUTE_FORCE_WINDOW=60
```

### Optional: Telegram Bot Setup

1. **Create a Telegram bot**
   - Message @BotFather on Telegram
   - Use `/newbot` command
   - Get your bot token

2. **Get your chat ID**
   - Message your bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your chat ID in the response

3. **Update .env file**
   - Add your bot token and chat ID

### Optional: OpenAI Setup

1. **Get OpenAI API key**
   - Visit: https://platform.openai.com/api-keys
   - Create a new API key

2. **Add to .env file**
   - Set `OPENAI_API_KEY=your_key_here`

## 📁 Project Structure

```
Final/
├── working_app.py         # Main Flask application (all-in-one)
├── config.py             # Configuration management
├── (models defined inline in working_app.py)
├── log_parser.py         # Log parsing engine
├── threat_detector.py    # Threat detection algorithms
├── risk_scorer.py        # Risk assessment system
├── alert_system.py       # Alert management
├── ai_explainer.py       # AI-powered explanations
├── test_system.py        # Comprehensive test suite
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── README.md            # This file
├── sample_logs/         # Sample log files for testing
│   ├── apache_access.log
│   ├── firewall.log
│   └── ssh_auth.log
├── logs/                # Runtime log directory (created automatically)
├── templates/           # HTML templates (created automatically)
│   └── dashboard.html
└── security_monitoring.db  # SQLite database (created automatically)
```

## 🔍 How It Works

### 1. Log Processing
- System monitors the `./logs` directory for new log files
- Supports multiple log formats with automatic pattern recognition
- Parses and normalizes log entries for analysis

### 2. Threat Detection
- **Brute Force Detection**: Identifies repeated failed login attempts
- **Port Scan Detection**: Detects systematic port enumeration
- **Pattern Matching**: Identifies SQL injection, XSS, and other attack patterns
- **Anomaly Detection**: Uses ML to detect unusual behavior patterns

### 3. Risk Assessment
- **Multi-factor scoring**: Combines threat type, history, location, and timing
- **Historical analysis**: Considers past threats from the same source
- **Geographic risk**: Incorporates location-based threat intelligence
- **Temporal patterns**: Analyzes timing of suspicious activities

### 4. Alert System
- **Real-time notifications**: Sends alerts via Telegram
- **Risk-based filtering**: Only alerts on significant threats
- **Daily summaries**: Provides comprehensive daily reports
- **Configurable thresholds**: Customizable alert sensitivity

### 5. AI Analysis
- **Threat explanations**: Generates detailed analysis of detected threats
- **Contextual insights**: Provides attack motivation and impact assessment
- **Recommendations**: Suggests specific mitigation strategies
- **Incident summaries**: Creates comprehensive incident reports

## 🎛️ Dashboard Features

### Real-time Monitoring
- **Live threat feed**: Shows recent security events
- **Risk level indicators**: Color-coded threat severity
- **Statistics overview**: System-wide security metrics
- **Interactive charts**: Visual threat analysis

### Threat Management
- **Detailed threat views**: Expandable threat information
- **Status management**: Mark threats as resolved or false positives
- **AI explanations**: In-depth threat analysis
- **Log correlation**: View associated log entries

### System Controls
- **Test functions**: Verify alert and AI systems
- **Auto-refresh**: Real-time dashboard updates
- **Responsive design**: Works on desktop and mobile

## 🧪 Testing

The system includes a comprehensive test suite:

```bash
python test_system.py
```

### Test Coverage
- ✅ Log parsing accuracy
- ✅ Threat detection algorithms
- ✅ Risk scoring calculations
- ✅ Alert system functionality
- ✅ AI explanation generation
- ✅ Database operations
- ✅ Dashboard API endpoints

## 📊 Sample Data

The system includes sample log files demonstrating:
- **Brute force attacks**: Multiple failed SSH login attempts
- **Port scanning**: Systematic port enumeration
- **Web attacks**: SQL injection and XSS attempts
- **Suspicious patterns**: Path traversal and command injection

## 🔧 Advanced Configuration

### Custom Detection Rules
Modify `threat_detector.py` to adjust:
- Detection thresholds
- Time windows
- Pattern matching rules
- ML model parameters

### Risk Scoring Weights
Adjust `config.py` to modify:
- Risk factor weights
- Geographic risk levels
- Severity multipliers
- Alert thresholds

### Log Format Support
Extend `log_parser.py` to add:
- New log patterns
- Custom parsing rules
- Additional metadata extraction

## 🌐 API Endpoints

### Threat Management
- `GET /api/threats` - Get recent threats
- `GET /api/threat/<id>` - Get threat details
- `POST /api/threat/<id>/update` - Update threat status

### System Statistics
- `GET /api/stats` - Get system statistics

### Testing
- `GET /api/test/alert` - Send test alert
- `GET /api/test/ai` - Test AI connection

## 🛡️ Security Considerations

### Production Deployment
- Change default Flask secret key
- Use HTTPS in production
- Implement proper authentication
- Regular security updates
- Monitor system logs

### Data Privacy
- Log data retention policies
- Secure database configuration
- API access controls
- Audit trail maintenance

## 🚀 Scaling the System

### Performance Optimization
- Database indexing
- Log rotation
- Caching strategies
- Load balancing

### Enterprise Features
- Multi-tenant support
- Advanced reporting
- Integration with SIEM systems
- Custom alert channels

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive comments
- Include error handling
- Document new features

## 📞 Support

### Troubleshooting
- Check log files for errors
- Verify environment configuration
- Run test suite for diagnostics
- Check API key validity

### Common Issues
- **Database errors**: Ensure proper file permissions
- **AI failures**: Verify OpenAI API key and credits
- **Alert issues**: Check Telegram bot configuration
- **Performance**: Monitor system resources

## 📄 License

This project is for educational purposes. Please ensure compliance with your organization's security policies when deploying in production environments.

---

## 🎓 Learning Outcomes

This project demonstrates advanced cybersecurity concepts:
- **Log Analysis**: Pattern recognition and parsing
- **Threat Intelligence**: Attack detection and classification
- **Machine Learning**: Anomaly detection algorithms
- **Risk Assessment**: Multi-factor security scoring
- **System Integration**: Real-time monitoring and alerting
- **AI Applications**: Natural language processing for security

Perfect for understanding modern security operations and building practical cybersecurity skills!

---

**Built with ❤️ for cybersecurity education**
