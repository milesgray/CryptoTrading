#!/usr/bin/env python3
"""
Service Runner for Crypto Sentiment Analyzer

Runs the sentiment analyzer as a containerized service with:
- Configuration management
- Scheduled analysis runs
- Health monitoring API
- Graceful shutdown handling
- Error recovery and retry logic
"""

import os
import sys
import signal
import threading
import time
import logging
import yaml
from datetime import datetime, timezone
import schedule
import psutil
from flask import Flask, jsonify

# Import our analyzer
from cryptotrading.sentiment.analyzer import CryptoSentimentAnalyzer

class SentimentAnalyzerService:
    """Service wrapper for the crypto sentiment analyzer"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.analyzer = None
        self.running = True
        self.last_run_status = {}
        self.health_status = {
            'status': 'starting',
            'last_health_check': None,
            'uptime_seconds': 0,
            'total_tweets_processed': 0,
            'errors_count': 0,
            'last_error': None
        }
        self.start_time = time.time()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger('sentiment_service')
        
        # Setup Flask app for health monitoring
        self.app = Flask(__name__)
        self.app.logger.disabled = True
        logging.getLogger('werkzeug').disabled = True
        
        self._setup_health_endpoints()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info("Sentiment Analyzer Service initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file or environment variables"""
        default_config = {
            'tokens': ['BTC', 'ETH'],  # Default tokens to analyze
            'analysis_interval_minutes': 30,
            'analysis_duration_minutes': 60,
            'max_following_users': 100,
            'mongodb': {
                'uri': os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'),
                'database': os.getenv('MONGODB_DATABASE', 'crypto_sentiment')
            },
            'twitter': {
                'bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
                'api_key': os.getenv('TWITTER_API_KEY'),
                'api_secret': os.getenv('TWITTER_API_SECRET'),
                'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
                'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            },
            'service': {
                'health_port': int(os.getenv('HEALTH_PORT', '8080')),
                'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                'max_retries': int(os.getenv('MAX_RETRIES', '3')),
                'retry_delay_seconds': int(os.getenv('RETRY_DELAY', '60'))
            }
        }
        
        # Try to load from YAML file if it exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with defaults
                    self._deep_merge(default_config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
        
        # Override with environment variables if they exist
        env_overrides = {
            'tokens': os.getenv('ANALYSIS_TOKENS', '').split(',') if os.getenv('ANALYSIS_TOKENS') else None,
            'analysis_interval_minutes': int(os.getenv('ANALYSIS_INTERVAL', '0')) or None,
            'analysis_duration_minutes': int(os.getenv('ANALYSIS_DURATION', '0')) or None,
            'max_following_users': int(os.getenv('MAX_FOLLOWING', '0')) or None
        }
        
        for key, value in env_overrides.items():
            if value is not None:
                default_config[key] = value
        
        return default_config
    
    def _deep_merge(self, base: dict, override: dict) -> None:
        """Deep merge override dict into base dict"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config['service']['log_level']
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('/app/logs/sentiment_service.log', mode='a')
            ]
        )
        
        # Create logs directory
        os.makedirs('/app/logs', exist_ok=True)
    
    def _setup_health_endpoints(self):
        """Setup Flask health monitoring endpoints"""
        
        @self.app.route('/health')
        def health_check():
            self.health_status['last_health_check'] = datetime.now(timezone.utc).isoformat()
            self.health_status['uptime_seconds'] = int(time.time() - self.start_time)
            
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return jsonify({
                'status': self.health_status['status'],
                'uptime_seconds': self.health_status['uptime_seconds'],
                'last_health_check': self.health_status['last_health_check'],
                'tweets_processed': self.health_status['total_tweets_processed'],
                'errors_count': self.health_status['errors_count'],
                'last_error': self.health_status['last_error'],
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3)
                },
                'last_run_status': self.last_run_status,
                'config': {
                    'tokens': self.config['tokens'],
                    'analysis_interval': self.config['analysis_interval_minutes']
                }
            })
        
        @self.app.route('/status')
        def status():
            return jsonify({
                'service': 'crypto-sentiment-analyzer',
                'version': '1.0.0',
                'status': self.health_status['status'],
                'running': self.running
            })
        
        @self.app.route('/metrics')
        def metrics():
            """Prometheus-style metrics endpoint"""
            metrics_text = f"""# HELP tweets_processed_total Total number of tweets processed
# TYPE tweets_processed_total counter
tweets_processed_total {self.health_status['total_tweets_processed']}

# HELP errors_total Total number of errors encountered
# TYPE errors_total counter
errors_total {self.health_status['errors_count']}

# HELP uptime_seconds Service uptime in seconds
# TYPE uptime_seconds gauge
uptime_seconds {int(time.time() - self.start_time)}

# HELP analysis_runs_total Total number of analysis runs
# TYPE analysis_runs_total counter
analysis_runs_total {len(self.last_run_status)}
"""
            return metrics_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
        @self.app.route('/config')
        def get_config():
            # Return config without sensitive information
            safe_config = self.config.copy()
            if 'twitter' in safe_config:
                for key in safe_config['twitter']:
                    if safe_config['twitter'][key]:
                        safe_config['twitter'][key] = '***HIDDEN***'
            return jsonify(safe_config)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def initialize_analyzer(self) -> bool:
        """Initialize the sentiment analyzer with retry logic"""
        for attempt in range(self.config['service']['max_retries']):
            try:
                self.analyzer = CryptoSentimentAnalyzer(
                    mongo_uri=self.config['mongodb']['uri'],
                    db_name=self.config['mongodb']['database']
                )
                self.logger.info("Sentiment analyzer initialized successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize analyzer (attempt {attempt + 1}): {e}")
                if attempt < self.config['service']['max_retries'] - 1:
                    time.sleep(self.config['service']['retry_delay_seconds'])
                else:
                    self.health_status['status'] = 'failed'
                    self.health_status['last_error'] = str(e)
                    self.health_status['errors_count'] += 1
        
        return False
    
    def run_scheduled_analysis(self):
        """Run analysis for all configured tokens"""
        if not self.running:
            return
        
        self.logger.info("Starting scheduled analysis run")
        run_start_time = time.time()
        
        for token in self.config['tokens']:
            if not self.running:
                break
                
            try:
                self.logger.info(f"Starting analysis for {token}")
                
                # Run analysis
                self.analyzer.run_analysis(
                    token_symbol=token,
                    duration_minutes=self.config['analysis_duration_minutes'],
                    max_following=self.config['max_following_users']
                )
                
                # Get aggregated results
                sentiment_data = self.analyzer.get_aggregated_sentiment(token, hours_back=1)
                
                # Update status
                self.last_run_status[token] = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'completed',
                    'tweets_analyzed': sentiment_data.get('total_tweets', 0),
                    'avg_sentiment': sentiment_data.get('avg_compound', 0),
                    'bullish_signal': sentiment_data.get('avg_bullish_signal', 0),
                    'bearish_signal': sentiment_data.get('avg_bearish_signal', 0)
                }
                
                self.health_status['total_tweets_processed'] += sentiment_data.get('total_tweets', 0)
                self.logger.info(f"Completed analysis for {token}: {sentiment_data.get('total_tweets', 0)} tweets")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {token}: {e}")
                self.health_status['errors_count'] += 1
                self.health_status['last_error'] = f"{token}: {str(e)}"
                
                self.last_run_status[token] = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'failed',
                    'error': str(e)
                }
        
        run_duration = time.time() - run_start_time
        self.logger.info(f"Analysis run completed in {run_duration:.2f} seconds")
        
        # Update health status
        if self.health_status['status'] != 'failed':
            self.health_status['status'] = 'healthy'
    
    def start_health_server(self):
        """Start the health monitoring server in a separate thread"""
        def run_server():
            try:
                self.app.run(
                    host='0.0.0.0',
                    port=self.config['service']['health_port'],
                    debug=False,
                    use_reloader=False
                )
            except Exception as e:
                self.logger.error(f"Health server error: {e}")
        
        health_thread = threading.Thread(target=run_server, daemon=True)
        health_thread.start()
        self.logger.info(f"Health server started on port {self.config['service']['health_port']}")
    
    def run(self):
        """Main service loop"""
        self.logger.info("Starting Crypto Sentiment Analyzer Service")
        
        # Initialize analyzer
        if not self.initialize_analyzer():
            self.logger.error("Failed to initialize analyzer, exiting")
            return 1
        
        # Start health monitoring server
        self.start_health_server()
        
        # Schedule analysis runs
        interval = self.config['analysis_interval_minutes']
        schedule.every(interval).minutes.do(self.run_scheduled_analysis)
        
        self.logger.info(f"Analysis scheduled every {interval} minutes for tokens: {self.config['tokens']}")
        
        # Run initial analysis
        self.run_scheduled_analysis()
        
        # Main service loop/
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            self.health_status['status'] = 'error'
            self.health_status['last_error'] = str(e)
            self.health_status['errors_count'] += 1
            return 1
        
        finally:
            self.logger.info("Shutting down service...")
            if self.analyzer:
                self.analyzer.cleanup()
            self.logger.info("Service shutdown complete")
        
        return 0

def main():
    """Main entry point"""
    service = SentimentAnalyzerService()
    return service.run()

if __name__ == "__main__":
    sys.exit(main())