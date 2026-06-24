import os
import sys
import time
import signal
import socket
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

# Resolve project root path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Get CLK_TCK for CPU calculation
try:
    CLK_TCK = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
except Exception:
    CLK_TCK = 100

class ServiceConfig:
    def __init__(self, name: str, display_name: str, description: str, script_path: str, default_port: Optional[int] = None, env_overrides: Optional[Dict[str, str]] = None):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.script_path = script_path
        self.default_port = default_port
        self.env_overrides = env_overrides or {}

class ServiceProcess:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.status = "STOPPED"  # STOPPED, RUNNING, ERROR, FINISHED
        self.last_error: Optional[str] = None
        
        # CPU tracking state
        self.last_cpu_time = 0.0
        self.last_cpu_measure_time = 0.0

    @property
    def pid(self) -> Optional[int]:
        if self.config.name == "serve":
            import os
            return os.getpid()
        return self.process.pid if self.process else None

    @property
    def uptime(self) -> float:
        if self.status == "RUNNING" and self.start_time:
            return time.time() - self.start_time
        return 0.0

    def to_dict(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "display_name": self.config.display_name,
            "description": self.config.description,
            "status": self.status,
            "pid": self.pid,
            "port": self.config.default_port,
            "uptime_seconds": round(self.uptime, 1),
            "cpu_percent": current_metrics.get("cpu", 0.0),
            "memory_mb": current_metrics.get("memory", 0.0),
            "last_error": self.last_error,
            "script_path": self.config.script_path
        }

class ServiceOrchestrator:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ServiceOrchestrator, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self.services: Dict[str, ServiceProcess] = {}
        
        # Define the 10 microservices
        configs = [
            ServiceConfig(
                name="serve",
                display_name="API Serving Gateway",
                description="FastAPI REST & WebSocket gateway for the dashboard, serving prices, order books, and managing other services.",
                script_path="services/serve/app.py",
                default_port=8362
            ),
            ServiceConfig(
                name="price",
                display_name="Price Ingestion Service",
                description="Subscribes to multiple exchanges, filters stale feeds, aggregates order books ($1M cap), and computes the Rollbit price index every 500ms.",
                script_path="services/price/service.py",
                default_port=8300
            ),
            ServiceConfig(
                name="retrieval",
                display_name="Pattern Retrieval & Forecast",
                description="Indexes historical price windows and uses fast vector similarity search (Annoy) to query current setups and compute consensus forecasts.",
                script_path="services/retrieval/main.py",
                default_port=8000
            ),
            ServiceConfig(
                name="embed",
                display_name="CNN Setup Embeddings API",
                description="Applies a trained Contrastive CNN encoder (SupCon Loss) to map 100-step price returns into 128-dimensional setup embeddings.",
                script_path="services/embed/server.py",
                default_port=8301
            ),
            ServiceConfig(
                name="sentiment",
                display_name="Twitter Sentiment Service",
                description="Ingests real-time Twitter streams, scores sentiment polarity using VADER, and stores aggregate sentiment indicators in MongoDB.",
                script_path="services/sentiment/service_runner.py"
            ),
            ServiceConfig(
                name="jepa",
                display_name="Koopman-JEPA Regime Classifier",
                description="Discovers dynamical market regime transitions using Joint Embedding Predictive Architectures (JEPA) and determines optimal leverage.",
                script_path="services/jepa/trading_integration.py"
            ),
            ServiceConfig(
                name="pressure",
                display_name="Order Book Pressure Service",
                description="Extracts high-frequency liquidity features (Order Flow Imbalance, Cumulative Volume Delta, Bid-Ask Pressure) from active feeds.",
                script_path="services/pressure/pressure_features.py"
            ),
            ServiceConfig(
                name="predict",
                display_name="Deep Learning Forecasting Engine",
                description="Runs TimesNet, Autoformer, and Transformer models to predict next-candle direction and long-term price forecasting signals.",
                script_path="services/predict/service.py"
            ),
            ServiceConfig(
                name="train",
                display_name="Model Training Pipeline",
                description="Triggers and monitors training runs for contrastive encoders, JEPA models, and forecasting networks with live loss plotting.",
                script_path="services/train/main.py"
            ),
            ServiceConfig(
                name="trade",
                display_name="Polymarket Trade Execution Broker",
                description="Mock execution broker that tracks balances, processes forecasting signals, records simulated trades, and calculates portfolio PnL.",
                script_path="services/trade/main.py"
            )
        ]

        for cfg in configs:
            self.services[cfg.name] = ServiceProcess(cfg)

        # The 'serve' service represents the current process, so mark it running
        serve_proc = self.services["serve"]
        serve_proc.status = "RUNNING"
        serve_proc.start_time = time.time()
        
        # Start background monitor thread for dead process detection
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a local port is already occupied."""
        if not port:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(('127.0.0.1', port)) == 0

    def _monitor_loop(self):
        """Periodically checks if running child processes have terminated."""
        while True:
            try:
                for name, s_proc in self.services.items():
                    if name == "serve":
                        continue
                    
                    if s_proc.status == "RUNNING" and s_proc.process:
                        poll_result = s_proc.process.poll()
                        if poll_result is not None:
                            # Process terminated
                            s_proc.stop_time = time.time()
                            if poll_result == 0:
                                s_proc.status = "FINISHED"
                            else:
                                s_proc.status = "ERROR"
                                s_proc.last_error = f"Process exited with non-zero code: {poll_result}"
            except Exception:
                pass
            time.sleep(1.0)

    def _get_process_metrics(self, pid: int, s_proc: ServiceProcess) -> Dict[str, float]:
        """Get CPU and memory metrics for a PID on Linux using /proc."""
        metrics = {"cpu": 0.0, "memory": 0.0}
        if not pid:
            return metrics
        
        # Verify process still exists
        try:
            os.kill(pid, 0)
        except OSError:
            return metrics

        # 1. Memory measurement (Resident Set Size via /proc/{pid}/statm or status)
        try:
            statm_path = f"/proc/{pid}/statm"
            if os.path.exists(statm_path):
                with open(statm_path, "r") as f:
                    fields = f.read().split()
                    if len(fields) >= 2:
                        # Index 1 is resident memory in pages. Multiply by page size.
                        pages = int(fields[1])
                        page_size = os.sysconf(os.sysconf_names['SC_PAGESIZE'])
                        metrics["memory"] = round((pages * page_size) / (1024 * 1024), 2)  # MB
        except Exception:
            pass

        # 2. CPU measurement (utime + stime via /proc/{pid}/stat)
        try:
            stat_path = f"/proc/{pid}/stat"
            if os.path.exists(stat_path):
                with open(stat_path, "r") as f:
                    content = f.read()
                    # stat values are space-separated, but command can contain spaces, e.g., (python3 -m)
                    # Find the last closing parenthesis to skip the comm field safely
                    rpar = content.rfind(')')
                    if rpar != -1:
                        fields = content[rpar+2:].split()
                        # Index 11 is utime (14th field in stat), index 12 is stime (15th field in stat)
                        utime = int(fields[11])
                        stime = int(fields[12])
                        total_jiffies = utime + stime
                        total_time = total_jiffies / CLK_TCK
                        
                        now = time.time()
                        if s_proc.last_cpu_measure_time > 0:
                            time_diff = now - s_proc.last_cpu_measure_time
                            cpu_diff = total_time - s_proc.last_cpu_time
                            if time_diff > 0.05:
                                # CPU percentage = (time_in_process / real_time_passed) * 100
                                # Cap it at 100 * CPU cores. For simplicity, return raw calculation.
                                metrics["cpu"] = round((cpu_diff / time_diff) * 100.0, 1)
                        
                        s_proc.last_cpu_time = total_time
                        s_proc.last_cpu_measure_time = now
        except Exception:
            pass

        return metrics

    def get_services_status(self) -> List[Dict[str, Any]]:
        """Return status and resources for all 10 services."""
        result = []
        for name, s_proc in self.services.items():
            metrics = {"cpu": 0.0, "memory": 0.0}
            if name == "serve":
                # Measures the current FastAPI server process
                metrics = self._get_process_metrics(os.getpid(), s_proc)
            elif s_proc.status == "RUNNING" and s_proc.pid:
                metrics = self._get_process_metrics(s_proc.pid, s_proc)
                
            result.append(s_proc.to_dict(metrics))
        return result

    def start_service(self, name: str) -> Dict[str, Any]:
        """Launch a service in the background."""
        if name not in self.services:
            return {"status": "error", "message": f"Unknown service: {name}"}
        
        s_proc = self.services[name]
        if name == "serve":
            return {"status": "error", "message": "API Serving Gateway is always running and cannot be started."}
        
        if s_proc.status == "RUNNING":
            return {"status": "already_running", "message": f"Service {name} is already running."}

        # Check if default port is occupied
        if s_proc.config.default_port and self._is_port_in_use(s_proc.config.default_port):
            s_proc.status = "ERROR"
            s_proc.last_error = f"Port {s_proc.config.default_port} is already in use by another process."
            return {"status": "error", "message": s_proc.last_error}

        # Prepare script command
        script_full_path = PROJECT_ROOT / s_proc.config.script_path
        if not script_full_path.exists():
            # If the script doesn't exist, create it if it's the trade mock
            if name == "trade":
                self._create_mock_trade_service()
            else:
                s_proc.status = "ERROR"
                s_proc.last_error = f"Script not found at: {script_full_path}"
                return {"status": "error", "message": s_proc.last_error}

        # Run command: use current python executable to run the script
        cmd = [sys.executable, str(script_full_path)]
        
        # Setup log file
        log_file_path = LOGS_DIR / f"{name}.log"
        # Overwrite log on start, or append? Let's overwrite for clean sessions but we can append.
        try:
            log_file = open(log_file_path, "w", buffering=1)
        except Exception as e:
            s_proc.status = "ERROR"
            s_proc.last_error = f"Failed to create log file: {str(e)}"
            return {"status": "error", "message": s_proc.last_error}

        # Setup environment variables
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Set PORT if applicable
        if s_proc.config.default_port:
            env["PORT"] = str(s_proc.config.default_port)
            # Embed server requires specific port configuration or overrides
            if name == "embed":
                env["PORT"] = "8301"  # Force port 8301 to prevent conflict with retrieval's 8000
        
        # Apply configurations from any service-specific .env or configs if provided
        for k, v in s_proc.config.env_overrides.items():
            env[k] = v

        try:
            # Spawn background process
            s_proc.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None # Run in a separate session so signals don't propagate
            )
            s_proc.start_time = time.time()
            s_proc.stop_time = None
            s_proc.status = "RUNNING"
            s_proc.last_error = None
            
            # Reset CPU measurement variables
            s_proc.last_cpu_time = 0.0
            s_proc.last_cpu_measure_time = 0.0
            
            # Close the log file reference in the main thread (Popen keeps it open in child)
            log_file.close()
            
            return {"status": "success", "message": f"Service {name} started successfully.", "pid": s_proc.pid}
        except Exception as e:
            s_proc.status = "ERROR"
            s_proc.last_error = f"Failed to spawn process: {str(e)}"
            return {"status": "error", "message": s_proc.last_error}

    def stop_service(self, name: str) -> Dict[str, Any]:
        """Stop a running service process gracefully using SIGINT or SIGTERM."""
        if name not in self.services:
            return {"status": "error", "message": f"Unknown service: {name}"}
        
        s_proc = self.services[name]
        if name == "serve":
            return {"status": "error", "message": "API Serving Gateway cannot be stopped via dashboard."}
        
        if s_proc.status != "RUNNING" or not s_proc.process:
            return {"status": "not_running", "message": f"Service {name} is not currently running."}

        try:
            # Try graceful shutdown via SIGINT first, then SIGTERM
            pid = s_proc.process.pid
            
            # In Linux, we can kill the entire process group using -pgid
            if os.name != 'nt':
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGINT)
                except Exception:
                    os.kill(pid, signal.SIGINT)
            else:
                s_proc.process.send_signal(signal.CTRL_C_EVENT)
            
            # Wait up to 3 seconds for graceful exit
            for _ in range(30):
                if s_proc.process.poll() is not None:
                    break
                time.sleep(0.1)
                
            # Force kill if still running
            if s_proc.process.poll() is None:
                if os.name != 'nt':
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        s_proc.process.kill()
                else:
                    s_proc.process.kill()
                s_proc.process.wait()
                s_proc.last_error = "Force killed (timed out during graceful shutdown)"
            
            s_proc.stop_time = time.time()
            s_proc.status = "STOPPED"
            s_proc.process = None
            return {"status": "success", "message": f"Service {name} stopped successfully."}
        except Exception as e:
            s_proc.status = "ERROR"
            s_proc.last_error = f"Error stopping service: {str(e)}"
            return {"status": "error", "message": s_proc.last_error}

    def restart_service(self, name: str) -> Dict[str, Any]:
        """Cycle a service (stop, then start)."""
        stop_res = self.stop_service(name)
        # Even if stop fails because it's not running, we try to start it
        time.sleep(0.5)
        start_res = self.start_service(name)
        return {
            "status": start_res["status"],
            "message": f"Restarted service {name}: {start_res.get('message', '')}"
        }

    def get_service_logs(self, name: str, limit: int = 100) -> List[str]:
        """Read the last N lines of a service's log file."""
        log_file_path = LOGS_DIR / f"{name}.log"
        
        # If running price logger locally, there is also record_output.log in src/ or logs/
        if not log_file_path.exists():
            # Check if there is an alternative log file
            alt_path = PROJECT_ROOT / "src" / "record_output.log"
            if name == "price" and alt_path.exists():
                log_file_path = alt_path
            else:
                return [f"[System] No logs found for service '{name}' yet. Start the service to write logs."]

        try:
            # Memory-efficient last N lines reader
            lines = []
            with open(log_file_path, "rb") as f:
                # Seek to end
                f.seek(0, os.SEEK_END)
                size = f.tell()
                
                # Read chunks backwards
                block_size = 4096
                data = b""
                pos = size
                
                while pos > 0 and len(data.split(b'\n')) <= limit + 1:
                    to_read = min(block_size, pos)
                    pos -= to_read
                    f.seek(pos, os.SEEK_SET)
                    data = f.read(to_read) + data
                
                # Split and decode
                decoded_lines = data.decode('utf-8', errors='ignore').split('\n')
                lines = decoded_lines[-limit-1:]
                # Remove empty last line
                if lines and not lines[-1]:
                    lines.pop()
            return lines
        except Exception as e:
            return [f"[Orchestrator Error] Failed to read logs: {str(e)}"]

    def update_service_config(self, name: str, config: Dict[str, str]) -> Dict[str, Any]:
        """Update service-specific environment configurations."""
        if name not in self.services:
            return {"status": "error", "message": f"Unknown service: {name}"}
        
        s_proc = self.services[name]
        s_proc.config.env_overrides.update(config)
        
        # Write to a service-specific .env or config if running
        return {
            "status": "success", 
            "message": f"Configuration for {name} updated. Restart the service to apply changes.",
            "current_config": s_proc.config.env_overrides
        }

    def _create_mock_trade_service(self):
        """Generates the mock Polymarket trade service script if it does not exist."""
        trade_dir = PROJECT_ROOT / "services" / "trade"
        trade_dir.mkdir(parents=True, exist_ok=True)
        
        script_content = """import os
import sys
import time
import json
import random
import datetime as dt
from datetime import datetime

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mock_polymarket_broker")

class MockPolymarketBroker:
    def __init__(self):
        self.usd_balance = 10000.0
        self.btc_position = 0.0
        self.eth_position = 0.0
        self.running = True
        
        self.portfolio_history = []
        self.trades = []
        
        # Load or set up mock price sources
        self.btc_price = 60000.0
        self.eth_price = 3500.0

    def run(self):
        logger.info("Mock Polymarket Trade Broker started.")
        logger.info(f"Initial USD Balance: ${self.usd_balance:,.2f}")
        
        last_log_time = time.time()
        
        while self.running:
            try:
                time.sleep(2.0)
                
                # Mock random price movements
                self.btc_price += random.uniform(-150, 150)
                self.eth_price += random.uniform(-10, 10)
                
                # Calculate total PnL
                portfolio_value = self.usd_balance + (self.btc_position * self.btc_price) + (self.eth_position * self.eth_price)
                
                # Random trade generation based on simulated signals
                if random.random() < 0.15:
                    self.execute_mock_trade()
                    
                # Periodic status report
                if time.time() - last_log_time > 10.0:
                    logger.info(f"PORTFOLIO STATUS | Value: ${portfolio_value:,.2f} | Cash: ${self.usd_balance:,.2f} | BTC: {self.btc_position:.4f} (@${self.btc_price:,.1f}) | ETH: {self.eth_position:.4f} (@${self.eth_price:,.1f})")
                    last_log_time = time.time()
                    
            except KeyboardInterrupt:
                logger.info("Graceful shutdown received.")
                break
            except Exception as e:
                logger.error(f"Broker error: {e}")

    def execute_mock_trade(self):
        assets = ["BTC", "ETH"]
        asset = random.choice(assets)
        price = self.btc_price if asset == "BTC" else self.eth_price
        side = "BUY" if random.random() > 0.4 else "SELL"
        
        if side == "BUY":
            # Spend 5-15% of cash
            spend = self.usd_balance * random.uniform(0.05, 0.15)
            if spend > 10.0:
                amount = spend / price
                self.usd_balance -= spend
                if asset == "BTC":
                    self.btc_position += amount
                else:
                    self.eth_position += amount
                
                trade = {
                    "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                    "side": "BUY",
                    "asset": asset,
                    "amount": round(amount, 4),
                    "price": round(price, 2),
                    "total": round(spend, 2)
                }
                self.trades.append(trade)
                logger.info(f"TRADE EXECUTED | BUY {amount:.4f} {asset} @ ${price:,.2f} (Total: ${spend:,.2f})")
        else:
            # Sell 20-50% of position
            pos = self.btc_position if asset == "BTC" else self.eth_position
            if pos > 0.001:
                amount = pos * random.uniform(0.20, 0.50)
                revenue = amount * price
                self.usd_balance += revenue
                if asset == "BTC":
                    self.btc_position -= amount
                else:
                    self.eth_position -= amount
                
                trade = {
                    "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                    "side": "SELL",
                    "asset": asset,
                    "amount": round(amount, 4),
                    "price": round(price, 2),
                    "total": round(revenue, 2)
                }
                self.trades.append(trade)
                logger.info(f"TRADE EXECUTED | SELL {amount:.4f} {asset} @ ${price:,.2f} (Total: ${revenue:,.2f})")

if __name__ == "__main__":
    broker = MockPolymarketBroker()
    broker.run()
"""
        script_path = trade_dir / "main.py"
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
