import os
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
