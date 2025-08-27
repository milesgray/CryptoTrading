import os
import asyncio
import logging
import signal
import time
from cryptotrading.rollbit.prices.price import PriceSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('price_system')

REFRESH_INTERVAL_MS = int(os.getenv("REFRESH_INTERVAL_MS", 500))

class PriceSystemService:
    def __init__(self):
        self.price_system = PriceSystem()
        self.start_time = time.time()
        self.running = True

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    async def run(self):
        """Main function to run the price system"""
        try:
            await self.price_system.initialize()
            logger.info("Price system initialized")
            while self.running:
                start_time = time.time()
                try:
                    await self.price_system.run()
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                finally:
                    # Calculate sleep time to maintain refresh interval
                    elapsed = (time.time() - start_time) * 1000  # in milliseconds
                    sleep_time = max(0, REFRESH_INTERVAL_MS - elapsed) / 1000  # in seconds
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return 1
        finally:
            await self.price_system.shutdown()
            logger.info("Price system shutdown complete")
            
        return 0

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

def main():
    """Main entry point"""
    service = PriceSystemService()
    return service.run() 

if __name__ == "__main__":
    asyncio.run(main())