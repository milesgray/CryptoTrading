import asyncio
import logging
from cryptotrading.rollbit.prices.price import PriceSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('price_system')

async def main():
    """Main function to run the price system"""
    price_system = PriceSystem()
    
    try:
        await price_system.initialize()
        await price_system.run()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        await price_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())