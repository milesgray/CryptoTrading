import asyncio
import json
from cryptotrading.data.postgres import get_connection, init_pool

async def main():
    await init_pool()
    async with get_connection() as conn:
        token = "ETH"
        # 1. Get matching symbols using TimescaleDB SkipScan which is extremely fast
        symbol_rows = await conn.fetch(
            "SELECT DISTINCT symbol FROM price_data WHERE symbol = $1 OR symbol LIKE $2",
            token, f"{token}/%"
        )
        matching_symbols = [r["symbol"] for r in symbol_rows]
        
        if not matching_symbols:
            print("No matching symbols found")
            return
            
        # 2. Get list of raw exchanges
        exchanges_rows = await conn.fetch(
            "SELECT DISTINCT exchange FROM price_data WHERE exchange LIKE 'exchange_raw_%';"
        )
        exchanges = [r["exchange"] for r in exchanges_rows]
        if not exchanges:
            print("No raw exchanges found")
            return
            
        # 3. Construct a fast UNION ALL query to fetch the latest row for each exchange
        queries = []
        for i, exchange in enumerate(exchanges):
            queries.append(f"""
                (SELECT exchange, time, close, metadata
                 FROM price_data
                 WHERE symbol = ANY($1) AND exchange = ${i+2}
                 ORDER BY time DESC
                 LIMIT 1)
            """)
        
        union_query = " UNION ALL ".join(queries)
        rows = await conn.fetch(union_query, matching_symbols, *exchanges)
        for r in rows:
            print(r["exchange"], r["time"], r["close"], r["metadata"].get("spread"))

if __name__ == "__main__":
    asyncio.run(main())
