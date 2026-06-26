"""
This module calculates multi-exchange meta-statistics from order book data 
collected across different cryptocurrency exchanges. These metrics help identify 
price fragmentation, cross-exchange arbitrage opportunities, and liquidity concentration.
"""

import math
from typing import List, Dict, Tuple, Optional

def calculate_multi_exchange_metrics(
    order_books: List[Dict],
    index_price: Optional[float] = None
) -> Dict[str, any]:
    """
    Calculate cross-exchange meta-statistics from a list of active order books.

    Args:
        order_books: A list of dictionaries representing exchange order books.
                     Each dictionary must have:
                     - 'exchange': (str) Identifier of the exchange (falls back to index if missing)
                     - 'bids': list of (price, size) tuples
                     - 'asks': list of (price, size) tuples
        index_price: The calculated composite index price. If provided, calculates deviation metrics.

    Returns:
        A dictionary containing calculated metrics:
        - 'price_dispersion': Standard deviation of mid-prices across exchanges.
        - 'average_bid_ask_spread': Average spread of all exchanges.
        - 'max_arbitrage_spread': Maximum crossed-book spread across exchanges (bid_A - ask_B > 0).
        - 'arbitrage_opportunities': List of potential cross-exchange arbitrage trades.
        - 'liquidity_concentration_hhi': Herfindahl-Hirschman Index of order book depth (0 to 10000).
        - 'index_deviation_std': Std dev of deviations of exchange mid-prices from the index price.
        - 'index_deviation_mad': Mean absolute deviation of exchange mid-prices from the index price.
    """
    metrics = {
        'price_dispersion': 0.0,
        'average_bid_ask_spread': 0.0,
        'max_arbitrage_spread': 0.0,
        'arbitrage_opportunities': [],
        'liquidity_concentration_hhi': 0.0,
        'index_deviation_std': 0.0,
        'index_deviation_mad': 0.0,
        'num_exchanges': len(order_books)
    }

    # Filter out empty or invalid books
    valid_books = []
    mid_prices = []
    spreads = []
    depths = []
    exchanges = []

    for idx, book in enumerate(order_books):
        bids = book.get('bids', [])
        asks = book.get('asks', [])
        exchange = book.get('exchange', f"exchange_{idx}")

        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            # Avoid crossed books within the same exchange
            if best_bid >= best_ask:
                continue

            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            total_depth = sum(b[1] for b in bids) + sum(a[1] for a in asks)

            valid_books.append({
                'exchange': exchange,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid': mid,
                'spread': spread,
                'total_depth': total_depth
            })

            mid_prices.append(mid)
            spreads.append(spread)
            depths.append(total_depth)
            exchanges.append(exchange)

    n = len(valid_books)
    if n == 0:
        return metrics

    # 1. Price Dispersion (Standard Deviation of Mid Prices)
    mean_mid = sum(mid_prices) / n
    var_mid = sum((x - mean_mid) ** 2 for x in mid_prices) / n
    metrics['price_dispersion'] = math.sqrt(var_mid)

    # 2. Average Bid-Ask Spread
    metrics['average_bid_ask_spread'] = sum(spreads) / n

    # 3. Global Arbitrage Spread (Cross-Exchange Crossed Books)
    max_arb = 0.0
    arb_opps = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # If exchange i's bid is higher than exchange j's ask, it is a cross-exchange arbitrage
            bid_i = valid_books[i]['best_bid']
            ask_j = valid_books[j]['best_ask']
            if bid_i > ask_j:
                spread_diff = bid_i - ask_j
                if spread_diff > max_arb:
                    max_arb = spread_diff
                arb_opps.append({
                    'buy_exchange': valid_books[j]['exchange'],
                    'sell_exchange': valid_books[i]['exchange'],
                    'spread': spread_diff,
                    'buy_price': ask_j,
                    'sell_price': bid_i
                })
    
    # Sort arbitrage opportunities by spread descending
    arb_opps.sort(key=lambda x: x['spread'], reverse=True)
    metrics['max_arbitrage_spread'] = max_arb
    metrics['arbitrage_opportunities'] = arb_opps

    # 4. Liquidity Concentration Herfindahl-Hirschman Index (HHI)
    # HHI is calculated by summing the squares of the percentage market share of each firm.
    total_market_depth = sum(depths)
    if total_market_depth > 0:
        hhi = 0.0
        for depth in depths:
            pct_share = (depth / total_market_depth) * 100.0
            hhi += pct_share ** 2
        metrics['liquidity_concentration_hhi'] = hhi

    # 5. Index Deviation metrics (if index_price is provided)
    if index_price is not None and index_price > 0:
        deviations = [mid - index_price for mid in mid_prices]
        metrics['index_deviation_std'] = math.sqrt(sum(d ** 2 for d in deviations) / n)
        metrics['index_deviation_mad'] = sum(abs(d) for d in deviations) / n

    return metrics
