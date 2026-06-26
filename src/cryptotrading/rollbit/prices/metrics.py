"""
This module calculates multi-exchange meta-statistics from order book data 
collected across different cryptocurrency exchanges. These metrics help identify 
price fragmentation, cross-exchange arbitrage opportunities, and liquidity concentration.
"""

import math
from typing import List, Dict, Tuple, Optional

def calculate_multi_exchange_metrics(
    order_books: List[Dict],
    index_price: Optional[float] = None,
    fee_rate: float = 0.0015,
    hhi_threshold_pct: float = 0.01
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
        fee_rate: The trading fee rate applied to both legs of the arbitrage trade (default 0.15% / 0.0015).
        hhi_threshold_pct: Percentage threshold from mid-price to define the order depth used for
                           calculating HHI concentration (default 1% / 0.01).

    Returns:
        A dictionary containing calculated metrics:
        - 'price_dispersion': Standard deviation of mid-prices across exchanges.
        - 'average_bid_ask_spread': Average spread of all exchanges.
        - 'max_arbitrage_spread': Maximum net crossed-book spread across exchanges (net_spread > 0).
        - 'arbitrage_opportunities': List of potential cross-exchange arbitrage trades (net of fees).
        - 'liquidity_concentration_hhi': Herfindahl-Hirschman Index of order book depth within threshold (0 to 10000).
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
            
            # Calculate depth within the percentage threshold of the mid price
            near_bid_depth = sum(size for price, size in bids if (mid - price) / mid <= hhi_threshold_pct)
            near_ask_depth = sum(size for price, size in asks if (price - mid) / mid <= hhi_threshold_pct)
            threshold_depth = near_bid_depth + near_ask_depth

            valid_books.append({
                'exchange': exchange,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid': mid,
                'spread': spread,
                'threshold_depth': threshold_depth
            })

            mid_prices.append(mid)
            spreads.append(spread)
            depths.append(threshold_depth)
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

    # 3. Global Arbitrage Spread (Cross-Exchange Crossed Books Net of Fees)
    max_net_arb = 0.0
    arb_opps = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # If exchange i's bid is higher than exchange j's ask, check for arbitrage
            bid_i = valid_books[i]['best_bid']
            ask_j = valid_books[j]['best_ask']
            if bid_i > ask_j:
                raw_spread = bid_i - ask_j
                fee_cost = (ask_j * fee_rate) + (bid_i * fee_rate)
                net_spread = raw_spread - fee_cost
                
                # Only include as opportunity if it is profitable net of fees
                if net_spread > 0:
                    if net_spread > max_net_arb:
                        max_net_arb = net_spread
                    arb_opps.append({
                        'buy_exchange': valid_books[j]['exchange'],
                        'sell_exchange': valid_books[i]['exchange'],
                        'raw_spread': raw_spread,
                        'spread': net_spread,  # Represent profitable net spread
                        'fee_cost': fee_cost,
                        'buy_price': ask_j,
                        'sell_price': bid_i
                    })
    
    # Sort arbitrage opportunities by net spread descending
    arb_opps.sort(key=lambda x: x['spread'], reverse=True)
    metrics['max_arbitrage_spread'] = max_net_arb
    metrics['arbitrage_opportunities'] = arb_opps

    # 4. Liquidity Concentration Herfindahl-Hirschman Index (HHI) within threshold
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
