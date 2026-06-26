"""
This module provides functions for calculating Rollbit's index price, 
funding payments, bust/stop loss prices, and market impact effects on closing prices.

Rollbit uses a composite index price derived from multiple cryptocurrency exchanges 
to ensure fair and reliable pricing for its trading games. This module provides
the tools to replicate these calculations.
"""

import math

from cryptotrading.util.status import StatusManager

def calculate_index_price(
        order_books: list[dict[str, list[tuple[float, float]]]], 
        min_valid_feeds: int = 6,
        status: StatusManager = StatusManager('formula.calculate_index_price'),
        return_book: bool = False,
        verbose: bool = False
) -> float:
    """
    Calculates the Rollbit index price based on a list of order books from different exchanges.

    Mathematical Formulation & Steps:
    1. Filter out stale feeds (assumed done upstream).
    2. Remove crossed feeds where `best_bid > best_ask` and calculate mid-prices:
       `mid_price = (best_bid + best_ask) / 2`
    3. Filter out outlier feeds whose mid-prices deviate by more than 10% from the median mid-price:
       `abs(mid_price - median_mid_price) <= 0.1 * median_mid_price`
    4. Ensure at least `min_valid_feeds` (default 6) are active.
    5. Aggregate remaining feeds into a composite order book. Individual order sizes are capped at 
       $1,000,000 to limit the influence of single massive orders:
       `size = min(raw_size, 1000000.0)`
    6. Define marginal buying and selling price functions:
       - `P_buy(s)`: The price at the level where the cumulative bid size first reaches or exceeds `s`.
       - `P_sell(s)`: The price at the level where the cumulative ask size first reaches or exceeds `s`.
       - `P_mid(s) = (P_buy(s) + P_sell(s)) / 2`
    7. Compute cumulative bid and ask size arrays. Let `v_i` be the sorted union of these cumulative sizes.
    8. Define maximum size `V` as `min(sum(bid_sizes), sum(ask_sizes))`.
    9. Calculate the scaling factor `L = 1 / V`.
    10. Compute exponential weights at each threshold `v_i` where `v_i <= V`:
        `w_i = L * exp(-L * v_i)`
    11. Sum the weighted marginal mid-prices and divide by total weights:
        `Index Price = sum(P_mid(v_i) * w_i) / sum(w_i)`

    Args:
        order_books: A list of dictionaries representing exchange order books. Each dict has keys:
                     'bids': list of (price, size) tuples, sorted descending.
                     'asks': list of (price, size) tuples, sorted ascending.
                     Sizes represent quantities in USD.
        min_valid_feeds: Minimum required valid feeds to output a price.
        status: StatusManager for progress and metric logging.
        return_book: If True, returns a dictionary containing the index price, total composite size, 
                     and the composite order book itself.
        verbose: Enable warning logging.

    Returns:
       If return_book is False: tuple (index_price, total_size) or None.
       If return_book is True: dict with keys 'price', 'size', and 'book' or None.

    Raises:
        TypeError: If order_books format or types are incorrect.
        ValueError: If price or size is not positive.
    """
    try:
        # Input type validation
        if not isinstance(order_books, list):
            raise TypeError("order_books must be a list of dictionaries.")
        for book in order_books:
            if not isinstance(book, dict):
                raise TypeError("Each element in order_books must be a dictionary.")
            if not ('bids' in book and 'asks' in book):
                raise TypeError("Each order book must contain 'bids' and 'asks' keys.")
            if not (isinstance(book['bids'], list) and isinstance(book['asks'], list)):
                raise TypeError("'bids' and 'asks' must be lists of [price, size] tuples.")
            for side in ['bids', 'asks']:
                for info in book[side]:
                    if len(info) == 2:
                        price, size = info
                    elif len(info) == 3:
                        price, size, _ = info
                    else:
                        continue
                    if not (isinstance(price, (int, float)) and isinstance(size, (int, float))):
                        raise TypeError("Prices and sizes in order books must be numbers.")
                    if price <= 0 or size <= 0:
                        raise ValueError("Prices and sizes must be positive numbers.")

        # 1. & 2.  Data is assumed to be already streamed and filtered for 30-second staleness.

        # 3. Remove invalid price feeds (crossed or outlier mid-prices).
        valid_order_books = []
        mid_prices = []
        for order_book in order_books:
            if order_book['bids'] and order_book['asks']:
                best_bid = order_book['bids'][0][0]
                best_ask = order_book['asks'][0][0]
                if best_bid > best_ask:  # Crossed book
                    continue
                mid_price = (best_bid + best_ask) / 2
                mid_prices.append(mid_price)
                valid_order_books.append(order_book)

        if len(mid_prices) < min_valid_feeds:
            return None  # 4. Not enough valid price feeds.

        median_mid_price = sorted(mid_prices)[len(mid_prices) // 2]
        filtered_order_books = [
            order_book for order_book, mid_price in zip(valid_order_books, mid_prices)
            if abs(mid_price - median_mid_price) <= 0.1 * median_mid_price
        ]

        if len(filtered_order_books) < min_valid_feeds:
            if verbose: status.warning(f"Not enough valid price feeds after filtering. {len(filtered_order_books)}")
            return None  # 4. Not enough valid price feeds after filtering.


        # 5. Combine order books.
        composite_order_book: dict[str, list[tuple[float, float]]] = {'bids': [], 'asks': []}
        for order_book in filtered_order_books:
            for side in ['bids', 'asks']:
                for info in order_book[side]:
                    if len(info) == 2:
                        price, size = info
                    elif len(info) == 3:
                        price, size, _ = info
                    else:
                        continue
                    size = min(size, 1000000.0)  # Cap individual order sizes.
                    composite_order_book[side].append((price, size))

        # Sort bids by price descending and asks by price ascending.
        composite_order_book['bids'].sort(key=lambda x: x[0], reverse=True)
        composite_order_book['asks'].sort(key=lambda x: x[0])


        # 6. Define marginal price functions.
        def p_buy(s: float) -> float:
            cumulative_size = 0.0
            for price, size in composite_order_book['bids']:
                cumulative_size += size
                if cumulative_size >= s:
                    return price
            return composite_order_book['bids'][-1][0] if composite_order_book['bids'] else 0.0 # Return worst bid if 's' exceeds total bids

        def p_sell(s: float) -> float:
            cumulative_size = 0.0
            for price, size in composite_order_book['asks']:
                cumulative_size += size
                if cumulative_size >= s:
                    return price
            return composite_order_book['asks'][-1][0] if composite_order_book['asks'] else float('inf') #return worst ask

        # 7. Define marginal mid-price function.
        def p_mid(s: float) -> float:
            return (p_buy(s) + p_sell(s)) / 2.0

        # 8. Calculate the weighted average of marginal mid-prices.

        # Calculate cumulative sizes for bids and asks.  These define our v_i.
        cumulative_bid_sizes = []
        cumulative_ask_sizes = []
        cumulative_bid_size = 0.0
        for _, size in composite_order_book['bids']:
            cumulative_bid_size += size
            cumulative_bid_sizes.append(cumulative_bid_size)
        cumulative_ask_size = 0.0
        for _, size in composite_order_book['asks']:
            cumulative_ask_size += size
            cumulative_ask_sizes.append(cumulative_ask_size)

        total_size = cumulative_bid_size + cumulative_ask_size

        # Combine and de-duplicate the cumulative sizes, then sort them.
        v_i = sorted(list(set(cumulative_bid_sizes + cumulative_ask_sizes)))

        # v_i must not be empty
        if not v_i:
            if verbose: status.warning("Empty v_i list.")
            return None

        # Calculate V (maximum size for mid-price calculation).
        V = min(sum(size for _, size in composite_order_book['bids']),
                sum(size for _, size in composite_order_book['asks']))

        if V == 0:  
            if verbose: status.warning("No volume in the order book.")
            return None  # No volume in the order book

        L = 1.0 / V
        
        total_weighted_price = 0.0
        total_weight = 0.0

        for v in v_i:
            if v > V: break  # important: do not calculate with sizes > V.
            weight = L * math.exp(-L * v)
            total_weighted_price += p_mid(v) * weight
            total_weight += weight

        if total_weight == 0.0:
            if verbose: status.warning("Total weight is zero.")
            return None  # Should not happen if V > 0

        index_price = total_weighted_price / total_weight
        
        if return_book:
            status.update_data({
                "price": index_price,
                "size": total_size, 
                "book": composite_order_book      
            })
            return {
                "price": index_price, 
                "size": total_size,
                "book": composite_order_book
            }
        status.update_data({
            "price": index_price,
            "size": total_size,            
        })
        return index_price, total_size
    except Exception as e:
        status.error(f"Error calculating index price: {str(e)}")
        return None


def calculate_funding_payment(
        bet_amount: float, 
        multiplier: float, 
        funding_rate: float, 
        hours_open: float,
        status: StatusManager = StatusManager('formula.calculate_funding_payment')
    ) -> float:
    """
    Calculates the funding payment for a given bet.

    Args:
        bet_amount: The initial amount of the bet (USD).
        multiplier: The bet multiplier.
        funding_rate: The hourly funding rate (e.g., 0.001 for 0.1%).
        hours_open: The number of hours the bet has been open.

    Returns:
        The funding payment amount.  Positive values represent payments
        *received* by the user; negative values represent payments *made*
        by the user.

    Example:
        calculate_funding_payment(500.0, 10.0, 0.001, 10.0)  # Returns -1.05
        calculate_funding_payment(500.0, 10.0, -0.001, 10.0) # Returns 1.05
    """
    if hours_open <= 8.0:
        return 0.0

    notional_value = bet_amount * multiplier
    funding_hours = int(hours_open - 8.0) # Only charge for full hours after the first 8.
    payment =  notional_value * funding_rate * funding_hours
    status.update_data({
        "payment": payment
    })
    return payment


def calculate_bust_or_stop_loss_price(
        p_open: float, 
        bet_multiplier: float, 
        trade_sign: int, 
        bust_buffer: float = 0.0, 
        stop_loss_price:float = None,
        status: StatusManager = StatusManager('formula.calculate_bust_or_stop_loss_price')
    ) -> float:
    """
    Calculates the trigger price for a bust or stop loss.

    Args:
        p_open: The opening price of the bet.
        bet_multiplier: The bet multiplier.
        trade_sign: 1 for long bets, -1 for short bets.
        bust_buffer: A buffer added to the bust price, set based on market conditions.
        stop_loss_price: The user-set stop-loss price. If None, calculate bust price.

    Returns:
        The trigger price for the bust or stop loss.

    Raises:
        ValueError: If trade_sign is not 1 or -1.

    Example:
        calculate_bust_or_stop_loss_price(100.0, 10.0, 1, 0.01)  # Returns 90.9
        calculate_bust_or_stop_loss_price(100.0, 10.0, -1, 0.01) # Returns 109.1
        calculate_bust_or_stop_loss_price(100.0, 10.0, 1, 0.01, 95.0) # Returns 95.0

    """
    if trade_sign not in (1, -1):
        raise ValueError("trade_sign must be 1 (long) or -1 (short).")

    if stop_loss_price is None:
        p_close = p_open * (1 - (1 / bet_multiplier))
    else:
        p_close = stop_loss_price

    p_trigger = p_close * (1 + trade_sign * bust_buffer)
    status.update_data({"p_trigger": p_trigger})
    return p_trigger


def calculate_closing_price(
        p_open: float, 
        p_close_ideal: float, 
        bet_amount: float, 
        bet_multiplier: float,
        base_rate: float, 
        rate_multiplier: float, 
        rate_exponent: float,
        position_multiplier: float,
        status: StatusManager = StatusManager('formula.calculate_closing_price')
    ) -> float:
    """
    Calculates the closing price for winning trades, considering market impact.

    Args:
        p_open: The Rollbit futures price at the time of bet opening.
        p_close_ideal: The Rollbit futures price at the time of bet closing (without market impact).
        bet_amount: The initial amount of the bet (USD).
        bet_multiplier: The bet multiplier.
        base_rate:  Formula parameter (see documentation).
        rate_multiplier: Formula parameter (see documentation).
        rate_exponent: Formula parameter (see documentation).
        position_multiplier: Formula parameter (see documentation).

    Returns:
        The adjusted closing price, considering market impact.

    Example:
        # Example with dummy values for the parameters.
        p_close = calculate_closing_price(
            p_open=100.0, p_close_ideal=110.0, bet_amount=1000.0, bet_multiplier=10.0,
            base_rate=0.99, rate_multiplier=100.0, rate_exponent=2.0, position_multiplier=5.0
        )

    """

    price_change_ratio = abs(p_close_ideal / p_open - 1)
    market_impact_denominator = (
        1
        + (1 / (price_change_ratio * rate_multiplier)) ** rate_exponent
        + (bet_amount * bet_multiplier) / (10**6 * price_change_ratio * position_multiplier)
    )
    p_close = p_open + (1 - base_rate) / market_impact_denominator * (p_close_ideal - p_open)
    status.update_data({"p_close": p_close})
    return p_close