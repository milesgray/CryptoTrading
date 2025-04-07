# Rollbit Futures Trading

This is an implementation of the algorithm described on [this page](https://rollbit.com/trading/price-formulation) - it describes exactly how the prices are determined for Rollbit's Crypto Future Trading 'Game'.  Rollbit is able to provide a unique trading experience with functionality and fees that should be impossible.  

- 0 price impact
- Instant resolution
- Only pay fee if you make money (0 cost opening trades)
- No funding cost for the first 8 hours
- See other's trades in real time

These are possible because unlike actual exchanges or other legitimate futures platforms, Rollbit's Futures Trading is a game that simulates the experience of opening perpetual future contracts without ever making an actual trade.  Since there is no real trade being made, there is no price impact, instant finality, no gas fees, etc.  Rollbit is a massively susccessful casino that is able to use their own bankroll to pay any 'wins', though the futures feature always makes money for them - but you don't need to worry about being able to close a big trade.  Unfortunately, they only support up to $1 million profit per trade.  You can have many trades open, though, so if you want to play big, just open many duplicate trades.


## Trading Price Formulation

### Rollbit Index Prices

To provide fair and reliable pricing for the trading games, Rollbit calculates a composite index price every **500 milliseconds** that is derived from real-time price feeds to the world's most liquid spot and derivative cryptocurrency exchanges. By incorporating many price sources, the Rollbit index is robust to market manipulation, technical issues, and other anomalous trading activity that may occur on individual exchanges from time to time.

## Index constituents

### Spot Exchanges

- Binance
- Coinbase Pro
- Kraken
- Huobi
- OKEx

### Derivative Exchanges

- Binance (Coin-Margined)
- Binance (USDT-Margined)
- Huobi (Coin-Margined)
- Huobi (USDT-Margined)
- OKEx (Coin-Margined)
- OKEx (USDT-Margined)

The index price methodology has been designed to satisfy two important statistical properties of time series: the Markov and martingale properties. The Markov property refers to the memoryless nature of a stochastic process, which ensures that the conditional probability distribution of future values only depends on the current value. The martingale property implies that the current value of a stochastic process is also the expected value of future values. These two properties make Rollbit's index prices unbiased estimators of future prices, so that users can bet on the changes in value of the underlying cryptocurrencies without having to worry about the microstructure effects of individual exchanges.

### The calculation steps are as follows:

1. Subscribe to as many levels of depth as available using each exchange's streaming APIs.

2. Remove any price feeds for which there have been no market data updates for the last 30 seconds.

3. Remove any price feeds with crossed buy and sell prices or whose top-of-book mid-price is more than 10% away from the median top-of-book mid-price across all price feeds.

4. Wait until there are at least 6 valid price feeds. If there are not enough price feeds, the Rollbit index price will not be updated.

5. Combine all resting limit orders from each price feed into a single composite order book. It is okay and expected that the price of some buy orders will exceed the price of some sell orders from other exchanges. Individual order sizes are capped to **$1 million** to limit the influence of a single large order.

6. Using the composite order book, a function is defined to represent the marginal price to buy or sell a given amount. For example, the marginal buying function is:

```
P_buy(s) = max{p_i | sum_{i in 1..N}{s_i} <= s}
```

where `p_i | i in 1..N` and `s_i | i in 1..N` are the buy prices and sizes sorted in increasing distance from the top-of-book. This function gives the maximum price one would pay to buy an amount `s`.

7. The marginal buy and sell price functions are then used to define a marginal mid-price function given a size:

```
P_mid(s) = (P_buy(s) + P_sell(s)) / 2
```

8. The final index price is then calculated as the weighted average of the marginal mid-prices at each size. The weights are chosen to be the probability density of the exponential distribution, which is monotonically decreasing, resulting in a higher emphasis on prices closer to the top-of-book. The weights are given by:

```
w_i = L * exp(-L * v_i)
```

where `v_i` are the sizes at which the mid-prices are calculated and are defined as the union of the cumulative buy and sell sizes from the composite order book. `L` is a scaling factor defined as `1 / V`, where `V` is the maximum size at which a mid-price is calculated and is defined as the minimum of the sum of buying and selling sizes in the composite order book.

The output of this calculation is a single index price that is used for both long and short bets in the Rollbit trading game. Unlike with most other trading platforms, Rollbit does not charge a bid-offer spread. Trading with a single price makes it possible to speculate on short-term price moves for which a bid-offer spread would be prohibitively expensive.

## Funding Payments

Keep your bet open for longer than 8 hours and you might incur or receive hourly funding payments, depending on market conditions. When the market is bullish, the funding rate is positive and long traders pay short traders. When the market is bearish, the funding rate is negative and short traders pay long traders.

For example, if you have an open long BTC bet for $500 with a 10x multiplier and the funding rate is 0.1%, you will be charged $0.21 per hour after the first 8 hours. If the bet were short rather than long, then you would receive funding payments.

If your bet is closed within **8 hours**, no funding payments will be made.

## Bust and Stop Loss Prices

Regardless of the liquidity in the underlying markets, Rollbit guarantees that busts and stop loss trades will consistently be filled at a price predetermined at entry. Bust losses are limited to the original bet amount and stop losses are limited by the user entry setting.

The trigger prices for busts and stop losses can be calculated by:
'''
P_trigger = P_close * (1 + trade_sign*bust_buffer)
'''

where 'P_close' is 'P_open*(1 - 1/bet_multiplier)' for busts or the stop loss price. The bust_buffer is a parameter set based on current market conditions. Its current value can be found [here](https://rollbit.com/public/prices).

## Market Impact Effects

When traded on a traditional exchange with a central limit order book, large taking orders will generally cross with resting orders beyond the best bid or offer, resulting in an average fill price worse than the top-of-book price. Market makers restrict the amount of liquidity that they're willing to offer at the top-of-book because of the additional risk that large orders represent.

Rather than require large orders to be filled at multiple prices, Rollbit instead uses a market impact formula to set the closing price for winning trades (losing manually-closed trades are unaffected) to replicate the market impact of large taking orders, but in a more consistent and predictable way. The easiest way to see the exact market impact effects is to use the ROI Calculator found on the futures trading page.

Users can also calculate the closing price for winning trades opened at time t and closed at time T themselves by using the formula:

```
P_close(T) = P(t) + ((1 - base_rate) / (1 + 1/abs((P(T)/P(t) - 1)*rate_multiplier)^rate_exponent + bet_amount*bet_multiplier/(10^6*abs(P(T)/P(t) - 1)*position_multiplier)))*(P(T) - P(t))
```

where `p(t)` is the Rollbit futures price at time `t`.

The formula parameters are set based on current market conditions and their current values can be found [here](https://rollbit.com/public/prices).