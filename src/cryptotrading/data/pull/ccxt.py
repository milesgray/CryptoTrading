import pandas as pd
from functools import partial
import ccxt
import numpy as np
import tqdm

from cryptotrading.util.ccxt import to_datetimes, to_datetime
from cryptotrading.util.proxy import get_proxy, PROXIES

CCXT_DEFAULT_EXCHANGE="binanceus"

def get_connection(exchange, proxy=None):
    if hasattr(ccxt, exchange):
        exchange_fn = partial(getattr(ccxt, exchange))
        return exchange_fn({"proxies": proxy}) if proxy else \
            exchange_fn({"proxies": PROXIES}) if PROXIES else \
            exchange_fn()
    else:
        return None

def pull_ohlcv_data(target_market,
                    n=1,
                    name="",
                    span="1h",
                    min_uts=-np.inf,
                    include_date=True,
                    include_y=True,
                    proxy=None,
                    exchange=CCXT_DEFAULT_EXCHANGE,
                    verbose=False):
    if verbose: print(f"Pulling {n} {span} data for {target_market} from {exchange}")
    connection = get_connection(exchange, proxy=proxy)
    if len(name) > 0:
        name = f"_{name}"
    since = None
    dfs = []
    if verbose:
        bar = tqdm.tqdm(total=n)
    i = 0
    while i < n:
        try:
            cex_x = connection.fetch_ohlcv(target_market, span, since=since)
            data = {
                f"uts": [xi[0]/1000 for xi in cex_x if xi[0]/1000 > min_uts],
                f"open{name}":[xi[1] for xi in cex_x if xi[0]/1000 > min_uts],
                f"high{name}":[xi[2] for xi in cex_x if xi[0]/1000 > min_uts],
                f"low{name}": [xi[3] for xi in cex_x if xi[0]/1000 > min_uts],
                f"close{name}": [xi[4] for xi in cex_x if xi[0]/1000 > min_uts],
                f"vol{name}": [xi[5] for xi in cex_x if xi[0]/1000 > min_uts]
            }
            if include_date:
                data[f'date'] = to_datetimes(data['uts'])
            if include_y:
                data[f'y_{name}'] = data[f"close{name}"]
            dfs.append(pd.DataFrame(data))

            dt = cex_x[-1][0] - cex_x[0][0]
            since = cex_x[0][0] - dt
            i = i + 1
            if verbose:
                bar.set_postfix({"since": since})
                bar.update(1)
        except Exception as e:
            if verbose: print(e)
            connection = get_connection(exchange, proxy=get_proxy() if proxy else None)
            # only increment i on data success

    return pd.concat(dfs) \
             .drop_duplicates() \
             .sort_values('uts') \
             .reset_index(drop=True)

def pull_ohlcv_markets(target_market,
                       n=1,
                       span='1h',
                       currency_name="USDC",
                       primary=False,
                       exchange_name="binanceus",
                       uts_index=True,
                       include_names=False,
                       proxy=None,
                       verbose=False):
    markets = get_markets(
        currency_name=currency_name,
        primary=primary,
        exchange_name=exchange_name,
        proxy=proxy,
        verbose=verbose)
    other_markets = [m for m in markets if target_market not in m]
    if verbose: print(f"Found {len(other_markets)} additional markets!\n{other_markets}")
    target_df = pull_ohlcv_data(target_market,
                                n=n,
                                span=span,
                                proxy=proxy,
                                exchange=exchange_name,
                                verbose=verbose)
    earliest_uts = target_df.iloc[0].uts
    if verbose: print(f"Data for {target_market} found starting at {to_datetime(earliest_uts)}")

    if uts_index:
        target_df = target_df.set_index("uts")
    if include_names:
        dfs = [(target_market, target_df)]
    else:
        dfs = [target_df]

    if len(other_markets):
        for market in markets:
            df = pull_ohlcv_data(market,
                        n=n,
                        span=span,
                        name="" if include_names else market.replace(currency_name, "").replace("/", ""),
                        min_uts=earliest_uts,
                        include_date=include_names,
                        include_y=include_names,
                        proxy=proxy,
                        exchange=exchange_name)
            if uts_index:
                df = df.set_index("uts")
            if include_names:
                dfs.append((market, df))
            else:
                dfs.append(df)
    return dfs

def get_ohlcv_data(target_market,
                   n=1,
                   span='1h',
                   currency_name="USDT",
                   primary=False,
                   exchange_name="binanceus",
                   axis=1,
                   proxy=None,
                   verbose=False):
    dfs = pull_ohlcv_markets(target_market,
                       n=n, span=span, currency_name=currency_name,
                       primary=primary, exchange_name=exchange_name,
                       uts_index=axis==1,
                       include_names=axis==0,
                       proxy=proxy, verbose=verbose)
    if axis == 1:
        return pd.concat(dfs, axis=1, ignore_index=False, sort=False, verify_integrity=True)
    else:
        new_dfs = []
        for (name, df) in dfs:
            df["name"] = name
            new_dfs.append(df)
        return pd.concat(new_dfs)

def get_markets(currency_name="USDC",
                primary=False,
                exchange_name="binanceus",
                proxy=None,
                verbose=False):
    connection = get_connection(exchange_name, proxy=proxy)
    markets = connection.load_markets()
    if verbose: print(f"{len(markets)} total markets on {exchange_name}")
    pattern = f"{currency_name}/" if primary else f"/{currency_name}"
    markets = [m for m in markets if pattern in m]
    if verbose: print(f"{len(markets)} {pattern} markets on {exchange_name}")
    return markets

def get_ticker_data(name, exchange=CCXT_DEFAULT_EXCHANGE, proxy=None):
    connection = get_connection(exchange, proxy=proxy)
    data = connection.fetch_ticker(name)
    return data