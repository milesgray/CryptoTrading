#@title utils
# helper functions: time
import logging
import datetime as dt
from pathlib import Path
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import ccxt

from .datetime import pretty_time, to_datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CCXT_DEFAULT_EXCHANGE = "binanceus"
PROXIES = None

def get_connection(exchange, proxy=None):
    if hasattr(ccxt, exchange):
        exchange_fn = partial(getattr(ccxt, exchange))
        return exchange_fn({"proxies": PROXIES}) if PROXIES else \
            exchange_fn({"proxies": proxy}) if proxy else exchange_fn()
    else:
        return None

def pull_ohlcv_data(
    target_market,
    n=1,
    name="",
    span="1s",
    min_uts=-np.inf,
    proxy=None,
    exchange=CCXT_DEFAULT_EXCHANGE,
    verbose=False,
):
    """
    Pull OHLCV data from CCXT exchange.
    
    Args:
        target_market: Trading pair (e.g., 'BTC/USDT')
        n: Number of data chunks to fetch
        name: Optional name for logging
        span: Timeframe (e.g., '1m', '1h', '1d')
        min_uts: Minimum Unix timestamp to include
        proxy: Proxy configuration
        exchange: CCXT exchange name
        verbose: Enable verbose logging
        
    Returns:
        List of fetched data chunks
    """
    if verbose: 
        logger.info(f"Pulling {n} {span} data for {target_market} from {exchange}")
    connection = get_connection(exchange, proxy=proxy)
    if len(name) > 0:
        name = f"_{name}"
    since = None
    results = []
    if verbose:
        bar = tqdm.tqdm(total=n)
    i = 0
    while i < n:
        try:
            cex_x = connection.fetch_ohlcv(target_market, span, since=since)
            
            results.append([(xi[4], dt.datetime.fromtimestamp(xi[0] / 1000, tz=dt.timezone.utc)) for xi in cex_x if xi[0]/1000 > min_uts])

            dt_ms = cex_x[-1][0] - cex_x[0][0]
            since = cex_x[0][0] - dt_ms
            i = i + 1
            if verbose:
                bar.set_postfix({"since": since})
                bar.update(1)
        except Exception as e:
            if verbose: 
                logger.error(f"Error fetching data: {e}", exc_info=True)
            # only increment i on data success

    return results

# helper-functions: higher level
def load_from_ohlc_data(file_name: str) -> tuple:
    """
    Load OHLC data from a file.
    
    Args:
        file_name: Path to the data file
        
    Returns:
        tuple: (list_of_unixtimes, list_of_close_prices)
    """
    with open(file_name, "r") as file:
        data_str = file.read().rstrip().replace('"', '')
    x = eval(data_str)  # list of lists
    uts = [xi[0]/1000 for xi in x]
    vals = [xi[4] for xi in x]
    return (uts, vals)


def filter_to_target_uts(
    target_uts: list[float],
    unfiltered_uts: list[float],
    unfiltered_vals: list[float]
) -> list[float]:
    """
    Return filtered_vals -- values at the target timestamps.
    
    Args:
        target_uts: List of target Unix timestamps
        unfiltered_uts: List of available Unix timestamps
        unfiltered_vals: List of values corresponding to unfiltered_uts
        
    Returns:
        list: Values filtered to match target timestamps
    """
    filtered_vals = [None] * len(target_uts)
    for i, target_ut in enumerate(target_uts):
        time_diffs = np.abs(np.asarray(unfiltered_uts) - target_ut)
        tol_s = 1  # should always align within e.g. 1 second
        target_ut_s = pretty_time(to_datetime(target_ut))
        assert min(time_diffs) <= tol_s, \
            f"Unfiltered times is missing target time: {target_ut_s}"
        j = np.argmin(time_diffs)
        filtered_vals[i] = unfiltered_vals[j]
    return filtered_vals


# helpers: save/load list
def save_list(list_: list, file_name: str):
    """
    Save a list to a file.
    
    Args:
        list_: List to save
        file_name: Path to save the file
    """
    p = Path(file_name)
    p.write_text(str(list_))


def load_list(file_name: str) -> list:
    """
    Load a list from a file.
    
    Args:
        file_name: Path to the file to load
        
    Returns:
        list: Loaded list
    """
    p = Path(file_name)
    s = p.read_text()
    list_ = eval(s)
    return list_


# helpers: prediction performance
def calc_nmse(y, yhat) -> float:
    """
    Calculate the normalized mean squared error.
    
    Args:
        y: Actual values
        yhat: Predicted values
        
    Returns:
        float: Normalized mean squared error
    """
    assert len(y) == len(yhat)
    mse_xy = np.sum(np.square(np.asarray(y) - np.asarray(yhat)))
    mse_x = np.sum(np.square(np.asarray(y)))
    nmse = mse_xy / mse_x
    return nmse


def plot_prices(cex_vals: list[float], pred_vals: list[float]):
    """
    Plot CEX values and predicted values.
    
    Args:
        cex_vals: List of CEX values
        pred_vals: List of predicted values
    """
    matplotlib.rcParams.update({'font.size': 22})
    x = [h for h in range(0, 12)]
    assert len(x) == len(cex_vals) == len(pred_vals)
    fig, ax = plt.subplots()
    ax.plot(x, cex_vals, '--', label="CEX values")
    ax.plot(x, pred_vals, '-', label="Pred. values")
    ax.legend(loc='lower right')
    plt.ylabel("ETH price")
    plt.xlabel("Hour")
    fig.set_size_inches(18, 18)
    plt.xticks(x)
    plt.show()

def extend_dates(df: pd.DataFrame, n_hours: int = 12) -> pd.DataFrame:
    """
    Extend a DataFrame with additional date rows.
    
    Args:
        df: DataFrame to extend
        n_hours: Number of hours to extend
        
    Returns:
        Extended DataFrame
    """
    end_time = dt.datetime.fromisoformat(df.iloc[-1].date) \
        if isinstance(df.iloc[-1].date, str) \
        else df.iloc[-1].date
    data = {
        "date": [end_time + dt.timedelta(hours=(h+1)) for h in range(n_hours)],
        "open":[0 for n in range(n_hours)],
        "high":[0 for n in range(n_hours)],
        "low": [0 for n in range(n_hours)],
        "close": [0 for n in range(n_hours)],
        "vol": [0 for n in range(n_hours)],
    }
    new_df = pd.DataFrame(data)
    return pd.concat([df, new_df])


    