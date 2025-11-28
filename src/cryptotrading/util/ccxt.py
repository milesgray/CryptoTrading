#@title utils
# helper functions: time

from pathlib import Path
import datetime as dt
from datetime import timezone
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def now():
    return dt.datetime.now()

def to_unixtime(value: dt.datetime):
    # must account for timezone, otherwise it's off
    ut = value.replace(tzinfo=timezone.utc).timestamp()
    value2 = dt.datetime.fromtimestamp(ut, tz=timezone.utc)  # to_datetime() approach
    assert value2 == value, f"value: {value}, value2: {value2}"
    return ut


def to_unixtimes(values: list) -> list:
    return [to_unixtime(v) for v in values]


def to_datetime(ut) -> dt.datetime:
    value = dt.datetime.fromtimestamp(ut, tz=timezone.utc)
    ut2 = value.timestamp()  # to_unixtime() approach
    assert ut2 == ut, f"ut: {ut}, ut2: {ut2}"
    return value


def to_datetimes(values: list) -> list[dt.datetime]:
    return [to_datetime(v) for v in values]


def round_to_nearest_hour(value: dt.datetime) -> dt.datetime:
    return (value.replace(second=0, microsecond=0, minute=0, hour=value.hour)
            + dt.timedelta(hours=value.minute//30))


def pretty_time(value: dt.datetime) -> str:
    return value.strftime('%Y/%m/%d, %H:%M:%S')


def print_datetime_info(descr: str, uts: list):
    dts = to_datetimes(uts)
    print(descr + ":")
    print(f"  starts on: {pretty_time(dts[0])}")
    print(f"    ends on: {pretty_time(dts[-1])}")
    print(f"  {len(dts)} datapoints")
    print(f"  time interval between datapoints: {(dts[1]-dts[0])}")


def target_12h_unixtimes(start_dt: dt.datetime) -> list:
    target_dts = [start_dt + dt.timedelta(hours=h) for h in range(12)]
    target_uts = to_unixtimes(target_dts)
    return target_uts


# helper-functions: higher level
def load_from_ohlc_data(file_name: str) -> tuple:
    """Returns (list_of_unixtimes, list_of_close_prices)"""
    with open(file_name, "r") as file:
        data_str = file.read().rstrip().replace('"', '')
    x = eval(data_str)  # list of lists
    uts = [xi[0]/1000 for xi in x]
    vals = [xi[4] for xi in x]
    return (uts, vals)


def filter_to_target_uts(
    target_uts: list,
    unfiltered_uts: list,
    unfiltered_vals: list
) -> list:
    """Return filtered_vals -- values at at the target timestamps"""
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
    """Save a file shaped: [1.2, 3.4, 5.6, ..]"""
    p = Path(file_name)
    p.write_text(str(list_))


def load_list(file_name: str) -> list:
    """Load from a file shaped: [1.2, 3.4, 5.6, ..]"""
    p = Path(file_name)
    s = p.read_text()
    list_ = eval(s)
    return list_


# helpers: prediction performance
def calc_nmse(y, yhat) -> float:
    assert len(y) == len(yhat)
    mse_xy = np.sum(np.square(np.asarray(y) - np.asarray(yhat)))
    mse_x = np.sum(np.square(np.asarray(y)))
    nmse = mse_xy / mse_x
    return nmse


def plot_prices(cex_vals, pred_vals):
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

def extend_dates(df, n_hours=12):
    end_time = dt.datetime.fromisoformat(df.iloc[-1].date) \
        if isinstance(df.iloc[-1].date, str) \
        else df.iloc[-1].date
    data = {
        "date": [end_time + dt.timedelta(hours=(n+1)) for n in range(n_hours)],
        "open":[0 for n in range(n_hours)],
        "high":[0 for n in range(n_hours)],
        "low": [0 for n in range(n_hours)],
        "close": [0 for n in range(n_hours)],
        "vol": [0 for n in range(n_hours)],
    }
    new_df = pd.DataFrame(data)
    return pd.concat([df, new_df])