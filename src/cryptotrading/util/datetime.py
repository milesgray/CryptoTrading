import datetime as dt

def now():
    """
    Get current UTC datetime.
    
    Returns:
        dt.datetime: Current UTC datetime
    """
    return dt.datetime.now()

def to_unixtime(value: dt.datetime) -> float:
    """
    Convert a datetime object to Unix timestamp.  Only works with UTC timezone.
    
    Args:
        value: datetime object in UTC
        
    Returns:
        float: Unix timestamp
    """
    # must account for timezone, otherwise it's off
    ut = value.replace(tzinfo=dt.timezone.utc).timestamp()
    value2 = dt.datetime.fromtimestamp(ut, tz=dt.timezone.utc)  # to_datetime() approach
    assert value2 == value, f"value: {value}, value2: {value2}"
    return ut


def to_unixtimes(values: list[dt.datetime]) -> list[float]:
    """
    Convert a list of datetime objects to Unix timestamps.
    
    Args:
        values: list of datetime objects in UTC
        
    Returns:
        list: Unix timestamps
    """
    return [to_unixtime(v) for v in values]


def to_datetime(ut: float) -> dt.datetime:
    """
    Convert a Unix timestamp to a datetime object in UTC.
    
    Args:
        ut: Unix timestamp
        
    Returns:
        datetime object in UTC
    """
    if isinstance(ut, str):
        ut = float(ut)
    value = dt.datetime.fromtimestamp(ut, tz=dt.timezone.utc)
    ut2 = value.timestamp()  # to_unixtime() approach
    assert ut2 == ut, f"ut: {ut}, ut2: {ut2}"
    return value


def to_datetimes(values: list[float]) -> list[dt.datetime]:
    """
    Convert a list of Unix timestamps to datetime objects in UTC.
    
    Args:
        values: list of Unix timestamps
        
    Returns:
        list: datetime objects in UTC
    """
    return [to_datetime(v) for v in values]


def round_to_nearest_hour(value: dt.datetime) -> dt.datetime:
    """
    Round a datetime object to the nearest hour.
    
    Args:
        value: datetime object
        
    Returns:
        datetime object rounded to the nearest hour
    """
    return (value.replace(second=0, microsecond=0, minute=0, hour=value.hour)
            + dt.timedelta(hours=value.minute//30))


def pretty_time(value: dt.datetime) -> str:
    """
    Format a datetime object as a string.
    
    Args:
        value: datetime object
        
    Returns:
        str: formatted datetime string
    """
    return value.strftime('%Y/%m/%d, %H:%M:%S')


def print_datetime_info(descr: str, uts: list[float]):
    """
    Print information about a list of Unix timestamps.
    
    Args:
        descr: Description of the data
        uts: List of Unix timestamps
    """
    dts = to_datetimes(uts)
    print(descr + ":")
    print(f"  starts on: {pretty_time(dts[0])}")
    print(f"    ends on: {pretty_time(dts[-1])}")
    print(f"  {len(dts)} datapoints")
    print(f"  time interval between datapoints: {(dts[1]-dts[0])}")


def target_12h_unixtimes(start_dt: dt.datetime) -> list[float]:
    """
    Generate 12 hourly timestamps starting from the given datetime.
    
    Args:
        start_dt: Starting datetime object
        
    Returns:
        list: 12 Unix timestamps
    """
    target_dts = [start_dt + dt.timedelta(hours=h) for h in range(12)]
    target_uts = to_unixtimes(target_dts)
    return target_uts

