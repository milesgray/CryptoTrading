import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_whale_positions(
    order_book_df: pd.DataFrame, 
    size_column: str,
    limit: int = 5,
    min_size_multiplier: int = 3
):
    """
    Find abnormally large positions ('whale' positions) in an order book.
    
    Parameters:
    order_book_df (pandas.DataFrame): DataFrame containing the order book data
    size_column (str): Column name that contains the order size
    limit (int, optional): Maximum number of whale positions to return. Default is 5.
    min_size_multiplier (float, optional): Minimum multiplier of median size to be considered a whale. Default is 3x.
    
    Returns:
    pandas.DataFrame: DataFrame containing the whale positions, sorted by size (largest first)
    """
    # Calculate basic statistics for the size column
    median_size = order_book_df[size_column].median()
    
    # Find positions that are significantly larger than the median
    # Using a simple multiplier approach rather than IQR which can be overly sensitive
    whale_threshold = median_size * min_size_multiplier
    whale_positions = order_book_df[order_book_df[size_column] >= whale_threshold].copy()
    
    # Sort by size (descending) to get the largest positions first
    whale_positions = whale_positions.sort_values(by=size_column, ascending=False)
    
    # Add a column showing how many times larger than the median this position is
    whale_positions['times_median_size'] = (whale_positions[size_column] / median_size).round(1)
    
    # Limit the number of positions returned if specified
    if limit is not None and limit > 0 and len(whale_positions) > limit:
        return whale_positions.head(limit)
    
    return whale_positions

def find_outliers(df, column, limit=None, direction='both'):
    """
    Find outliers in a DataFrame column with the option to limit the number returned.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the data
    column (str): Column name to check for outliers
    limit (int, optional): Maximum number of outliers to return. If None, returns all outliers.
    direction (str, optional): Which outliers to find - 'both', 'upper', or 'lower'
    
    Returns:
    pandas.DataFrame: DataFrame containing the outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers based on direction parameter
    if direction == 'both':
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
    elif direction == 'upper':
        outliers = df[df[column] > upper_bound].copy()
    elif direction == 'lower':
        outliers = df[df[column] < lower_bound].copy()
    else:
        raise ValueError("direction must be 'both', 'upper', or 'lower'")
    
    # Sort outliers by distance from the median to get the most extreme ones first
    median = df[column].median()
    outliers['distance'] = abs(outliers[column] - median)
    outliers = outliers.sort_values('distance', ascending=False)
    
    # Remove the distance column before returning
    outliers = outliers.drop('distance', axis=1)
    
    # Limit the number of outliers if specified
    if limit is not None and limit > 0:
        return outliers.head(limit)
    
    return outliers

def condense_order_book(bids, asks, num_buckets=10, size_column='size', price_column='price'):
    """
    Condense order book data into buckets distributed by size.
    
    Parameters:
    -----------
    bids : pandas.DataFrame or list of dicts
        Bid orders with price and size columns
    asks : pandas.DataFrame or list of dicts
        Ask orders with price and size columns
    num_buckets : int
        Number of buckets to create
    size_column : str
        Column name for the size values
    price_column : str
        Column name for the price values
        
    Returns:
    --------
    dict containing:
        'bid_buckets': List of (price_range, avg_price, total_size) for each bid bucket
        'ask_buckets': List of (price_range, avg_price, total_size) for each ask bucket
        'bid_outliers': List of (price, size) for outlier bids
        'ask_outliers': List of (price, size) for outlier asks
    """
    # Convert to DataFrames if lists are provided
    if isinstance(bids, list | np.ndarray):
        if isinstance(bids[0], list | tuple):
            bids = pd.DataFrame({"price": [b[0] for b in bids], "size": [b[1] for b in bids]})
        else:
            bids = pd.DataFrame(bids)            
    if isinstance(asks, list | np.ndarray):
        if isinstance(asks[0], list | tuple):
            asks = pd.DataFrame({"price": [a[0] for a in asks], "size": [a[1] for a in asks]})
        else:
            asks = pd.DataFrame(asks)
    
    # Ensure required columns exist
    required_columns = [size_column, price_column]
    if not all(col in bids.columns for col in required_columns) or \
    not all(col in asks.columns for col in required_columns):
        print(f"{bids.columns} || {asks.columns}")
        raise ValueError(f"Both bid and ask data must contain columns: {required_columns}")
    
    
    # Function to bucket non-outlier data
    def create_buckets(df, column, num_buckets, direction='upper'):
        if df.empty:
            return []
        
        # Remove outliers
        outliers = find_outliers(df, column, limit=5, direction=direction)
        non_outliers = df[~df.index.isin(outliers.index)].copy()
        
        if non_outliers.empty:
            return [], outliers[[price_column, size_column]].values.tolist()
        
        # Create buckets based on size distribution
        size_distribution = non_outliers[column].values
        
        # If we have fewer unique values than buckets, adjust num_buckets
        unique_sizes = len(np.unique(size_distribution))
        actual_num_buckets = min(num_buckets, unique_sizes)
        
        if actual_num_buckets <= 1:
            # If only one bucket, put everything in it
            avg_price = non_outliers[price_column].mean()
            total_size = non_outliers[size_column].sum()
            min_price = non_outliers[price_column].min()
            max_price = non_outliers[price_column].max()
            buckets = [(f"{min_price:.2f}-{max_price:.2f}", avg_price, total_size)]
        else:
            # Use percentile-based bucketing for more even distribution
            percentiles = np.linspace(0, 100, actual_num_buckets + 1)
            bucket_thresholds = np.percentile(size_distribution, percentiles)
            
            buckets = []
            for i in range(actual_num_buckets):
                lower = bucket_thresholds[i]
                upper = bucket_thresholds[i + 1]
                
                # Handle edge case for the last bucket to include the maximum value
                if i == actual_num_buckets - 1:
                    bucket_items = non_outliers[(non_outliers[column] >= lower) & 
                                            (non_outliers[column] <= upper)]
                else:
                    bucket_items = non_outliers[(non_outliers[column] >= lower) & 
                                            (non_outliers[column] < upper)]
                
                if not bucket_items.empty:
                    avg_price = bucket_items[price_column].mean()
                    total_size = bucket_items[size_column].sum()
                    min_price = bucket_items[price_column].min()
                    max_price = bucket_items[price_column].max()
                    buckets.append((f"{min_price:.2f}-{max_price:.2f}", avg_price, total_size))
        
        return buckets, outliers[[price_column, size_column]].values.tolist()
    
    # Process bids and asks
    bid_buckets, bid_outliers = create_buckets(bids, size_column, num_buckets, direction="upper")
    ask_buckets, ask_outliers = create_buckets(asks, size_column, num_buckets, direction="upper")
    
    return {
        'bid_buckets': bid_buckets,
        'ask_buckets': ask_buckets,
        'bid_outliers': bid_outliers,
        'ask_outliers': ask_outliers
    }

def visualize_order_book(condensed_data, title="Order Book Visualization"):
    """
    Visualize the condensed order book data.
    
    Parameters:
    -----------
    condensed_data : dict
        Output from condense_order_book function
    title : str
        Title for the plot
    """
    bid_buckets = condensed_data['bid_buckets']
    ask_buckets = condensed_data['ask_buckets']
    bid_outliers = condensed_data['bid_outliers']
    ask_outliers = condensed_data['ask_outliers']
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Function to plot buckets
    def plot_buckets(buckets, color, label_prefix, offset=0):
        y_positions = np.arange(len(buckets)) + offset
        price_ranges = [bucket[0] for bucket in buckets]
        sizes = [bucket[2] for bucket in buckets]
        
        # Plot bars for buckets
        bars = ax.barh(y_positions, sizes, color=color, alpha=0.7, 
                       height=0.8, label=f"{label_prefix} Buckets")
        
        # Add labels
        for i, (bar, price_range) in enumerate(zip(bars, price_ranges)):
            ax.text(bar.get_width() + bar.get_width() * 0.02, 
                   y_positions[i], 
                   f"{price_range}", 
                   va='center')
        
        return max(y_positions) if y_positions.size > 0 else 0
    
    # Function to plot outliers
    def plot_outliers(outliers, color, marker, label, y_start):
        if outliers:
            # Extract prices and sizes
            prices = [item[0] for item in outliers]
            sizes = [item[1] for item in outliers]
            
            # Plot outliers with distinct markers
            y_positions = np.linspace(y_start + 1, y_start + 3, len(outliers))
            scatter = ax.scatter(sizes, y_positions, marker=marker, s=100, 
                                color=color, label=label, edgecolors='black')
            
            # Add labels for each outlier
            for i, (price, size) in enumerate(zip(prices, sizes)):
                ax.text(size + size * 0.05, y_positions[i], 
                       f"Price: {price:.2f}, Size: {size}", 
                       va='center')
        
        return y_start + 3 if outliers else y_start
    
    # Plot bid buckets and outliers
    max_bid_y = plot_buckets(bid_buckets, 'green', 'Bid')
    max_bid_outlier_y = plot_outliers(bid_outliers, 'darkgreen', 'o', 'Bid Outliers', max_bid_y)
    
    # Add spacing between bids and asks
    spacing = 2
    
    # Plot ask buckets and outliers
    max_ask_y = plot_buckets(ask_buckets, 'red', 'Ask', max_bid_outlier_y + spacing)
    max_ask_outlier_y = plot_outliers(ask_outliers, 'darkred', 's', 'Ask Outliers', max_ask_y)
    
    # Set labels and title
    ax.set_ylabel('Price Ranges')
    ax.set_xlabel('Size')
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig