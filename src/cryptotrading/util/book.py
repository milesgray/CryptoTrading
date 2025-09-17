import pandas as pd
import numpy as np

def order_book_to_df(
    bids, 
    asks, 
    side_column='side',
    price_column='price',
    size_column='size'
) -> pd.DataFrame:
    # Convert to DataFrames if lists are provided
    if isinstance(bids, list | np.ndarray):
        if isinstance(bids[0], list | tuple):
            bids = pd.DataFrame({
                price_column: [b[0] for b in bids], 
                size_column: [b[1] for b in bids],
                side_column: ["b" for b in bids]})
        else:
            bids = pd.DataFrame(bids)            
    if isinstance(asks, list | np.ndarray):
        if isinstance(asks[0], list | tuple):
            asks = pd.DataFrame({
                price_column: [a[0] for a in asks], 
                size_column: [a[1] for a in asks], 
                side_column: ["a" for a in asks]})
        else:
            asks = pd.DataFrame(asks)
    
    return pd.concat([bids, asks])