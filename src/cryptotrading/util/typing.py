import numpy as np

# Helper function to safely convert to float
def to_float(x) -> float:
    """
    Safely convert various data types to float with fallback handling.

    Args:
        x: Value to convert (can be numpy array, scalar, etc.)

    Returns:
        float: Converted value or 0.0 as fallback
    """
    try:
        # Handle None case
        if x is None:
            return 0.0

        # Try direct conversion first
        result = float(x)

        # Check if result is finite
        if not np.isfinite(result):
            return 0.0

        return result

    except (TypeError, ValueError, OverflowError):
        # If direct conversion fails, handle special cases
        if hasattr(x, 'item'):
            # For numpy arrays with item() method
            try:
                result = float(x.item())
                return result if np.isfinite(result) else 0.0
            except (ValueError, TypeError, OverflowError):
                pass

        # For numpy arrays or lists with at least one element
        if hasattr(x, '__getitem__') and len(x) > 0:
            return to_float(x[0])

        # If all else fails, return 0.0 as a fallback
        return 0.0

def get_first_float(x) -> float:
    """
    Safely get the first element of a list or numpy array and convert it to float.

    Args:
        x: List or numpy array to get the first element from

    Returns:
        float: First element converted to float or 0.0 as fallback
    """
    if hasattr(x, '__getitem__') and len(x) > 0:
        return to_float(x[0])
    return to_float(x)