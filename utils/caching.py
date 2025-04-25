"""
Caching utilities for diabetes clustering analysis.
"""

import os
import pickle
import hashlib
import inspect
import logging
from functools import wraps


CACHING_ENABLED = True


def generate_cache_key(func, args, kwargs, exclude_params=None):
    """Generate a unique key for caching based on function name and parameters.

    Args:
        func: The function being cached
        args: Positional arguments
        kwargs: Keyword arguments
        exclude_params: List of parameter names to exclude from cache key generation
    """
    exclude_params = exclude_params or []

    # Get function source code to ensure changes to function implementation change the cache key
    func_source = inspect.getsource(func)

    # Get function signature to identify parameter names
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Map positional args to their parameter names
    args_dict = {param_names[i]: arg for i, arg in enumerate(args) if i < len(param_names)}

    # Combine with kwargs
    all_params = {**args_dict, **kwargs}

    # Filter out excluded parameters
    filtered_params = {k: v for k, v in all_params.items() if k not in exclude_params}

    # Convert filtered parameters to string representation
    params_str = str(sorted(filtered_params.items()))

    # Create hash from combination of function source and filtered parameters
    key_str = f"{func.__name__}_{func_source}_{params_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_result(cache_dir="cache", exclude_params=None):
    """Decorator to cache function results based on input parameters.

    Args:
        cache_dir: Directory to store cache files
        exclude_params: List of parameter names to exclude from cache key generation
    """
    os.makedirs(cache_dir, exist_ok=True)
    if not exclude_params:
        exclude_params = ["output_dir"]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key (excluding specified parameters)
            cache_key = generate_cache_key(func, args, kwargs, exclude_params)
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{cache_key}.pkl")

            # Check if cache exists
            if os.path.exists(cache_file) and CACHING_ENABLED:
                logging.info(f"Loading cached result for {func.__name__}")
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                return result

            # Execute function if no cache exists
            logging.info(f"No cache found for {func.__name__}, executing function")
            result = func(*args, **kwargs)

            # Save result to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator


# Example usage:
# @cache_result(exclude_params=["output_dir"])
# def create_dimension_reduction_visualizations(X_processed, output_dir="output"):
#     # Function implementation
#     pass