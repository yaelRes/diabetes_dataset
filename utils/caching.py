"""
Caching utilities for diabetes clustering analysis.
"""

import os
import pickle
import hashlib
import inspect
import logging
from functools import wraps


def generate_cache_key(func, args, kwargs):
    """Generate a unique key for caching based on function name and parameters."""
    # Get function source code to ensure changes to function implementation change the cache key
    func_source = inspect.getsource(func)

    # Convert args and kwargs to string representation
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items()))

    # Create hash from combination of function source and parameters
    key_str = f"{func.__name__}_{func_source}_{args_str}_{kwargs_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_result(cache_dir="cache"):
    """Decorator to cache function results based on input parameters."""
    os.makedirs(cache_dir, exist_ok=True)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(func, args, kwargs)
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{cache_key}.pkl")

            # Check if cache exists
            if os.path.exists(cache_file):
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
