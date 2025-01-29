"""
This module hosts various utility decorators
"""
import warnings
import functools

def deprecated(reason: str, removal_version: str = None):
    """
    A decorator to mark functions or classes as deprecated.

    Parameters
    ----------
    reason : str
        A message explaining why the function/class is deprecated and what to use instead.
    removal_version : str, optional
        The version when this function/class is expected to be removed.

    Returns
    -------
    function
        The wrapped function with a deprecation warning.

    Examples
    --------
    @deprecated("Use 'new_function' instead.", "2.0.0")
    def old_function():
        pass
    """
    def decorator(func):
        message = f"'{func.__name__}' is deprecated. {reason}"
        if removal_version:
            message += f" It will be removed in version {removal_version}."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
