"""
Utility decorators for function and class enhancements.

Includes decorators for marking deprecations and other utilities.
"""
import warnings
import functools

def deprecated(reason: str, removal_version: str = None):
    """
    Mark a function or class as deprecated.

    Parameters
    ----------
    reason : str
        Explanation of why it is deprecated and what to use instead.
    removal_version : str, optional
        Expected version when it will be removed.

    Returns
    -------
    function
        The decorated function with a deprecation warning.

    Examples
    --------
    >>> @deprecated("Use 'new_function' instead.", "2.0.0")
    >>> def old_function():
    >>>     pass
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