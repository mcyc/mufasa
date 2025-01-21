"""
This module provides tools to monitor memory usage and
estimate the memory allocations needed

"""

import psutil  # To detect system memory
import os
import threading
import functools
import time

from spectral_cube import DaskSpectralCube
from dask.array import Array as daArray

from .mufasa_log import get_logger
logger = get_logger(__name__)

def monitor_peak_memory(output_attr=None, key=None, unit='GB'):
    """
    Decorator to monitor and record the peak memory usage of a function.

    This decorator measures the peak memory usage, including intermediate peaks,
    during the execution of a function. It supports multi-threaded tasks and
    can store results in an instance attribute or print them.

    Parameters
    ----------
    output_attr : str or None, optional
        The name of the instance attribute to store the peak memory usage.
        If `None`, the memory usage is printed. If specified and a key is
        provided, the attribute is treated as a dictionary. Default is `None`.
    key : str or None, optional
        The dictionary key to use when storing memory usage in the attribute
        specified by `output_attr`. If `None`, the value is stored directly
        in the attribute. Default is `None`.
    unit : {'KB', 'MB', 'GB', 'TB'}, default='GB'
        The unit for reporting memory usage. If an unrecognized unit is given,
        a warning is logged, and the default ('GB') is used.

    Returns
    -------
    callable
        A decorator that monitors peak memory usage for the decorated function.

    Raises
    ------
    ValueError
        If `output_attr` is specified as a non-dictionary attribute and a `key`
        is also provided.

    Examples
    --------
    Monitor memory usage and print results:

    >>> @monitor_peak_memory()
    ... def my_function(self):
    ...     # Function logic here
    ...     pass

    Store memory usage in an instance attribute:

    >>> @monitor_peak_memory(output_attr='memory_usage', key='task1')
    ... def my_function(self):
    ...     # Function logic here
    ...     pass
    >>> self.memory_usage['task1']  # Access the stored peak memory

    Use a custom unit for memory usage:

    >>> @monitor_peak_memory(unit='MB')
    ... def my_function(self):
    ...     pass
    """
    ipw = 3
    if unit == 'KB':
        ipw = 1
    elif unit == 'MB':
        ipw = 2
    elif unit == 'TB':
        ipw = 4
    elif unit != 'GB':
        logger.warning(f"unit {unit} is not reconized, defaulting to GB")
        unit = 'GB'

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            process = psutil.Process(os.getpid())
            peak_memory = [0]  # Use a list to allow modification within the thread
            # determine unit conversion
            def monitor():
                """Continuously monitor memory usage and record the peak."""
                while monitoring[0]:
                    try:
                        current_memory = process.memory_info().rss / (1024 ** ipw)  # Memory in MB
                        peak_memory[0] = max(peak_memory[0], current_memory)
                        time.sleep(0.1)  # Sample every 100ms
                    except psutil.NoSuchProcess:
                        break

            # Start monitoring in a separate thread
            monitoring = [True]
            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.start()

            try:
                # Execute the function
                result = func(self, *args, **kwargs)
            finally:
                # Stop monitoring and wait for the thread to finish
                monitoring[0] = False
                monitor_thread.join()

            # Store peak memory in the specified attribute or print it
            if output_attr is not None:
                if not hasattr(self, output_attr):
                    setattr(self, output_attr, {} if key else None)
                attr_value = getattr(self, output_attr)

                if isinstance(attr_value, dict) and key is not None:
                    attr_value[key] = peak_memory[0]
                elif key is None:
                    setattr(self, output_attr, peak_memory[0])
                else:
                    raise ValueError(f"{output_attr} must be a dictionary when key is specified.")
            else:
                print(f"Peak memory usage for '{func.__name__}': {peak_memory[0]:.2f} {unit}")

            return result

        return wrapper

    return decorator


def peak_memory(output_container=None):
    """
    Decorator to monitor and display the peak memory usage of a function,
    including intermediate peaks, and handle multi-threaded tasks.

    Args:
        output_container (list or dict, optional): A mutable object where the
            peak memory usage will be stored. If None, it defaults to printing
            the peak memory usage.

    Returns:
        function: A wrapped function that reports peak memory usage.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            peak_memory = [0]  # Use a list to allow modification within the thread

            def monitor():
                """Continuously monitor memory usage and record the peak."""
                while monitoring[0]:
                    try:
                        current_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB
                        peak_memory[0] = max(peak_memory[0], current_memory)
                        time.sleep(0.1)  # Sample every 100ms
                    except psutil.NoSuchProcess:
                        break

            # Start monitoring in a separate thread
            monitoring = [True]
            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.start()

            try:
                # Execute the function
                result = func(*args, **kwargs)
            finally:
                # Stop monitoring and wait for the thread to finish
                monitoring[0] = False
                monitor_thread.join()

            # Store peak memory in the provided container or print it
            if output_container is not None:
                if isinstance(output_container, list) and len(output_container) > 0:
                    output_container[0] = peak_memory[0]
                elif isinstance(output_container, dict):
                    output_container['peak_memory'] = peak_memory[0]
                else:
                    raise ValueError("output_container must be a list or dict.")
            else:
                print(f"Peak memory usage for '{func.__name__}': {peak_memory[0]:.2f} MB")

            return result

        return wrapper

    return decorator


def calculate_target_memory(multicore, use_total=False, max_usable_fraction=0.85):
    """
    Calculate target memory per core for chunked computations.

    Parameters
    ----------
    multicore : int
        Number of cores available for computation. Memory is divided evenly
        among these cores.
    use_total : bool, optional, default=False
        If `True`, calculate based on the total system memory. If `False`, use
        available memory instead.
    max_usable_fraction : float, optional, default=0.85
        Maximum fraction of memory to allocate for computation. For example,
        `0.85` means 85% of memory is considered usable.

    Returns
    -------
    target_memory_mb : float
        Target memory per core, in megabytes (MB).

    Examples
    --------
    Calculate target memory per core using available system memory:

    >>> calculate_target_memory(multicore=4, use_total=False, max_usable_fraction=0.9)
    4096.0  # Example value, system-dependent

    Notes
    -----
    - If `use_total=True`, the calculation includes memory currently in use by other processes.
    - Ensure `multicore > 0` to avoid division errors.
    - The calculated memory assumes even distribution across all cores.
    """
    memory_info = psutil.virtual_memory()
    memory_to_use = memory_info.total if use_total else memory_info.available
    usable_memory = memory_to_use * max_usable_fraction

    # Divide usable memory by the number of cores
    target_memory_mb = usable_memory / multicore / (1024 * 1024)  # Convert to MB
    return target_memory_mb


def calculate_dask_memory_limit(n_workers):
    """
    Mimic Dask's default memory limit setting.

    Parameters:
    - n_workers (int): Number of workers to divide the memory among.

    Returns:
    - memory_limit (float): Memory limit per worker in bytes.
    """
    if n_workers <= 0:
        raise ValueError("Number of workers must be greater than 0.")

    # Get total system memory in bytes
    total_memory = psutil.virtual_memory().total

    # Divide by number of workers
    memory_limit_per_worker = total_memory / n_workers

    return memory_limit_per_worker


def get_system_free_memory():
    mem = psutil.virtual_memory()
    return mem.available / 1e9  # Convert to GB

def get_size_mb(array):
    # calculate the size of a ndarray in MB
    size_in_bytes = array.size * array.itemsize
    return size_in_bytes / (1024 ** 2)

def tmp_save_gauge(cube, factor=20, max_mem_gb=0.3):
    """
    Return whether or not it's worth DaskSpectralCube results temporary
    Based on how much free memory is currenlty left

    Note: when data is chunked properly, mufasa shouldn't need memory larger than 20 times cube size

    """

    if isinstance(cube, DaskSpectralCube):
        data = cube._data
    elif isinstance(cube, daArray):
        data = cube
    else:
        raise TypeError(f"cube type {type(cube)} is invalid")

    data_size = get_size_mb(data) # in MB
    mem_free = get_system_free_memory()*1e3 # in MB

    # advice to save results temporary if
    return data_size*1e3 > max_mem_gb or (data_size * factor > mem_free)