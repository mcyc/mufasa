import psutil  # To detect system memory
import os
import threading
import functools
import time

def monitor_peak_memory(output_container=None):
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


def monitor_peak_memory_new(sampling_interval=0.1):  # Default to 100ms
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            peak_memory = [0]

            def monitor():
                while monitoring[0]:
                    try:
                        current_memory = process.memory_info().rss / (1024 ** 2)
                        peak_memory[0] = max(peak_memory[0], current_memory)
                        time.sleep(sampling_interval)
                    except psutil.NoSuchProcess:
                        break

            monitoring = [True]
            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.start()

            try:
                result = func(*args, **kwargs)
            finally:
                monitoring[0] = False
                monitor_thread.join()

            print(f"Peak memory usage for '{func.__name__}': {peak_memory[0]:.2f} MB")
            return result
        return wrapper
    return decorator


def calculate_target_memory(multicore, use_total=False, max_usable_fraction=0.85):
    """
    Calculate the target memory per chunk based on system memory and the number of cores.

    This function dynamically computes the target memory available for each chunk
    by considering the system's total or available memory, the number of cores in use,
    and a specified fraction of memory to utilize.

    Parameters
    ----------
    multicore : int
        The number of cores available for computation. The memory will be divided evenly
        across these cores.
    use_total : bool, optional
        If True, the calculation is based on the total system memory. Otherwise,
        it uses the available memory. Default is False.
    max_usable_fraction : float, optional
        The maximum fraction of memory to allocate for computation. For example,
        a value of 0.85 means 85% of the memory is considered usable. Default is 0.85.

    Returns
    -------
    target_memory_mb : float
        The target memory per chunk, in megabytes (MB), calculated as:
        `(usable_memory / multicore)`, where usable memory is determined based on
        the input parameters.

    Examples
    --------
    >>> calculate_target_memory(multicore=4, use_total=False, max_usable_fraction=0.9)
    4096.0  # Example value, system-dependent

    Notes
    -----
    - If `use_total` is set to True, the calculation includes memory currently in use by other processes.
    - Ensure that `multicore` is greater than zero to avoid division errors.
    - The returned memory value is specific to each core and assumes tasks are distributed evenly.
    """
    memory_info = psutil.virtual_memory()
    memory_to_use = memory_info.total if use_total else memory_info.available
    usable_memory = memory_to_use * max_usable_fraction

    # Divide usable memory by the number of cores
    target_memory_mb = usable_memory / multicore / (1024 * 1024)  # Convert to MB
    return target_memory_mb