import psutil  # To detect system memory

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