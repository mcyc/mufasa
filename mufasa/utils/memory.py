import psutil  # To detect system memory

def calculate_target_memory(multicore, use_total=False, max_usable_fraction=0.85):
    """
    Dynamically calculate target memory per chunk based on system memory and cores.

    Parameters:
    ----------
    multicore : int
        Number of cores being used.
    use_total : bool
        Whether to use total memory instead of available memory.
    max_usable_fraction : float
        Maximum fraction of memory to use (default: 90%).

    Returns:
    -------
    target_memory_mb : float
        Target memory per chunk in MB.
    """
    memory_info = psutil.virtual_memory()
    memory_to_use = memory_info.total if use_total else memory_info.available
    usable_memory = memory_to_use * max_usable_fraction

    # Divide usable memory by the number of cores
    target_memory_mb = usable_memory / multicore / (1024 * 1024)  # Convert to MB
    return target_memory_mb