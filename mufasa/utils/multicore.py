from multiprocessing import cpu_count

from mufasa_log import get_logger

def validate_n_cores(n_cores):
    default = cpu_count() - 1

    if n_cores is None: return default

    if isinstance(n_cores, bool):
        if n_cores: return default
        else: return 1

    if isinstance(n_cores, int):
        if 1 <= n_cores <= cpu_count(): return default
        else: raise ValueError(f'n_cores should be between 1 and multiprocessing.cpu_count(). Value given was {n_cores}')

    raise ValueError(f'n_cores must be one of (None, True, False, integer). Value given was {n_cores}')
