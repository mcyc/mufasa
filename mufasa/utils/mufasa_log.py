import logging
from astropy import log as astropy_log
from logging import INFO, WARNING, DEBUG, ERROR

logging_format = dict(format='%(asctime)s - %(levelname)s: %(message)s [%(name)s]', datefmt='%m/%d %I:%M%p')

def reset_logger(logger, handler_class):
    for handler in logger.handlers:
        if isinstance(handler, handler_class):
            logger.removeHandler(handler)


def init_logging(logfile='mufasa.log', console_level=logging.INFO, file_level=logging.DEBUG, log_pyspeckit_to_file=False):
    '''
    :param logfile: file to save to (default mufasa.log)
    :param console_level: minimum logging level to print to screen (default logging.INFO)
    :param file_level: minimum logging level to save to file (default logging.INFO)
    :param log_pyspeckit_to_file: whether to include psypeckit outputs in log file (default False)
    
    Note that technically log_pyspeckit_to_file applies to not just pyspeckit but all modules using astropy.log
    However, astropy says that you shouldn't use it, and you should write your own
    '''
    log_formatter = logging.Formatter(logging_format['format'], datefmt=logging_format['datefmt'])
    log_filter = ContextFilter()

    # set up the main logger instance, all other logger are children of this
    root_logger = logging.getLogger('mufasa')
    root_logger.setLevel(min(console_level, file_level)) # ensure that all desired log levels are caught
    # root_logger.addFilter(log_filter)

    # reset logger so that multiple mufasa calls don't append to the same file when they shouldn't
    reset_logger(root_logger, logging.FileHandler)
    reset_logger(root_logger, logging.StreamHandler)
    
    # set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_formatter)
    console_handler.addFilter(log_filter)
    root_logger.addHandler(console_handler)
    
    # set up file handler
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(log_formatter)
    # file_handler.addFilter(log_filter)
    root_logger.addHandler(file_handler)

    # deal with astropy
    astropy_log.propagate = False # don't send astropy logs through the MUFASA logger
    astropy_log.removeHandler(astropy_log.handlers[0]) # don't print astropy logs directly to console
    astropy_log.addHandler(console_handler)
    if log_pyspeckit_to_file:
        astropy_log.addHandler(file_handler)
    else: 
        # override user input for log entries with severity warning or higher
        # haven't actually tested if warnings/errors successfully pass through
        override_file_handler = logging.FileHandler(logfile)
        override_file_handler.setLevel(logging.WARNING)
        override_file_handler.addFilter(log_filter)
        astropy_log.addHandler(override_file_handler)

    return root_logger


def get_logger(module_name):
    return logging.getLogger(module_name)


class ContextFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'origin'):
            record.name = record.origin # files using astropy.log have name overwritten with 'astropy'
        elif hasattr(record, 'funcName'):
            # sometimes the console output (but not file) has a second log record with funcName twice, so trying to avoid that
            if record.funcName in record.name: return None
            else:
                record.name = ''.join([record.name,'.', record.funcName]) 
        return record
