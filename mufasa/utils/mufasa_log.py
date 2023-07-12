import logging
from astropy import log as astropy_log
from logging import INFO, WARNING, DEBUG, ERROR

logging_format = dict(format='%(asctime)s - %(levelname)s: %(message)s [%(name)s]', datefmt='%m/%d %I:%M%p')

def init_logging(logfile='mufasa.log', console_level=INFO, file_level=DEBUG, astropy_console_level=ERROR, astropy_file_level=WARNING):
    '''
    :param logfile: file to save to (default mufasa.log)
    :param console_level: minimum logging level to print to screen (default INFO)
    :param file_level: minimum logging level to save to file (default INFO)
    :param astropy_console_level: minimum logging level to print astropy/pyspeckit records to screen (default ERROR)
    :param astropy_file_level: at what log severity to save astropy/pyspeckit records (default WARNING)
    '''
    # filter_warnings()
    log_formatter = logging.Formatter(logging_format['format'], datefmt=logging_format['datefmt'])
    log_filter = OriginContextFilter()

    # set up the main logger instance, all other logger are children of this
    root_logger = logging.getLogger('mufasa')
    root_logger.setLevel(min(console_level, file_level)) # ensure that all desired log levels are caught

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
    root_logger.info(f'Logging from mufasa to {logfile}.')

    # deal with astropy
    astropy_log.propagate = False # don't send astropy logs through the MUFASA logger
    astropy_log.removeHandler(astropy_log.handlers[0]) # don't print astropy logs directly to console
    astropy_console_handler = logging.StreamHandler()
    astropy_console_handler.setLevel(astropy_console_level)
    astropy_console_handler.setFormatter(log_formatter)
    astropy_console_handler.addFilter(log_filter)
    astropy_log.addHandler(astropy_console_handler)

    # log astropy to file at specified level
    astropy_file_handler = logging.FileHandler(logfile)
    astropy_file_handler.setLevel(astropy_file_level)
    astropy_file_handler.addFilter(log_filter)
    astropy_log.addHandler(astropy_file_handler)

    logging.captureWarnings(True)
    warning_logger = logging.getLogger('py.warnings')
    warning_logger.addFilter(WarningContextFilter())
    warning_logger.addHandler(file_handler)
    warning_logger.addHandler(console_handler)
    warning_logger.propagate = False

    return root_logger

def get_logger(module_name):
    return logging.getLogger(module_name)

class OriginContextFilter(logging.Filter):
    # format the origin of the log record to be reasonably specific
    def filter(self, record):
        if hasattr(record, 'origin'):
            record.name = record.origin # files using astropy.log have name overwritten with 'astropy'
        elif hasattr(record, 'funcName'):
            # sometimes the console output (but not file) has a second log record with funcName twice, so trying to avoid that
            if record.funcName in record.name: return None
            else:
                record.name = ''.join([record.name,'.', record.funcName]) 
        return record

class WarningContextFilter(logging.Filter):
    # format records which are captured from warnings.warn to look like the other records, rather than their default
    def filter(self, record):

        try: 
            path, lineno, category, msg = (record.msg).split(':')
            path = path[path.index('site-packages')+14:-3]

            msg = msg.replace('\n', '')
            if 'warnings.warn' in msg:
                msg = msg[:msg.index('warnings.warn')-1]
            elif 'logger.warning' in msg:
                msg = msg[:msg.index('logger.warning')-1]
            if 'warn(' in msg:
                msg = msg[:msg.index('warn(')]
                
            record.msg = msg.strip()
            record.pathname = path
            record.lineno = lineno
            record.name = '.'.join(path.split('/'))
            record.funcName = f'Line {lineno}'
            return record
        except ValueError:
            # if splitting the record based on ':' fails, just return the record as-is
            return record
