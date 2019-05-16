from ... import __version__
import logging
import argparse

def add(parser):
    #GENERIC OPTIONS
    parser.add_argument('--version', action='version', version='%(prog)s '+str(__version__))

    _LOG_LEVEL_STRINGS = ['ERROR','WARNING', 'INFO', 'DEBUG']
    def _log_level_string_to_int(log_level_string):
        if not log_level_string in _LOG_LEVEL_STRINGS:
            message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, _LOG_LEVEL_STRINGS)
            raise argparse.ArgumentTypeError(message)
        log_level_int = getattr(logging, log_level_string, logging.ERROR)
        # check the logging log_level_choices have not changed from our expected values
        assert isinstance(log_level_int, int)
        return log_level_int

    parser.add_argument('--log-level',
    	default='WARNING',
    	dest='log_level',
    	type=_log_level_string_to_int,
    	nargs='?',
    	help='Set the logging output level. {0}'.format(_LOG_LEVEL_STRINGS)
    )

    #return argument group
    return None
