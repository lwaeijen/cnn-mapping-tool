class Logger(object):
    def __init__(self, logger=None, **kwargs):
        self.logger=logger

    def log(self, msg, level='info'):
        #If there is not proper logger, just use printing
        if not self.logger:
            print msg
            return

        #Figure out message level
        lvl = level.lower()
        if lvl in ['info']:
            self.logger.info(msg)
        elif lvl in ['debug']:
            self.logger.debug(msg)
        else:
            self.warning("Unknown log level for message:%s"%msg)

    def debug(self, msg):
        return self.log(msg, level='debug')

    def info(self, msg):
        return self.log(msg, level='info')
