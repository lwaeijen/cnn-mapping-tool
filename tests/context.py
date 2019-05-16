import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cnn_implementer

from subprocess import Popen, PIPE
def run_cmd(cmd, log=None):
    if log: log.debug("Executing: %s"%(cmd))
    proc=Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    if log:
        if proc.returncode:
            #not succesfull, dump stderr
            for line in stderr.split('\n'):
                if line.strip():
                    log.error('stderr: '+line)
        else:
            #succesfull
            for line in stdout.split('\n'):
                if line.strip():
                    log.debug('stdout: '+line)
    return stdout, stderr, proc.returncode
