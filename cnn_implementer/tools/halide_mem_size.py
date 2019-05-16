import argparse
import re
import csv
import sys
import logging

def stdin_lines():
    remaining=''
    while True:
        output = sys.stdin.read(256)
        if output == '':
            break

        #prefix output with whatever was remaining from previous iteration
        output = remaining + output

        #find the last newline
        newline_idx=output.rfind('\n')

        #everything after the last new line will be kept until next iteration
        remaining=output[newline_idx+1:]

        #if there was at least a valid newline, we can yield the output until the last newline
        if newline_idx!=-1:
            for line in output[0:newline_idx].split('\n'):
                yield line

    #finish up the remainder
    for line in remaining.split('\n'):
        yield line

def main(args=None):
    # Construct the argument parser
    parser = argparse.ArgumentParser(description="Halide Trace Analysis Tool: give memory usage halide program trace")

    parser.add_argument('-i', '--input', dest='inFile', action='store', required=False, default=None,
        help="Halide trace"
    )
    parser.add_argument('-o', '--output', dest='outFile', action='store', required=False, default=None,
        help="Csv file to store memory usage"
    )

    _LOG_LEVEL_STRINGS = ['ERROR', 'INFO', 'DEBUG']

    def _log_level_string_to_int(log_level_string):
        if not log_level_string in _LOG_LEVEL_STRINGS:
            message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, _LOG_LEVEL_STRINGS)
            raise argparse.ArgumentTypeError(message)

        log_level_int = getattr(logging, log_level_string, logging.ERROR)
        # check the logging log_level_choices have not changed from our expected values
        assert isinstance(log_level_int, int)

        return log_level_int

    parser.add_argument('--log-level',
    	default='ERROR',
    	dest='log_level',
    	type=_log_level_string_to_int,
    	nargs='?',
    	help='Set the logging output level. {0}'.format(_LOG_LEVEL_STRINGS)
    )

    # Parse arguments
    args = parser.parse_args()


    #Construct logger
    logging.basicConfig(level=args.log_level, format="%(message)s")
    log = logging.getLogger()

    #compile regex
    regex=re.compile("\s*(?P<name>[\w\d_\$]+):.*(peak|stack):\s*(?P<size>\d+).*")

    #scan trace for sizes
    if args.inFile:
        with open(args.inFile, 'rt') as f:
            log.info("Parsing from file: %s"%(args.inFile))
            entries=map(lambda m: m.groupdict(), filter(None,map(lambda l: regex.match(l), f.readlines())))
    else:
        log.info("Reading from stdin")
        entries=[]
        for line in stdin_lines():
            m=regex.match(line)
            if m:
                entries+=[m.groupdict()]

            #pass through everything
            sys.stdout.write(line+'\n')
            sys.stdout.flush()

    #potentially log to output
    log.info("Memory sizes:")
    for e in entries:
        log.info(' +%s: %s'%(e['name'],e['size']))

    #write to output
    if args.outFile:
        log.info("Storing in file: %s"%(args.outFile))
        with open(args.outFile, 'wt') as f:
            writer=csv.writer(f, delimiter=',')
            for e in entries:
                writer.writerow(map(lambda k: e[k], ['name','size']))
