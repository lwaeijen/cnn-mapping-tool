import subprocess
import logging
import argparse
import csv
from threading import Thread, Lock
import multiprocessing
import sys
try:
    #python3
    from queue import Queue
except:
    #python 2
    from Queue import Queue

def main(args=None):

    # Construct the argument parser
    parser = argparse.ArgumentParser(description="Halide Trace Analysis Tool: give access counts from trace-enabled halide program")

    parser.add_argument('-o', '--output', dest='outFile', action='store', required=False, default=None,
        help="Csv file to store access counts"
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
    logging.basicConfig(level=args.log_level, format="%(threadName)s:%(message)s")
    log = logging.getLogger()

    #Internal settings
    num_threads = multiprocessing.cpu_count()
    q = Queue(maxsize=num_threads*2)
    global_counts={}
    global_counts_lock = Lock()

    #thread function to process lines from the halide program
    def process_lines(q):

        def ensure(buf):
            if buf not in local_counts:
                local_counts[buf]={
                        'loads':0,
                        'stores':0
                }

        #daemon loop
        while True:

            #clear local counts every iteration
            local_counts={}

            #get linebuf from the queue
            linebuf=q.get()

            #decode to regular string (python 3)
            #linebuf=linebuf.decode('utf8')

            #process all lines in the buffer
            for line in linebuf.split('\n'):

                if line[0:4]=='Load':
                    if '.' not in line:
                        print 'ERROR PARSING:', line
                    else:
                        buf=line[5:line.index('.', 5)]
                        ensure(buf)
                        local_counts[buf]['loads']+=1
                elif line[0:5]=='Store':
                    buf=line[6:line.index('.', 6)]
                    ensure(buf)
                    local_counts[buf]['stores']+=1
                else:
                    sline=line.strip()
                    if sline!='':
                        #This is not a load or store line, pass it through
                        sys.stdout.write(line+'\n')
                        sys.stdout.flush()

            #get lock on global counts and merge local counts
            with global_counts_lock:
                for buf in local_counts:
                    if buf not in global_counts:
                        global_counts[buf]={
                            'loads':0,
                            'stores':0
                        }
                    for access_type in local_counts[buf]:
                        global_counts[buf][access_type]+=local_counts[buf][access_type]

            #signal that the linebuf is processed
            q.task_done()


    #launch worker threads
    log.info('Launching %d threads to process command output'%(num_threads))
    for i in range(num_threads):
        worker = Thread(target=process_lines, args=(q,))
        worker.setDaemon(True)
        worker.start()


    #read lines from stdin and put into queue for processing
    log.info('Processing stdin')
    remaining=''
    while True:
        output = sys.stdin.read(8192*32)
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
            q.put(output[0:newline_idx])

    #perhaps there was some trailing output that didn't terminate with a newline, yield it here
    if remaining!='':
        q.put(remaining)
    log.info('Done reading from stdin')

    #wait for all lines to get processed
    log.info('Waiting for threads to finish processing')
    q.join()

    #store to csv file
    if args.outFile:
        with open(args.outFile, 'wt') as f:
            writer=csv.writer(f, delimiter=',')
            for buf in global_counts:
                for d in ['loads', 'stores']:
                    writer.writerow([buf+'_'+d, global_counts[buf][d]])
        log.info("Saved to %s"%(args.outFile))

    #nicely format results and report to command line
    s=['Accesses:']
    for buf in global_counts:
        s+=['+'+buf]
        s+=[' -loads:  '+str(global_counts[buf]['loads'])]
        s+=[' -stores: '+str(global_counts[buf]['stores'])]
    log.info('\n'.join(s))

if __name__ == '__main__':
    main()
