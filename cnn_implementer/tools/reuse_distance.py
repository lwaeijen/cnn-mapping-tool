import subprocess
import logging
import argparse
import csv
from threading import Thread, Lock
import multiprocessing
import sys
import bisect
from copy import deepcopy
try:
    #python3
    from queue import Queue
except:
    #python 2
    from Queue import Queue

def main(args=None):

    # Construct the argument parser
    parser = argparse.ArgumentParser(description="Halide Trace Analysis Tool: give reuse distance information from trace-enabled halide program")

    parser.add_argument('-o', '--output', dest='outFile', action='store', required=False, default=None,
        help="Csv file to store reuse distance frequencies"
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


    with multiprocessing.Manager() as manager:

        hist = manager.dict()
        load_counter=manager.Value(int, 0)
        load_counter.value=0

        #Internal settings
        q = Queue(maxsize=1)



        def index(a, x):
            'Locate the leftmost value exactly equal to x'
            i = bisect.bisect_left(a, x)
            if i != len(a) and a[i] == x:
                return i
            raise ValueError

        #thread function to process lines from the halide program
        def process_lines(q, load_counter, hist):

            #keep track of last access
            last_access_addr2time={}
            last_access_sorted_time=[]
            lhist={'inf':0}
            load_cnt=0


            #daemon loop
            while True:


                #get linebuf from the queue
                linebuf=q.get()

                #decode to regular string (python 3)
                linebuf=linebuf.decode('utf8')

                #process all lines in the buffer
                for line in linebuf.split('\n'):
                    if line.startswith('Store'): # and ')' in line:

                        #Note: uncomment for intermediate storing for backup
                        #if load_cnt&0xFFFFFF==0:
                        #    with open('tmp_histogram.txt','wt') as f:
                        #        f.write(str(lhist))

                        buf=line[6:line.index(')')+1]

                       # #never accessed before
                        if buf not in last_access_addr2time:
                            #+1 cold miss
                            lhist['inf']+=1

                        #accessed before, update hist
                        else:
                            #get index of last time this element was accesses
                            last_time_idx=index(last_access_sorted_time, last_access_addr2time[buf])

                            ##number of unique accesses ince then
                            unique=len(last_access_sorted_time)-last_time_idx

                            ##remove old last access time
                            del last_access_sorted_time[last_time_idx]

                            ##update the histogram
                            if unique not in lhist:
                                lhist[unique]=0
                            lhist[unique]+=1

                        #store current access as last access
                        last_access_addr2time[buf]=load_cnt

                        #update last access time
                        last_access_sorted_time+=[load_cnt]

                        #increment counter
                        load_cnt+=1


                    elif line[0:4]=='Load':
                        #we don't care about loads here
                        pass
                    else:
                        sline=line.strip()
                        if sline!='':
                            #This is not a load or store line, pass it through
                            sys.stdout.write(line+'\n')
                            sys.stdout.flush()

                #share results with parent
                #note: explicit assignment to trigger proper sharing with manager object
                for k, v in lhist.items():
                    hist[k]=v
                load_counter.value=load_cnt

                #signal that the linebuf is processed
                q.task_done()


        #launch worker thread
        log.info('Launch worker thread to process lines')
        worker = Thread(target=process_lines, args=(q,load_counter, hist))
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
                for dist in sorted(hist.keys()):
                    freq=hist[dist]
                    writer.writerow([dist,freq])
            log.info("Saved to %s"%(args.outFile))

        #nicely format results and report to command line
        s=['Total loads: %d'%load_counter.value]
        for dist in sorted(hist.keys()):
            freq=hist[dist]
            s+=[' %s: %s '%(dist, freq)]
        log.info('\n'.join(s))

if __name__ == '__main__':
    main()
