import json
from itertools import islice
from math import log, ceil, floor

#check if matplotlib exists, o.w. warn user and exit
import pkgutil;
if not pkgutil.find_loader("matplotlib"):
    print "Matplotlib not found, please install to continue with the plot program, for example by: \"pip install matplotlib\""
    exit(-1)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def load_costs(fname):
    with open(fname, 'rt') as f:
        configs=json.loads(f.read())
    return [ cfg['networkcost'] for cfg in configs]

def transpose(l):
    #take a list of dictionaries and turn it into a dictionary of lists
    d={}
    for e in l:
        for k in e.keys():
            if k not in d:
                d[k]=[]
            d[k]+=[e[k]]
    return d

def pareto_lines(x_arr,y_arr):
    #take pareto points sorted on x and calc all lines to illustrate the pareto front
    points=[(x,y) for x,y in zip(x_arr, y_arr)]
    for (ax,ay),(bx, by) in window(points):
        yield (ax,ay),(bx,ay) #horizontal line piece
        yield (bx,ay),(bx,by) #vertical line piece

def plot_pareto(x,y, point_color='blue', line_color='black', zorder=1):
    #simple scatter plot of the points
    plt.scatter(x,y, zorder=zorder, c=point_color)

    #add lines to illustrate pareto front
    for (ax,ay),(bx, by) in pareto_lines(x,y):
        plt.plot((ax,bx), (ay,by), c=line_color, zorder=zorder+1)


def bsize_vs_accesses(trends, fname=None, colors=['red','blue','purple','pink'], names=None, max_bsize=None, min_bsize=None):
    #clear the figure from any existing data/plots
    plt.clf()

    assert(len(colors)>=len(trends) and "Please specify more colors!!")

    #patch up names for the zip
    if names==None:
        names=[None]*len(trends)

    #list with handles for the legend
    legend_handles=[]

    #loop over all trends
    min_log_bsize=None
    max_log_bsize=None
    for costs, color, name in zip(trends, colors, names):

        #sort points by buffer size and make arrays per dimensions
        costs=transpose(sorted(costs, key=lambda e: e['buffer_size']))

        #shorthands
        bsize=costs['buffer_size']
        acc=costs['accesses']

        #trim to min and max sizes if specified
        if min_bsize:
            i=0
            while bsize[i] < min_bsize*1024:
                i+=1
                if i>=len(bsize):
                    break
            bsize=bsize[i:]
            acc=acc[i:]
        if max_bsize:
            i=len(bsize)-1
            while bsize[i] > max_bsize*1024:
                i-=1
                if i==0:
                    break
            bsize=bsize[:i]
            acc=acc[:i]

        #transform to two-log
        log_bsize=map(lambda b: log(b,2), bsize)

        #find min and max of blog far axis scaling
        min_log_bsize = min(log_bsize) if not min_log_bsize else min(min_log_bsize, min(log_bsize))
        max_log_bsize = max(log_bsize) if not max_log_bsize else max(max_log_bsize, max(log_bsize))

        #plot the pareto curve
        plot_pareto(log_bsize, acc, zorder=2,point_color='black', line_color=color)

        if name!=None:
            legend_handles+=[mlines.Line2D([], [], color=color, marker='.', markersize=15, label=name)]

    #add labels
    plt.xlabel("Buffer size [kB]")
    plt.ylabel("External Accesses [#]")

    #set ticks x ticks, match labels to non-log number
    bsize_range=range(int(floor(min_log_bsize)), int(ceil(max_log_bsize))+1)
    plt.xticks(bsize_range, map(lambda x: "%.2f"%round(float(x),2), map(lambda x: float((2**x))/1024.0, bsize_range)),rotation=45)

    #set y axis to 10 log
    ax = plt.gca()
    ax.set_yscale('log')

    #enable grid
    ax.grid(True, linestyle='-')
    ax.grid(True, which='minor', axis='y', linewidth=0.5, linestyle='--')

    #add legend
    if legend_handles!=[]:
        ax.legend(handles=legend_handles) #, loc='upper right')

    #make sure the figure fits (makes room for all labels etc)
    plt.tight_layout()

    if fname:
        #save to file
        plt.savefig(fname)


def main(args=None):
    import argparse
    import os

    #Construct the parser
    parser = argparse.ArgumentParser(description="CNN plot tool")

    parser.add_argument('-c', '--costs', nargs='+', dest='fcost', required=True, action='store',
        help="File with configurations and their associated costs"
    )

    parser.add_argument('-l', '--legend', nargs='+', dest='series', required=False, action='store',
        help="list of series names"
    )

    parser.add_argument('-b', '--bsize-vs-accesses', dest='fbva', required=False, action='store', default=None,
        help="Filename for buffer size vs accesses plot"
    )

    parser.add_argument('-u', '--max-bsize', dest='max_bsize', type=float, required=False, action='store', default=None,
        help="Maximum buffer size to be plotted"
    )

    parser.add_argument('-m', '--min-bsize', dest='min_bsize', type=float, required=False, action='store', default=None,
        help="Minimum buffer size to be plotted"
    )

    parser.add_argument('-f', '--fontsize', dest='fontsize', required=False, action='store', default=18,
        help="Fontsize for labels and text in the figure"
    )

    parser.add_argument('-s', '--silent', dest='silent', required=False, action='store_true', default=False,
        help="Do not show plots"
    )

    #colors
    #colors=['red','blue','green']

    #theme colors blue, red, green, purple
    colors=['#268bd2', '#cb4b16', '#859900', '#6c71c4']

    # Parse arguments
    args = parser.parse_args()

    #set fontsize
    font = {
        #'family' : 'normal',
        #'weight' : 'bold',
        'size'   : args.fontsize
    }
    matplotlib.rc('font', **font)

    #load costs from file
    trends= [ load_costs( fcost ) for fcost in args.fcost ]

    #names of the series in legend
    series=map(os.path.basename, args.fcost) if not args.series else args.series

    #plot buffer size vs accesses
    if args.fbva or (not args.silent):
        bsize_vs_accesses(trends, args.fbva, names=series, colors=colors, min_bsize=args.min_bsize, max_bsize=args.max_bsize)
        if not args.silent:
            plt.show()
