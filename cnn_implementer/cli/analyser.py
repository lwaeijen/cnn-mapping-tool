import arguments
from ..model import NetworkConfigs, Network, Segment
import logging
import argparse
from ..optimizers import foldBuffers, unfoldBuffers

def analyser_driver(args, net, points, logger=None):

    #for all points
    for point in points:

        #update the configurations by applying any optimizations and analysing the costs
        for segment_cfg in point.segmentconfigs:
            segment=Segment(network=net, layers=segment_cfg.layers)

            #perfrom optimizations
            if not args.analyseOnly:
                if args.bufferFolding:
                    #apply buffer folding, possibly forcing the size
                    foldBuffers(net, segment_cfg, force=args.forceBufferFolding)
                else:
                    #flag is set to not use folding, unfold any possibly folded buffers
                    unfoldBuffers(net, segment_cfg)


            #analyse the cost of this segment
            segment.eval_config(segment_cfg, ignore_output=args.ignoreOutput, noceil=args.noCeil)

        #depenings on settings we output a summary
        s=["Total costs:"]
        s+=["  accesses:    %d"%(point.cost.accesses)]
        s+=["  buffer size: %d"%(point.cost.buffer_size)]
        s+=["  MACs:        %d"%(point.cost.macs)]
        logger.info('\n'.join(s))

        s=["-"*20]
        s+=["Detailed:"]
        for segment_cfg in point.sorted_segments(net):
            if not set(['convolution', 'input']).isdisjoint(set([ net.nodes[lyr]['type'] for lyr in segment_cfg.layers])):
                s+=[' +'+segment_cfg.name]
                s+=['   -data accesses:      '+str(segment_cfg.cost.data_accesses)]
                s+=['   -weight accesses:    '+str(segment_cfg.cost.weight_accesses)]
                s+=['   -data buffer size:   '+str(segment_cfg.cost.data_buffer_size)]
                s+=['   -weight buffer size: '+str(segment_cfg.cost.weight_buffer_size)]
                s+=['   -MACs:               '+str(segment_cfg.cost.macs)]
        logger.info('\n'.join(s))


    #write configs to file
    if args.output:
        points.save(args.output)

    return points

def standalone():
    #Construct the parser
    parser = argparse.ArgumentParser(description="CNN network config analysis and optimization tool")

    #Add arguments
    arguments.generic.add(parser)

    #Standalone arguments
    group=parser.add_argument_group('Analyser options')
    group.add_argument('-n', '--net', dest='net', required=True, action='store',
        help="Network model stored with the frontend"
    )
    group.add_argument('-p', '--points', dest='points', action='store', required=True,
        help="Design points found by DSE"
    )
    group.add_argument('-i', '--index', dest='pointIdx', action='store', required=False, default=None,type=int,
        help="Index of design point to implement (default is all points in the input file)"
    )
    group.add_argument('-o', '--output', dest='output', action='store', required=False, default=None,
        help="Output file to store network config with costs"
    )
    group.add_argument('-a', '--analyse-only', dest='analyseOnly', action='store_true', required=False, default=False,
        help="Only perform analyses and do not apply any optimizations"
    )

    #optimization arguments
    group=parser.add_argument_group('Optimization switches')
    group.add_argument('--opt-no-buffer-folding', dest='bufferFolding', action='store_false', default=True,
        help="Disable the folding of buffers"
    )
    group.add_argument('--opt-no-force-buffer-folding', dest='forceBufferFolding', action='store_false', default=True,
        help="Do not force folded buffer size to minimum (allows halide to use circular buffer with power of two sizes)"
    )

    #Validation arguments
    group=parser.add_argument_group('Experimental switches')
    group.add_argument('--exp-ignore-output-buffers', dest='ignoreOutput', action='store_true', default=False,
        help="Ignore the contributions of the output buffer for experimental purposes only!"
    )
    group.add_argument('--exp-no-ceil', dest='noCeil', action='store_true', default=False,
        help="Use regular division instead of ceiling for tighter, but possibly overly optimistic models"
    )

    # Parse arguments
    args = parser.parse_args()

    #Construct the logger
    logging.basicConfig(level=args.log_level, format='%(message)s')
    logger = logging.getLogger()

    #Sanity check on arguments
    if not args.output:
        logger.warning("No output file specified, running this command will have no effect")

    #load the network
    net=Network.load(args.net)

    #Load configurations
    points=NetworkConfigs(fname=args.points)

    #Select point to implement
    if args.pointIdx!=None:
        assert(args.pointIdx>=0 and args.pointIdx<len(points) and "Invalid point index")
        points=NetworkConfigs([points[args.pointIdx]])

    #run thebackend
    code=analyser_driver(args, net, points, logger)

if __name__ == '__main__':
    standalone()
