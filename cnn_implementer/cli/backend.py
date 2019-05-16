import arguments
from .. import backends
from ..model import NetworkConfigs, Network
import logging
import argparse
import os

def backend_driver(args, net, point, logger=None):
    #code generation with backend
    if args.backend == 'halide':

        #Get halide backend to implement the selected point
        halide=backends.Halide(
            logger=logger,
        )

        #generate the code for the selected point
        code=halide.implement(
            net,
            point,
            DEBUG=args.halideDebug,
            PROFILING=args.halideProfiling,
            TRACING=args.halideTracing
        )

        #write code to file
        if args.halideOutput:
            with open(args.halideOutput, 'wt') as f:
                f.write(code)

        return code

def standalone():
    #Construct the parser
    parser = argparse.ArgumentParser(description="CNN implementation tool")

    #Add arguments
    arguments.generic.add(parser)
    group=arguments.backend.add(parser)

    #Standalone arguments
    group.add_argument('-n', '--net', dest='net', required=True, action='store',
        help="Network model stored with the frontend"
    )
    group.add_argument('-p', '--points', dest='points', action='store', required=True,
        help="Design points found by DSE"
    )
    group.add_argument('-i', '--index', dest='pointIdx', action='store', required=False, default=0,type=int,
        help="Index of design point to implement (default=0)"
    )

    # Parse arguments
    args = parser.parse_args()

    #Construct the logger
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger()

    #Sanity check on arguments
    if not args.halideOutput:
        logger.warning("No output file specified, running this command will have no effect")

    #load the network
    net=Network.load(args.net)

    #Load configurations
    points=NetworkConfigs(fname=args.points)

    #Select point to implement
    assert(args.pointIdx>=0 and args.pointIdx<len(points) and "Invalid point index")
    point=points[args.pointIdx]

    #run thebackend
    files=backend_driver(args, net, point, logger)

if __name__ == '__main__':
    standalone()
