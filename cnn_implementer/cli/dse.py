import arguments
import argparse
from .. import dse
from ..model import Network
import logging

def dse_driver(args, net, logger=None):

    if args.dseStrategy.lower()=='random':
        #init the design space using the random search method
        ds = dse.RandomSearch(
            net,
            TESTING = args.dseTesting,
            logger=logger,
        )
    elif args.dseStrategy.lower()=='naive':
        #perform a search over the naive schedules (only vary compute and store levels)
        ds = dse.Naive(
            net,
            logger=logger,
        )
    else:
        #init the design space using the bruteforce method
        ds = dse.BruteForce(
            net,
            TESTING = args.dseTesting,
            logger=logger,
        )

    #run the explorations
    ds.explore(
        pareto = args.dseParetoOnly,

        min_fused_layers = args.dseMinFused,
        max_fused_layers = args.dseMaxFused,

        #fold into circular buffers or not
        buffer_folding = args.dseBufferFolding,

        #fold to minimum or alow to fold to next power of 2 for performance but more memory usage
        force_fold= args.dseForceBufferFolding,

        #ignore the output buffers contribution while exploring
        ignore_output=args.ignoreOutput,

        #do not use the ceiling operator
        noceil=args.noCeil,

        #tiling can be omitted by setting it to the size of the featuremaps
        notiling = args.dseNoTiling
    )

    #report results to user
    logger.info("Found %s points"%(len(list(ds.points))))

    #Save the space if required
    if args.dseOutputPoints:
        logger.info("Saving found points to \"%s\""%(args.dseOutputPoints))
        ds.points.save(args.dseOutputPoints)

    #return design points
    return ds.points


def standalone():

    #Construct the parser
    parser = argparse.ArgumentParser(description="CNN design space exploration tool")
    arguments.generic.add(parser)
    group=arguments.dse.add(parser)

    #Standalone arguments
    group.add_argument('-n', '--net', dest='net', required=True, action='store',
        help="Network model stored with the frontend"
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
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger()

    #Argument sanity checking
    if not args.dseOutputPoints:
        logger.warning("No points output file file specified, command will have no effect")

    #load network
    net=Network.load(args.net)

    #run the design space exporation
    points=dse_driver(args, net, logger)


if __name__ == '__main__':
    standalone()
