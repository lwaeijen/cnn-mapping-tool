import arguments
import argparse
from .. import frontends
import logging
import os

def frontend_driver(args, logger=None):

    #Frontend load design space
    if args.frontend == 'caffe':

        #Load network structure from file
        net = frontends.caffe.loadNet(args.caffeDeploy, logger=logger)

        #store loaded network to dotfile if required
        if args.networkDotfile:
            net.write_dot(args.networkDotfile)

        #if caffe model is specified, load weights and bias values as well
        if args.caffeModel:
            outdir=os.path.dirname(args.modelOutfile)
            net.weights=frontends.caffe.loadWeights(args.caffeModel, net, outdir, logger)
        else:
            if logger: logger.warning("No .caffemodel specified. Will not generate weight and bias binary files.")

        #store the model to a binary format for later use
        if args.modelOutfile:
            net.save(args.modelOutfile)

        #return the loaded net
        return net

def standalone():
    #Construct the parser
    parser = argparse.ArgumentParser(description="CNN implementation tool")

    #Add arguments
    arguments.generic.add(parser)
    arguments.frontend.add(parser, standalone=True)

    # Parse arguments
    args = parser.parse_args()

    #Construct the logger
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger()

    #Argument sanity checks
    if not args.networkDotfile and not args.modelOutfile:
        logger.warning("No model output or dotfile specified. This command will not generate any output files.")

    #run the frontend
    net=frontend_driver(args, logger)

if __name__ == '__main__':
    standalone()
