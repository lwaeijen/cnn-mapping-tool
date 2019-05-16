import argparse
from ..frontends.caffe import caffe_pb2
import google.protobuf.text_format as txtf
import logging


def main():

    _LOG_LEVEL_STRINGS = ['ERROR', 'INFO', 'DEBUG']

    # Construct the argument parser
    parser = argparse.ArgumentParser(description="Caffe format conversion tool. Can convert prototxt into caffemodel and vice versa")

    parser.add_argument('-i', '--input', dest='inFile', action='store', required=True,
        help="Input file with either .prototxt or .caffemodel extension"
    )

    parser.add_argument('-o', '--output', dest='outFile', action='store', required=False, default=None,
        help="Output file, if ommited will be same filename as input with different extension"
    )

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

    # Construct logger
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger()

    #sanity check on input file
    if not (args.inFile.endswith('prototxt') or args.inFile.endswith('caffemodel')):
        raise TypeError('Input file does not have the correct extension, should be either .prototxt or .caffemodel')

    #default output if needed
    if not args.outFile:
        if args.inFile.endswith('prototxt'):
            args.outFile=args.inFile[0:-9]+'.caffemodel'
        else:
            args.outFile=args.inFile[0:-11]+'.prototxt'

    #create caffe net representation
    net=caffe_pb2.NetParameter()

    #from prototxt to caffemodel (binary protobuf)
    if args.inFile.endswith('.prototxt'):
        logger.info('Opening input file %s'%args.inFile)
        with open(args.inFile, 'rt') as f:
            txtf.Merge(f.read(), net)

        logger.debug('Parsed Network:'+str(net))

        logger.info('Writing to %s'%args.outFile)
        with open(args.outFile, 'wt') as f:
            f.write(net.SerializeToString())

    #from caffemodel to prototxt
    else:
        logger.info('Opening input file %s'%args.inFile)
        with open(args.inFile, 'rb') as f:
            net.ParseFromString(f.read())

        logger.debug('Parsed Network:'+str(net))

        logger.info('Writing to %s'%args.outFile)
        with open(args.outFile, 'wt') as f:
            f.write(str(net))
