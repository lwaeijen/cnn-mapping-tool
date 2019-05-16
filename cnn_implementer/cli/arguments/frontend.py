import argparse

# Frontend options
def add(parser, standalone=False):

    frontend = parser.add_argument_group('Frontend Options')

    frontend.add_argument('-f', '--frontend', dest='frontend', choices=[
            'caffe',
        ],
        action='store',
        default="caffe",
        help="Frontend selection (default=caffe)"
    )

    frontend.add_argument('--caffe-deploy', dest='caffeDeploy', action='store', default=None, required=standalone,
        help=".prototxt Network deploy description"
    )

    frontend.add_argument('--caffe-model', dest='caffeModel', action='store', default=None,
        help=".caffemodel with trained weights"
    )

    frontend.add_argument('--write-network-dotfile', dest='networkDotfile', action='store', default=None,
        help='Write out loaded network to dotfile'
    )

    frontend.add_argument('-o', '--write-model', dest='modelOutfile', action='store', default=None,required=True,
        help='Write out network model to specified file'
    )

    #return argument group
    return frontend
