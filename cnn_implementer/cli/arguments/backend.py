
def add(parser):

    #Backend Options
    backend=parser.add_argument_group('Backend Options')

    backend.add_argument('-b', '--backend', dest='backend', choices=[
            'halide',
            'none',
        ],
        action='store',
        default="halide",
        help="Backend selection for code emision (default=halide)"
    )

    backend.add_argument('--halide-code', dest='halideOutput', action='store', default=None,
        help="Output directory for generated halide code"
    )

    backend.add_argument('--halide-debug-code', dest='halideDebug', action='store_true', default=False,
        help="Insert debug messages in generated halide code"
    )

    backend.add_argument('--halide-profile-code', dest='halideProfiling', action='store_true', default=False,
        help="Enable profiling of generated halide code"
    )

    backend.add_argument('--halide-trace-code', dest='halideTracing', action='store_true', default=False,
        help="Insert load and store tracing into generated halide code"
    )

    #return argument group
    return backend
