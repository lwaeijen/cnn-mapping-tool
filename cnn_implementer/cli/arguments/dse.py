

def add(parser):
    # Design space options
    dseg = parser.add_argument_group('Design Space Exploration Options')

    dseg.add_argument('--dse-save-points', dest='dseOutputPoints', action='store', default=None,
        help="Save design points to json file"
    )

    dseg.add_argument('--dse-all', dest='dseParetoOnly', action='store_false', default=True,
        help="Keep all design points during exploration, not only pareto points"
    )

    dseg.add_argument('--dse-testing', dest='dseTesting', action='store_true', default=False,
        help="Use pseudorandom costs instead of real formulas"
    )

    dseg.add_argument('--dse-no-tiling', dest='dseNoTiling', action='store_true', default=False,
        help="No tiling is applied by setting the tile size equal to the output dimensions of each feature map"
    )

    dseg.add_argument('--dse-no-buffer-folding', dest='dseBufferFolding', action='store_false', default=True,
        help="Disable the folding of buffers (for testing purposes)"
    )

    dseg.add_argument('--dse-no-force-buffer-folding', dest='dseForceBufferFolding', action='store_false', default=True,
        help="Do not enforce minimal size buffer folds (allows halide to fold buffer sizes to powers of two for efficiency)"
    )

    dseg.add_argument('--dse-min-fused-layers', dest='dseMinFused', action='store', type=int, default=1,
        help="Minimim number of fused layers"
    )

    dseg.add_argument('--dse-max-fused-layers', dest='dseMaxFused', action='store', type=int, default=-1,
        help="Maximum number of fused layers"
    )

    dseg.add_argument('--dse-strategy', dest='dseStrategy', action='store', default="bruteforce",
        choices=[
            "bruteforce",
            "random",
            "naive",
        ],
        help="Search strategy to use. When \"random\" is speficied 1 randomly generated point will be returned. Default=bruteforce"
    )

    #return argument group
    return dseg
