import os
import jinja2
import networkx as nx
from ..utils import Logger
from math import ceil, floor
from ..model import Segment

#Add function to Segments that generates unique names for internal nodes
#Function is specific for halide backend, hence it is added here and not in the original definition of Segment
def halide_name(self, layer_name):
    def sanitize(s):
        return s.replace('/','_')

    #if not on of our nodes, don't rename
    if layer_name not in self.nodes():
        return sanitize(layer_name)

    #if an output, don't rename
    if layer_name in self.produces:
        return sanitize(layer_name)

    # This is an internal node, there might be other segments with similar named layers
    # return a name with the segment id added
    return sanitize(layer_name+'_seg'+str(self.id))

Segment.halide_name=halide_name

class Halide(Logger):
    def __init__(self, **kwargs):
        #init super
        super(Halide, self).__init__(**kwargs)

        self.template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        self.main_template='main.c.j2'

    def implement(self, network, networkConfig, DEBUG=False, PROFILING=False, TRACING=False):

        self.network = network

        #contains the filenames of the binaries
        self.params = self.network.weights

        #sanity check
        assert(self.params and "Can not implement a network if the weights are not specified!")

        #convert config segments to real segments
        segments=[]
        inputs=0
        info_msg=['Halide Backend implementing the selected segments:']
        for segment_cfg in networkConfig.sorted_segments(self.network):

            #Init segment with required layers
            segment = Segment(network=network, layers=segment_cfg.layers)

            #convert to old-style tuple
            segments+=[(segment, segment_cfg)]

            #some extra bookkeeping
            for layer in segment.get_nodes():
                if layer['type'] in ['input']:
                    inputs+=1

            #tell the user what we are up to
            info_msg+=['  - '+str(segment)]
        self.info('\n'.join(info_msg))

        #render main code
        code = self.render({
            'segments':      segments,
            'number_inputs': inputs,
            'debug':         DEBUG,
            'PROFILING':     PROFILING,
            'TRACING':       TRACING,
        })

        #return dictionary with filenames
        return code

    def render(self, context):
        #helper function to translate internal model dimensions to halide backend dimensions
        def rename(dim):
            rt = {
                'x':  'n',
                'y':  'm',
                'zo': 'o',
                'zi': 'i',

                'x_i':  'n_i',
                'y_i':  'm_i',
                'zo_i': 'o_i',
                'zi_i': 'i_i',

                'x_o':  'n_o',
                'y_o':  'm_o',
                'zo_o': 'o_o',
                'zi_o': 'i_o',
            }
            assert(dim in rt and "Error: unknown dimension, check your config file and also make sure it is analysed to ensure proper compute levels are defined")
            return rt[dim]

        def rename_order(order, postfix='', exclude=[]):
            return [ rename(dim)+postfix for dim in order if rename(dim)[0] not in exclude]

        def jinja_debug(text):
            print 'Halide Backend Template Debug:', text
            return

        #context to pass to jinja
        ctxt={
            'nx': nx,
            'network': self.network,
            'rename': rename,
            'rename_order': rename_order,
            'len': len,
            'type': type,
            'range': range,
            'enumerate': enumerate,
            'max': max,
            'min': min,
            'params': self.params,
            'map': map,
            'str': str,
            'int': int,
            'float': float,
            'floor': floor,
            'ceil': ceil,
            'list': list,
        }
        ctxt.update(context)

        #set the environment
        env=jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir)
        )

        #add filter
        env.filters['dbg']=jinja_debug

        #load template and render
        return env.get_template(self.main_template).render(ctxt)
