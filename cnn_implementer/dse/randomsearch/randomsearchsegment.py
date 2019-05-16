from ..dsesegment import DSESegment
from random import choice, shuffle, randint
from random import seed as rseed
from ...model import LayerConfig, SegmentConfig
import networkx as nx
from math import ceil, log

class RandomSearchSegment(DSESegment):

    def config(self,
            notiling=False,  #if set to true tiles will be set to the featuremap dimensions
            seed=None,
            **kwargs
        ):

        #Seed the rng if requested
        if seed:
            rseed(seed)

        #Get output layer
        out = self.last

        #randomly select a config for this segment
        cfg={
            'Tx':               choice([2**p for p in range(int(ceil(log(out['xo'],2)))+1)]) if not notiling else out['xo'],
            'Ty':               choice([2**p for p in range(int(ceil(log(out['yo'],2)))+1)]) if not notiling else out['yo'],
            'Tzo':              choice([2**p for p in range(int(ceil(log(out['zo'],2)))+1)]) if not notiling else out['zo'],
        }
        if out['type'].lower() in ['convolution']:
            cfg['Tzi']= choice([2**p for p in range(int(ceil(log(out['zi']/out['groups'],2)))+1)]) if not notiling else out['zi']/out['groups']

        #Fix loop order
        inner=['zo_i','y_i','x_i']+ ( ['zi_i'] if out['type'].lower() in ['convolution'] else [])
        shuffle(inner)

        outer_loops=['zo_o','y_o','x_o']+ ( ['zi_o'] if out['type'].lower() in ['convolution'] else [])
        outer=choice(outer_loops)

        cfg['order']=inner+[outer]+filter(lambda lvl: lvl!=outer, outer_loops)

        #For each convolution layer there is a store data and store weights level
        store_keys=[]
        for l in self.nodes():
            if self.nodes[l]['type'] in ['convolution']:
                data_st=randint(0,5)
                cfg[str(l)+'.store_at']          = cfg['order'][data_st]
                cfg[str(l)+'.store_at_weights']  = cfg['order'][randint(data_st,5)]


        #NOTE: simply copy paste from bruteforce, code can be optimized by merging with preceeding code
        #translate dict into new-style segment config
        layer_configs=[]
        for l in self.nodes():
            if self.nodes[l]['type'] in ['convolution']:
                layer_configs += [LayerConfig(
                    l,                          #name
                    cfg[l+'.store_at'],         #store level
                    cfg[l+'.store_at_weights'],
                    cfg[l+'.store_at'],         #NOTE: COMPUTE level!! (for now always the same as store)
                    cfg[l+'.store_at_weights']
                )]
            else:
                #still need to add a layerconfig for this layer, but init with none
                layer_configs += [LayerConfig(l, None, None, None, None)]

        config=SegmentConfig(

            #Segment name
            self.name,

            #order
            cfg['order'],

            #tiling
            {
                'Tx'  : cfg['Tx'],
                'Ty'  : cfg['Ty'],
                'Tzo' : cfg['Tzo'],
                'Tzi' : cfg['Tzi'] if 'Tzi' in cfg else -1,
            },

            #output_store_at
            choice(cfg['order'][:len(inner)+1]),

            #layer configs
            layer_configs
        )

        return config
