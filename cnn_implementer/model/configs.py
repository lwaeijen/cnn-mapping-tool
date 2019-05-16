from costs import *
from ..utils import pareto_cost_config
from copy import deepcopy
import networkx as nx

class LayerConfig(Packing):

    def regular_init(self, layer_name, store_at, store_at_weights, compute_at, compute_at_weights):
        self.name               = layer_name
        self.store_at           = store_at
        self.store_at_weights   = store_at_weights
        self.compute_at         = compute_at
        self.compute_at_weights = compute_at_weights
        self.folds              = {}

        self.cost=LayerCost(0,0,0)

    def pack(self):
        packed={}
        if self.folds!={}:
            packed['storage_folds']=self.folds
        if self.store_at:
            packed['store_at']=self.store_at
        if self.store_at_weights:
            packed['store_at_weights']=self.store_at_weights
        if self.compute_at:
            packed['compute_at']=self.compute_at
        if self.compute_at_weights:
            packed['compute_at_weights']=self.compute_at_weights
        if self.cost:
            packed['costs']=self.cost.pack()
        return packed

    def unpack(self, packed):
        self.cost=LayerCost(packed=packed['costs']) if 'costs' in packed else LayerCost(0,0,0)
        self.store_at = packed['store_at'] if 'store_at' in packed else None
        self.store_at_weights = packed['store_at_weights'] if 'store_at_weights' in packed else None
        self.compute_at = packed['compute_at'] if 'compute_at' in packed else None
        self.compute_at_weights = packed['compute_at_weights'] if 'compute_at_weights' in packed else None
        self.name=packed['name'] if 'name' in packed else None
        self.folds=packed['storage_folds'] if 'storage_folds' in packed else {}


class SegmentConfig(Packing):

    def regular_init(self, segment_name, order, tiling, output_store_at, layerconfigs):
        self.name = segment_name
        self.order  = order
        self.tiling = tiling
        self.layers = set()
        self.output_store_at = output_store_at

        #list of layers
        self.__layerconfigs=[]
        #internal dict of layers
        self.__layersbyname={}

        #init segment costs
        self.cost=SegmentCost(self.__layerconfigs)

        self.add_layers(layerconfigs)

    @property
    def layerconfigs(self):
        return self.__layerconfigs

    @property
    def cost_tuple(self):
        return (self.cost.accesses, self.cost.buffer_size, self.cost.macs)

    def add_layers(self, layerconfigs):
        for lyr in layerconfigs:
            self.add_layer(lyr)

    def add_layer(self, layerconfig):
        #update our cost object
        self.cost.add_layer(layerconfig.cost)

        #add layer to ourselve
        self.__layerconfigs+=[layerconfig]
        self.layers.add(layerconfig.name)
        self.__layersbyname[layerconfig.name]=layerconfig

    #To help integration into legacy code
    def __getitem__(self, key):
        if key == 'order':
            return self.order
        if key in self.tiling:
            return self.tiling[key]
        if key in self.__layersbyname:
            return self.__layersbyname[key]
        #super(SegmentConfig, self).__getitem__(self, key)


    def pack(self):
        packed={
            #'name'   : self.name,
            'order'  : self.order,
            'tiling' : self.tiling,
            'output_store_at': self.output_store_at,
            'layers' : dict( (lyr.name,  lyr.pack()) for lyr in self.__layerconfigs)
        }
        if self.cost:
            packed['segmentcost']=self.cost.pack()
        return packed

    def unpack(self, packed):
        self.name   = packed['name']
        self.order  = packed['order']
        self.tiling = packed['tiling']
        self.output_store_at = packed['output_store_at'] if 'output_store_at' in packed else None

        #init structures
        self.layers = set()
        self.__layerconfigs=[]
        self.__layersbyname={}

        if 'segmentcost' in packed:
            self.cost=SegmentCost(packed=packed['segmentcost'])
        else:
            self.cost=SegmentCost(self.__layerconfigs)

        #add name to packed
        packed_layers=[]
        for name, pLyrCfg in packed['layers'].items():
            pLyrCfg['name']=name
            packed_layers+=[pLyrCfg]

        #init layers
        self.add_layers( [ LayerConfig(packed = pLyrCfg) for pLyrCfg in packed_layers ] )

class NetworkConfig(Packing):

    def regular_init(self, network_name='Unamed Network', segmentconfigs=[]):
        self.name               = network_name
        self.segmentconfigs     = []
        self.cost               = NetworkCost()
        self.layers             = set()
        self.produces           = set()

        #add segments
        self.add_segments(segmentconfigs)

    #return segments in topological order according to given network
    def sorted_segments(self, net):
        top_sort=list(nx.topological_sort(net))
        def cmp(cfg_a, cfg_b):
            first_a=min([ top_sort.index(lyr) for lyr in cfg_a.layers])
            first_b=min([ top_sort.index(lyr) for lyr in cfg_b.layers])
            return first_a-first_b
        return sorted(self.segmentconfigs, cmp=cmp)


    @property
    def cost_tuple(self):
        return (self.cost.accesses, self.cost.buffer_size, self.cost.macs)

    def add_segments(self, cfgs):
        for cfg in cfgs:
            self.add_segment(cfg)

    def add_segment(self, segmentconfig):
        #update cost object
        self.cost.add_segment(segmentconfig.cost)

        #add segmentconfig to self
        self.segmentconfigs+=[segmentconfig]
        self.layers |= segmentconfig.layers

    def remove_segments(self, cfgs):
        for cfg in cfgs:
            self.remove_segment(cfg)

    def remove_segment(self, cfg):
        self.cost.remove_segment(cfg.cost)
        self.segmentconfigs.remove(cfg)
        self.layers -= cfg.layers

    def merge(self, networkconfig):
        self.add_segments(networkconfig.segmentconfigs)


    def pack(self):
        packed={
            'name'     : self.name,
            'segments' : dict( (segment.name, segment.pack()) for segment in self.segmentconfigs)
        }
        if self.cost:
            packed['networkcost']=self.cost.pack()
        return packed

    def unpack(self, packed):
        #declare vars
        self.segmentconfigs     = []
        self.layers             = set()
        self.produces           = set()

        #unpack costs
        if 'networkcost' in packed:
            self.cost=NetworkCost(packed=packed['networkcost'])
        else:
            self.cost=NetworkCost()

        #unpack name
        self.name               = packed['name']

        #add names back to packedsegments
        packed_segments=[]
        for name, pSegCfg in packed['segments'].items():
            pSegCfg['name']=name
            packed_segments+=[pSegCfg]

        #add segments
        self.add_segments([ SegmentConfig(packed = pSegCfg) for pSegCfg in packed_segments ])


class NetworkConfigs(Packing):

    def regular_init(self, networkconfigs):
        self.configs = networkconfigs

    def pack(self):
        return [ cfg.pack() for cfg in self.configs]

    def unpack(self, packed):
        self.configs = [ NetworkConfig(packed=pcfg) for pcfg in packed ]

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx):
        return self.configs[idx]

    def __setitem__(self, idx, val):
        self.configs[idx]=val

    def __iadd__(self, item):
        self.configs+=item
        return self

    def __iter__(self):
        return self.configs.__iter__()

    def filter_pareto(self):

        #first select only points with unique costs
        unique_costs=set()
        selected_configs=[]
        points=[]
        for cfg in self.configs:

            #get costs
            cost=cfg.cost_tuple

            #only add point if costs are unique
            l=len(unique_costs)
            unique_costs.add(cost)
            if len(unique_costs)>l:
                points+=[(cost, len(selected_configs))]
                selected_configs+=[cfg]

        # do pareto filtering
        points, _ = pareto_cost_config(points)

        #map back pareto points to configs
        self.configs = [ selected_configs[cfg_idx] for cost, cfg_idx in points]

        return self

    def crossproduct(self, networkconfigs):
        new_configs=[]
        for a in self.configs:
            for b in networkconfigs.configs:
                yield NetworkConfig(a.name, a.segmentconfigs+b.segmentconfigs)
