import networkx as nx
from ..utils import Storable
from .layerdag import LayerDAG
from copy import deepcopy
from .. import analysers
from itertools import chain, permutations, product, count

class Network(LayerDAG, Storable):
    # NOTE: The graph of the network should be acyclic for this framework to function properly

    def __init__(self, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)
        self.__layers_by_type={}
        self.weights=None
        self.fused={}

    def layers(self, *args, **kwargs):
        for n in self.nodes(*args, **kwargs):
            yield self.nodes()[n]

    def layer(self, name):
        return self.nodes()[name]

    def layers_by_type(self, type):
        if not type in self.__layers_by_type:
            return []
        return self.__layers_by_type[type]

    def remove_layer(self, name):
        #remove layer from network, but connect input to outputs

        #first connect all inputs to all ouputs
        for p in self.predecessors(name):
            for s in self.successors(name):
                self.add_edge(p,s)

        self.delete_layer(name)

    def delete_layer(self, name):
        #remove from auxilary structures
        lyr = self.layer(name)
        if 'type' in lyr:
            self.__layers_by_type[lyr['type']].remove(name)

        #delete the original node from graph
        self.remove_node(name)

    def connect(self, src_name, dest_name):
        self.add_edge(src_name,dest_name)

    def disconnect(self, src_name, dest_name):
        if self.has_edge(src_name, dest_name):
            self.remove_edge(src_name,dest_name)

    def add_layer(self,
            name,           # Name of this layer
            top=[],         # names of layers connected to the top of this layer
            bottom=[],      # names of layers connected to the bottom of this layer
            **kwargs        # other properties that are to be added to the layer
        ):

        #Make sure there is some type defined, and types are always all lower case
        if 'type' not in kwargs:
            kwargs['type']='undef'
        kwargs['type']=kwargs['type'].lower()

        # Add to graph
        self.add_node(name, layer_name=name, **kwargs)

        #Add edges from predecessors to this node
        for pre in bottom:
            self.add_edge(pre, name)

        # Some extra bookkeeping
        if kwargs['type'] not in self.__layers_by_type:
            self.__layers_by_type[kwargs['type']]=[]
        self.__layers_by_type[kwargs['type']]+=[name]

    def __str__(self):
        return '\n'.join(map(lambda l: l+str(self.layer(l)), self.nodes()))

    def __repr__(self):
        return self.__str__()
