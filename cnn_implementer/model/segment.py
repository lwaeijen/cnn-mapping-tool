import networkx as nx
from layerdag import LayerDAG
from itertools import count
from .. import analysers

class Segment(LayerDAG):
    _ids=count(0)

    # Create the segment with the given layers
    def __init__(self, name=None, network=None, layers=[], TESTING=False, **kwargs):

        #Generate a unique ID for this segment
        self.id=next(self._ids)

        #Set the complete network
        self.net=network

        #other parameters
        self.TESTING=TESTING

        #set per layer what the inputs and outputs outside of the segment are
        self.inputs=set()        #external nodes that are required by one or more internal nodes
        self.outputs=set()       #external nodes that require one or more internal nodes
        self.produces=set()      #internal nodes that are required by one ore more external nodes
        self.consumers=set()     #internal nodes that use data from external nodes

        #create a subgraph and set it as this segments graph
        if network:
            super(Segment, self).__init__(self.net.subgraph(layers))
        else:
            super(Segment, self).__init__(**kwargs)

        #Compute name for this segment
        if len(self)==0:
            self.name=name if name else 'Unnamed Segment'
        elif len(self)==1:
            self.name=list(self.nodes())[0]
            self.last_name=self.name
            self.last=self.nodes[self.last_name]
        else:
            topsort=list(nx.topological_sort(self))
            self.name='fused('+','.join(topsort)+')'
            for name in reversed(topsort):
                self.last_name=name
                if self.nodes[name]['type'] not in ['virtual_slice', 'slice']:
                    break
            self.last=self.nodes[self.last_name]

        #Initialize the inputs, outputs, produces and consumers sets
        for l in layers:

            #external nodes that are required by one or more internal nodes
            self.node[l]['ext_inputs']=filter(lambda s: s not in layers, self.net.predecessors(l))
            self.inputs.update(self.node[l]['ext_inputs'])

            #external nodes that require one or more internal nodes
            self.node[l]['ext_outputs']=filter(lambda s: s not in layers, self.net.successors(l))
            self.outputs.update(self.node[l]['ext_outputs'])

            #internal nodes that use data from external nodes
            if len(self.node[l]['ext_inputs'])!=0:
                self.consumers.add(l)

            #internal nodes that are required by one ore more external nodes
            if len(self.node[l]['ext_outputs'])!=0:
                self.produces.add(l)


        #last segment has no other segments depending, add produces by other means
        if len(self.produces)==0:
            for l in layers:
                null=True
                for _ in self.net.successors(l):
                    null=False
                    break
                if null:
                    self.produces.add(l)

    def __repr__(self):
        return self.name


    #evaluate one configuration of the segment
    def eval_config(self, cfg, **kwargs):

        # Special testing mode which returns 3D random tuples
        if self.TESTING:
            r_acc, r_data_bs, r_weight_bs, r_macs =  self.rng.randnints(0,1000,4)
            #Just set the cost of the last layer, rest will be zero, so it determines cost of whole segment
            cfg[self.last_name].cost.accesses=r_acc
            cfg[self.last_name].cost.weight_buffer_size=r_weight_bs
            cfg[self.last_name].cost.data_buffer_size=r_data_bs
            cfg[self.last_name].cost.macs=r_macs


        #set output tile sizes
        self.last['Tzo'] = cfg['Tzo']
        self.last['Tzi'] = cfg['Tzi']

        #Helper function to calculate tiles in the network
        def calc_tiles(node,Tx, Ty, sKx=1, sKy=1,depth=0):
            # The fundamental problem with this way of working is the assumption
            # that each layer has only 1 input layer
            # This works for the current networks, but need to be adjusted for
            # different types of networks later


            #get node name and predecessors
            predecessors = [self.nodes[pred_name] for pred_name in self.predecessors(node['layer_name'])]

            #set sizes based on request of predecessor
            node['Txo'] = Tx
            node['Tyo'] = Ty

            #set scaled kernel sizes (minimum kernel storage)
            if node['type'] in ['convolution']:
                sKx=node['Kx']+(sKx-1)*node['Sx']
                if 'sKx' not in node:
                    node['sKx']=sKx
                node['sKx']=max(node['sKx'], sKx)

                sKy=node['Ky']+(sKy-1)*node['Sy']
                if 'sKy' not in node:
                    node['sKy']=sKy
                node['sKy']=max(node['sKy'], sKy)

            #calculate input sizes
            if node['type'] in ['convolution']:
                node['Txi'] = node['Kx']+(node['Txo']-1)*node['Sx']
                node['Tyi'] = node['Ky']+(node['Tyo']-1)*node['Sy']
            elif node['type'] in ['input']:
                pass #no more inputs to this layer
            else:
                #default: assume one-to-one relation
                node['Txi']=node['Txo']
                node['Tyi']=node['Tyo']


            #Z dimension depends on depth
            if depth==0:
                node['Tzo'] = cfg['Tzo']
                node['Tzi'] = cfg['Tzi']
            elif depth==1:
                node['Tzo'] = cfg['Tzi']
                #when there are multiple inputs, just sum for now (N.B. probably incorrect!)
                if node['type']  in ['convolution']:
                    node['Tzi'] = sum(node['input_sizes'].values())
                else:
                    node['Tzi'] = 0
                    for pred in predecessors:
                        if pred['type']=='slice':
                            start, end = pred['slices'][node['layer_name']]
                            node['Tzi'] += abs(end-start)
                        else:
                            node['Tzi'] += pred['zo']
            else:
                #after depth 1 we always need the full width
                if node['type'] not in ['slice']:
                    node['Tzo'] = node['zo']

                #when there are multiple inputs, just sum for now (N.B. probably incorrect!)
                if node['type']  in ['convolution']:
                    node['Tzi'] = sum(node['input_sizes'].values())
                else:
                    node['Tzi'] = 0
                    for pred in predecessors:
                        if pred['type']=='slice':
                            start, end = pred['slices'][node['layer_name']]
                            node['Tzi'] += abs(end-start)
                        else:
                            node['Tzi'] += pred['zo']

            #also set the depth of the node wrt to the scheduled output node
            if node['type'] in ['convolution']:
                node['depth']=depth
                depth+=1

            # Propagate tile sizes through layers
            for pred in predecessors:
                calc_tiles(pred, node['Txi'], node['Tyi'], sKx, sKy, depth=depth)

        #Calculate input tile sizes for the last layer
        calc_tiles(self.last, cfg['Tx'], cfg['Ty'])

        #et execution order
        self.last['order']=cfg['order']

        #set the store levels of each convolution layer
        for n in self.nodes.itervalues():
            #skip if not a convolution layer
            if n['type'] not in ['convolution']:
                continue

            #set store and compute levels
            n['store_at']=cfg[n['layer_name']].store_at
            n['store_at_weights']=cfg[n['layer_name']].store_at_weights
            n['compute_at']=cfg[n['layer_name']].compute_at
            n['compute_at_weights']=cfg[n['layer_name']].compute_at_weights


        #Run all analysers to set costs of this config

        #Memory accesses
        analysers.memory.accesses(self, cfg, **kwargs)

        #Evaluate buffer size
        analysers.memory.footprint(self, cfg, **kwargs)

        #Evaluate MACs
        analysers.compute.macs(self, cfg, **kwargs)

        #strictly no need to return, cfg is updated with costs
        return cfg


    def get_nodes(self):
        for _, node in super(Segment, self).nodes(data=True):
            yield node
