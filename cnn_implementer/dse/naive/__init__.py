from ..designspace import DesignSpace
from naivesegment import NaiveSegment as Segment
from ...model.configs import NetworkConfigs, NetworkConfig
import networkx as nx
from itertools import product

class Naive(DesignSpace):

    def __init__(self, net, **kwargs):
        #init super
        super(Naive, self).__init__(net, **kwargs)

        #dictionary to keep track of segments
        self.segments={}

        #dictionary to keep track of explore segment configurations
        self.segment_configs={}

        #var to hold explored points
        self.points=None

    def networkConfigs(self, pareto=True, **kwargs):
        if pareto:
            return self.__networkConfigs_pareto(**kwargs)
        else:
            return self.__networkConfigs_all(**kwargs)

    #generate pareto network configurations, minimizes the number of computations but requires more memory than the non pareto version
    def __networkConfigs_pareto(self,
            min_fused_layers=1,    #Minimum number of layers to fuse (must be >=1)
            max_fused_layers=-1,   #Maximum number of layers to fuse together (negative means no limit)
            **kwargs
        ):

        #We will be using this list a couple of times, so just load it into mem at once
        sorted_nodes=list(nx.topological_sort(self.net))

        #set max fused layers in case no limit is given
        if max_fused_layers<=-1:
            max_fused_layers=len(sorted_nodes)

        def generate_segmentations(layers):

            #if there are no more layers left yield an empty list
            if len(layers)==0:
                yield []

            #loop over valid segment sizes for the head
            for l in range(min_fused_layers, min(max_fused_layers, len(layers))+1):
                head=layers[0:l]
                tail=layers[l:]

                # we can not mix element wise and convolution for now
                # For the time being apply an extremely crude filter
                types=[ self.net.nodes[l]['type'] for l in head]
                if 'eltwise' in types and 'convolution' in types:
                    #no point in growing this segment any more
                    #self.info('cannot mix conv and element wise, ignoring config')
                    return

                #get key for this particular segment
                key=frozenset(head)
                key2=frozenset([key])

                #If this is a new segment, create it and put in our list
                if key2 not in self.segment_configs:

                    #Create the head segment
                    head_segment = Segment(network=self.net, layers=head, TESTING=self.TESTING)

                    # not sure how to fuse into a layer that is not a convolution, reject these segments as well
                    if len(head_segment.nodes)>1 and head_segment.last['type'] not in ['convolution']:
                        #self.info('removing segment fused('+','.join(head_segment.nodes())+') because the final layer is not a supported fusion layer')
                        continue #is fixable by adding more nodes

                    # not sure how to deal with segment that have more than one production, also reject this segment for now
                    if len(head_segment.produces)>1:
                        #self.info('removing segment fused('+','.join(head_segment.nodes())+') because the it has multiple productions')
                        return #can not get better by adding more nodes

                    #add explored segment configurations to our dictionary
                    self.info("Exploring segment %s"%(head_segment))
                    self.segment_configs[key2]=NetworkConfigs([NetworkConfig(network_name=self.net.name, segmentconfigs=[cfg]) for cfg in head_segment.explore(pareto=True, **kwargs)])
                    self.info("Found %d implementation options"%(len(self.segment_configs[key2])))

                #recurs on tail
                for p in generate_segmentations(tail):
                    yield [key]+p


        #iterate over all segments
        finalCfgs=NetworkConfigs([])
        for segmentation in generate_segmentations(sorted_nodes):


            cfgs=NetworkConfigs([NetworkConfig(network_name=self.net.name, segmentconfigs=[])])
            accu_key_mut=set()
            for segment_key in segmentation:
                accu_key_mut.add(segment_key)
                accu_key=frozenset(accu_key_mut)
                key2=frozenset([segment_key])
                if accu_key not in self.segment_configs:
                    self.info("merging configs of segment \"(%s)\" into cumulative configs \"%s\""%(','.join(segment_key), '^'.join(map(lambda k: ','.join(k), accu_key_mut))))
                    self.segment_configs[accu_key]=NetworkConfigs(list(cfgs.crossproduct(self.segment_configs[key2]))).filter_pareto()
                cfgs=self.segment_configs[accu_key]
            finalCfgs+=cfgs
        return finalCfgs.filter_pareto()

    #generate ALL network configurations, but using generators so low memory requirements
    def __networkConfigs_all(self,
            min_fused_layers=1,    #Minimum number of layers to fuse (must be >=1)
            max_fused_layers=-1,   #Maximum number of layers to fuse together (negative means no limit)
            **kwargs
        ):

        #We will be using this list a couple of times, so just load it into mem at once
        sorted_nodes=list(nx.topological_sort(self.net))

        #set max fused layers in case no limit is given
        if max_fused_layers<=-1:
            max_fused_layers=len(sorted_nodes)

        def generate_network_configs(layers, cfg):

            #if there are no more layers left yield an empty list
            if len(layers)==0:
                yield cfg

            #loop over valid segment sizes for the head
            for l in range(min_fused_layers, min(max_fused_layers, len(layers))+1):
                head=layers[0:l]
                tail=layers[l:]

                # we can not mix element wise and convolution for now
                # For the time being apply an extremely crude filter
                types=[ self.net.nodes[l]['type'] for l in head]
                if 'eltwise' in types and 'convolution' in types:
                    #no point in growing this segment any more
                    #self.info('cannot mix conv and element wise, ignoring config')
                    return

                #get key for this particular segment
                key=frozenset(head)

                #If this is a new segment, create it and put in our list
                if key not in self.segments:

                    #Create the head segment
                    head_segment = Segment(network=self.net, layers=head, TESTING=self.TESTING)

                    # not sure how to fuse into a layer that is not a convolution, reject these segments as well
                    if len(head_segment.nodes)>1 and head_segment.last['type'] not in ['convolution']:
                        #self.info('removing segment fused('+','.join(head_segment.nodes())+') because the final layer is not a supported fusion layer')
                        continue #is fixable by adding more nodes

                    # not sure how to deal with segment that have more than one production, also reject this segment for now
                    if len(head_segment.produces)>1:
                        #self.info('removing segment fused('+','.join(head_segment.nodes())+') because the it has multiple productions')
                        return #can not get better by adding more nodes

                    #add generated segment to our dictionary
                    self.info("Exploring segment %s"%(head_segment))
                    self.segments[key]=list(head_segment.explore(pareto=False, unique=True, **kwargs))
                    self.info("Found %d implementation options"%(len(self.segments[key])))


                #for every segment config in the head, recurs on the tail
                for head_segment_cfg in self.segments[key]:

                    #Add head to network config
                    cfg.add_segment(head_segment_cfg)

                    #recurs on tail
                    for complete_cfg in generate_network_configs(tail,cfg):
                        yield complete_cfg

                    #remove head again from the network config
                    #(cfg should return to original state as before the "add_Segment")
                    cfg.remove_segment(head_segment_cfg)

        #return the generator
        return generate_network_configs(sorted_nodes, NetworkConfig([]))



    def explore(self, **kwargs):
        #debug
        #self.points=self.networkConfigs(pareto=True,min_fused_layers=1, max_fused_layers=1, tiling={ 'xo':32, 'yo':4, 'zo':8, 'zi':16})

        self.points=self.networkConfigs(**kwargs)
        return self.points
