from random import seed as rseed
from random import randint, random
from randomsearchsegment import RandomSearchSegment as Segment
from ..designspace import DesignSpace
from ...model import NetworkConfig, NetworkConfigs
import networkx as nx

class RandomSearch(DesignSpace):

    def getPoint(self, seed=None, **kwargs):

        #seed if requested
        if seed:
            rseed(seed)

        #Create total config
        return NetworkConfig(segmentconfigs=[ seg.config(**kwargs) for seg in self.__segments(**kwargs)])


    def explore(self, count=1, seed=None, **kwargs):
        #seed once if required
        if seed: rseed(seed)

        #collect "count" number of points
        self.points=NetworkConfigs([self.getPoint(**kwargs) for _ in xrange(count)])

        #return
        return self.points


    #Iterator over segments
    def __segments(self,
            min_fused_layers=1,    #Minimum number of layers to fuse (must be >=1)
            max_fused_layers=-1,   #Maximum number of layers to fuse together (negative means no limit)
            seed=None,
            **kwargs
        ):

        # seed rng if requested
        if seed:
            rseed(seed)

        #We will be using this list a couple of times, so just load it into mem at once
        sorted_nodes=list(nx.topological_sort(self.net))

        #Limites on the size of segments to be fused
        min_fused_layers=max(1,min_fused_layers)
        max_fused_layers=max(1,max_fused_layers) if max_fused_layers>0 else len(self.net)

        def __next_segment(startIdx=0):

            #reached the end of the network, we're done here
            if len(sorted_nodes)<=startIdx:
                return True, []

            #randomly select lenght of this segment
            seg_len = randint(min_fused_layers, max(1, min(max_fused_layers, len(sorted_nodes)-startIdx-1)))

            def trySegment(startIdx, endIdx):
                #get selected layers
                layers=sorted_nodes[startIdx:endIdx]

                #figure out if this is a legal segment
                types=[ self.net.nodes[l]['type'] for l in layers]
                if 'eltwise' in types and 'convolution' in types:
                    return False, []

                #generate segment
                seg=Segment(network=self.net, layers=sorted_nodes[startIdx:endIdx], TESTING=self.TESTING)

                # not sure how to fuse into a layer that is not a convolution, reject this segment
                if len(seg.nodes)>1 and seg.last['type'] not in ['convolution']:
                    return False, []

                # not sure how to deal with segment that have more than one production, also reject this segment for now
                if len(seg.produces)>1:
                    return False, []

                #if we made it here, the segment is valid
                #try to generate a next one
                success, next_segments,  = __next_segment(endIdx)

                #The search bottomed, return success and hope it's True ;)
                return success, [seg]+next_segments


            #First let's try the selected lenght and shrink if it doesn't work out
            for endIdx in reversed(range(1,seg_len+1)):
                endIdx+=startIdx
                succes, segments = trySegment(startIdx, endIdx)
                if succes:
                    return succes, segments

            #if we reach here we were unable to get a valid configuration by shrinking
            #try to increase to max seg lenght instead
            for endIdx in range(seg_len+1,min(max_fused_layers, len(sorted_nodes)-startIdx)):
                endIdx+=startIdx
                succes, segments = trySegment(startIdx, endIdx)
                if succes:
                    return succes, segments

            #no way to get a valid segment at this point, backtrack
            return False, []

        #recursively try to find a configuration
        succes, segments = __next_segment()

        assert(succes and "Could not find a valid segmentation with given restrictions")

        #succes, return the list of segments
        return segments
