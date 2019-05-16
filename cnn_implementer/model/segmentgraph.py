from ..utils import DAG, Logger
from operator import or_

# Augmented graph that connects segments
class SegmentGraph(DAG, Logger):

    def __init__(self, segmentIterator=None, **kwargs):

        #Sanity check
        assert(segmentIterator!=None and "Must pass a segment iterator/generator to SegmentGraph")

        #init supers
        Logger.__init__(self,**kwargs)
        super(SegmentGraph, self).__init__(**kwargs)

        #all segments without internal inputs (i.e. starting points of the network)
        self.inputs=set()

        #all segments without outputs (i.e., end points of the network)
        self.outputs=set()

        #first add all segments to the graph without connections
        for segment in segmentIterator:
            self.add_node(segment.name, data=segment)

        #Get the set of all nodes produced in this network
        self.produces=reduce(or_, map(lambda n: n.produces, self.get_nodes())) if len(self) else set()

        #Init structure to map layers to segments that produce them
        self.segments_by_production=dict((prod, set()) for prod in self.produces)

        #sort segments by the outputs they generate
        #also fill the self.inputs and self.outputs sets
        for segment in self.get_nodes():

            for n in segment.produces:
                self.segments_by_production[n].add(segment.name)

            #if this has no inputs connect it to our single source
            if len(segment.inputs)==0:
                self.inputs.add(segment.name)

            #if this has no outputs connect it to our single sink
            if len(segment.outputs)==0:
                self.outputs.add(segment.name)

        #Add edges between all the segments
        for segment in self.get_nodes():
            for inp in segment.inputs:
                if inp not in self.segments_by_production:
                    #this node apparently is only internal to segments, no need to connect it
                    continue
                for tgt in self.segments_by_production[inp]:
                    self.debug(' '.join(['connecting', tgt, 'to', segment.name, 'because it generates', inp]))

                    #Finally make the connection, set traversed to False for later processing
                    self.add_edge(tgt, segment.name)


    #Helper function to get node segments returned
    def get_nodes(self, **kwargs):
        for _,seg in super(SegmentGraph, self).nodes(data='data',**kwargs):
            yield seg

