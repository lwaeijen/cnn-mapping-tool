import networkx as nx
from ..model import SegmentGraph
from dsesegment import DSESegment as Segment

class DSESegmentGraph(SegmentGraph):
    def __init__(self, source=None, sink=None, **kwargs):

        #init super
        super(DSESegmentGraph, self).__init__(**kwargs)

        #init without source and sink nodes
        self.source=None
        self.sink=None

        #add sink and source segments if specified
        if source!=None:
            self.addSource(source.name, source)
        if sink!=None:
            self.addSink(sink.name, sink)

        #Clean up any dead paths which can never form a proper path through the network
        self.__removeDeadPaths()

    def addSource(self, name, segment):
        #Add source node to this segment graph
        self.source=name

        assert(self.source not in self.nodes())
        self.add_node(self.source, data=segment)

        #Connect source and sink nodes
        for inp in self.inputs:
            self.add_edge(self.source, inp)
        self.inputs=set(self.source)

    def addSink(self, name, segment):
        #Add sink node to this segment graph
        self.sink=name

        assert(self.sink not in self.nodes())
        self.add_node(self.sink,   data=segment)

        for out in self.outputs:
            self.add_edge(out, self.sink)
        self.outputs=set(self.sink)

    def __removeDeadPaths(self):
        # clean up some dead paths which can never be part of a valid path through the segmentgraph

        # Can only clean up if we have a source and sink node
        if (not self.source) or (not self):
            return

        # anything not reachable by dfs from the source can be removed
        to_remove=set(self.nodes)-set(nx.dfs_tree(self, self.source).nodes)

        # also anything not reachable by dfs on the reversed graphy from the sink is a dead end
        to_remove.update(set(self.nodes)-set(nx.dfs_tree(self.reverse(copy=False), self.sink).nodes))

        #do the removal of useless segments
        if len(to_remove)!=0:
            self.debug('The following segments are removed since they are never part of a legal path through the network:')
        for n in to_remove:
            self.debug('  - '+str(n))
            self.remove_node(n)
