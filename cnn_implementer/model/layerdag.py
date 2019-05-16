from ..utils import DAG

class LayerDAG(DAG):
    #DAG with specific functions when the nodes are layers


    #Find successors of certain type
    def successors_type(self, node_name, type):
        # Do a DFS to find the successor nodes of a given type
        #
        # successors_type('A', type=2) on the graph below would result in:
        # [C,D]
        #
        # A(type=0)-->B(type=1)-->C(type=2)
        # |
        # |--> D(type=2)
        ss=[]
        for s in self.successors(node_name):
            node_s=self.nodes[s]
            if node_s['type'] not in type:
                ss+=self.successors_type(s, type)
            else:
                ss+=[node_s]
        return ss
