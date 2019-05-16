import networkx as nx

class DAG(nx.DiGraph):

    def write_dot(self, fname):
        # Write to dot file"
        try:
            import pydot
        except:
            assert("Pydot is required to store graphs to dot file" and False)
        nx.nx_pydot.write_dot(self, fname)


