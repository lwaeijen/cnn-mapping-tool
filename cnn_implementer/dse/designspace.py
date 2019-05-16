from ..utils import Storable
from ..utils import Logger

class DesignSpace(Storable, Logger):

    def __init__(self, network, TESTING=False, **kwargs):

        #init supers
        super(DesignSpace, self).__init__(**kwargs)

        #The network that is being explored
        self.net = network

        #Set testing var
        self.TESTING=TESTING


    #explore should be implemented by subclasses
    #returns a list of network configurations and their costs
    def explore(self, **kwargs):
        assert("Can not do a design space exploration with this parent class")

    #Some helper functions
    def unique_layer_name(self, start_name='layer', append='0'):
        #guarantee unique layer name in this network
        name=start_name
        while name in self.net.nodes():
            name+=append
        return name
