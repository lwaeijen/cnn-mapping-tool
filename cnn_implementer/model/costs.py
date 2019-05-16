from ..utils import Packing

class LayerCost(Packing):

    def regular_init(self,weight_buffer_size, data_buffer_size, macs):
        self.__weight_buffer_size=weight_buffer_size
        self.__data_buffer_size=data_buffer_size
        self.__buffer_size=0
        self.__macs=macs
        self.__dirty=True

        self.associated_segment_cost=None


    @property
    def weight_buffer_size(self):
        return self.__weight_buffer_size

    @weight_buffer_size.setter
    def weight_buffer_size(self, weight_buffer_size):
        self.__weight_buffer_size = weight_buffer_size
        self.set_dirty()

    @property
    def data_buffer_size(self):
        return self.__data_buffer_size

    @data_buffer_size.setter
    def data_buffer_size(self, data_buffer_size):
        self.__data_buffer_size = data_buffer_size
        self.set_dirty()

    @property
    def macs(self):
        return self.__macs

    @macs.setter
    def macs(self, val):
        self.__macs=val
        if self.associated_segment_cost:
            self.associated_segment_cost.set_dirty()

    @property
    def buffer_size(self):
        if self.__dirty:
            self.__buffer_size=self.__weight_buffer_size+self.__data_buffer_size
            self.__dirty=False
        return self.__buffer_size

    def set_dirty(self):
        #set dirty bit, we need to recalculate any 'calculated properties' when requested
        #using this dirty mechanism saves some calculations (basically we get lazy evaluation)
        self.__dirty=True

        if self.associated_segment_cost:
            self.associated_segment_cost.set_dirty()

    def set_parent(self, asc):
        self.associated_segment_cost=asc

    def pack(self):
        return {
            'weight_buffer_size' : self.__weight_buffer_size,
            'data_buffer_size'   : self.__data_buffer_size,
            'macs'               : self.__macs,
        }

    def unpack(self, packed):
       self.__weight_buffer_size = packed['weight_buffer_size']
       self.__data_buffer_size   = packed['data_buffer_size']
       self.__macs               = packed['macs']
       self.__dirty=True

class SegmentCost(Packing):
    def regular_init(self, layerconfigs={}):
        self.__weight_accesses=0
        self.__data_accesses=0
        self.__output_accesses=0

        self.__layerconfigs=[]
        self.__buffer_size=0
        self.__output_buffer_size=0
        self.__macs=0
        self.__dirty=False
        self.__associated_network_cost=None

        #add layers
        self.add_layers(layerconfigs)


    def pack(self):
        return {
            'data_accesses'      : self.__data_accesses,
            'weight_accesses'    : self.__weight_accesses,
            'output_accesses'    : self.__output_accesses,
            'buffer_size'        : self.buffer_size,
            'output_buffer_size' : self.__output_buffer_size,
            'macs'               : self.macs,
        }

    def unpack(self, packed):
        self.__layerconfigs=[]
        self.__data_accesses        = packed['data_accesses'] if 'data_accesses' in packed else 0
        self.__weight_accesses      = packed['weight_accesses'] if 'weight_accesses' in packed else 0
        self.__output_accesses      = packed['output_accesses'] if 'output_accesses' in packed else 0
        self.__buffer_size          = packed['buffer_size'] if 'buffer_size' in packed else None
        self.__output_buffer_size   = packed['output_buffer_size'] if 'output_buffer_size' in packed else 0
        self.__macs                 = packed['macs'] if 'macs' in packed else None
        self.__dirty                =True
        self.__associated_network_cost=None

    def is_dirty(self):
        return self.__dirty

    @property
    def layercosts(self):
        return self.__layerconfigs

    @property
    def data_accesses(self):
        return self.__data_accesses

    @data_accesses.setter
    def data_accesses(self,val):
        self.__data_accesses = val
        self.set_dirty()

    @property
    def weight_accesses(self):
        return self.__weight_accesses

    @weight_accesses.setter
    def weight_accesses(self,val):
        self.__weight_accesses = val
        self.set_dirty()

    @property
    def output_accesses(self):
        return self.__output_accesses

    @output_accesses.setter
    def output_accesses(self,val):
        self.__output_accesses = val
        self.set_dirty()

    def add_layers(self, layerconfigs):
        for lc in layerconfigs:
            self.add_layer(lc)

    def add_layer(self, layerconfig):
        #adding a layer means we need to recalc costs
        self.set_dirty()

        #set us as parent for layer config so we get dirt callbacks
        layerconfig.set_parent(self)

        #add layer config to our list
        self.__layerconfigs+=[layerconfig]

    def set_dirty(self):
        #set dirty bit, we need to recalculate any 'calculated properties' when requested
        #using this dirty mechanism saves some calculations (basically we get lazy evaluation)
        self.__dirty=True

        #notify parent we need an update
        if self.__associated_network_cost:
            self.__associated_network_cost.set_dirty()

    def set_parent(self, asc):
        self.associated_network_cost=asc

    @property
    def buffer_size(self):
        self.__clean()
        return self.__buffer_size

    @property
    def data_buffer_size(self):
        #this is only used for reporting, let's just calculate it everytime
        return sum([ lc.data_buffer_size for lc in self.__layerconfigs])

    @property
    def weight_buffer_size(self):
        #this is only used for reporting, let's just calculate it everytime
        return sum([ lc.weight_buffer_size for lc in self.__layerconfigs])

    @property
    def output_buffer_size(self):
        return self.__output_buffer_size

    @output_buffer_size.setter
    def output_buffer_size(self,val):
        self.__output_buffer_size = val
        self.set_dirty()

    @property
    def macs(self):
        self.__clean()
        return self.__macs

    @property
    def accesses(self):
        self.__clean()
        return self.__accesses

    def __clean(self):
        if self.__dirty:
            self.__accesses=self.__data_accesses+self.__weight_accesses+self.__output_accesses
            self.__buffer_size=sum([lc.buffer_size for lc in self.__layerconfigs]) + self.__output_buffer_size
            self.__macs=sum([lc.macs for lc in self.__layerconfigs])
            self.__dirty=False


class NetworkCost(Packing):
    def regular_init(self, segmentcosts=[]):
        self.__segmentcosts=[]
        self.__accesses=0
        self.__macs=0
        self.__buffer_size=0
        self.__dirty=False

        self.add_segments(segmentcosts)

    def pack(self):
        return {
            'accesses'   : self.__accesses,
            'buffer_size': self.__buffer_size,
            'macs'       : self.__macs,
        }

    def unpack(self, packed):
        self.__dirty=True
        self.__segmentcosts=[]

        self.__accesses      = packed['accesses'] if 'accesses' in packed else 0
        self.__buffer_size   = packed['buffer_size'] if 'buffer_size' in packed else None
        self.__macs          = packed['macs'] if 'macs' in packed else None
        self.set_dirty()

    @property
    def accesses(self):
        self.__clean()
        return self.__accesses

    @property
    def macs(self):
        self.__clean()
        return self.__macs

    @property
    def buffer_size(self):
        self.__clean()
        return self.__buffer_size

    def set_dirty(self):
        #set dirty bit, we need to recalculate any 'calculated properties' when requested
        #using this dirty mechanism saves some calculations (basically we get lazy evaluation)
        self.__dirty=True

    def __clean(self):
        if self.__dirty:
            self.__buffer_size=max([sc.buffer_size for sc in self.__segmentcosts])
            self.__macs=sum([sc.macs for sc in self.__segmentcosts])
            self.__accesses=sum([sc.accesses for sc in self.__segmentcosts])
            self.__dirty=False

    def add_segments(self, segmentcosts):
        for sc in segmentcosts:
            self.add_segment(sc)

    def add_segment(self, segmentcost):
        #set ourselves as parent so we get notified when our child gets dirty
        segmentcost.set_parent(self)
        self.__segmentcosts+=[segmentcost]
        self.set_dirty()

    def remove_segment(self, segmentcost):
        #set ourselves as parent so we get notified when our child gets dirty
        segmentcost.set_parent(None)
        self.__segmentcosts.remove(segmentcost)
        self.set_dirty()

    def merge(self, networkcost):
        self.add_segments(networkcosts.__segmentcosts)
