from math import ceil, log, floor

def macs(segment, segmentCfg, **kwargs):
    return segmentCfg

#from . import *
#from math import ceil
#from ..model import Network
#from functools import reduce
#
#
#
#class ComputeAnalyser(Analyser):
#    def __call__(self):
#        macs=self.macs()
#        return {
#            'macs': macs,
#        }
#
#    def macs(self):
#        macs={}
#
#        #some shorthands
#        last=self.nw.last[0]
#        order=last.order
#        tiling=last.tiling
#        dimensions=last.dim
#
#        #first get the output macs (easy)
#        output_macs=reduce(lambda a,b: a*b, [dimensions[d] for d in ['o','m','n']])*(dimensions['i']*last.kernel_volume())
#        macs[last.name]=output_macs
#
#        for lyr in self.nw.layers:
#            if lyr.type.lower() in ['convolution'] and lyr!=last:
#                store_idx_current =order.index(lyr.store_at)
#                if len(lyr.bottom_ptr)!=1:
#                    print 'ERROR: support only layers with single input layer (bottom) for now'
#                    exit(-1)
#                input_lyr=lyr.bottom_ptr[0].get()
#                store_idx_input =order.index(input_lyr.store_at)
#
#                #determine number of blocks required:
#                blocks=1
#                for dim in ['m','n','o']:
#                    if order.index(dim+'_i') < store_idx_current:
#                        blocks*=int(ceil(dimensions[dim]/tiling[dim]))
#                    else:
#                        blocks*=dimension[dim]
#                blocks*=dimensions['i']
#
#                #determine size of intermediate blocks
#                size=1
#                seen=[]
#                for idx in range(0,store_idx_current):
#                    dim=order[idx][0]
#                    if dim =='i':
#                        continue
#                    if dim in ['m','n']:
#                        seen+=[dim]
#                        kdim=dict((k,v) for k,v in zip(['m','n'],['k','l']))[dim]
#                        tile=tiling[dim]+lyr.fused_tile_border(kdim)
#                        size*=tile
#                for dim in ['n', 'm']:
#                    if dim not in seen:
#                        size*=3 #minimal size is 3 in these dimensions
#                #print('size', size)
#
#                #get number of macs
#                lyr_macs=blocks*size*lyr.kernel_volume()*lyr.dimension['i']
#
#                #LEGACY CODE: fit towards what halide does without guidance
#                #halide_correction=1
#                #if order.index('i_i')>order.index('o_i') and store_idx_inter>order.index('i_i'):
#                #    halide_correction=1+(float(s['tile_i']-1)/float(s['tile_i']))*(s['tile_o']-1)
#                #halide_macs=output_macs+int(intermediate_macs*halide_correction)
#
#                macs[lyr.name]=lyr_macs
#
#        return macs
