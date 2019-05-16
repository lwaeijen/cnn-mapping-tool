from math import ceil, log, floor
import networkx as nx
from ..utils import next_pow
from operator import mul

def foldDimensionsWeights(order, store_level):

    #shorthands
    o=order
    sl=store_level
    sli=o.index(sl)

    #get first dimension below the store to fold (not x and y are not real fold dims)
    foldDims=[] if sli==0 else [o[sli-1]]

    #we can possibly fold both zo_i and zo_o if they follow eachother directly
    if sli>=2 and (foldDims[0] in ['zi_i', 'zo_i']):
        foldDims+=[o[sli-2]]

    #x and y will be added in this process, but they are not real folddims, filter them out for the sanity check
    return filter(lambda d: d in ['zi_i', 'zo_i'], foldDims)

def foldDimensionsData(order, store_level):
    #shorthands
    o=order
    sl=store_level
    sli=o.index(sl)

    #get first dimension below the store to fold (not x and y are not real fold dims)
    foldDims=[] if sli==0 else [o[sli-1]]

    #if the folding dimension is i, we can also fold 'm' or 'n' if its below it
    if sli>=2 and foldDims[0]=='zi_i':
        foldDims+=[o[sli-2]]

    #remove any fake fold dims for assertion
    return filter(lambda x: x in ['x_i', 'y_i','zi_i'], foldDims)

def foldDimensionsOutput(order, store_level):
    foldDims=[]
    for dim in reversed(order[:order.index(store_level)]):
        if dim in ['zi_i', 'zi_o']:
            break
        foldDims+=[dim]
    return foldDims

def footprint(segment, segmentCfg, **kwargs):
    #should we ignore the output buffer contribution?
    ignore_output = kwargs['ignore_output'] if 'ignore_output' in kwargs else False

    #do not use ceiling operator
    noceil = kwargs['noceil'] if 'noceil' in kwargs else False


    seg=segment
    nw=seg.net
    last=seg.last

    #print 'predicting', segment
    #shorthands
    o=last['order']
    #print o

    #start with the output buffer of this segment
    if last['type'].lower() in ['convolution']:
        sl=segmentCfg.output_store_at
        sl='zo_o'
        foldDims=foldDimensionsOutput(o,sl)

        #TODO: account for the case where Ti==Di!!!

        size=1
        seen=[]
        if ignore_output:
            segmentCfg.cost.output_buffer_size=0
        else:

            for dim in reversed(o[:o.index(sl)]):
                if dim in foldDims+['zi_i', 'zi_o']:
                    #folddims and zi have no contribution
                    continue

                dim, out_in = dim.split('_')
                dim+='o' if dim in ['x','y'] else ''

                if out_in in 'o':
                    size*=last[dim]
                    seen+=[dim]
                elif dim not in seen:
                    size*=last['T'+dim]

            segmentCfg.cost.output_buffer_size=int(size*4)



        #output_strategy='no_partial_transfers'
        #if output_strategy=='no_partial_transfers':
        #    size=1.0
        #    if last['Tzi'] == last['zi']:
        #        #if the tilesize of zi equals the full dimension, results in o are finished the moment we complete the zi_i loop
        #        for dim in o[0:o.index('zi_i')]:
        #            dim=dim.split('_')[0]
        #            if dim[-1]!='o':
        #                dim+='o'
        #            size*=float(last['T'+dim])
        #    else:
        #        #Zi is tiled, results are only complete when we reach the zi_o loop
        #        for dim in o[0:o.index('zi_o')]:
        #            dim, in_out=dim.split('_')

        #            #zi does not contribute to the output buffer size
        #            if dim in ['zi']:
        #                continue

        #            if dim[-1]!='o':
        #                dim+='o'

        #            if in_out=='i':
        #                size*=float(last['T'+dim])
        #            else:
        #                size*=float(last[dim]/last['T'+dim])

        #    segmentCfg.cost.output_buffer_size=int(size*4)

    #Loop over all layers in this segment
    for n in seg.get_nodes():

        #shorthand
        layer_cfg = segmentCfg[n['layer_name']]

        if n['type'].lower() in ['convolution']:

            #shorthands
            sw=n['store_at_weights']
            cl=n['compute_at_weights']
            depth=n['depth']

            ######################
            # weights buffer size
            #

            #figure out which buffers are supposed to fold
            foldDims=foldDimensionsWeights(o, sw)

            #sanity check the compute level is sane. If this is verified we can do the analysis
            min_cl=min(map(lambda d: o.index(d), foldDims)) if foldDims!=[] else 0
            assert(o.index(cl)>=min_cl and "The compute level is placed too low, can not analyze")
            assert(o.index(cl)<=o.index(sw) and "The compute level can not be placed above the store level")

            #see what folding dimensions remain based on the compute level
            foldDims=filter(lambda d: o.index(d)>=o.index(cl), foldDims)

            #Required buffer size after folding
            size=n['Kx']*n['Ky']

            if depth==0:
                for dim in o[0:o.index(sw)]:
                    if dim[0] == 'z':
                        if dim not in foldDims:
                            D=n['T'+dim[0:2]]
                            if dim=='zi':
                                D/=n['groups']
                            size*=D

            elif depth==1:
                #Always need to store all inputs above depth 1
                size*=n['zi']
                #if the zi_i of the last layer is below our store, we need to store the full weight set over our own zo
                if 'zi_i' in o[0:o.index(sw)]:
                    size*=n['Tzo']

            else:
                #always full weight set is required
                size*=n['Tzo']*n['zi']

            #update data buffer size of this layer for this config
            layer_cfg.cost.weight_buffer_size=size*4


            ######################
            # data section
            #

            #shorthands
            st=n['store_at']
            cl=n['compute_at']

            #figure out which buffers are supposed to fold
            foldDims=foldDimensionsData(o, st)

            #sanity check the compute level is sane. If this is verified we can do the analysis
            min_cl=min(map(lambda d: o.index(d), foldDims)) if foldDims!=[] else 0
            assert(o.index(cl)>=min_cl and "The compute level is placed too low, can not analyze")
            assert(o.index(cl)<=o.index(st) and "The compute level can not be placed above the store level")

            #see what folding dimensions remain based on the compute level
            foldDims=filter(lambda d: o.index(d)>=o.index(cl), foldDims)

            #compute buffer sizes
            size=1
            if depth==0:
                seen=[]
                for dim in o[0:o.index(st)]:
                    if dim[0] in ['x', 'y']:
                        seen+=[dim[0]]
                        if dim in foldDims:
                            #halide by default uses next power of 2 if force folding is not enabled
                            if dim in layer_cfg.folds:
                                size*=n['K'+dim[0]]
                                #sanity check: assert(n['K'+dim[0]]==layer_cfg.fold[dim])
                            else:
                                size*=next_pow(n['K'+dim[0]],2)
                        else:
                            size*=n['T'+dim[0]+'o']*n['stride_'+dim[0]]+n['K'+dim[0]]-n['stride_'+dim[0]]
                    else:
                        if dim[0:2]=='zo':
                            #zo only contributes in the special case that there are groups, because then neighboring Z will not share the same input maps!
                            if n['groups']!=1:
                                size*= n['Tzo']/(n['zo']/n['groups'])*n['zi']/n['groups']
                        else:
                            if dim not in foldDims:
                                size*=min(n['T'+dim[0:2]], n['zi']/n['groups'])

                for d in ['x','y']:
                    if d not in seen:
                        size*=n['K'+d]


            elif depth==1:
                for dim in o[0:o.index(st)]:
                    if dim[0] in ['x', 'y']:
                        size*=n['K'+dim[0]]
            else:
                #figure out what the succesors kernel size is (maximum in each dimension if multiple)
                max_Kx=0
                max_Ky=0
                for s in nw.successors_type(n['layer_name'], type='convolution'):
                    s_name = s['layer_name']
                    if s_name in segment.nodes:
                        #internal nodes should use the fused kernel size
                        s=segment.nodes[s_name]
                        max_Kx=max(max_Kx, s['sKx'])
                        max_Ky=max(max_Ky, s['sKy'])
                    else:
                        #external nodes use regular kernel size
                        # NOTE: perhaps this should be 1 since we don't care about space for external (
                        # (current node is an output buffer apparently)
                        max_Kx=max(max_Kx, s['Kx'])
                        max_Ky=max(max_Ky, s['Ky'])

                #need to store over full range always at this depth
                size=n['zo']

                #figure out if we need to store the kernel size or Tiles in x or y
                seen=[]
                for dim in o[0:o.index(st)]:
                    if dim[0] in ['x', 'y']:
                        size*=n['T'+dim[0]+'o']
                        seen+=[dim[0]]
                if 'x' not in seen:
                    size*=max_Kx
                if 'y' not in seen:
                    size*=max_Ky

            #update data buffer size of this layer for this config
            layer_cfg.cost.data_buffer_size=(size)*4


    #segmentCfg is updated with costs, return not strictly required here
    return segmentCfg


def accesses(segment,segmentCfg,**kwargs):
    #NOTE: for now we only need to support no fusion, some shortcuts are taken in the code below

    #should we ignore the contributions of the output buffers
    ignore_output = kwargs['ignore_output'] if 'ignore_output' in kwargs else False

    #do not use ceiling operator
    noceil = kwargs['noceil'] if 'noceil' in kwargs else False

    #start from last layer and work our way to the input
    last=segment.last

    #shorthands
    order=last['order']
    o=order

    data_accesses=0
    weight_accesses=0

    n=last

    #calc the total number of tiles
    if n['type'].lower() in ['convolution']:


        #start with the output buffer
        sl=segmentCfg.output_store_at
        #TODO: account for the case where Ti==Di!!!

        volume=1
        seen=[]
        sl='zo_o'
        if ignore_output:
            segmentCfg.cost.output_accesses=0
        else:
            for dim in reversed(o[:o.index(sl)]):
                if dim in ['zi_i', 'zi_o']:
                    #zi dimensions has no contribution to the volume of the transfers
                    continue

                dim, out_in = dim.split('_')
                dim+='o' if dim in ['x','y'] else ''
                if out_in in 'o':
                    volume*=last[dim]
                    seen+=[dim]
                elif dim not in seen:
                    volume*=last['T'+dim]
            times=1
            for dim in o[o.index(sl):]:
                dim, out_in = dim.split('_')
                dim+='o' if dim in ['x','y'] else ''
                if out_in in 'i':
                    times*=last[dim]
                    seen+=[dim]
                elif dim not in seen:
                    #volume*=int(float(last[dim])/float(last['T'+dim])) #Quite possibly always more precise...
                    if noceil:
                        times*=float(last[dim])/float(last['T'+dim])
                    else:
                        times*=int(ceil(float(last[dim])/float(last['T'+dim])))

            #count every time twice, since we need to load and store. Note, this counts each points one time too often, since the first time there is no load so we subtract the total volume once
            segmentCfg.cost.output_accesses=int((times*volume*2)-reduce(mul, map(lambda d: last[d], ['xo','yo','zo'])))


        #next is the the data
        sl = n['store_at']
        sli = order.index(sl)

        ###
        #volume of a transfer:
        vol=1
        for d in ['x','y']:
            #if y or x is below the store level, the volume size *= Tx+Kx-1   -- else it's Kx
            vol*= n['T'+d+'o']*n['stride_'+d]+n['K'+d]-n['stride_'+d] if order.index(d+'_i') < sli else n['K'+d]

        if order.index('zi_i') < sli:
            #if Tzi >= number of inputs in a group, reduce the volume accordingly
            vol*=min(n['Tzi'],n['zi']/n['groups'])

        #if there are groups, it's not the same i that is shared, but the volume increases
        if n['groups']!=1 and order.index('zo_i') < sli:
            #Tzo/(zo/groups) = how many groups of output maps per tile
            #zi.groups = how many input maps per group of o
            #mult: number of input maps needed for 1 tile of o (usually depends on Ti, but for groups that's not true
            vol*=n['Tzo']/(n['zo']/n['groups'])*n['zi']/n['groups']

        ####
        # Times it is transferred
        times=1

        for dim in order[0:sli]:
            dim=dim.split('_')[0]
            if len(dim)==1:
                dim+='o'
            D=n[dim]
            T=n['T'+dim]

            #if zi compensate for any grouping
            if dim=='zi':
                D/=n['groups']
                T=min(T, D)

            #divide and ceil if required
            D_T=float(D)/float(T)
            times*= D_T if noceil else int(ceil(D_T))

        for dim in order[sli:4]:
            dim=dim.split('_')[0]
            if len(dim)==1:
                dim+='o'
            D=n[dim]
            if dim=='zi':
                D/=n['groups']
            times*=D
        data_accesses=vol * times


        #Accesses for applying any relu or sigmoid operations
        #if n['sigmoid'] or n['relu']:
        #    data_accesses+=n['zo']*n['xo']*n['yo']


        #################
        #Continue with the accesses of the weights
        sl = n['store_at_weights']
        sli = order.index(sl)

        ###
        #volume of a transfer
        size=n['Kx']*n['Ky']

        weight_accesses=size*n['zo']*n['zi']/n['groups']
        for dim in ['x', 'y']:
            if order.index(dim+'_i')<sli:
                if noceil:
                    weight_accesses*=float(n[dim+'o'])/float(n['T'+dim+'o'])
                else:
                    weight_accesses*=int(ceil(float(n[dim+'o'])/float(n['T'+dim+'o'])))
            else:
                weight_accesses*=n[dim+'o']

    #update accesses of this layer
    segmentCfg.cost.data_accesses=int(data_accesses)
    segmentCfg.cost.weight_accesses=int(weight_accesses)

    return segmentCfg
