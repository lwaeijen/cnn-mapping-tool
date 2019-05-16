from ..analysers.memory import foldDimensionsData, foldDimensionsWeights

def foldBuffers(net, segment_cfg, force=True):
    #fold buffers by setting the compute level
    #force emits the size the buffers should fold to
    o=segment_cfg['order']

    # loop over all layers in segment
    for layer_cfg in segment_cfg.layerconfigs:

        #get node in network
        n = net.nodes[layer_cfg.name]

        #only applicable for convolution layers
        if n['type'].lower() not in ['convolution']:
            continue

        #shorthands
        sl=layer_cfg.store_at
        slw=layer_cfg.store_at_weights

        #set compute to lowest fold dim of data
        fd_data=foldDimensionsData(o, sl)+[sl]
        layer_cfg.compute_at=fd_data[0]

        #annotate the size of x and y folded buffers for halide code gen
        if force:
            n=net.nodes[layer_cfg.name]
            for dim in fd_data:
                if dim in ['x_i', 'y_i']:
                    layer_cfg.folds[dim]=n['K'+dim[0]]

        #set compute to lowest fold dim of weights
        fd_weights=foldDimensionsWeights(o, slw)+[slw]
        layer_cfg.compute_at_weights=fd_weights[0]

def unfoldBuffers(net, segment_cfg):
    #unfold buffers by setting the compute level to the store level
    o=segment_cfg['order']

    # loop over all layers in segment
    for layer_cfg in segment_cfg.layerconfigs:

        #get node in network
        n = net.nodes[layer_cfg.name]

        #only applicable for convolution layers
        if n['type'].lower() not in ['convolution']:
            continue

        #shorthands
        slw=layer_cfg.store_at_weights

        #set compute level equal to store levels
        layer_cfg.compute_at=layer_cfg.store_at
        layer_cfg.compute_at_weights=layer_cfg.store_at_weights

        #no forced folding anymore
        layer_cfg.folds={}
