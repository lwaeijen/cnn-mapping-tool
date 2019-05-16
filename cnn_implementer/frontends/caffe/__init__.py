from . import caffe_pb2
import google.protobuf.text_format as txtf
from ...model import Network
from math import ceil, floor
import itertools
import struct
import os
from collections import OrderedDict

def get_layer_type(t):
    #Get type of layer
    if type(t)==int:
        #translate integer type to string types
        return {
            3: 'concat',
            4: 'convolution',
            5: 'data',
            6: 'dropout',
            14:'innerproduct',
            15:'lrn',
            17:'pooling',
            18:'relu',
            20:'softmax',
            21:'loss',
            22:'split',
        }[t]
    else:
        return t.lower()


def loadNet(fname, logger=None):
    def debug(*msgs):
        #emulate print behaviour
        msg=' '.join(map(str,msgs))
        if logger: logger.debug(msg)

    def info(*msgs):
        #emulate print behaviour
        msg=' '.join(map(str,msgs))
        if logger: logger.info(msg)

    def warn(*msgs):
        #emulate print behaviour
        msg=' '.join(map(str,msgs))
        if logger: logger.warning(msg)

    def err(*msgs):
        #emulate print behaviour
        msg=' '.join(map(str,msgs))
        if logger: logger.error(msg)

    #create caffe net representation
    net=caffe_pb2.NetParameter()

    #parse deploy file
    info("Loading network from: %s"%(fname))
    with open(fname, 'rt') as f:
        txtf.Merge(f.read(), net)

    #Translate caffe network to internal representation
    nw = Network(name=net.name)

    #list of layers that can be inlined as supported by the backend
    supported_fusion_layers=['convolution', 'innerproduct', 'pooling']

    #Layers that still need to be fused
    to_fuse={}

    #Add all layers except activation layers
    #(Activations layers are added as a loop on top of convolution layers, in our model activation is an attribute of convolution layers)
    layers = net.layer if len(list(net.layer))!=0 else net.layers #apparently there are different caffe versions about


    #since we treat the blobs as layers, some patching is required
    #A more structural solution is to make a graph of blobs, but that requires a significant rewrite of the whole tool

    #collect translation table from blob name to layer name
    tops={}

    for l in layers:
        ltype=get_layer_type(l.type)

        #no renaming for slice layers! (At least not how we defined and use them...)
        if ltype in ['slice']:
            for t in l.top:
                tops[t]=t
            continue

        for t in l.top:

            #exclude inplace layers
            if t in l.bottom:
                continue

            #warn if we are redefining
            if t in tops and tops[t]!=l.name:
                warn("Redefining top %s of layer %s which was previously defined as %s. Resulting network might be incorrect"%(t,l.name,tops[t]))

            #Debug messages if actually rename a top
            if t!=l.name:
                debug("Renaming top blob '%s' of layer %s to %s"%(t,l.name,l.name))

            #assign this layer's name to the top
            tops[t]=l.name

    #often omitted in deploy networks, to avoid a warning insert it here
    if 'data' not in tops:
        tops['data']='data'

    #rename bottoms to point to layer names rather than blobs using the tops lookup table
    for l in layers:
        lb=len(l.bottom)
        for idx in range(lb):
            b=l.bottom[idx]
            if b in tops:
                l.bottom[idx]=tops[b]
            else:
                warn('No translation found for bottom', b, 'of layer', l.name)

    #clean up tops
    del tops

    for l in layers:

        info("Translating layer %s into internal network format"%(l.name))

        #depending on the caffe model version we need some translation here
        t=get_layer_type(l.type)
        debug("Detected layer type: %s"%(t))

        #sanity checking
        if(t not in [
            'convolution',
            'eltwise',
            'dropout',
            'innerproduct',
            'pooling',
            'relu',
            'softmax',
            'batchnorm',
            'scale',
            'input',
            'sigmoid',
            'slice',
            'lrn',
            'concat']):
            err("Unsupported layer type",t,"encountered")
            raise TypeError


        if t=='dropout':
            info("Ignoring dropout layer \"%s\", since dropout is not used during inference"%(l.name))
            continue

        #Set Activation layers apart to add to the network later
        if t in ['relu', 'sigmoid', 'batchnorm', 'scale']:
            to_fuse[l.name]=l
            continue

        #basic params
        kwargs={
            'name'    :  l.name,
            'top'     :  list(l.top),
            'bottom'  :  list(l.bottom),
            'type'    :  t,
        }

        #Pooling
        if t in ['pooling']:
           kwargs['pooling']={
               'type':         ['max','ave','stochastic'][l.pooling_param.pool] if type(l.pooling_param.pool)==type(1) else str(l.pooling_param.pool).lower(),
               'kernel_size':  l.pooling_param.kernel_size if not l.pooling_param.global_pooling else "global",
               'K':            l.pooling_param.kernel_size if not l.pooling_param.global_pooling else "global",
               'stride':       l.pooling_param.stride,
               'S':            l.pooling_param.stride,
               'pad':          l.pooling_param.pad,
           }

        if t=='softmax':
            #default params are good enough
            pass

        if t in ['innerproduct']:
            #fully connected layer, we'll just disguise this as a regular convolutional layer
            #by setting the kernel size equal to the input later in the calcDim function
            # Set convultion parameters:
            kwargs['padding_y']       = 0
            kwargs['padding_x']       = 0
            kwargs['stride_y']        = 1
            kwargs['stride_x']        = 1
            kwargs['Sy']              = kwargs['stride_y']
            kwargs['Sx']              = kwargs['stride_x']
            kwargs['zo']              = int(l.inner_product_param.num_output)
            kwargs['kernel_y']        = -1
            kwargs['kernel_x']        = -1
            kwargs['Kx']              = kwargs['kernel_x']
            kwargs['Ky']              = kwargs['kernel_y']
            kwargs['relu']            = False
            kwargs['sigmoid']         = False
            kwargs['fully_connected'] = True
            kwargs['groups']          = 1
            kwargs['batchnorm']       = False
            kwargs['scale']           = False

        if t in ['slice']:
            assert(l.slice_param.axis==1 and "For now only support slice layer with axis==1")
            # for each slice add a 'virtual_slice' layer
            for sl, start, end in zip(l.top, [0]+list(l.slice_param.slice_point), list(l.slice_param.slice_point)+[None]):
                nw.add_layer(
                name=sl,
                start=start,
                end=end,
                zo=end-start if end != None else None,
                zi=end-start if end != None else None,
                bottom=l.bottom,
                parent=l.bottom[0],
                type='virtual_slice',
            )
            continue

        if t in ['concat']:
            assert(l.concat_param.axis==1 and "For now only support concat layer with axis==1")
            #for dimension calculation later
            kwargs['seen']=set()
            kwargs['zo']=0
            kwargs['inputs']=list(l.bottom)
            #will hold the input sizes after calcDim
            kwargs['input_sizes']={}
            kwargs['input_concat_range']={}

        #extra params for convolution layers
        if t in ['convolution']:
            p=l.convolution_param

            # Padding
            kwargs['padding_y'] = 0
            kwargs['padding_x'] = 0
            if hasattr(p, 'pad') and len(p.pad):
                if len(p.pad)==1:
                    kwargs['padding_y'] = p.pad[0]
                    kwargs['padding_x'] = p.pad[0]
                else:
                    kwargs['padding_y'] = p.pad[0]
                    kwargs['padding_x'] = p.pad[1]

            # Stride
            kwargs['stride_y'] = 1
            kwargs['stride_x'] = 1
            if hasattr(p, 'stride') and len(p.stride):
                if len(p.stride)==1:
                    kwargs['stride_y'] = p.stride[0]
                    kwargs['stride_x'] = p.stride[0]
                else:
                    kwargs['stride_y'] = p.stride[0]
                    kwargs['stride_x'] = p.stride[1]

            #Add some shorthand aliases
            kwargs['Sy']=kwargs['stride_y']
            kwargs['Sx']=kwargs['stride_x']

            # kernel size
            if len(p.kernel_size)==1:
                kwargs['kernel_y'] = int(p.kernel_size[0])
                kwargs['kernel_x'] = int(p.kernel_size[0])
            else:
                kwargs['kernel_y'] = int(p.kernel_size[0])
                kwargs['kernel_x'] = int(p.kernel_size[1])

            #holder for all z-input dimensions
            kwargs['input_sizes']={}

            #Add some shorthand aliases
            kwargs['Ky']=kwargs['kernel_y']
            kwargs['Kx']=kwargs['kernel_x']

            #number of output feature maps
            kwargs['zo'] = int(p.num_output)

            #Relu is an attribute of convolution layers, default is False
            #Relu layers are processed layer and thus possibly set this parameter to True
            kwargs['relu']=False
            kwargs['sigmoid']=False

            #number of groups
            kwargs['groups']=int(p.group)

            #just a regular layer
            kwargs['fully_connected']=False

            #batchnorm and scaling
            kwargs['batchnorm'] = False
            kwargs['scale'] = False

        #extra params for input layers (untested!)
        elif t in ['input']:
            kwargs['xo']=l.input_param.shape[0].dim[-1]
            kwargs['yo']=l.input_param.shape[0].dim[-2]
            kwargs['zo']=l.input_param.shape[0].dim[-3]

            #input layers can only be connected to themselves
            #sane prototxt should define this, but the caffe upgrade_net_proto_text tools can rename the input layer to "input" without updating the connections
            #hardpatching that here
            kwargs['name']=kwargs['top'][0]

        elif t in ['lrn']:
            kwargs['K']=l.lrn_param.local_size
            kwargs['alpha']=l.lrn_param.alpha
            kwargs['beta']=l.lrn_param.beta


        #finally add the layer to the network
        nw.add_layer(
            **kwargs
        )

    #In deployment specs the input layer is specified differently
    #If present we'll add it as a layer here
    if hasattr(net, 'input') and net.input!=[]:

        #Sanity check + extract name of the input layer
        if len(net.input)>1:
            raise Exception("Only single input networks supported")
        name=str(net.input[0])

        #add to network
        nw.add_layer(
            name=name,
            top=[],
            bottom=[],
            type='input',

            #dimension of output of this layer
            zo=net.input_dim[1],
            yo=net.input_dim[2],
            xo=net.input_dim[3],
        )

    #fusing decorator
    def fusing(func):
        def wrapper(layer):
            tgt_name=layer.bottom[0]

            #fall through until we find the real target
            while tgt_name in nw.fused:
                tgt_name_new=nw.fused[tgt_name]

                #detect unintended loops
                if tgt_name==tgt_name_new:
                    warning("Detected loop in network, continuing but results might be incorrect")
                    break
                tgt_name=tgt_name_new

            #if target is another layer that still needs to be inserted do nothing for now
            if tgt_name in to_fuse:
                tgt = to_fuse[tgt_name]
                return

            #get the tgt from the network
            tgt = nw.layer(tgt_name)

            #sanity check
            if tgt['type'] not in supported_fusion_layers:
                #exception for elementwise which does support relu fusion
                if not ( (tgt['type'] in ['eltwise']) and layer.type.lower() in ['relu'] ):
                    err("Unable to fuse layer %s into layer %s of type %s. Fusion is only supported in the following layer types:%s"%(l.name, tgt_name, tgt['type'], ''.join(map(lambda s: '\n  - '+s, supported_fusion_layers))))
                    exit(-1)

            #perform the fusion
            debug("Fusing", layer.name, "into",tgt_name,'of type',tgt['type'])
            func(layer, tgt)

            #update structures
            nw.fused[layer.name]=tgt_name
            del to_fuse[layer.name]

            #redefine connections in the network when it was not defined as inplace
            if layer.name in nw:
                for old_succ in list(nw.successors(layer.name)):
                    for t in layer.top:
                        nw.disconnect(l.name, old_succ)
                        nw.connect(tgt_name, old_succ)
                nw.delete_layer(layer.name)

        return wrapper

    @fusing
    def fuse_batchnorm(layer, tgt):
        tgt['batchnorm']=True
        eps=layer.batch_norm_param.eps
        tgt['batchnorm_eps']=eps if eps else None #backend should translated "None" to minimum for the given platform

    @fusing
    def fuse_scaling(layer, tgt):
        tgt['scale']={
            'bias': layer.scale_param.bias_term if hasattr(layer, 'scale_param') and hasattr(layer.scale_param, 'bias_term') else False,
        }

    @fusing
    def fuse_act(layer, tgt):
        t=get_layer_type(layer.type)
        tgt[t]=True

    #perform layer fusions / inlining
    old_len=len(to_fuse)
    info("Start fusing %d inplace layers"%(old_len))
    while len(to_fuse)!=0: #we'll keep at it untill all are fused (alternative implementation could be recursive approach)
        for name, l in to_fuse.items():
            t=get_layer_type(l.type)
            if t in ['scale']:
                fuse_scaling(l)
            elif t in ['batchnorm']:
                fuse_batchnorm(l)
            elif t in ['relu', 'sigmoid']:
                fuse_act(l)
            else:
                err("Unsupported fusion of layer type %s"%(t))
                exit(-1)
        if old_len==len(to_fuse):
            err("Did not manage to fuse the following layers:\n%s"%('\t\n'.join(to_fuse)))
            exit(-1)
        old_len=len(to_fuse)

    ########################################3
    # Add dimensions to all layers

    # First find the input layer
    input_lyrs=nw.layers_by_type('input')
    if len(input_lyrs)==0:
        raise Exception("Error: No input layer found")
    if len(input_lyrs)>1:
        raise Exception("Not supported: More than one input layer found")
    in_layer=input_lyrs[0]

    ## Recursive function to calculate the dimensions of each layer
    def calcDims(name,xi=0,yi=0,zi=0,seen=set(), input_name=None):
        #get real layer object
        layer = nw.layer(name)

        #structure to hold all calculated dimenions
        calc_dimensions={}

        #get the layer type
        t=layer['type'].lower()

        if t in ['virtual_slice']:
            if not layer['end']:
                layer['end']=zi
                layer['zo']=layer['end']-layer['start']
                layer['zi']=layer['end']-layer['start']


        #if a concatenation, we need to collect all inputs and then continue
        if t in ['concat']:
            layer['seen'].add(input_name)
            layer['xo']=xi
            layer['yo']=yi
            layer['input_sizes'][input_name]={}
            layer['input_sizes'][input_name]['xi']=xi
            layer['input_sizes'][input_name]['yi']=yi
            layer['input_sizes'][input_name]['zi']=zi
            layer['zo']+=zi
            if set(layer['seen'])!=set(layer['inputs']):
                #keep going until we have concated all inputs along the z-axis
                return
            #We have all the layers, let's calculate the concatenation range of each concat layer
            # (mainly for easy backend support)
            start=0
            for inp in layer['inputs']:
                end=start+layer['input_sizes'][inp]['zi']
                layer['input_concat_range'][inp]=(start,end)
                start=end

        if t in ['innerproduct']:
            #convert to "regular" convolution
            layer['type']='convolution'
            layer['kernel_x']=xi
            layer['kernel_y']=yi
            layer['Kx']=xi
            layer['Ky']=yi

        if t in ['pooling']:
            #With global pooling we set the kernel equal to the featuremap size
            if layer['pooling']['K']=='global':
                layer['pooling']['K']=xi
                layer['pooling']['kernel_size']=xi

            #ceil due to inconsistent caffe implementations....https://github.com/BVLC/caffe/issues/1318
            calc_dimensions['xo']=int(ceil((float(xi)-float(layer['pooling']['K']))/float(layer['pooling']['S'])))+1 if layer['pooling']['pad']==0 else xi
            calc_dimensions['yo']=int(ceil((float(yi)-float(layer['pooling']['K']))/float(layer['pooling']['S'])))+1 if layer['pooling']['pad']==0 else yi

        #if convolution layer, set the dimension
        if t in ['convolution', 'innerproduct']:
            #Calculate output dim based on stride and padding
            calc_dimensions['xo']=int(floor(float(xi+2*layer['padding_x']-layer['kernel_x'])/float(layer['stride_x'])))+1
            calc_dimensions['yo']=int(floor(float(yi+2*layer['padding_y']-layer['kernel_y'])/float(layer['stride_y'])))+1

            #input has a simple dependency on the output
            #Note: this input includes possible padding
            calc_dimensions['xi']=calc_dimensions['xo']+layer['kernel_x']-1
            calc_dimensions['yi']=calc_dimensions['yo']+layer['kernel_y']-1

        if t in ['convolution', 'innerproduct', 'eltwise','relu', 'pooling']:
            #inputs feature maps always matches output of previous
            calc_dimensions['zi']=zi

        if t in ['eltwise', 'virtual_slice']:
           calc_dimensions['xo']=xi
           calc_dimensions['yo']=yi

        if t in ['eltwise', 'pooling']:
           calc_dimensions['zo']=zi

        if t in ['convolution']:
            #for convolutions the number of input feature maps are stored as well
            layer['input_sizes'][input_name]=zi

        if t in ['softmax', 'lrn']:
            calc_dimensions['zi']=zi
            calc_dimensions['yi']=yi
            calc_dimensions['xi']=xi
            calc_dimensions['zo']=zi
            calc_dimensions['yo']=yi
            calc_dimensions['xo']=xi

        #If this layer was calculated before, just do a sanity check if the calculated values match
        #layers are visited more than once if in the network they take input from multiple layers, such as the element wise layer
        if name in seen:
            for dim, val in calc_dimensions.items():
                if val != layer[dim]:
                    raise TypeError('Malformed network, input dimensions in merging layer do not match')
            #all good (and no need to further propagate since we have been there before)
            return

        #first time we encounter this layer, list it
        seen.update([name])

        #commit calculated values
        debug("Calculated dimensions of %s with type %s: %s"%(name, t, str(calc_dimensions)))

        for dim, val in calc_dimensions.items():
            layer[dim]=val

        #update all successors recursively
        for s in nw.successors(name):
            #the pooled size is passed on to the successor layers
            #for our models though the unpooled xo and yo will be used
            calcDims(s,layer['xo'],layer['yo'],layer['zo'],seen=seen, input_name=name)

    #Call function starting at input layer
    info("Calculating dimensions of all layers...")
    calcDims(in_layer)
    info("Done loading network structure")

    ########################################3

    #remove the (virtual) slices and concat layers
    #for t in ['virtual_slice', 'slice', 'concat']:
    #for t in ['slice']:
    #    #Note: the layers_by_type list is modified by the remove_layer function,
    #    #the extra call to list ensures we get the complete original list to loop over
    #    for lyr in list(nw.layers_by_type(t)):
    #        nw.remove_layer(lyr)

    #return internal network model
    return nw

def loadWeights(fname, nw, outdir='.', logger=None):

    #hello user
    if logger: logger.info("Loading weights from file: %s"%(fname))

    #create caffe net representation
    net=caffe_pb2.NetParameter()

    #parse trained network from binary file
    with open(fname, 'rb') as f:
        net.ParseFromString(f.read())

    #some inconsistencies with caffe here
    if hasattr(net, 'layer') and len(net.layer)!=0:
        layers=net.layer
    elif hasattr(net, 'layers') and len(net.layers)!=0:
        layers=net.layers
    else:
        if logger: logger.warn("Unable to read caffemodel file. Possibly corrupted or unsupported caffe version")

    #Sanity check it
    assert(len(layers)!=0 and 'No weights loaded, possibly due to outdated caffemodel file. Please see the upgrade_net_proto_binary tool supplied with caffe')

    #function to sanitize layers names for storing
    def sanitize_name(s):
        r={
            ' ':'',
            '/':'_',
        }
        for k,rep in r.items():
            s=s.replace(k,rep)
        return s

    #function to save iterable as binary file
    def save_bin(fname, data, fmt_spec='f', bsize=2048):
        it=iter(data)
        with open(fname, 'wb') as f:
            #write in slices to reduce overhead somewhat
            #largest overhead is due to protobuf internally first converting to strings and now having to convert that back to floats...
            while True:
                buf=tuple(itertools.islice(it, bsize))
                if not buf:
                    return
                f.write(struct.pack(fmt_spec*len(buf), *buf))

    #collect weights and bias parameters of convolution layers
    params={}
    warned=False
    for layer in layers:

        #depending on the caffe model version we need some translation here
        t=get_layer_type(layer.type)
        if logger: logger.debug("Parsed layer %s with type %s"%(layer.name, t))

        #warn when we find unidentified layers
        if len(layer.blobs) and t not in ['convolution', 'innerproduct', 'scale', 'batchnorm']:
            if logger: logger.warn("Found data for layer %s of type %s which is not supported by this framework, ignoring..."%(layer.name, t))

        if t in ['scale']:
            name = layer.name if layer.name not in nw.fused else nw.fused[layer.name]
            if name not in params: params[name]={}

            scale=layer.blobs[0]
            if len(scale.shape.dim)!=1:
                if logger: logger.error('Could not parse scale for layer %s - malformed dimensions'%(layer.name))
                raise

            #construct filename
            sname=sanitize_name(name)
            code_fname=sname+'_scale.bin'
            path_fname=os.path.abspath(os.path.join(outdir, code_fname))

            #keep user happy
            if logger: logger.info("Writing scale for layer %s to %s"%(name, path_fname))

            #save to binary file
            save_bin(path_fname, scale.data)

            #set filename for backend (not including any path directives here, code should be executed in same dir as bin files)
            params[name]['scale']=code_fname

            #can be a bias as well
            if len(layer.blobs)==2:
                bias=layer.blobs[1]
                if len(bias.shape.dim)!=1:
                    if logger: logger.error('Could not parse scale bias for layer %s - malformed dimensions'%(name))
                    raise

                #construct filename
                sname=sanitize_name(name)
                code_fname=sname+'_scale_bias.bin'
                path_fname=os.path.abspath(os.path.join(outdir, code_fname))

                #keep user happy
                if logger: logger.info("Writing scale bias for layer %s to %s"%(name, path_fname))

                #save to binary file
                save_bin(path_fname, bias.data)

                #set filename for backend (not including any path directives here, code should be executed in same dir as bin files)
                params[name]['scale_bias']=code_fname

        if t in ['batchnorm']:
            name = layer.name if layer.name not in nw.fused else nw.fused[layer.name]
            if name not in params: params[name]={}

            if len(layer.blobs)!=3:
                if logger: logger.error('Could not parse scale for layer %s - malformed dimensions'%(name))
                raise

            #the order is apparently strict caffemodels ???
            mean=layer.blobs[0]
            var=layer.blobs[1]
            #moving_avg=layer.blobs[2] //NOTE: not used during inference, so not sure why this is kept anyway

            #MEAN
            #construct filename
            sname=sanitize_name(name)
            code_fname=sname+'_batchnorm_mean.bin'
            path_fname=os.path.abspath(os.path.join(outdir, code_fname))

            #keep user happy
            if logger: logger.info("Writing batchnorm mean for layer %s to %s"%(name, path_fname))

            #save to binary file
            save_bin(path_fname, mean.data)

            #set filename for backend (not including any path directives here, code should be executed in same dir as bin files)
            params[name]['batchnorm_mean']=code_fname

            #VARIANCE
            #construct filename
            sname=sanitize_name(name)
            code_fname=sname+'_batchnorm_variance.bin'
            path_fname=os.path.abspath(os.path.join(outdir, code_fname))

            #keep user happy
            if logger: logger.info("Writing batchnorm variance for layer %s to %s"%(name, path_fname))

            #save to binary file
            save_bin(path_fname, var.data)

            #set filename for backend (not including any path directives here, code should be executed in same dir as bin files)
            params[name]['batchnorm_var']=code_fname


        #for convolution and innerproduct (i.e. fully connected we load weights and biases)
        if t in ['convolution', 'innerproduct']:
            if layer.name not in params: params[layer.name]={}

            #index the length of the blobs in case the blob dims are missing
            blob_lengths=[len(blob.data) for blob in layer.blobs]

            #functions to process bias and weights
            def process_data(data, type='bias'):
                #construct filename
                sname=sanitize_name(layer.name)
                code_fname=sname+'_'+str(type)+'.bin'
                path_fname=os.path.abspath(os.path.join(outdir, code_fname))

                #keep user happy
                if logger: logger.info("Writing %s for layer %s to %s"%(type, layer.name, path_fname))

                #save to binary file
                save_bin(path_fname, data)

                #set filename for backend (not including any path directives here, code should be executed in same dir as bin files)
                params[layer.name][type]=code_fname

            def bias(data):
                process_data(data,'bias')

            def weights(data):
                process_data(data,'weights')

            #iterate over the blobs
            for blob in layer.blobs:

                #in some cases the dim is not set (probably broken by the caffe upgrade_net tools)
                if len(blob.shape.dim)==0:
                    if not warned:
                        if logger: logger.warning('Badly structured caffemodel file. Bias and weight data dimensions are missing. This might be caused when the caffemodel is upgraded using the caffe "upgrade_net_proto_binary" tool. Continuing with guess for dimensions...')
                        warned=True
                    #best guessing follows
                    if len(blob.data)==min(blob_lengths):
                        #guess bias
                        bias(blob.data)
                    else:
                        #guess weights
                        weights(blob.data)

                #bias
                elif len(blob.shape.dim)==1:
                    bias(blob.data)

                #weights (usually 4, but if kernel size =1 (or fully connected layers) they can be squashed)
                elif 2<=len(blob.shape.dim) and len(blob.shape.dim)<=4:
                    weights(blob.data)

                #unknown...
                else:
                    raise TypeError('Could not parse weights file - malformed dimensions')

    if logger: logger.info("Done loading weights and biases")
    return params
