#!/usr/bin/env python
import os
import struct
import jinja2

def render(ctxt={}):
    #load template and render
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader('.')
    ).get_template('traffic.prototxt.j2').render(ctxt)

def read_short(fname):
    with open(fname, 'rb') as f:
        return [struct.unpack('h', raw)[0] for raw in iter(lambda: f.read(2), '')]

def read_uchar(fname):
    with open(fname, 'rb') as f:
        return [struct.unpack('B', raw)[0] for raw in iter(lambda: f.read(1), '')]

weights={}
for i in range(1,4+1):
    weights[i]=map(lambda w: float(w)/(256.0), read_short('weight0%d.bin'%(i)))

bias={}
for i in range(1,4+1):
    bias[i]=map(lambda s: float(s)/(256.0) ,read_short('bias0%d.bin'%(i)))

fixact=read_uchar('fixact01.bin')
#print [ p/255.0 for p in fixact]

conv2_map=[
    [0,1,2],
    [1,2,3],
    [2,3,4],
    [3,4,5],
    [0,4,5],
    [0,1,5],

    [0,1,2,3],
    [1,2,3,4],
    [2,3,4,5],
    [0,3,4,5],
    [0,1,4,5],
    [0,1,2,5],
    [0,1,3,4],
    [1,2,4,5],
    [0,2,3,5],

    [0,1,2,3,4,5],
]

#add padding to conv2.
#Unconnected feature maps get a kernel with all zeroes for weights
weights2=[]
ptr=0
for layer in conv2_map:
    for i in range(6):
        if i in layer:
            weights2+=weights[2][ptr*(6*6):(ptr+1)*(6*6)]
            ptr+=1
        else:
            weights2+=[0]*(6*6)
weights[2]=weights2

#swap store order of 'o' and 'i' dimensions
def swapDims(src, irange, orange, kernel_size):
    tmp=[]
    for i in range(irange):
        for o in range(orange):
            offset=(o*irange*kernel_size)+i*(kernel_size)
            tmp+=src[offset:offset+kernel_size]
    return tmp

##NOTE: never here    weights[1]=swapDims(weights[1],1,6,6*6)

#weights[2]=swapDims(weights[2],6,16,36)

#split the weights of conv3 into two parts
a=weights[3][0:8*40*25]
b=weights[3][8*40*25:]
weights[3][0]=a
weights[3][1]=b


#weights[3][0]=swapDims(weights[3][0],8,40,25)
#weights[3][1]=swapDims(weights[3][1],8,40,25)

#weights[4]=swapDims(weights[4],80,8,1)

prototxt=render({
    'weights':weights,
    'bias': bias,
})

with open('traffic.weights.prototxt','wt') as f:
    f.write(prototxt)
