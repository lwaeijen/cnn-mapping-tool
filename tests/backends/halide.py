from context import cnn_implementer as cnn
from context import run_cmd
from unittest import TestCase
import tempfile
import logging
import os
import shutil
import filecmp
import pytest

logging.basicConfig(level=logging.CRITICAL)
log = logging.getLogger()

def compile_halide(src, tgt, log=None):
    return run_cmd("g++ %(src)s -lHalide -lpthread -ldl -lz -lpng -ljpeg `llvm-config --system-libs 2> /dev/null` -std=c++11 -o %(tgt)s"%{'src':src, 'tgt':tgt}, log)

class TestHalideCodeGenerationTraffic(TestCase):
    def __init__(self, *args, **kwargs):
        #print "Testing Halide Code Generation for Traffic Sign Detection Network"
        super(self.__class__, self).__init__(*args, **kwargs)
        self.fname='./examples/caffe/traffic/traffic.prototxt'
        self.fname_weights='./examples/caffe/traffic/traffic.caffemodel'
        self.fname_input=os.path.abspath('./examples/caffe/traffic/test046.png')
        self.fname_reference=os.path.abspath('./examples/caffe/traffic/ref.txt')


    def test_halide_codegen(self):
        try:
            #create a temporary directory to hold code and binary weights/bias
            self.tempDir=tempfile.mkdtemp()

            #load network
            self.net = cnn.frontends.caffe.loadNet(self.fname)
            self.net.weights = cnn.frontends.caffe.loadWeights(self.fname_weights, self.net, outdir=self.tempDir)

            #get design space
            self.ds = cnn.dse.RandomSearch(self.net, TESTING=True, logger=log)

            point=self.ds.getPoint(
                seed=1337,
                max_fused_layers=1,
            )

            #Get halide backend to implement the seleted point
            halide=cnn.backends.Halide()
            code=halide.implement(self.net, point, DEBUG=True)

            #TODO: some basic asserts could be used here to verify it at least looks like code

        finally:
            #Finally clean up temp dir
            if self.tempDir and os.path.exists(self.tempDir):
                shutil.rmtree(self.tempDir)


    @pytest.mark.halide
    def test_halide_correctness(self):
        try:
            #create a temporary directory to hold code and binary weights/bias
            self.tempDir=tempfile.mkdtemp()

            #load network
            self.net = cnn.frontends.caffe.loadNet(self.fname)
            self.net.weights = cnn.frontends.caffe.loadWeights(self.fname_weights, self.net, outdir=self.tempDir)

            #get design space
            self.ds = cnn.dse.RandomSearch(self.net, TESTING=True, logger=log)

            #different exploration
            point=self.ds.getPoint(
                seed=1337,
                max_fused_layers=2,
            )

            #Get halide backend to implement the seleted point
            halide=cnn.backends.Halide()
            code=halide.implement(self.net, point, DEBUG=False)


            #write code to output dir
            fname_code=os.path.join(self.tempDir, 'halide.cpp')
            with open(fname_code, 'wt') as f:
                f.write(code)

            #compile to binary
            fname_bin=os.path.join(self.tempDir, 'halide.exe')
            stdout, stderr, ret =  compile_halide(fname_code, fname_bin, log=log)
            self.assertEqual(ret, 0)

            #execute program
            fname_output=os.path.join(self.tempDir, 'output.txt')
            stdout, stderr, ret = run_cmd('%(bin)s %(input)s %(output)s'%{
                'bin':fname_bin,
                'input':self.fname_input,
                'output':fname_output
            }, log=log)
            self.assertEqual(ret, 0)

            #compare output with reference
            self.assertTrue(filecmp.cmp(fname_output, self.fname_reference))

        finally:
            #Finally clean up temp dir
            if self.tempDir and os.path.exists(self.tempDir):
                shutil.rmtree(self.tempDir)


class TestHalideCodeGenerationVDSR(TestCase):
    def __init__(self, *args, **kwargs):
        #print "Testing Halide Code Generation for VDSR Network"
        super(self.__class__, self).__init__(*args, **kwargs)
        self.fname='./examples/caffe/VDSR/VDSR_net_deploy.prototxt'
        self.fname_weights='./examples/caffe/VDSR/VDSR_net_deploy.caffemodel'
        self.fname_input=os.path.abspath('./examples/caffe/VDSR/blr_256.png')
        self.fname_reference=os.path.abspath('./examples/caffe/VDSR/blh_256_ref.png')

    def test_halide_codegen(self):
        try:
            #create a temporary directory to hold code and binary weights/bias
            self.tempDir=tempfile.mkdtemp()

            #load network
            self.net = cnn.frontends.caffe.loadNet(self.fname)
            self.net.weights = cnn.frontends.caffe.loadWeights(self.fname_weights,self.net, outdir=self.tempDir)

            #get design space
            self.ds = cnn.dse.RandomSearch(self.net, TESTING=True, logger=log)

            #do basic DSE
            point=self.ds.getPoint(
                seed=42,
                max_fused_layers=2,
                tiling={
                'xo': 2,
                'yo': 4,
                'zo': 8,
                'zi': 4,
            })

            #Get halide backend to implement the seleted point
            halide=cnn.backends.Halide()
            code=halide.implement(self.net, point, DEBUG=True)

            #TODO: some basic asserts could be used here to verify it at least looks like code

        finally:
            #Finally clean up temp dir
            if self.tempDir and os.path.exists(self.tempDir):
                shutil.rmtree(self.tempDir)

    @pytest.mark.halide
    def test_halide_correctness(self):
        try:
            #create a temporary directory to hold code and binary weights/bias
            self.tempDir=tempfile.mkdtemp()

            #load network
            self.net = cnn.frontends.caffe.loadNet(self.fname)
            self.net.weights = cnn.frontends.caffe.loadWeights(self.fname_weights, self.net, outdir=self.tempDir)

            #get design space
            self.ds = cnn.dse.RandomSearch(self.net, TESTING=True, logger=log)

            #get a point
            point=self.ds.getPoint(
                seed=42,
                max_fused_layers=1, #N.B. only 1 to speed up processing
                tiling={
                'xo': 64,
                'yo': 64,
                'zo': 64,
                'zi': 64,
            })

            #Get halide backend to implement the seleted point
            halide=cnn.backends.Halide()
            code=halide.implement(self.net, point, DEBUG=False)


            #write code to output dir
            fname_code=os.path.join(self.tempDir, 'halide.cpp')
            with open(fname_code, 'wt') as f:
                f.write(code)

            #compile to binary
            fname_bin=os.path.join(self.tempDir, 'halide.exe')
            stdout, stderr, ret =  compile_halide(fname_code, fname_bin, log=log)
            self.assertEqual(ret, 0)

            #execute program
            fname_output=os.path.join(self.tempDir, 'output.png')
            stdout, stderr, ret = run_cmd('%(bin)s %(input)s %(output)s'%{
                'bin':fname_bin,
                'input':self.fname_input,
                'output':fname_output
            }, log=log)
            self.assertEqual(ret, 0)

            #compare output with reference
            self.assertTrue(filecmp.cmp(fname_output, self.fname_reference))

        finally:
            #Finally clean up temp dir
            if self.tempDir and os.path.exists(self.tempDir):
                shutil.rmtree(self.tempDir)
