from context import cnn_implementer as cnn
from unittest import TestCase

class TestLoadingSimple(TestCase):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.fname='./examples/caffe/simple/simple.prototxt'
        self.name='SimpleNet'

    def get_net(self):
        return cnn.frontends.caffe.loadNet(self.fname)

    def test_name(self):
        net = self.get_net()
        self.assertEqual(net.name, self.name)

    def test_layer_count(self):
        net = self.get_net()
        print net.nodes()
        self.assertEqual(len(net.nodes()), 3)

    def test_dimensions_input(self):
        net = self.get_net()
        first=net.layer('data')
        self.assertEqual(first['xo'],20)
        self.assertEqual(first['yo'],20)
        self.assertEqual(first['zo'],1)

class TestLoadingTraffic(TestCase):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.fname='./examples/caffe/traffic/traffic.prototxt'
        self.name='TrafficSignMPeemen'

    def get_net(self):
        return cnn.frontends.caffe.loadNet(self.fname)

    def test_name(self):
        net = self.get_net()
        self.assertEqual(net.name, self.name)

    def test_layer_count(self):
        net = self.get_net()
        self.assertEqual(len(net.nodes()), 9)
        self.assertEqual(len(net.layers_by_type('convolution')), 5)

    def test_dimensions(self):
        net = self.get_net()
        ref={
            'data': (1280, 720, 1),
            'conv1': (638,358,6),
            'conv2': (317,177,16),
            'conv3a': (313,173,40),
            'conv3b': (313,173,40),
            'conv3':  (313,173,80),
            'fc4_traffic_output': (313,173,8),
        }
        for layer, dims in ref.items():
            for d, ref_val in zip(['xo','yo','zo'], dims):
                self.assertEqual(net.layer(layer)[d],ref_val,
                    msg="Dimension %s of layer %s does not match reference value %d (got: %d)"%(
                        d,
                        layer,
                        ref_val,
                        net.layer(layer)[d]
                    )
                )

class TestLoadingVDSR(TestCase):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.fname='./examples/caffe/VDSR/VDSR_net_deploy.prototxt'
        self.name='VDSR'

    def get_net(self):
        return cnn.frontends.caffe.loadNet(self.fname)

    def test_name(self):
        net = self.get_net()
        self.assertEqual(net.name, self.name)

    def test_layer_count(self):
        net = self.get_net()
        self.assertEqual(len(net.nodes()), 22)

    def test_dimensions_input(self):
        net = self.get_net()
        first=net.layer('data')
        self.assertEqual(first['xo'],256)
        self.assertEqual(first['yo'],256)
        self.assertEqual(first['zo'],1)

    def test_dimensions(self):
        net = self.get_net()
        groundtruth={
            'conv1' : (256, 256),
            'conv2' : (256, 256),
            'conv3' : (256, 256),
            'conv4' : (256, 256),
            'conv5' : (256, 256),
            'conv6' : (256, 256),
            'conv7' : (256, 256),
            'conv8' : (256, 256),
            'conv9' : (256, 256),
            'conv10': (256, 256),
            'conv11': (256, 256),
            'conv12': (256, 256),
            'conv13': (256, 256),
            'conv14': (256, 256),
            'conv15': (256, 256),
            'conv16': (256, 256),
            'conv17': (256, 256),
            'conv18': (256, 256),
            'conv19': (256, 256),
            'conv20': (256, 256),
        }
        for lyr in net.layers():
            if lyr['type'].lower() in ['convolution']:
                for offset, dim in enumerate(['xo','yo']):
                    self.assertEqual(
                        lyr[dim],groundtruth[lyr['layer_name'].lower()][offset],
                        msg='Dimension %s does not match in layer %s (%d!=%d)'%(
                            dim,
                            lyr['layer_name'],
                            lyr[dim],
                            groundtruth[lyr['layer_name'].lower()][offset]
                        )
                    )

