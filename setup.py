#!/usr/bin/env python
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

from codecs import open
from os.path import abspath, dirname, join
from subprocess import call, CalledProcessError
import sys

__version__ = "1.0.0"

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
        long_description = file.read()

class PyTest(TestCommand):
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(['./tests', '-m',"not halide"])
        sys.exit(errno)

class PyTestHalide(TestCommand):
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(['./tests', '-m',"halide"])
        sys.exit(errno)

setup(name='cnn_implementer',
    version=__version__,
    description='LLVM for CNNs',
    long_description=long_description,
    url='OMITED FOR REVIEW',
    author='OMITTED FOR REVIEW',
    author_email='OMITTED FOR REVIEW',
    license='MIT',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'protobuf',
        'jinja2',
        'networkx',
        'dill',
        'pydot',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'cnn-frontend=cnn_implementer.cli.frontend:standalone',
            'cnn-dse=cnn_implementer.cli.dse:standalone',
            'cnn-backend=cnn_implementer.cli.backend:standalone',

            'cnn-opt=cnn_implementer.cli.analyser:standalone',
            'cnn-plot=cnn_implementer.tools.plot:main',

            'caffe-fmt-conv=cnn_implementer.tools.caffe_convert:main',
            'halide-access-count=cnn_implementer.tools.access_count:main',
            'halide-mem-size=cnn_implementer.tools.halide_mem_size:main',
            'halide-reuse-distance=cnn_implementer.tools.reuse_distance:main',
        ],
    },
    tests_require = ['pytest'],
    cmdclass = {
        'test' : PyTest,
        'test_halide' : PyTestHalide,
    },
    include_package_data=True,
    zip_safe=False,
)
