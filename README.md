# CNN Implementer
-----------------

This python package provides a framework including commandline tool to transform a highlevel CNN description, such as a caffe network, and translate it to an efficient implementation for a target (software and or hardware).
The philosophy behind this closely resembles that of LLVM, in that there are frontends which read in various network descriptions from existing frameworks (e.g. Caffe), a common intermediate network representation which can be analysed using various generic analysis passes, and different backends fo various targets.

More information about this tool and the theory behind it can be found in:</br>
[Automatic memory-efficient scheduling of CNNs - Waeijen et al.](https://research.tue.nl/en/publications/automatic-memory-efficient-scheduling-of-cnns)


## Usage
--------

To use the commandline interface, provide a network in the caffe format to generate a \*.net file:
```
cnn-frontend --caffe-deploy examples/caffe/traffic/traffic.prototxt --caffe-model examples/caffe/traffic/traffic.caffemodel --write-network-dotfile traffic.dot --write-model traffic.net
```
Apart from the binary traffic.net file, also a dot file of the network graph is generated.

To generate a random schedule for the network, use the ```cnn-dse``` tool:
```
cnn-dse --dse-save-points traffic_point.json -n traffic.net --dse-strategy random
```
This will generate a single schedule in traffic_point.json, which is human readable and can be modified if desired.
In particular after manual modification it is recommended to run the ```cnn-opt``` tool to optimize compute levels and validate schedule correctness to some degree.
The ```cnn-opt``` tool also provides flags to disable the actual optimizations, but still insert some required extra fields to the schedule for proper code generation.
See ```cnn-opt --help``` for more information.

Instead of generating a random point, it is also possible to perform a full design space exploration using
```
cnn-dse --dse-save-points traffic_points.json -n traffic.net
```
The generated points file will contain all the pareto points of the internal buffer size versus external accesses space.
Note: A full DSE can take a long time depending on the network. Options to restrict the design space are available to limit the space and speed up the exploration. Please check ```cnn-dse --help```.

To visualize the pareto front generated in the previous step, the ```cnn-plot``` utility can be used:
```
cnn-plot -c traffic_points.json -l ExploredPoints
```
If desired multiple point files can be specified after the "-c" argument to plot multiple fronts. The arguments after "-l" are the corresponding labels in the legend of the plot.
When desired the output can also be saved to a file by adding "-s -b graph.pdf"

Finally a selected schedule can be implemented in halide using the backend tool.
```
cnn-backend -p traffic_points.json -n traffic.net -i INDEX_OF_SELECTED_POINT --halide-code traffic.cpp
```
The index of the selected network is the index of the selected point in the traffic_points.json file.
When this argument is ommitted the first schedule in the file will be implemented.
The result is a single cpp file which implements the entire network as a halide function.
To call this function pass a halide buffer with the input image.
For profiling, debugging and tracing memory accesses see the additional flags of the backend tool.

## Installation
---------------
To install this package directly from gitlab:
```
sudo pip install git+ssh://git@OMITTED_FOR_BLIND_REVIEW/cnn-framework.git
```
When installing on a machine where you don't have sudo rights, you can install into your home directory using:
```
pip install git+ssh://git@OMITTED_FOR_BLIND_REVIEW/cnn-framework.git --user
```

After the installation completes you should be able to issue the ```cnn-opt --version``` command.

## Development
--------------
First install the requirements:
```sudo apt install python2 python-dev make python-virtualenv```

To setup the development environment use the makefile:
```make install```
This will install the package in a virtualenvironment using symlinks. I.e., if you modify the sources there is no need to reinstall, all changes are immediate.
To use the environment source the "activation" file that is created by the make command:
```source ./activate```
After this command the cnn commandline tools will be available to you.

To test changes to your code, simply run:
```make test```
This will run all tests specified in the "tests" directory verifying functionality
The same tests are executed by our gitlab runner when you commit your code to our server.
Please make sure all tests pass before creating a merge request to the master branch.
