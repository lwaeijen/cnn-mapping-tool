# Sets up a testing environment
VENV=.venv
ACT=./activate
ACTIVATE_SCRIPT=$(VENV)/bin/activate
UPDATED_TOOLS=$(VENV)/.pip_update_done
INSTALL_DONE=$(VENV)/.install_done

#PATHS for network examples
SOURCE_DIR=./examples/
BUILD_DIR=./build/


#For debugging we add dependencies to all python files
PYTHON_DEPS=$(wildcard cnn_implementer/backends/templates/*.c.j2) $(shell find cnn_implementer -name "*.py")


#################
#
# PYTHON TESTING
#

#Default python tests
.PHONY:test
test:$(INSTALL_DONE)
	bash -c "source $(ACT) && ./setup.py test"

#run halide to check if output is valid
#by default these tests are excluded from the generic tests as it requires halide to be setup
.PHONY:halide
halide:$(INSTALL_DONE)
	bash -c "source $(ACT) && ./setup.py test_halide"


################
#
# Caffe networks
#

#Caffe Network prototxts
CAFFE_SOURCE_FILES=caffe/traffic/traffic.prototxt caffe/simple/simple.prototxt caffe/VDSR/VDSR.prototxt
CAFFE_TARGETS=$(addprefix $(BUILD_DIR), $(CAFFE_SOURCE_FILES:.prototxt=.exe))

#Tool settings
LOG_LEVEL=DEBUG
GENERIC_FLAGS=--log-level=$(LOG_LEVEL)
BACKEND_HALIDE_FLAGS= $(GENERIC_FLAGS) #--halide-debug-code

#"Build" all caffe networks executables
.PHONY:caffe
caffe:$(CAFFE_TARGETS)



#Specific rules for the traffic sign network
.PHONY:traffic traffic_memsize traffic_accesses
traffic:traffic_memsize
traffic_memsize:$(BUILD_DIR)caffe/traffic/traffic_memsize
traffic_accesses:$(BUILD_DIR)caffe/traffic/traffic_trace.csv
TRAFFIC_INPUT_IMG=$(subst $(BUILD_DIR),$(SOURCE_DIR),./$(dir $@))test046.png
%traffic_profile.txt:%traffic_profile.exe
	./$< $(TRAFFIC_INPUT_IMG) $(dir $@)traffic_detections.txt 2>&1 | tee $@
%traffic_trace.csv:%traffic_trace.exe $(INSTALL_DONE)
	bash -c "source $(ACT) && halide-access-count -o $@ -c './$< $(TRAFFIC_INPUT_IMG) $(dir $@)traffic_detections_trace.txt'"
.PRECIOUS: %traffic_profile.txt %traffic_trace.csv


#Specific rules for the VDSR network
.PHONY:VDSR VDSR_memsize VDSR_accesses
VDSR:VDSR_memsize
VDSR_memsize:$(BUILD_DIR)caffe/VDSR/VDSR_memsize
VDSR_accesses:$(BUILD_DIR)caffe/VDSR/VDSR_trace.csv
VDSR_INPUT_IMG=$(subst $(BUILD_DIR),$(SOURCE_DIR),./$(dir $@))blr_256.png
%VDSR_profile.txt:%VDSR_profile.exe
	./$< $(VDSR_INPUT_IMG) $(dir $@)blh_256.png 2>&1 | tee $@
%VDSR_trace.csv:%VDSR_trace.exe $(INSTALL_DONE)
	bash -c "source $(ACT) && halide-access-count -o $@ -c './$< $(VDSR_INPUT_IMG) $(dir $@)blh_256_trace.png'"
.PRECIOUS: %VDSR_profile.txt %VDSR_trace.csv


#Specific rules for the simple network
.PHONY:simple simple_memsize simple_accesses
simple:simple_memsize
simple_memsize:$(BUILD_DIR)caffe/simple/simple_memsize
simple_accesses:$(BUILD_DIR)caffe/simple/simple_trace.csv
SIMPLE_INPUT_IMG=$(subst $(BUILD_DIR),$(SOURCE_DIR),./$(dir $@))lena_20x20.png
%simple_profile.txt:%simple_profile.exe
	./$< $(SIMPLE_INPUT_IMG) $(dir $@)lena_out.png 2>&1 | tee $@
%simple_trace.csv:%simple_trace.exe $(INSTALL_DONE)
	bash -c "source $(ACT) && halide-access-count -o $@ -c './$< $(SIMPLE_INPUT_IMG) $(dir $@)lena_out_trace.png'"
.PRECIOUS: %simple_profile.txt %simple_trace.csv


###############################################
#
# Generic build rules for the example networks
#

#generate network from caffe prototxt
$(BUILD_DIR)%.net $(BUILD_DIR)%_network.dot:$(SOURCE_DIR)%.prototxt $(INSTALL_DONE) $(PYTHON_DEPS)
	@mkdir -p $(dir $@)
	bash -c "source $(ACT) && cnn-frontend --frontend=caffe --caffe-deploy=$< --write-model=$(BUILD_DIR)$*.net --write-network-dotfile=$(BUILD_DIR)$*.dot"

#generate design points by DSE
%_points.json %_segmentgraph.dot:%.net $(INSTALL_DONE) $(PYTHON_DEPS)
	bash -c "source $(ACT) && cnn-dse --dse-save-points=$*_points.json --dse-write-segment-dot=$*_segmentgraph.dot --net=$<"

#generate halide code for point
%.cpp:%_points.json %.net $(INSTALL_DONE) $(PYTHON_DEPS)
	bash -c "source $(ACT) && cnn-backend $(BACKEND_HALIDE_FLAGS) --net=$*.net --point=$*_points.json --halide-code=$@"

#generate halide code with profiling enabled for point
%_profile.cpp:%_points.json %.net $(INSTALL_DONE) $(PYTHON_DEPS)
	bash -c "source $(ACT) && cnn-backend $(BACKEND_HALIDE_FLAGS) --halide-profile-code --net=$*.net --point=$*_points.json --halide-code=$@"

#generate halide code with tracing for point
%_trace.cpp:%_points.json %.net $(INSTALL_DONE) $(PYTHON_DEPS)
	bash -c "source $(ACT) && cnn-backend $(BACKEND_HALIDE_FLAGS) --halide-trace-code --net=$*.net --point=$*_points.json --halide-code=$@"

#compile halide code
%.exe:%.cpp
	g++ $< -lHalide -lpthread -ldl -lz -lpng -ljpeg $(shell llvm-config --system-libs 2> /dev/null) -std=c++11 -o $@

#Get memory size by parsing halide profile information
%_memsize:%_profile.txt $(INSTALL_DONE)
	bash -c "source $(ACT) && halide-mem-size -i $< > $@"

#Prevent deletion of intermediate files
.PRECIOUS: $(BUILD_DIR)%.net %_points.json %.cpp %_profile.cpp %_trace.cpp %.exe %_memsize


#######################################################
#
# targets for installation of the virtual environment
#
.PHONY:install
install:$(INSTALL_DONE)
$(INSTALL_DONE):$(ACT) $(UPDATED_TOOLS)
	bash -c "source $(ACT) && pip install -e . && touch $@"

$(UPDATED_TOOLS):$(ACT)
	# In particular on our bare bone CI test runners the setuptools can be outdated without forcing an update
	bash -c "source $(ACT) && pip install -U pip wheel setuptools && touch $@"

$(ACT):$(ACTIVATE_SCRIPT)
	ln -fs $< $@

$(ACTIVATE_SCRIPT):
	bash -c "export LC_ALL=C && virtualenv -p python2 $(VENV) || ( rm -rf $(VENV); exit 1 )"


###########
#
# Cleaning
#
#Cleaning of built networks
.PHONY:clean
clean:
	rm -rf $(BUILD_DIR)

#Clean full virtual environment
.PHONY:realclean
realclean:clean
	rm -rf $(VENV) $(ACT)
