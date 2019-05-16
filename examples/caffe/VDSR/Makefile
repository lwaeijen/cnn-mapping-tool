HALIDE_PATH=~/halide/
HALIDE_LD=LD_LIBRARY_PATH=$(HALIDE_PATH)/bin

all:log
	@cat log

log:exe
	$(HALIDE_LD) ./exe > log 2>&1;

code.cpp:
	cnn-cli -i "VDSR_net_deploy.prototxt" -o "code.cpp"

exe:code.cpp
	$(HALIDE_LD) g++ code.cpp -g -I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools -L $(HALIDE_PATH)/bin -lHalide -lpthread -ldl -std=c++11 -o exe

clean:
	rm -rf exe code.cpp log
