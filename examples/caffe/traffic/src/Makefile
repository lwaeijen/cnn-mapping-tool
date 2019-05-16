all:traffic.caffemodel traffic.prototxt

traffic.weights.prototxt:$(wildcard *.bin) convert_bins.py traffic.prototxt.j2
	./convert_bins.py

traffic.caffemodel:traffic.weights.prototxt
	bash -c "source ../../../../activate && caffe-fmt-conv -i $< -o $@"

#for now let's see if a copy is just sufficient
traffic.prototxt:traffic.weights.prototxt
	cp $< $@


test:output.pgm
output.pgm:cnnlayersfix.exe
	./$<

cnnlayersfix.exe:cnnlayersfix.c
	gcc $< -o $@ -lm


clean:
	@rm -f traffic.prototxt traffic.weights.prototxt traffic.caffemodel
