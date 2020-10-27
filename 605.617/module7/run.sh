nvcc assignment.cu -lcudart -lcuda -o runMultiStreams

	./runMultiStreams 1 1 1 > multipleStreams_dev11_add_output.txt
	./runMultiStreams 2 1 2 > multipleStreams_dev12_subtract_output.txt