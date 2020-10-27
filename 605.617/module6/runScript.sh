nvcc assignment_128.cu -o assignment_128Threads_executable
	./assignment_128Threads_executable 1 >> registerVsSharedMemory_output_add_128Threads.txt
	./assignment_128Threads_executable 4 >> registerVsSharedMemory_output_divide_128Threads.txt
	
nvcc assignment_256.cu -o assignment_256Threads_executable
	./assignment_256Threads_executable 1 >> registerVsSharedMemory_output_add_256Threads.txt
	./assignment_256Threads_executable 4 >> registerVsSharedMemory_output_divide_256Threads.txt