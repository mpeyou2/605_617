nvcc assignment.cu -o assignment_executable
	./assignment_executable 0 >> sharedMemory_output.txt
	./assignment_executable 1 >> constantMemory_output.txt
