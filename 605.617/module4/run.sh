nvcc assignment.cu -w -o assignment_output
	./assignment_output 32 32 1 5 >> mod4_output.txt
	./assignment_output  32 32 0 5 >> mod4_output.txt
	./assignment_output  64 32 1 5 >> mod4_output.txt
	./assignment_output 64 32 0 5 >> mod4_output.txt
	./assignment_output  4096 128 0 1 >> mod4_output.txt
	./assignment_output 4096 128 1 1 >> mod4_output.txt