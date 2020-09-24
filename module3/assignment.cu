//Based on the work of Andrew Krepps
#include <stdio.h>


#include <stdio.h>

#define ARRAY_SIZE N
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

////////////////////////OPERATIONS//////////////////////////////////////////////

//ADD=1
__global__  void add(int * array1,int * array2,int * array3)
{
	const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	array3[i]=array1[i]+array2[i];
}

//SUBTRACT=2
__global__  void subtract(int * array1,int * array2,int * array3)
{
	const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	array3[i]=array1[i]-array2[i];
}


//MULTIPLY=3
__global__  void multiply(int * array1,int * array2,int * array3)
{
	
	const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	array3[i]=array1[i]*array2[i];
}


//MOD=4
__global__  void mod(int * array1,int * array2,int * array3)
{
	const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	array3[i]=array1[i]%array2[i];
}


//////////////////////////GPU FUNCTION//////////////////////////////////
void main_sub(int N, int BLOCK_SIZE, int NUM_BLOCKS, int whichOperation)
{
	/* Declare  statically four arrays of ARRAY_SIZE each */
	int array1[ARRAY_SIZE];
	int array2[ARRAY_SIZE];
	int array3[ARRAY_SIZE];

	for(int i = 0; i < ARRAY_SIZE; i++)
	{
		array1[i] = i;
		array2[i] = (rand()%4);

		//Check that array1 and array 2 inputs are correct
		//printf("ARRAY1 at %i\nARRAY2 at %i\n\n", array1[i], array2[i]);
	}



	/* Declare pointers for GPU based params */
	int *gpu_block1;
	int *gpu_block2;
	int *gpu_block3;

	cudaMalloc((void **)&gpu_block1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block2, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block3, ARRAY_SIZE_IN_BYTES);


	cudaMemcpy( gpu_block1, array1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_block2, array2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_block3, array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	/* Execute our kernel */
	switch(whichOperation) {

   //ADD
   	case 1  :
   		printf("///////////////////////OUTPUT ADD///////////////\n");
    	add<<<NUM_BLOCKS, BLOCK_SIZE>>>(gpu_block1,gpu_block2,gpu_block3);
    	break; 
   	//SUBTRACT
   	case 2  :
   		printf("///////////////////////OUTPUT SUBTRACT///////////////\n");
    	subtract<<<NUM_BLOCKS, BLOCK_SIZE>>>(gpu_block1,gpu_block2,gpu_block3);
    	break;
   	//MULTIPLY 
   	case 3  :
   		printf("///////////////////////OUTPUT MULTIPLY///////////////\n");
    	multiply<<<NUM_BLOCKS, BLOCK_SIZE>>>(gpu_block1,gpu_block2,gpu_block3);
    	break;
	//MOD 
   	case 4  :
   		printf("///////////////////////OUTPUT MOD///////////////\n");
    	mod<<<NUM_BLOCKS, BLOCK_SIZE>>>(gpu_block1,gpu_block2,gpu_block3);
    	break; 
   
	}
	

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( array1, gpu_block1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( array2, gpu_block2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( array3, gpu_block3, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block1);
	cudaFree(gpu_block2);
	cudaFree(gpu_block3);

	/* Iterate through the arrays and print */
	for(int i = 0; i < ARRAY_SIZE; i+=4)
	{
		printf("Index %i:\t %i\t\tIndex %i:\t %i\t\tIndex %i:\t %i\t\tIndex %i:\t %i\n", i, array3[i], i+1, array3[i+1],i+2, array3[i+2], i+3, array3[i+3]);
	}
}

//////////////////////////MAIN///////////////////////////////////

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	main_sub(totalThreads,blockSize,numBlocks, 1);
	main_sub(totalThreads,blockSize,numBlocks, 2);
	main_sub(totalThreads,blockSize,numBlocks, 3);
	main_sub(totalThreads,blockSize,numBlocks, 4);

}

