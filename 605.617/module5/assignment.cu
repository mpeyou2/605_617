//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 16
#define BLOCK_SIZE 16
#define NUM_BLOCKS N/BLOCK_SIZE

#define ARRAY_SIZE N
#define ARRAY_SIZE_IN_BYTES (sizeof(int) * (ARRAY_SIZE))

__constant__ int opNum = 1;
__constant__ int gpu_array1_c[ARRAY_SIZE_IN_BYTES];
__constant__ int gpu_array2_c[ARRAY_SIZE_IN_BYTES];

////////////////////////OPERATIONS//////////////////////

//SHARED MEMORY
__global__  void operations_shared(int * array1, int * array2, int *array3)
{
	int i = threadIdx.x;

	__shared__ int tmpArray1_s[ARRAY_SIZE];
	__shared__ int tmpArray2_s[ARRAY_SIZE];
	__shared__ int tmpArray3_s[ARRAY_SIZE];

	tmpArray1_s[i] = array1[i];
	tmpArray2_s[i] = array2[i];
	tmpArray3_s[i] = array3[i];

	if (opNum ==1)
	{ tmpArray3_s[i]=tmpArray1_s[i]+tmpArray2_s[i];}
	else if (opNum ==2)
	{ tmpArray3_s[i]=tmpArray1_s[i]-tmpArray2_s[i];}
	else if (opNum ==3)
	{ tmpArray3_s[i]=tmpArray1_s[i]*tmpArray2_s[i];}
	else //if (opNum ==4)
	{ tmpArray3_s[i]=tmpArray1_s[i]%tmpArray2_s[i];}

	__syncthreads();

	array1[i] = tmpArray1_s[i];
	array2[i] = tmpArray2_s[i];
	array3[i] = tmpArray3_s[i];

}

//CONSTANT MEMORY
__global__  void operations_constant(int* array3)
{
	int marker = threadIdx.x+blockDim.x*blockIdx.x;

	if (opNum ==1){ array3[marker]=gpu_array1_c[marker]+gpu_array2_c[marker];}
	else if (opNum ==2){array3[marker]=gpu_array1_c[marker]-gpu_array2_c[marker];}
	else if (opNum ==3){ array3[marker]=gpu_array1_c[marker]*gpu_array2_c[marker];}
	else { array3[marker]=gpu_array1_c[marker]%gpu_array2_c[marker];}

	__syncthreads();


}

//////////////////////////MAIN CPU FUNCTION////////////////////////////
int main(int argc, char** argv)
{

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int isConst = 1;
	
	if (argc >= 2) {
		isConst = atoi(argv[1]);
	}				

	/* Declare statically 3 array sized ARRAY_SIZE*/

	
	int* host_array1;
	int* host_array2;
	int* host_array3;

	host_array1=(int*)malloc(ARRAY_SIZE_IN_BYTES);
	host_array2=(int*)malloc(ARRAY_SIZE_IN_BYTES);
	host_array3=(int*)malloc(ARRAY_SIZE_IN_BYTES);

	/* Declare pointers for GPU based params */
	int *gpu_array1;
	int *gpu_array2;
	int *gpu_array3;

	for(int i= 0; i < ARRAY_SIZE; i++)
	{
		host_array1[i] = i;
		host_array2[i] = 1; 
		host_array3[i] = 0;

		//Check thathost_array1 and array 2 inputs are correct
		//printf("ARRAY1 at %u\nARRAY2 at %u\nARRAY3 at %u\n\n",host_array1[i], host_array2[i],host_array3[i]);
	}


	cudaMalloc((void**)&gpu_array1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_array2, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_array3, ARRAY_SIZE_IN_BYTES);

	cudaEventRecord(start);
	/////////////////USE SHARED MEMORY///////////////
	if (isConst==0)
	{

	cudaMemcpy( gpu_array1,host_array1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_array2,host_array2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_array3,host_array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	operations_shared<<<NUM_BLOCKS,BLOCK_SIZE>>>(gpu_array1,gpu_array2,gpu_array3);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f\n", milliseconds);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy(host_array1, gpu_array1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array2, gpu_array2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array3, gpu_array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);


	cudaFree(gpu_array1);
	cudaFree(gpu_array2);
	cudaFree(gpu_array3);
	}
	////////////////////USE CONSTANT MEMORY////////////////////////////////
	else if (isConst==1)
	{
		cudaMemcpyToSymbol( gpu_array1_c,host_array1, ARRAY_SIZE_IN_BYTES);
		cudaMemcpyToSymbol( gpu_array2_c,host_array2, ARRAY_SIZE_IN_BYTES);
		cudaMemcpy( gpu_array3,host_array3,ARRAY_SIZE_IN_BYTES,cudaMemcpyHostToDevice);

		operations_constant<<<NUM_BLOCKS,BLOCK_SIZE>>>(gpu_array3);

		
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Time elapsed: %f\n", milliseconds);

		cudaMemcpy(host_array3, gpu_array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

		
		cudaFree(gpu_array1);
		cudaFree(gpu_array2);
		cudaFree(gpu_array3);
	}	
	for( int k=0; k<ARRAY_SIZE; k++)
	{
		printf("\nINDEX: %i\tVALUE:%i\n",k, host_array3[k]);

	}
	
	
}

