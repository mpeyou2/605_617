//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 1024
#define BLOCK_SIZE 16
#define NUM_BLOCKS N/BLOCK_SIZE

#define ARRAY_SIZE N
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))


typedef unsigned short int u16;
typedef unsigned int u32;
int opNum=1;

__constant__ u32 gpu_array1[ARRAY_SIZE];
__constant__ u32 gpu_array2[ARRAY_SIZE];
__constant__ u32 const_data_gpu[ARRAY_SIZE];
__device__ static u32 gmem_data_gpu[KERNEL_LOOP];
static u32 const_data_host[KERNEL_LOOP];


////////////////////////OPERATIONS//////////////////////
__device__  void operations(u32 * array1, u32 * array2, u32 *array3)
{
	const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (opNum ==1)
	{ array3[i]=array1[i]+array2[i];}
	else if (opNum ==2)
	{ array3[i]=array1[i]-array2[i];}
	else if (opNum ==3)
	{ array3[i]=array1[i]*array2[i];}
	else //if (opNum ==4)
	{ array3[i]=array1[i]%array2[i];}
}
/////////////////COPIES DATA TO SHARED////////////////////////
__device__ void copy_data_to_shared(const u32 * data, u32 * tmp_array, const u32 tid)
{
	// Copy data into temp store
	for(u32 i = 0; i<ARRAY_SIZE; i++)
	{
		tmp_array[i+tid] = data[i+tid];
	}
	__syncthreads();
}


////////////////MAIN GLOBAL FUNCTIONS/////////////////


//SHARED//
__global__ void gpu_shared_data(u32 *host_array1,u32 * array2, u32 *host_array3)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ u32 array_tmp1[ARRAY_SIZE];
	__shared__ u32 array_tmp2[ARRAY_SIZE];
	__shared__ u32 array_tmp3[ARRAY_SIZE];

	copy_data_to_shared(array1, array_tmp1, tid);
	copy_data_to_shared(array2, array_tmp2, tid);
	copy_data_to_shared(array3, array_tmp3, tid);

	operations(array_tmp1, array_tmp2, array_tmp3, )
}

//////////////////////////MAIN CPU FUNCTION////////////////////////////
void main_sub()
{				
	/* Declare  statically four arrays of ARRAY_SIZE each */
	int host_array1[ARRAY_SIZE];
	int host_array2[ARRAY_SIZE];
	int host_array3[ARRAY_SIZE];

	for(u32 i= 0; i < ARRAY_SIZE; i++)
	{
		host_array1[i] = i;
		host_array2[i] = 1; //(rand()%4);

		//Check thathost_array1 and array 2 inputs are correct
		//printf("ARRAY1 at %u\nARRAY2 at %u\n\n",host_array1[i], array2[i]);
	}



	/* Declare pointers for GPU based params */
	u32 *gpu_block1;
	u32 *gpu_block2;
	u32 *gpu_block3;

	cudaMalloc((void **)&gpu_block1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block2, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block3, ARRAY_SIZE_IN_BYTES);

	int constant_var = 1;
	/*if(constant_var == 1)
	{
		//CONSTANT MEMORY

		cudaMemcpy( gpu_array1,host_array1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
		cudaMemcpy( gpu_array2,host_array2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
		cudaMemcpy( gpu_block3,host_array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );


		//execute kernel for constant data 
		printf("\n\n using constant memory\n\n");
		add<<<NUM_BLOCKS,BLOCK_SIZE>>>(gpu_array1,gpu_array2,gpu_block3);
	}

	else{*/
	//SHARED MEMORY
		//execute kernel for shared data
		printf("\n\n using shared memory\n\n");
		gpu_shared_data<<<NUM_BLOCKS,BLOCK_SIZE>>>(gpu_block1,gpu_block2,gpu_block3);

	//}

	  
	

	/* Free the arrays on the GPU as now we're done with them */
	//cudaMemcpy(host_array1, gpu_block1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	//cudaMemcpy(host_array2, gpu_block2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	//cudaMemcpy(host_array3, gpu_block3, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_block1);
	cudaFree(gpu_block2);
	cudaFree(gpu_block3);

	/* Iterate through the arrays and print */
	for(int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Index %i:\t %u\n",i,host_array3[i]);
	}
}

//////////////////////////MAIN INPUT///////////////////////////////////

int main()
{
	int totalThreads = (1 << 20);
	int blockSize = 256;

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	main_sub();
}

