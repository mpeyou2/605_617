//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h> 
#include <cuda.h> 

#define N 32*32
#define BLOCK_SIZE 16
#define NUM_BLOCKS N/BLOCK_SIZE

#define ARRAY_SIZE N
#define ARRAY_SIZE_IN_BYTES (sizeof(int) * (ARRAY_SIZE))

///generate data//
__host__ void generateData(int * host_data_ptr, int arrayNum)
{
        for(unsigned int i=0; i < N; i++)
        {
        	if(arrayNum ==1)
        	{
        		host_data_ptr[i] = i;
        	}
        	else if(arrayNum ==2)
        	{
        		host_data_ptr[i] = 1;
        	}
        	else
        	{
        		host_data_ptr[i] = rand()%3;
        	}
                
        }
}
//KERNEL// 
__global__ void operation(int *device_a, int *device_b, int *device_result, int opNum)
{

	int threadId = threadIdx.x + blockIdx.x * blockDim.x ;

	if (threadId < ARRAY_SIZE) 
        if (opNum ==1){ device_result[threadId]= device_a[threadId]+device_b[threadId];}
		else if (opNum ==2){device_result[threadId]= device_a[threadId]-device_b[threadId];}
		else if (opNum ==3){device_result[threadId]= device_a[threadId]*device_b[threadId];}
		else { device_result[threadId]= device_a[threadId]%device_b[threadId];} 
} 

//****************************************************************************

void main_streams(int opNum, int dev_num1, int dev_num2)
{  

	cudaDeviceProp prop; 
	int *host_a, *host_b, *host_result; 
  	int *device_a, *device_b, *device_result; 
  	int whichDevice; 

  	cudaGetDeviceCount( &whichDevice); 
  	cudaGetDeviceProperties( &prop, whichDevice); 

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaStream_t stream; 
  	cudaStreamCreate(&stream);
	
///////////Declare Arrays && allocate data/////////

	//Allocate Host Memory
	cudaHostAlloc((void **)&host_a, ARRAY_SIZE_IN_BYTES, cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_b, ARRAY_SIZE_IN_BYTES, cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_result, ARRAY_SIZE_IN_BYTES, cudaHostAllocDefault);

	//Allocate Device Memory

	cudaMalloc((void**)&device_a, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&device_b, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&device_result, ARRAY_SIZE_IN_BYTES);

///////////Fill host arrays with values/////////

	generateData(host_a, 1);
	generateData(host_b, 2);

	cudaEventRecord(start);
	
///////////copy over memory /////////

	cudaMemcpyAsync(device_a, host_a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice, stream); 
  	cudaMemcpyAsync(device_b, host_b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice, stream); 

///////////execute operation/////////
	
  	operation <<<ARRAY_SIZE, dev_num1, dev_num2, stream>>>(device_a, device_b, device_result, opNum);

  	cudaMemcpyAsync(host_result, device_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost, stream);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f\n", milliseconds);

///////////free memory /////////


	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_result);
	

	printf("\n/////////////////MULTIPLE STREAMS RESULTS//////////////////\n");
	for( int k=0; k<ARRAY_SIZE; k++)
	{
		printf("\nINDEX: %i\tVALUE:%i\n",k, host_result[k]);
	}
}

int main(int argc, char** argv){

	int opNum=1; 
	int dev_num1 = 1;
	int dev_num2 = 1;

	if (argc >= 2) {
		opNum = atoi(argv[1]);
	}

	if (argc >= 3) {
		dev_num1= atoi(argv[2]);
	}

	if (argc >= 4) {
		dev_num2 = atoi(argv[3]);
	}

	main_streams(opNum, dev_num1, dev_num2);

	return 0;
}

