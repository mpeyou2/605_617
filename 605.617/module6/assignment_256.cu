//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 256
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

//REGISTER MEMORY
__global__  void operations_reg(int* array1,int* array2,int* array3, int opNum)
{
	int marker = threadIdx.x+blockDim.x*blockIdx.x;

	int tmp_data1 = array1[marker];
	int tmp_data2 = array2[marker];
	int tmp_data3 = array3[marker];

	if (opNum ==1){ tmp_data3=tmp_data1+tmp_data2;}
	else if (opNum ==2){tmp_data3=tmp_data1-tmp_data2;}
	else if (opNum ==3){tmp_data3=tmp_data1*tmp_data2;}
	else { tmp_data3=tmp_data1%tmp_data2;}

	array3[marker]=tmp_data3;
}
//SHARED MEMORY
__global__  void operations_shared(int * array1, int * array2, int *array3, int opNum)
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

//****************************************************************************

void main_register(int opNum)
{  

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
///////////Declare Arrays/////////

	
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

	cudaMalloc((void**)&gpu_array1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_array2, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_array3, ARRAY_SIZE_IN_BYTES);

///////////Fill host arrays with values/////////

	generateData(host_array1, 1);
	generateData(host_array2, 2);
	generateData(host_array3, 3);

	cudaEventRecord(start);
	
///////////copy over memory /////////
	cudaMemcpy( gpu_array1,host_array1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_array2,host_array2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_array3,host_array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

///////////execute operation/////////
	operations_reg<<<NUM_BLOCKS,BLOCK_SIZE>>>(gpu_array1,gpu_array2,gpu_array3,opNum);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f\n", milliseconds);

///////////copy memory back /////////
	cudaMemcpy(host_array3, gpu_array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost); //only need array 3


	cudaFree(gpu_array1);
	cudaFree(gpu_array2);
	cudaFree(gpu_array3);
	

	printf("\n/////////////////REGISTER MEMORY RESULTS//////////////////\n");
	for( int k=0; k<ARRAY_SIZE; k++)
	{
		printf("\nINDEX: %i\tVALUE:%i\n",k, host_array3[k]);

	}
	
	
}

//****************************************************************************
/////////////////////////USE SHARED MEMORY/////////////////////////
void main_shared( int opNum)
{

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
///////////Declare Arrays/////////

	
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

	cudaMalloc((void**)&gpu_array1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_array2, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_array3, ARRAY_SIZE_IN_BYTES);

///////////Fill host arrays with values/////////

	generateData(host_array1, 1);
	generateData(host_array2, 2);
	generateData(host_array3, 3);

	cudaEventRecord(start);
	
///////////copy over memory /////////
	cudaMemcpy( gpu_array1,host_array1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_array2,host_array2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_array3,host_array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

///////////execute operation/////////
	operations_shared<<<NUM_BLOCKS,BLOCK_SIZE>>>(gpu_array1,gpu_array2,gpu_array3,opNum);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f\n", milliseconds);

///////////copy memory back /////////
	cudaMemcpy(host_array3, gpu_array3, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost); //only need array 3


	cudaFree(gpu_array1);
	cudaFree(gpu_array2);
	cudaFree(gpu_array3);
	

	printf("\n/////////////////SHARED MEMORY RESULTS//////////////////\n");
	for( int k=0; k<ARRAY_SIZE; k++)
	{
		printf("\nINDEX: %i\tVALUE:%i\n",k, host_array3[k]);

	}
	
	
}

int main(int argc, char** argv){

	int opNum=1; 
	

	if (argc >= 2) {
		opNum = atoi(argv[1]);
	}

	main_shared(opNum);
	main_register(opNum);

	return 0;
}

