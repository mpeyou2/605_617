//Based on the work of Andrew Krepps
#include <stdio.h>


#include <stdlib.h>

#define ARRAY_SIZE N
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))
#define FIRST_ASCII_SYMBOL 65
//65=A
#define LAST_ASCII_SYMBOL 122
//122 = z
#define SHIFT 4
#define LETTER_RANGE LAST_ASCII_SYMBOL - FIRST_ASCII_SYMBOL 

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

//Caesar Cipher = 5 
__global__  void cipherEncrypt(int *message, int* cipher_key, int* encodedMsg)
{

   const unsigned int z = (blockIdx.x * blockDim.x) + threadIdx.x;  
   //shift all letter values to zero
   char zeroed_char = message[z] - FIRST_ASCII_SYMBOL;

   //make the cipher key
   	cipher_key[z] = ((zeroed_char + SHIFT) % LETTER_RANGE)+ FIRST_ASCII_SYMBOL;

   	char cipher_char = (char) cipher_key[z]+ FIRST_ASCII_SYMBOL;

	//change back to ascii and store in encodedMsg 
	encodedMsg[z] = (int) zeroed_char +SHIFT+ FIRST_ASCII_SYMBOL;

}


//////////////////////////GPU FUNCTION//////////////////////////////////
void main_sub(int N, int BLOCK_SIZE, int NUM_BLOCKS, int whichOperation, int pinnable)
{
	
	printf("/////NUM THREADS:%i\t BLOCK SIZE:%i \t",N,BLOCK_SIZE);

	//create timing 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

///////////////DECLARE PAGABLE MEMORY/////////////////
	int *h_pagable1; //pagable
	int *h_pagable2;
	int *h_pagable3;

	//PAGEABLE MEMORY
	/* Declare  statically 3 arrays of ARRAY_SIZE (N) each */
	h_pagable1 = (int*)malloc(ARRAY_SIZE*sizeof(int));	
	h_pagable2 = (int*)malloc(ARRAY_SIZE*sizeof(int));
	h_pagable3 = (int*)malloc(ARRAY_SIZE*sizeof(int));

/////////////////FILL ARRAYS////////////////
// fill the arrays with values described by module3; 
//comment out for arrays of //txt

	//using cipher
	if (whichOperation == 5)
	{
		for(int k=0; k < ARRAY_SIZE; k++)
		{
			h_pagable1[k] = (char)(k + 64);//just fill with alphabet


			h_pagable2[k] = 0; //cipher starts with all 0s

		}



	}
	//doing operation
	else{
		for(int i = 0; i < N; i++)
		{
			h_pagable1[i] = i;
			h_pagable2[i] = (rand()%4);

			//Check that array1 and array 2 inputs are correct
			//printf("ARRAY1 at %i\nARRAY2 at %i\n\n", h_pagable1[i], h_pagable[i]);
		}
	}
///////////////DECLARE DEVICE MEMORY/////////////////
	int *d_1;  //device memory
 	int *d_2;
	int *d_3;

	cudaMalloc((void**)&d_1, ARRAY_SIZE_IN_BYTES);  // device 
	cudaMalloc((void**)&d_2, ARRAY_SIZE_IN_BYTES); 
	cudaMalloc((void**)&d_3, ARRAY_SIZE_IN_BYTES);


///////////////DECLARE PINNED MEMORY && COPY DATA FROM CPU TO GPU/////////
	int *h_pinnable1; //pinnable memory
	int *h_pinnable2; 
	int *h_pinnable3;

	//USING PINNABLE MEMORY
	if (pinnable ==1) 
	{
		printf("Memory type: Pinned\t");
		cudaMallocHost((void**)&h_pinnable1, ARRAY_SIZE_IN_BYTES); // host pinned
		cudaMallocHost((void**)&h_pinnable2, ARRAY_SIZE_IN_BYTES);
		cudaMallocHost((void**)&h_pinnable3, ARRAY_SIZE_IN_BYTES);

//cudaMemcpy( array1, gpu_block1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost )
		
		memcpy( h_pinnable1, h_pagable1, ARRAY_SIZE_IN_BYTES );
		memcpy( h_pinnable2, h_pagable2, ARRAY_SIZE_IN_BYTES );
		memcpy( h_pinnable3, h_pagable3, ARRAY_SIZE_IN_BYTES );


		cudaMemcpy( d_1, h_pinnable1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
		cudaMemcpy( d_2, h_pinnable2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
		cudaMemcpy( d_3, h_pinnable3, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	}
	////USING ONLY PAGABLE MEMORY
	else{
		printf("Memory type: Pagable\t");
		cudaMemcpy( d_1, h_pagable1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
		cudaMemcpy( d_2, h_pagable2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
		cudaMemcpy( d_3, h_pagable3, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	}


///////////////////EXECUTE KERNEL////////////////////////////////
	
	cudaEventRecord(start);

	switch(whichOperation) {

   //ADD
   	case 1  :
   		printf("Operation: ADD///////////\n");
    	add<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_1,d_2,d_3);
    	break; 
   	//SUBTRACT
   	case 2  :
   		printf("Operation: SUBTRACT///////////\n");
    	subtract<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_1,d_2,d_3);
    	break;
   	//MULTIPLY 
   	case 3  :
   		printf("Operation: MUTIPLY///////////\n");
    	multiply<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_1,d_2,d_3);
    	break;
	//MOD 
   	case 4  :
   		printf("Operation: MOD///////////\n");
    	mod<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_1,d_2,d_3);
    	break; 
    //caesar cipher
    case 5   :
    	printf("Operation:///////////\n");
    	cipherEncrypt<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_1,d_2,d_3);
    	break;
   
	}

///////////////COPY BACK DATA FROM GPU TO CPU////////////////////////
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f\n", milliseconds);

	////////////////////


	cudaMemcpy( h_pagable1, d_1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_pagable2, d_2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_pagable3, d_3, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );

///////////////PRINT RESULTS////////////////////////////////////////


/* Iterate through the arrays and print */

	for(int i = 0; i < ARRAY_SIZE; i++)
	{	

		if (whichOperation ==5)
		{   

			char ogLetter = (char) h_pagable1[i];
			char cipherLetter = (char) h_pagable2[i]+FIRST_ASCII_SYMBOL;
			printf("\n\nOG Letter int was: %i\nOG Letter char was: %c\nCipher int is: %i\nEncoded int is:%i\nEncoded char is now: %c\n", h_pagable1[i], ogLetter, h_pagable2[i], h_pagable3[i], cipherLetter);

		}
		else{

			printf("Index %i:\t %i\n", i, h_pagable3[i]);
		}
	}


////////////////FREE MEMORY///////////////////////////////////////
	/* Free the arrays on the GPU as now we're done with them */
	cudaFree(d_1);
	cudaFree(d_2);
	cudaFree(d_3);


	cudaFreeHost(h_pinnable1);
	cudaFreeHost(h_pinnable2);
	cudaFreeHost(h_pinnable3);

	free(h_pagable1);
	free(h_pagable2);
	free(h_pagable3);

	
}

//////////////////////////MAIN///////////////////////////////////

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	int operationNum = 0;
	int pinnable = 0;

	//total threads
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	//block size
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}
	//using pinned memory?
	if (argc >= 4) {
		pinnable = atoi(argv[3]);
	}
	//operation/kernel execution number
	if (argc >= 5) {
		 operationNum = atoi(argv[4]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
    //int N, int BLOCK_SIZE, int NUM_BLOCKS, int whichOperation, int pinnable

	main_sub(totalThreads,blockSize,numBlocks, operationNum, pinnable);

}

