#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include "global.h"


#define RECORD_LENGTH 64
#define NUM_BUCKETS 1000000

#define BLOCK_ID (gridDim.y * blockIdx.x + blockIdx.y)
#define THREAD_ID (threadIdx.x)
#define TID (BLOCK_ID * blockDim.x + THREAD_ID)

typedef struct
{
	pagingConfig_t* pconfig;
	cudaStream_t* serviceStream;
} dataPackage_t;


__global__ void kernel(char* records, int numRecords, int* recordSizes, int numThreads, pagingConfig_t* pconfig, hashtableConfig_t* hconfig)
{
	int index = TID;
	
	for(int i = index; i < numRecords; i += numThreads)
	{
		char* record = &records[i * RECORD_LENGTH];
		int recordSize = recordSizes[i];
		int value = 1;

		addToHashtable((void*) record, recordSize, (void*) &value, sizeof(int), hconfig, pconfig);
	}
}

void* recyclePages(void* arg)
{
	pagingConfig_t* pconfig = ((dataPackage_t*) arg)->pconfig;
	cudaStream_t* serviceStream = ((dataPackage_t*) arg)->serviceStream;

	pageRecycler(pconfig, serviceStream);
	return NULL;
}


int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("Usage: %s <numinputs>\n", argv[0]);
		return 1;
	}	

	int numRecords = atoi(argv[1]);

	dim3 grid(1, 1, 8);
	dim3 block(1, 1, 512);
	int numThreads = grid.x * block.x;
	numRecords = (numRecords % numThreads == 0)? numRecords : (numRecords + (numThreads - (numRecords % numThreads)));
	

	printf("@INFO: Allocating %dMB for input data\n", (numRecords * RECORD_LENGTH) / (1 << 20));
	char* records = (char*) malloc(numRecords * RECORD_LENGTH);
	int* recordSizes = (int*) malloc(numRecords * sizeof(int));

	srand(time(NULL));

	for(int i = 0; i < numRecords; i ++)
	{
		recordSizes[i] = rand() % (RECORD_LENGTH - 8);
		if(recordSizes[i] < 10)
			recordSizes[i] = 3;
	}

	for(int i = 0; i < numRecords; i ++)
	{
		records[i * RECORD_LENGTH + 0] = 'w';
		records[i * RECORD_LENGTH + 1] = 'w';
		records[i * RECORD_LENGTH + 2] = 'w';
		records[i * RECORD_LENGTH + 3] = '.';

		int j = 4;
		for(; j < recordSizes[i] - 4; j ++)
			records[i * RECORD_LENGTH + j] = rand() % 26 + 97;

		records[i * RECORD_LENGTH + j + 0] = '.';
		records[i * RECORD_LENGTH + j + 1] = 'c';
		records[i * RECORD_LENGTH + j + 2] = 'o';
		records[i * RECORD_LENGTH + j + 3] = 'm';
	}

	char* drecords;
	cudaMalloc((void**) &drecords, numRecords * RECORD_LENGTH);
	cudaMemcpy(drecords, records, numRecords * RECORD_LENGTH, cudaMemcpyHostToDevice);

	int* drecordSizes;
	cudaMalloc((void**) &drecordSizes, numRecords * sizeof(int));
	cudaMemcpy(drecordSizes, recordSizes, numRecords * sizeof(int), cudaMemcpyHostToDevice);

	//============ initializing the hash table and page table ==================//
	largeInt availableGPUMemory = (1 << 29);
	unsigned minimumQueueSize = 20;
	pagingConfig_t* pconfig = (pagingConfig_t*) malloc(sizeof(pagingConfig_t));
	memset(pconfig, 0, sizeof(pagingConfig_t));
	initPaging(availableGPUMemory, minimumQueueSize, pconfig);

	hashtableConfig_t* hconfig = (hashtableConfig_t*) malloc(sizeof(hashtableConfig_t));
	hashtableInit(NUM_BUCKETS, 64, hconfig);
	
	
	pagingConfig_t* dpconfig;
	cudaMalloc((void**) &dpconfig, sizeof(pagingConfig_t));
	cudaMemcpy(dpconfig, pconfig, sizeof(pagingConfig_t), cudaMemcpyHostToDevice);

	hashtableConfig_t* dhconfig;
	cudaMalloc((void**) &dhconfig, sizeof(hashtableConfig_t));
	cudaMemcpy(dhconfig, hconfig, sizeof(hashtableConfig_t), cudaMemcpyHostToDevice);
	
	cudaStream_t serviceStream;
	cudaStreamCreate(&serviceStream);
	
	//==========================================================================//
	


	//====================== Calling the kernel ================================//

	kernel<<<grid, block>>>(drecords, numRecords, drecordSizes, numThreads, dpconfig, dhconfig);

	
	pthread_t thread;
	dataPackage_t argument;

	argument.pconfig = pconfig;
	argument.serviceStream = &serviceStream;

	pthread_create(&thread, NULL, recyclePages, &argument);

	while(cudaSuccess != cudaStreamQuery(serviceStream))
		usleep(300);	
	cudaThreadSynchronize();
	
	
}
