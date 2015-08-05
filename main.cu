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


__global__ void kernel(char* records, int numRecords, int* recordSizes, int numThreads, pagingConfig_t* pconfig, hashtableConfig_t* hconfig, int* status)
{
	int index = TID;
	
	for(int i = index; i < numRecords; i += numThreads)
	{
		char* record = &records[i * RECORD_LENGTH];
		int recordSize = recordSizes[i];
		recordSize = (recordSize % 8 == 0)? recordSize : (recordSize + (8 - (recordSize % 8)));
		largeInt value = 1;

		if(addToHashtable((void*) record, recordSize, (void*) &value, sizeof(largeInt), hconfig, pconfig) == true)
			status[index * 2] ++;
		else
			status[index * 2 + 1] ++;
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
	cudaError_t errR;
	int numRecords = 4500000;
	if(argc == 2)
	{
		numRecords = atoi(argv[1]);
	}	

	dim3 grid(8, 1, 1);
	dim3 block(512, 1, 1);
	int numThreads = grid.x * block.x;
	numRecords = (numRecords % numThreads == 0)? numRecords : (numRecords + (numThreads - (numRecords % numThreads)));
	printf("@INFO: Number of records: %d (%d per thread)\n", numRecords, numRecords / numThreads);
	
	

	printf("@INFO: Allocating %dMB for input data\n", (numRecords * RECORD_LENGTH) / (1 << 20));
	char* records = (char*) malloc(numRecords * RECORD_LENGTH);
	int* recordSizes = (int*) malloc(numRecords * sizeof(int));

	srand(time(NULL));

	for(int i = 0; i < numRecords; i ++)
	{
		recordSizes[i] = rand() % (RECORD_LENGTH - 8);
		if(recordSizes[i] < 14)
			recordSizes[i] = 14;
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

	printf("Some records:\n");
	for(int i = 0; i < 20; i ++)
	{
		for(int j = 0; j < recordSizes[i]; j ++)
		{
			printf("%c", records[i * RECORD_LENGTH + j]);
		}
		printf("\n");
	}

	printf("@INFO: done initializing the input data\n");

	char* drecords;
	cudaMalloc((void**) &drecords, numRecords * RECORD_LENGTH);
	cudaMemcpy(drecords, records, numRecords * RECORD_LENGTH, cudaMemcpyHostToDevice);

	int* drecordSizes;
	cudaMalloc((void**) &drecordSizes, numRecords * sizeof(int));
	cudaMemcpy(drecordSizes, recordSizes, numRecords * sizeof(int), cudaMemcpyHostToDevice);

	printf("@INFO: done allocating input records on GPU\n");

	//============ initializing the hash table and page table ==================//
	largeInt availableGPUMemory = (1 << 28);
	pagingConfig_t* pconfig = (pagingConfig_t*) malloc(sizeof(pagingConfig_t));
	memset(pconfig, 0, sizeof(pagingConfig_t));
	
	printf("@INFO: calling initPaging\n");
	initPaging(availableGPUMemory, pconfig);

	hashtableConfig_t* hconfig = (hashtableConfig_t*) malloc(sizeof(hashtableConfig_t));
	printf("@INFO: calling hashtableInit\n");
	hashtableInit(NUM_BUCKETS, hconfig);
	
	
	printf("@INFO: transferring config structs to GPU memory\n");
	pagingConfig_t* dpconfig;
	cudaMalloc((void**) &dpconfig, sizeof(pagingConfig_t));
	cudaMemcpy(dpconfig, pconfig, sizeof(pagingConfig_t), cudaMemcpyHostToDevice);

	hashtableConfig_t* dhconfig;
	cudaMalloc((void**) &dhconfig, sizeof(hashtableConfig_t));
	cudaMemcpy(dhconfig, hconfig, sizeof(hashtableConfig_t), cudaMemcpyHostToDevice);

	int* dstatus;
	cudaMalloc((void**) &dstatus, numThreads * 2 * sizeof(int));
	cudaMemset(dstatus, 0, numThreads * 2 * sizeof(int));
	
	cudaStream_t serviceStream;
	cudaStreamCreate(&serviceStream);
	cudaStream_t execStream;
	cudaStreamCreate(&execStream);
	
	//==========================================================================//
	
	 struct timeval partial_start, partial_end;//, exec_start, exec_end;
        time_t sec;
        time_t ms;
        time_t diff;
	


	//====================== Calling the kernel ================================//

	printf("@INFO: calling kernel\n");
	errR = cudaGetLastError();
	printf("@REPORT: CUDA Error before calling the kernel: %s\n", cudaGetErrorString(errR));

        gettimeofday(&partial_start, NULL);

	kernel<<<grid, block, 0, execStream>>>(drecords, numRecords, drecordSizes, numThreads, dpconfig, dhconfig, dstatus);

	
	pthread_t thread;
	dataPackage_t argument;

	argument.pconfig = pconfig;
	argument.serviceStream = &serviceStream;

	pthread_create(&thread, NULL, recyclePages, &argument);

	while(cudaErrorNotReady == cudaStreamQuery(execStream))
		usleep(300);	
	cudaThreadSynchronize();

	gettimeofday(&partial_end, NULL);
        sec = partial_end.tv_sec - partial_start.tv_sec;
        ms = partial_end.tv_usec - partial_start.tv_usec;
        diff = sec * 1000000 + ms;

        printf("\n%10s:\t\t%0.0f\n", "Total time", (double)((double)diff/1000.0));


	errR = cudaGetLastError();
	printf("@REPORT: CUDA Error at the end of the program: %s\n", cudaGetErrorString(errR));

	int* status = (int*) malloc(numThreads * 2 * sizeof(int));
	cudaMemcpy(status, dstatus, numThreads * 2 * sizeof(int), cudaMemcpyDeviceToHost);

	int totalSuccess = 0, totalFailed = 0;
	for(int i = 0; i < numThreads; i ++)
	{
		totalSuccess += status[i * 2];
		totalFailed += status[i * 2 + 1];
	}

	printf("Total success: %d\n", totalSuccess);
	printf("Total failed: %d\n", totalFailed);


	cudaMemcpy(pconfig->hpages, pconfig->pages, pconfig->totalNumPages * sizeof(page_t), cudaMemcpyDeviceToHost);
	int numUsed = 0;
	int numNotUsed = 0;
	int maxUsed = 0;
	for(int i = 0; i < pconfig->totalNumPages; i ++)
	{
		if(pconfig->hpages[i].used > 0)
			numUsed ++;
		else
			numNotUsed ++;

		if(pconfig->hpages[i].used > maxUsed)
			maxUsed = pconfig->hpages[i].used;
	}

	printf("@INFO: numUsed: %d\n", numUsed);
	printf("@INFO: numNotUsed: %d\n", numNotUsed);
	printf("@INFO: maxUsed: %d\n", maxUsed);

	return 0;
}
