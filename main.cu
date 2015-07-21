#include <stdio.h>


#define RECORD_LENGTH 64

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
	pagingConfig_t* pconfig = (pagingConfig_t*) malloc(sizeof(pagingConfig_t));
	initPaging(int hashBuckets, largeInt availableGPUMemory, int minimumQueueSize, largeInt freeMemBaseAddr, pagingConfig_t* pconfig)

	
	pagingConfig_t* dpconfig;
	cudaMalloc((void**) dpconfig, sizeof(pagingConfig_t));
	cudaMemcpy(dpconfig, pconfig, sizeof(pagingConfig_t), cudaMemcpyHostToDevice);
	
	//==========================================================================//
	


	//====================== Calling the kernel ================================//

	kernel<<<grid, block>>>(drecords, drecordSizes

	
	
}
