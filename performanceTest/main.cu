#include "global.h"
#include "hashGlobal.h"
#include "kernel.cu"

#define TEXTITEMSIZE 1
#define DATAITEMSIZE 1
#define RECORD_SIZE 64
#define GB 1073741824

#define EPOCHCHUNK 30

#define NUMTHREADS (MAXBLOCKS * BLOCKSIZE)

#define MULTI_VALUE 1



__global__ void insert_kernel(	input_t* data,
			unsigned size,
			unsigned numThreads,
			multipassConfig_t* mbk,
			char* states)
{
	int index = TID;
	int* myNumbers = mbk->dmyNumbers;
	input_t value;
	value.data1 = 1;

	for(unsigned i = index; i < size; i += numThreads)
	{
		int sizeIdentifier = (i / numThreads) % 16;
		int keySize = 8 + sizeIdentifier * 8;

		if(insert_basic((void*) &(data[i]), keySize, (void*) &value, keySize, mbk) == true)
		{
			myNumbers[index * 2] ++;
			//states[i] = SUCCEED;
		}
		else
		{
			myNumbers[index * 2 + 1] ++;
		}

	}
	//printf("%lld\n", sum);
}

__global__ void lookup_kernel(	input_t* data,
			unsigned size,
			unsigned numThreads,
			multipassConfig_t* mbk,
			char* states)
{
	int index = TID;
	int* myNumbers = mbk->dmyNumbers;
	input_t value;
	value.data1 = 1;

	for(unsigned i = index; i < size; i += numThreads)
	{
		int sizeIdentifier = (i / numThreads) % 16;
		int keySize = 8 + sizeIdentifier * 8;

		if(lookup_basic((void*) &(data[i]), keySize, mbk) != NULL)
		{
			myNumbers[index * 2] ++;
			//states[i] = SUCCEED;
		}
		else
		{
			myNumbers[index * 2 + 1] ++;
		}

	}
	//printf("%lld\n", sum);
}

__global__ void insert_multi_value_kernel(	input_t* data,
						unsigned size,
						unsigned numThreads,
						multipassConfig_t* mbk,
						char* states)
{
	int index = TID;
	int* myNumbers = mbk->dmyNumbers;
	input_t value;
	value.data1 = 1;
	largeInt sum = 0;

	for(unsigned i = index; i < size; i += numThreads)
	{
		//int sizeIdentifier = (i / numThreads) % 16;
		//int keySize = 8 + sizeIdentifier * 8;
		int keySize = sizeof(input_t);

		if(insert_multi_value((void*) &(data[i]), keySize, (void*) &value, keySize, mbk) == true)
		{
			myNumbers[index * 2] ++;
			//states[i] = SUCCEED;
		}
		else
		{
			myNumbers[index * 2 + 1] ++;
		}

	}
	//printf("%lld\n", sum);
}



int main(int argc, char** argv)
{
	cudaError_t errR;
	cudaThreadExit();

	if (argc < 2)
	{
		printf("USAGE: %s <input elements in million>\n", argv[0]);
		exit(1);
	}

	unsigned inputDataSize = atoi(argv[1]);
	inputDataSize *= 1000000;

	input_t* inputData = (input_t*) malloc(inputDataSize * sizeof(input_t));
	
	srand(time(NULL));
	for(unsigned i = 0; i < inputDataSize; i ++)
		inputData[i].data1 = rand() % (inputDataSize / 3);


	input_t* dinputData;
	cudaMalloc((void**) &dinputData, inputDataSize * sizeof(input_t));
	cudaMemcpy(dinputData, inputData, inputDataSize * sizeof(input_t), cudaMemcpyHostToDevice);


	dim3 grid(16, 1, 1);
	dim3 block(384, 1, 1);
	int numThreads = block.x * grid.x * grid.y;

	//============ initializing the hash table and page table ==================//
	int pagePerGroup = 25;
	multipassConfig_t* mbk = initMultipassBookkeeping(	numThreads,
								inputDataSize,
								pagePerGroup);
	multipassConfig_t* dmbk;
	cudaMalloc((void**) &dmbk, sizeof(multipassConfig_t));
	cudaMemcpy(dmbk, mbk, sizeof(multipassConfig_t), cudaMemcpyHostToDevice);

	//==========================================================================//

	struct timeval partial_start, partial_end;//, bookkeeping_start, bookkeeping_end, passtime_start, passtime_end;
	time_t sec;
	time_t ms;
	time_t diff;

	errR = cudaGetLastError();
	printf("#######Error before calling the kernel is 2: %s\n", cudaGetErrorString(errR));

	//int passNo = 1;
	//bool failedFlag = false;

	gettimeofday(&partial_start, NULL);

	insert_multi_value_kernel<<<grid, block>>>(dinputData, 
				inputDataSize, 
				numThreads,
				dmbk,
				mbk->dstates);
	cudaThreadSynchronize();

	gettimeofday(&partial_end, NULL);
	sec = partial_end.tv_sec - partial_start.tv_sec;
	ms = partial_end.tv_usec - partial_start.tv_usec;
	diff = sec * 1000000 + ms;
	printf("\n%10s:\t\t%0.1fms\n", "Insert total time", (double)((double)diff/1000.0));


	errR = cudaGetLastError();
	printf("Error after calling the kernel is: %s\n", cudaGetErrorString(errR));


#if 0
	cudaMemset(mbk->dmyNumbers, 0, 2 * numThreads * sizeof(int));

	gettimeofday(&partial_start, NULL);

	lookup_kernel<<<grid, block>>>(dinputData, 
				inputDataSize, 
				numThreads,
				dmbk,
				mbk->dstates);
	cudaThreadSynchronize();

	gettimeofday(&partial_end, NULL);
	sec = partial_end.tv_sec - partial_start.tv_sec;
	ms = partial_end.tv_usec - partial_start.tv_usec;
	diff = sec * 1000000 + ms;
	printf("\n%10s:\t\t%0.1fms\n", "Lookup total time", (double)((double)diff/1000.0));
#endif


	cudaMemcpy(mbk, dmbk, sizeof(multipassConfig_t), cudaMemcpyDeviceToHost);
	unsigned totalCounter = mbk->counter1 + mbk->counter2 + mbk->counter3;
	printf("Counter 1: %0.1f\%\n", ((float) (mbk->counter1) / (float) totalCounter) * 100.0);
	printf("Counter 2: %0.1f\%\n", ((float) (mbk->counter2) / (float) totalCounter) * 100.0);
	printf("Counter 3: %0.1f\%\n", ((float) (mbk->counter3) / (float) totalCounter) * 100.0);


	int* dmyNumbers = mbk->dmyNumbers;
	int* myNumbers = mbk->myNumbers;
	cudaMemcpy(myNumbers, dmyNumbers, 2 * numThreads * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemset(dmyNumbers, 0, 2 * numThreads * sizeof(int));

	largeInt totalSuccess = 0;
	largeInt totalFailed = 0;
	for(int i = 0; i < numThreads; i ++)
	{
		totalSuccess += myNumbers[i * 2];
		totalFailed += myNumbers[i * 2 + 1];
	}

	printf("Total success: %lld\n", totalSuccess);
	printf("Total failure: %lld\n", totalFailed);



	cudaMemcpy((void*) mbk->hashTableOffset, mbk->dbuffer, mbk->totalNumPages * PAGE_SIZE, cudaMemcpyDeviceToHost);
	hashBucket_t** buckets = (hashBucket_t**) malloc(NUM_BUCKETS * sizeof(hashBucket_t*));
	cudaMemcpy(buckets, mbk->buckets, NUM_BUCKETS * sizeof(hashBucket_t*), cudaMemcpyDeviceToHost);
	

	int totalDepth = 0;
	int totalValidBuckets = 0;
	int totalEmpty = 0;
	int maximumDepth = 0;
#ifdef MULTI_VALUE
	int totalValueDepth = 0;
#endif
	for(int i = 0; i < NUM_BUCKETS; i ++)
	{
		hashBucket_t* bucket = buckets[i];
		if(bucket == NULL)
			totalEmpty ++;
		else
			totalValidBuckets ++;

		int localMaxDepth = 0;
		while(bucket != NULL)
		{
#ifdef MULTI_VALUE
			int valueDepth = 0;
			valueHolder_t* valueHolder = bucket->valueHolder;
			while(valueHolder != NULL)
			{
				valueDepth ++;
				valueHolder = valueHolder->next;
			}
			totalValueDepth += valueDepth;
#endif
			totalDepth ++;
			localMaxDepth ++;
			bucket = bucket->next;
		}
		if(localMaxDepth > maximumDepth)
			maximumDepth = localMaxDepth;
	}

	float emptyPercentage = ((float) totalEmpty / (float) NUM_BUCKETS) * (float) 100;
	float averageDepth = (float) totalDepth / (float) totalValidBuckets;
	printf("Empty percentage: %0.1f\n", emptyPercentage);
	printf("Average depth: %0.1f\n", averageDepth);
	printf("Max depth: %d\n", maximumDepth);

#ifdef MULTI_VALUE
	float averageValueDepth = (float) totalValueDepth / (float) totalDepth;
	printf("Average value depth: %0.1f\n", averageValueDepth);
#endif


	return 0;
}
