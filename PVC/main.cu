//Author: Reza Mokhtari
//Data: 11/4/2013
//Title: Wordcount

/* Description: This applications takes a document as input and counts the number of
 * occurrences of each url in it.
*/

#include "global.h"

#define HASHWALKSTEP 1333
#define NUMHASHROWS 10000000
#define DISPLAYRESULTS 
#define CHUNKMB 350

#define CACHEENABLED 1
__device__ int isEqualGrouping(char* first, char* second, int len, char* myTrainSpace)
{
	for(int i = 0; i < len; i ++)
	{
		//ACCESS 18
		collectAddresses((largeInt) &first[i], 18, myTrainSpace);
		char c = first[i];
		//ACCESS 19
		collectAddresses((largeInt) &second[i], 19, myTrainSpace);
		char d = second[i];

		
		if(c != d)
			return 0;
	}
	return 1;
	
}

__device__ int isEqualSimulate(char* first, char* second, int len, largeInt* addrLocator1, largeInt* addrLocator2, int* hits1, int* hits2, int* trainingAccesses)
{
	for(int i = 0; i < len; i ++)
	{
		//ACCESS 18
		simulateCacheHitAndMiss((largeInt) &first[i], 0, addrLocator1, hits1, trainingAccesses);
		char c = first[i];
		//ACCESS 19
		simulateCacheHitAndMiss((largeInt) &second[i], 1, addrLocator2, hits2, trainingAccesses);
		char d = second[i];
		
		if(c != d)
			return 0;
	}
	return 1;
	
}

__device__ int isEqual(char* first, char* second, int len, int cacheLineId1, int cacheLineId2,
			char* myCache, largeInt* addrLocator1, largeInt* addrLocator2, short* bitmap1, short* bitmap2)
{
	for(int i = 0; i < len; i ++)
	{
		//ACCESS 18
#ifdef CACHEENABLED
		char c = *((char*) cacheRead((char*) &first[i], cacheLineId1, myCache, addrLocator1, bitmap1));
#else
		char c = first[i];
#endif
		//ACCESS 19
#ifdef CACHEENABLED
		char d = *((char*) cacheRead((char*) &second[i], cacheLineId2, myCache, addrLocator2, bitmap2));
#else
		char d = second[i];
#endif
		
		if(c != d)
			return 0;
	}
	return 1;
	
}


__device__ unsigned int hashFuncSimulate(char* str, int len, unsigned numBuckets, int counter, largeInt* addrLocator1, int* hits1, int* trainingAccesses)
{                       
        unsigned hash = 2166136261;
        unsigned FNVMultiple = 16777619;

	for(int i = 0; i < len; i ++)
        {
		//ACCESS 2
		simulateCacheHitAndMiss((largeInt) &str[i], 0, addrLocator1, hits1, trainingAccesses);
		char c = str[i];

		hash += counter;
		hash += (int) c;
                hash = hash * FNVMultiple;  /* multiply by the magic number */
		hash += len;
		hash -= (int) c;
		hash *= (len + counter);
        }

        return hash % numBuckets;
}

__device__ unsigned int hashFunc(char* str, int len, unsigned numBuckets, int counter, int cacheLineId1, char* myCache, largeInt* addrLocator1, short* bitmap1)
{                       
        unsigned hash = 2166136261;
        unsigned FNVMultiple = 16777619;

	for(int i = 0; i < len; i ++)
        {
		//ACCESS 2
#ifdef CACHEENABLED
		char c = *((char*) cacheRead((char*) &str[i], cacheLineId1, myCache, addrLocator1, bitmap1));
#else
		char c = str[i];
#endif

		hash += counter;
		hash += (int) c;
                hash = hash * FNVMultiple;  /* multiply by the magic number */
		hash += len;
		hash -= (int) c;
		hash *= (len + counter);
        }

        return hash % numBuckets;
}


__device__ inline void addToHashTableSimulate(char* url, int urlSize, hashEntry* hashTable, int numBuckets, int hIndex, volatile unsigned int* volatile locks, 
						largeInt* addrLocator1, largeInt* addrLocator2, int* hits1, int* hits2, int* trainingAccesses)
{
	int oldValue = 1;
	int numConflict = 0;
	while(oldValue == 1)
	{
		hashEntry* possibleMatch = &hashTable[hIndex];

		//ACCESS 3
		simulateCacheHitAndMiss((largeInt) &(possibleMatch->valid), 1, addrLocator2, hits2, trainingAccesses);
		int possibleValid = possibleMatch->valid;
		if(possibleValid  == 1)
		{
			//ACCESS 4
			simulateCacheHitAndMiss((largeInt) &(possibleMatch->length), 1, addrLocator2, hits2, trainingAccesses);
			int possibleLength = possibleMatch->length;
			//ACCESS 5
			simulateCacheHitAndMiss((largeInt) &(possibleMatch->url), 1, addrLocator2, hits2, trainingAccesses);
			char* possibleUrl = possibleMatch->url;
			if(urlSize == possibleLength && isEqualSimulate(url, possibleUrl, urlSize, addrLocator1, addrLocator2, hits1, hits2, trainingAccesses))
			{
				//ACCESS 6
				simulateCacheHitAndMiss((largeInt) &(possibleMatch->counter), 1, addrLocator2, hits2, trainingAccesses);
				int tempCounter = possibleMatch->counter;
				tempCounter ++;
				//ACCESS 7, WRITE
				simulateCacheHitAndMiss((largeInt) &(possibleMatch->counter), 1, addrLocator2, hits2, trainingAccesses);
				possibleMatch->counter = tempCounter;
				oldValue = 0;
			}
			else
			{
				//hIndex = (hIndex+ HASHWALKSTEP) % numBuckets;
				numConflict ++;
				hIndex = hashFuncSimulate(url, urlSize, NUMHASHROWS, numConflict, addrLocator1, hits1, trainingAccesses);
			}
		}
		else
		{
			while(oldValue == 1)
			{
				oldValue = atomicCAS((unsigned int*) &locks[hIndex], 0, 1);

				if(oldValue == 0)
				{
					hashEntry* possibleMatch = &hashTable[hIndex];

					//ACCESS 8
					simulateCacheHitAndMiss((largeInt) &(possibleMatch->valid), 1, addrLocator2, hits2, trainingAccesses);
					int possibleValid = possibleMatch->valid;
					if(possibleValid == 1)
					{
						//ACCESS 9
						simulateCacheHitAndMiss((largeInt) &(possibleMatch->length), 1, addrLocator2, hits2, trainingAccesses);
						int possibleLength = possibleMatch->length;

						//ACCESS 10
						simulateCacheHitAndMiss((largeInt) &(possibleMatch->url), 1, addrLocator2, hits2, trainingAccesses);
						char* possibleUrl = possibleMatch->url;
						if(urlSize == possibleLength && isEqualSimulate(url, possibleUrl, urlSize, addrLocator1, addrLocator2, hits1, hits2, trainingAccesses))
						{
							//ACCESS 11
							simulateCacheHitAndMiss((largeInt) &(possibleMatch->counter), 1, addrLocator2, hits2, trainingAccesses);
							int tempCounter = possibleMatch->counter;
							tempCounter ++;
							//ACCESS 12, WRITE
							simulateCacheHitAndMiss((largeInt) &(possibleMatch->counter), 1, addrLocator2, hits2, trainingAccesses);
							possibleMatch->counter = tempCounter;

							atomicExch((unsigned int*) &locks[hIndex], 0);
						}
						else
						{
							atomicExch((unsigned int*) &locks[hIndex], 0);
							hIndex = (hIndex+ HASHWALKSTEP) % numBuckets;
							oldValue = 1;
						}
					}
					else
					{
						//ACCESS 13, WRITE
						simulateCacheHitAndMiss((largeInt) &(possibleMatch->counter), 1, addrLocator2, hits2, trainingAccesses);
						possibleMatch->counter = 1;
						//ACCESS 14, WRITE
						simulateCacheHitAndMiss((largeInt) &(possibleMatch->length), 1, addrLocator2, hits2, trainingAccesses);
						possibleMatch->length = urlSize;
						for(int j = 0; j < urlSize; j ++)
						{
							//ACCESS 15
							simulateCacheHitAndMiss((largeInt) &url[j], 0, addrLocator1, hits1, trainingAccesses);
							char c = url[j];

							//ACCESS 16, WRITE
							simulateCacheHitAndMiss((largeInt) &(possibleMatch->url[j]), 1, addrLocator2, hits2, trainingAccesses);
							possibleMatch->url[j] = c;
						}
						//ACCESS 17
						simulateCacheHitAndMiss((largeInt) &(possibleMatch->valid), 1, addrLocator2, hits2, trainingAccesses);
						possibleMatch->valid = 1;

						atomicExch((unsigned int*) &locks[hIndex], 0);
					}

				}
			}	
		}
	}
	//if(numConflict > 1000)
		//printf("Conflicts: %d\n", numConflict);

	return;
}


__device__ inline void addToHashTable(char* url, int urlSize, hashEntry* hashTable, int numBuckets, int hIndex, volatile unsigned int* volatile locks, 
		int cacheLineId1, int cacheLineId2, char* myCache, largeInt* addrLocator1, largeInt* addrLocator2, short* bitmap1, short* bitmap2)
{

	int oldValue = 1;
	int numConflict = 0;
	while(oldValue == 1)
	{
		hashEntry* possibleMatch = &hashTable[hIndex];

		//ACCESS 3
#ifdef CACHEENABLED
		int possibleValid = *((int*) cacheRead((char*) &(possibleMatch->valid), cacheLineId2, myCache, addrLocator2, bitmap2));
#else
		int possibleValid = possibleMatch->valid;
#endif
		if(possibleValid  == 1)
		{
			//ACCESS 4
#ifdef CACHEENABLED
			int possibleLength = *((int*) cacheRead((char*) &(possibleMatch->length), cacheLineId2, myCache, addrLocator2, bitmap2));
#else
			int possibleLength = possibleMatch->length;
#endif
			//ACCESS 5
			//char* possibleUrl = *((char**) cacheRead((char*) &(possibleMatch->url), cacheLineId2, myCache, addrLocator2, bitmap2));
			char* possibleUrl = possibleMatch->url;

			if(urlSize == possibleLength && isEqual(url, possibleUrl, urlSize, cacheLineId1, cacheLineId2, myCache, addrLocator1, addrLocator2, bitmap1, bitmap2))
			{
				//ACCESS 6
#ifdef CACHEENABLED
				int tempCounter = *((int*) cacheRead((char*) &(possibleMatch->counter), cacheLineId2, myCache, addrLocator2, bitmap2));
#else
				int tempCounter = possibleMatch->counter;
#endif
				tempCounter ++;
				//ACCESS 7, WRITE
#ifdef CACHEENABLED
				*((int*) cacheWrite((char*) &(possibleMatch->counter), 15, cacheLineId2, myCache, addrLocator2, bitmap2)) = tempCounter;
#else
				possibleMatch->counter = tempCounter;
#endif
				oldValue = 0;
			}
			else
			{
				//hIndex = (hIndex+ HASHWALKSTEP) % numBuckets;
				numConflict ++;
				hIndex = hashFunc(url, urlSize, NUMHASHROWS, numConflict, cacheLineId1, myCache, addrLocator1, bitmap1);
			}
		}
		else
		{
			while(oldValue == 1)
			{
				oldValue = atomicCAS((unsigned int*) &locks[hIndex], 0, 1);

				if(oldValue == 0)
				{
					hashEntry* possibleMatch = &hashTable[hIndex];

					//ACCESS 8
#ifdef CACHEENABLED
					int possibleValid = *((int*) cacheRead((char*) &(possibleMatch->valid), cacheLineId2, myCache, addrLocator2, bitmap2));
#else
					int possibleValid = possibleMatch->valid;
#endif
					if(possibleValid == 1)
					{
						//ACCESS 9
#ifdef CACHEENABLED
						int possibleLength = *((int*) cacheRead((char*) &(possibleMatch->length), cacheLineId2, myCache, addrLocator2, bitmap2));
#else
						int possibleLength = possibleMatch->length;
#endif

						//ACCESS 10
						//char* possibleUrl = *((char**) cacheRead((char*) &(possibleMatch->url), sharedCacheSegmentAssignments[10], myCache, addrLocator1, addrLocator2, bitmap1, bitmap2));
						char* possibleUrl = possibleMatch->url;
						if(urlSize == possibleLength && isEqual(url, possibleUrl, urlSize, cacheLineId1, cacheLineId2, myCache, addrLocator1, addrLocator2, bitmap1, bitmap2))
						{
							//ACCESS 11
#ifdef CACHEENABLED
							int tempCounter = *((int*) cacheRead((char*) &(possibleMatch->counter), cacheLineId2, myCache, addrLocator2, bitmap2));
#else
							int tempCounter = possibleMatch->counter;
#endif
							tempCounter ++;
							//ACCESS 12, WRITE
#ifdef CACHEENABLED
							*((int*) cacheWrite((char*) &(possibleMatch->counter), 15, cacheLineId2, myCache, addrLocator2, bitmap2)) = tempCounter;
#else
							possibleMatch->counter = tempCounter;
#endif

							atomicExch((unsigned int*) &locks[hIndex], 0);
						}
						else
						{
							atomicExch((unsigned int*) &locks[hIndex], 0);
							hIndex = (hIndex+ HASHWALKSTEP) % numBuckets;
							oldValue = 1;
						}
					}
					else
					{
						//ACCESS 13, WRITE
#ifdef CACHEENABLED
						*((int*) cacheWrite((char*) &(possibleMatch->counter), 15, cacheLineId2, myCache, addrLocator2, bitmap2)) = 1;
#else
						possibleMatch->counter = 1;
#endif
						//ACCESS 14, WRITE
#ifdef CACHEENABLED
						*((int*) cacheWrite((char*) &(possibleMatch->length), 15, cacheLineId2, myCache, addrLocator2, bitmap2)) = urlSize;
#else
						possibleMatch->length = urlSize;
#endif
						for(int j = 0; j < urlSize; j ++)
						{
							//ACCESS 15
#ifdef CACHEENABLED
							char c = *((char*) cacheRead((char*) &(url[j]), cacheLineId1, myCache, addrLocator1, bitmap1));
#else
							char c = url[j];
#endif

							//ACCESS 16, WRITE
#ifdef CACHEENABLED
							*((char*) cacheWrite((char*) &(possibleMatch->url[j]), 1, cacheLineId2, myCache, addrLocator2, bitmap2)) = c;
#else
							possibleMatch->url[j] = c;
#endif
						}
						//ACCESS 17, WRITE
#ifdef CACHEENABLED
						*((int*) cacheWrite((char*) &(possibleMatch->valid), 15, cacheLineId2, myCache, addrLocator2, bitmap2)) = 1;
#else
						possibleMatch->valid = 1;
#endif

						atomicExch((unsigned int*) &locks[hIndex], 0);
					}

				}
			}	
		}
	}
	//if(numConflict > 1000)
		//printf("Conflicts: %d\n", numConflict);

	return;
}

//unoptimized version
__global__ void urlCountKernel(char* data, largeInt fileSize, largeInt* startIndexes, hashEntry* hashTable, unsigned int* locks, int maxCacheLine)
{

	int index = TID;
	ptr_t start = startIndexes[index];
	ptr_t end = startIndexes[index + 1];
	
 	char *nKey = (char*)&data[start];
        char* p = nKey;

	char* url;
	int urlSize;

#if 1
	//========== Cache stuff ================//
	extern __shared__ char sbuf[];
	char* myCache = &sbuf[threadIdx.x * 16 * maxCacheLine];
	__shared__ int sharedCacheSegmentAssignments[NUMIDENTIFIERS];
	short bitmap1 = 0, bitmap2 = 0;
	int cacheLineId1 = -1;
	int cacheLineId2 = -1;
	int hits1 = 0;
	int hits2 = 0;
	largeInt addrLocator1 = 0, addrLocator2 = 0;
	int trainingAccesses;
	//=======================================//
#endif

#ifdef CACHEENABLED
	if(threadIdx.x < 32)
	{
		trainingAccesses = 0;
		while((start < end) && trainingAccesses < 200)
		{
			url = p;
			urlSize = 0;
			//ACCESS 0
			simulateCacheHitAndMiss((largeInt) p, 0, &addrLocator1, &hits1, &trainingAccesses);
			char c = *p;

			for(; c != '\t' && start < end; start++ , urlSize++)
			{
				p++;

				//ACCESS 1
				simulateCacheHitAndMiss((largeInt) p, 0, &addrLocator1, &hits1, &trainingAccesses);
				c = *p;
			}

			p++;
			start++;

			int hIndex = hashFuncSimulate(url, urlSize, NUMHASHROWS, 0, &addrLocator1, &hits1, &trainingAccesses);

			addToHashTableSimulate(url, urlSize, hashTable, NUMHASHROWS, hIndex, locks, &addrLocator1, &addrLocator2, &hits1, &hits2, &trainingAccesses);
			//if(index == 0)
			//printf("hIndex: %d\n", hIndex);

			//do something with the url

			//ACCESS 20
			simulateCacheHitAndMiss((largeInt) p, 0, &addrLocator1, &hits1, &trainingAccesses);
			c = *p;

			for (; c != '\n' && start < end; start++)
			{
				p++;
				//ACCESS 21
				simulateCacheHitAndMiss((largeInt) p, 0, &addrLocator1, &hits1, &trainingAccesses);
				c = *p;
			}

			p++;
			start++;
		}

		if(threadIdx.x == 0)
		{
			cacheAssignments(hits1, hits2, sharedCacheSegmentAssignments, maxCacheLine, trainingAccesses);
			//printf("hits1: %d, hits2: %d, cacheLineId1: %d, cacheLineId2: %d, totalAccess: %d\n", hits1, hits2, 
				//sharedCacheSegmentAssignments[0], sharedCacheSegmentAssignments[1], trainingAccesses);
		}

	}
#endif
	
	__syncthreads();
	cacheLineId1 = sharedCacheSegmentAssignments[0];
	cacheLineId2 = sharedCacheSegmentAssignments[1];

	while(start < end)
	{

		url = p;
		urlSize = 0;
		//ACCESS 0
#ifdef CACHEENABLED
		char c = *((char*) cacheRead((char*) p, cacheLineId1, myCache, &addrLocator1, &bitmap1));
#else
		char c = *p;
#endif

		for(; c != '\t' && start < end; start++ , urlSize++)
		{
			p++;

			//ACCESS 1
#ifdef CACHEENABLED
			c = *((char*) cacheRead((char*) p, cacheLineId1, myCache, &addrLocator1, &bitmap1));
#else
			c = *p;
#endif

		}

		p++;
		start++;

		int hIndex = hashFunc(url, urlSize, NUMHASHROWS, 0, cacheLineId1, myCache, &addrLocator1, &bitmap1);
		addToHashTable(url, urlSize, hashTable, NUMHASHROWS, hIndex, locks, cacheLineId1, cacheLineId2, myCache, &addrLocator1, &addrLocator2, &bitmap1, &bitmap2);
		//if(index == 0)
		//printf("hIndex: %d\n", hIndex);

		//do something with the url

		//ACCESS 20
#ifdef CACHEENABLED
		c = *((char*) cacheRead((char*) p, cacheLineId1, myCache, &addrLocator1, &bitmap1));
#else
		c = *p;
#endif

		for (; c != '\n' && start < end; start++)
		{
			p++;
			//ACCESS 21
#ifdef CACHEENABLED
			c = *((char*) cacheRead((char*) p, cacheLineId1, myCache, &addrLocator1, &bitmap1));
#else
			c = *p;
#endif
		}

		p++;
		start++;

	}
}


int countToNextLine(char* start)
{
	int counter = 0;
	while(start[counter] != '\n')
		counter ++;

	return counter + 1;
}


int main(int argc, char** argv)
{
	cudaError_t errR;
	cudaThreadExit();

	int fd;
	char *fdata;
	struct stat finfo;
	char *fname;

	if (argc < 4)
	{
		printf("USAGE: %s <inputfilename> <numBlocks> <numThreadsPerBlock>\n", argv[0]);
		exit(1);
	}

	fname = argv[1];
	fd = open(fname, O_RDONLY);
	fstat(fd, &finfo);
	size_t fileSize = (size_t) finfo.st_size;
	printf("Allocating %lluMB for the input file.\n", ((long long unsigned int)fileSize) / (1 << 20));
	fdata = (char *) malloc(finfo.st_size);


	largeInt maxReadSize = MAXREAD;
        largeInt readed = 0;
        largeInt toRead = 0;

        if(fileSize > maxReadSize)
        {
                largeInt offset = 0;
                while(offset < fileSize)
                {
                        toRead = (maxReadSize < (fileSize - offset))? maxReadSize : (fileSize - offset);
                        readed += pread(fd, fdata + offset, toRead, offset);
                        printf("read: %lliMB\n", toRead / (1 << 20));
                        offset += toRead;
                }
        }
        else
	{
                readed = read (fd, fdata, fileSize);
	}

        if(readed != fileSize)
	{
                printf("Not all of the input file is read. Read: %lluMB, total: %luMB\n", readed, fileSize);
		return 1;
	}


	int numBlocks = atoi(argv[2]);
	int numBlockThreads = atoi(argv[3]);

	dim3 grid(numBlocks, 1, 1);
	dim3 block(numBlockThreads, 1, 1);
	int numThreads = grid.x * grid.y * block.x * block.y;
	printf("numThreads: %d\n", numThreads);

	//================= startIndexes =====================//
	size_t startIndexSize = (numThreads + 1) * sizeof(largeInt);

	largeInt* tempstartIndex = (largeInt*) malloc(startIndexSize + MEMORY_ALIGNMENT);
	largeInt* startIndexes = (largeInt*) ALIGN_UP(tempstartIndex, MEMORY_ALIGNMENT);
	memset((void*) startIndexes, 0, startIndexSize);
	cudaHostRegister((void*) startIndexes, startIndexSize, CU_MEMHOSTALLOC_DEVICEMAP);

	largeInt* tempstartIndex2 = (largeInt*) malloc(startIndexSize + MEMORY_ALIGNMENT);
	largeInt* startIndexes2 = (largeInt*) ALIGN_UP(tempstartIndex2, MEMORY_ALIGNMENT);
	memset((void*) startIndexes2, 0, startIndexSize);
	cudaHostRegister((void*) startIndexes2, startIndexSize, CU_MEMHOSTALLOC_DEVICEMAP);
	//====================================================//


	int threadChunkSize;
	largeInt* dstartIndexes;
	largeInt* dstartIndexes2;
	cudaMalloc((void**) &dstartIndexes, (numThreads + 1) * sizeof(largeInt));
	cudaMalloc((void**) &dstartIndexes2, (numThreads + 1) * sizeof(largeInt));
	//======================================================//
	
	//============== Hash table and locks===================//
	size_t hashTableSize = NUMHASHROWS * sizeof(hashEntry);
	printf("Allocating %luMB for hash table.\n", hashTableSize / (1 << 20));
	
	hashEntry* dhashTable;
	cudaMalloc((void**) &dhashTable, hashTableSize);
	cudaMemset(dhashTable, 0, hashTableSize);

	unsigned int* dcolumnLocks;
	cudaMalloc((void**) &dcolumnLocks, NUMHASHROWS * sizeof(int));
	cudaMemset(dcolumnLocks, 0, NUMHASHROWS * sizeof(int));


	char* dtrainSpace;
	cudaMalloc((void**) &dtrainSpace, numThreads * TRAININGSPACESIZE * NUMIDENTIFIERS * sizeof(char));
	cudaMemset(dtrainSpace, 0, numThreads * TRAININGSPACESIZE * NUMIDENTIFIERS * sizeof(char));
	//======================================================//

	errR = cudaGetLastError();
	printf("Error after allocating memory spaces is: %s\n", cudaGetErrorString(errR));
	if(errR != cudaSuccess)
		exit(1);
	
	struct timeval partial_start, partial_end;
	time_t sec;
	time_t ms;
	time_t diff;

	char* ddata;
	cudaMalloc((void**) &ddata, fileSize);


	threadChunkSize = fileSize / numThreads;
	assert(threadChunkSize > 0);

	for(int i = 0; i < numThreads; i ++)
		startIndexes[i] = i * threadChunkSize;

	startIndexes[numThreads] = fileSize;

	for(int i = 1; i < numThreads; i ++)
	{
		int disToNextWhiteSpace = countToNextLine(&fdata[startIndexes[i]]);
		startIndexes[i] += disToNextWhiteSpace;
	}

	cudaMemcpy(dstartIndexes, startIndexes, (numThreads + 1) * sizeof(largeInt), cudaMemcpyHostToDevice);
	cudaMemcpy(ddata, fdata, fileSize, cudaMemcpyHostToDevice);

	printf("Getting to run the kernel\n");

	int shMemPerThread = 0;
	int maxCacheLines = 0;
	int totalShMemSize = (48 * 1024) - 512;
	if(grid.x <= 8)
	{
		shMemPerThread = totalShMemSize / block.x;
		maxCacheLines = shMemPerThread / CACHESEGMENTSIZE;
	}
	else
	{
		int runningBlocksPerSM = (grid.x / 8);
		if(grid.x % 8 != 0)
			runningBlocksPerSM ++;
		if(runningBlocksPerSM > 16)
			runningBlocksPerSM = 16;

		if((runningBlocksPerSM * block.x) > 2048)
		{
			runningBlocksPerSM = (2048 / block.x) * block.x;
		}

		shMemPerThread = totalShMemSize / (runningBlocksPerSM * block.x);
		maxCacheLines = shMemPerThread / CACHESEGMENTSIZE;
	}
	
	
	int sharedSize = 16 * block.x * maxCacheLines;
	printf("maxCacheLines: %d\n", maxCacheLines);
	printf("sharedSize: %d\n", sharedSize);



	gettimeofday(&partial_start, NULL);

	urlCountKernel<<<grid, block, sharedSize>>>(ddata, fileSize, dstartIndexes, dhashTable, dcolumnLocks, maxCacheLines);
	cudaThreadSynchronize();

	gettimeofday(&partial_end, NULL);
	sec = partial_end.tv_sec - partial_start.tv_sec;
	ms = partial_end.tv_usec - partial_start.tv_usec;
	diff = sec * 1000000 + ms;




	printf("\n%10s:\t\t%0.0f\n", "Time elapsed", (double)((double)diff/1000.0));

	//==========================================================================//




	cudaThreadSynchronize();


	errR = cudaGetLastError();
	printf("Error after executing kernel: %s\n", cudaGetErrorString(errR));


	hashEntry* hhashTable = (hashEntry*) malloc(hashTableSize);
	cudaMemcpy(hhashTable, dhashTable, hashTableSize, cudaMemcpyDeviceToHost);


#ifdef DISPLAYRESULTS

	unsigned showCounter = 0;

	for(int i = 0; i < hashTableSize / sizeof(hashEntry); i ++)
	{
		if(hhashTable[i].valid == 1)
		{
			if(showCounter < 20)
			{
				printf("%20s", hhashTable[i].url);
				printf("\t count: %d\n", hhashTable[i].counter);
			}
			showCounter ++;
		}
	}
	printf("Number of urls: %u\n", showCounter);

#endif


	return 0;
}
