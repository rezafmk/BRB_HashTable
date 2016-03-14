#include "hashGlobal.h"

void hashtableInit(unsigned numBuckets, multipassConfig_t* mbk, unsigned groupSize)
{
	mbk->numBuckets = numBuckets;
	mbk->groupSize = groupSize;
	int numGroups = (numBuckets + (groupSize - 1)) / groupSize;
	cudaMalloc((void**) &(mbk->groups), numGroups * sizeof(bucketGroup_t));
	cudaMalloc((void**) &(mbk->buckets), numBuckets * sizeof(hashBucket_t*));
	cudaMalloc((void**) &(mbk->locks), numBuckets * sizeof(unsigned));
	cudaMalloc((void**) &(mbk->isNextDeads), numBuckets * sizeof(short));
	
	cudaMemset(mbk->groups, 0, numGroups * sizeof(bucketGroup_t));
	cudaMemset(mbk->buckets, 0, numBuckets * sizeof(hashBucket_t*));
	cudaMemset(mbk->locks, 0, numBuckets * sizeof(unsigned));
	cudaMemset(mbk->isNextDeads, 0, numBuckets * sizeof(short));
}

__device__ unsigned int hashFunc(char* str, int len, unsigned numBuckets)
{
        unsigned hash = 2166136261;
        unsigned FNVMultiple = 16777619;

        for(int i = 0; i < len; i ++)
        {
                char c = str[i];

                hash += (int) c;
                hash = hash * FNVMultiple;  /* multiply by the magic number */
                hash += len;
                hash -= (int) c;
        }

        return hash % numBuckets;
}


__device__ void resolveSameKeyAddition(void const* key, void* value, void* oldValue)
{
	*((int*) oldValue) += 1;
}

__device__ hashBucket_t* containsKey(hashBucket_t* bucket, void* key, int keySize, multipassConfig_t* mbk)
{
	while(bucket != NULL)
	{
		char* oldKey = (char*) ((largeInt) bucket + sizeof(hashBucket_t));
		bool success = true;
		//OPTIMIZE: do the comparisons 8-byte by 8-byte
#if 1
		int i = 0;
		for(; i < keySize/ALIGNMET && success; i ++)
		{
			if(((largeInt*) oldKey)[i] != ((largeInt*) key)[i])
				success = false;
		}
		i *= ALIGNMET;
		for(; i < keySize && success; i ++)
		{
			if(oldKey[i] != ((char*) key)[i])
				success = false;
		}
#endif
#if 0

		for(int i = 0; i < keySize; i ++)
		{
			if(oldKey[i] != ((char*) key)[i])
			{
				success = false;
				break;
			}
		}
#endif
		if(success)
			break;

		if(bucket->isNextDead == 0 && bucket->next != NULL)
			bucket = (hashBucket_t*) ((largeInt) bucket->next - mbk->hashTableOffset + (largeInt) mbk->dbuffer);
		else
			bucket = NULL;
	}

	return bucket;
}

__device__ bool atomicAttemptIncRefCount(int* refCount)
{
	int oldRefCount = *refCount;
	int assume;
	bool success;
	do
	{
		success = false;
		assume = oldRefCount;
		if(oldRefCount >= 0)
		{
			oldRefCount = (int) atomicCAS((unsigned*) refCount, (unsigned) oldRefCount, oldRefCount + 1);
			success = true;
		}
	} while(oldRefCount != assume);

	return success;
}

__device__ int atomicDecRefCount(int* refCount)
{
	int oldRefCount = *refCount;
	int assume;
	do
	{
		assume = oldRefCount;
		if(oldRefCount >= 0) // During normal times
		{
			oldRefCount = (int) atomicCAS((unsigned*) refCount, (unsigned) oldRefCount, (unsigned) (oldRefCount - 1));
		}
		else // During failure
		{
			oldRefCount = (int) atomicCAS((unsigned*) refCount, (unsigned) oldRefCount, (unsigned) (oldRefCount + 1));
		}

	} while(oldRefCount != assume);

	return oldRefCount;
}

__device__ bool atomicNegateRefCount(int* refCount)
{
	int oldRefCount = *refCount;
	int assume;
	do
	{
		assume = oldRefCount;
		if(oldRefCount >= 0)
			oldRefCount = (int) atomicCAS((unsigned*) refCount, (unsigned) oldRefCount, ((oldRefCount * (-1)) - 1));

	} while(oldRefCount != assume);

	return (oldRefCount >= 0);
	
}

__device__ bool addToHashtable(void* key, int keySize, void* value, int valueSize, multipassConfig_t* mbk)
{
	bool success = true;
	unsigned hashValue = hashFunc((char*) key, keySize, mbk->numBuckets);

	unsigned groupNo = hashValue / mbk->groupSize;
	//unsigned groupNo = hashValue / GROUP_SIZE;

	bucketGroup_t* group = &(mbk->groups[groupNo]);
	
	hashBucket_t* existingBucket;

	int keySizeAligned = (keySize % ALIGNMET == 0)? keySize : keySize + (ALIGNMET - (keySize % ALIGNMET));
	int valueSizeAligned = (valueSize % ALIGNMET == 0)? valueSize : valueSize + (ALIGNMET - (valueSize % ALIGNMET));

	unsigned oldLock = 1;

	do
	{
		oldLock = atomicExch((unsigned*) &(mbk->locks[hashValue]), 1);

		if(oldLock == 0)
		{
			hashBucket_t* bucket = NULL;
			if(mbk->buckets[hashValue] != NULL)
				bucket = (hashBucket_t*) ((largeInt) mbk->buckets[hashValue] - mbk->hashTableOffset + (largeInt) mbk->dbuffer);
			//First see if the key already exists in one of the entries of this bucket
			//The returned bucket is the 'entry' in which the key exists
			if(mbk->isNextDeads[hashValue] != 1 && (existingBucket = containsKey(bucket, key, keySize, mbk)) != NULL)
			{
				void* oldValue = (void*) ((largeInt) existingBucket + sizeof(hashBucket_t) + keySizeAligned);
				resolveSameKeyAddition(key, value, oldValue);
			}
			else
			{
				hashBucket_t* newBucket = (hashBucket_t*) multipassMalloc(sizeof(hashBucket_t) + keySizeAligned + valueSizeAligned, group, mbk, groupNo);
				if(newBucket != NULL)
				{
					//TODO reduce the base offset if not null
					//newBucket->next = (bucket == NULL)? NULL : (hashBucket_t*) ((largeInt) bucket - (largeInt) mbk->dbuffer);
					newBucket->next = NULL;
					if(bucket != NULL)
						newBucket->next = (hashBucket_t*) ((largeInt) bucket - (largeInt) mbk->dbuffer + mbk->hashTableOffset);
					if(mbk->isNextDeads[hashValue] == 1)
						newBucket->isNextDead = 1;
					newBucket->keySize = (short) keySize;
					newBucket->valueSize = (short) valueSize;
						
					mbk->buckets[hashValue] = (hashBucket_t*) ((largeInt) newBucket - (largeInt) mbk->dbuffer + mbk->hashTableOffset);
					mbk->isNextDeads[hashValue] = 0;

					//TODO: this assumes that input key is aligned by ALIGNMENT, which is not a safe assumption
					for(int i = 0; i < (keySizeAligned / ALIGNMET); i ++)
						*((largeInt*) ((largeInt) newBucket + sizeof(hashBucket_t) + i * ALIGNMET)) = *((largeInt*) ((largeInt) key + i * ALIGNMET));
					for(int i = 0; i < (valueSizeAligned / ALIGNMET); i ++)
						*((largeInt*) ((largeInt) newBucket + sizeof(hashBucket_t) + keySizeAligned + i * ALIGNMET)) = *((largeInt*) ((largeInt) value + i * ALIGNMET));
				}
				else
				{
					success = false;
				}
			}

			atomicExch((unsigned*) &(mbk->locks[hashValue]), 0);
		}
	} while(oldLock == 1);

	return success;
}

__global__ void setGroupsPointersDead(multipassConfig_t* mbk, unsigned numBuckets)
{
	int index = TID;
	if(index < numBuckets)
	{
		mbk->isNextDeads[index] = 1;
	}
	
}



multipassConfig_t* initMultipassBookkeeping(int* hostCompleteFlag, 
						int* gpuFlags, 
						int flagSize,
						int numThreads,
						int epochNum,
						int numRecords,
						int pagePerGroup)
{
	
	multipassConfig_t* mbk = (multipassConfig_t*) malloc(sizeof(multipassConfig_t));
	mbk->hostCompleteFlag = hostCompleteFlag;
	mbk->gpuFlags = gpuFlags;
	mbk->flagSize = flagSize;
	mbk->numThreads = numThreads;
	mbk->epochNum = epochNum;
	mbk->numRecords = numRecords;


	mbk->availableGPUMemory = (500 * (1 << 20));
	mbk->hhashTableBufferSize = MAX_NO_PASSES * mbk->availableGPUMemory;
	mbk->hhashTableBaseAddr = malloc(mbk->hhashTableBufferSize);
	memset(mbk->hhashTableBaseAddr, 0, mbk->hhashTableBufferSize);

	//This is how we decide the number of groups: based on the number of available pages, we make sure 
	//group size is calculated in a way that a given number of `pagePerGroup` pages are assigned to each group
	int availableNumPages = mbk->availableGPUMemory / PAGE_SIZE;
	mbk->groupSize = (pagePerGroup * NUM_BUCKETS) / availableNumPages;
	mbk->numGroups = (NUM_BUCKETS + (mbk->groupSize - 1)) / mbk->groupSize;
	//mbk->numGroups = (NUM_BUCKETS + (GROUP_SIZE - 1)) / GROUP_SIZE;
	printf("############# groupSize: %d, number of groups: %d\n", mbk->groupSize, mbk->numGroups);


	cudaMalloc((void**) &(mbk->dfailedFlag), sizeof(bool));
	cudaMemset(mbk->dfailedFlag, 0, sizeof(bool));


	cudaMalloc((void**) &(mbk->depochSuccessStatus), epochNum * sizeof(char));
	cudaMemset(mbk->depochSuccessStatus, 0, epochNum * sizeof(char));
	mbk->epochSuccessStatus = (char*) malloc(epochNum * sizeof(char));


	// Calling initPaging
	initPaging(mbk->availableGPUMemory, mbk);
	mbk->hashTableOffset = (largeInt) mbk->hhashTableBaseAddr;

	hashtableInit(NUM_BUCKETS, mbk, mbk->groupSize);
	
	
	printf("@INFO: transferring config structs to GPU memory\n");

	cudaMalloc((void**) &(mbk->dstates), mbk->numRecords * sizeof(char));
	cudaMemset(mbk->dstates, 0, mbk->numRecords * sizeof(char));


	mbk->myNumbers = (int*) malloc(2 * numThreads * sizeof(int));
	cudaMalloc((void**) &(mbk->dmyNumbers), 2 * numThreads * sizeof(int));
	cudaMemset((mbk->dmyNumbers), 0, 2 * numThreads * sizeof(int));

	size_t total, free;
	cudaMemGetInfo(&free, &total);
	printf("total memory: %luMB, free: %luMB\n", total / (1 << 20), free / (1 << 20));


	printf("@INFO: number of page: %d\n", (int)(mbk->availableGPUMemory / PAGE_SIZE));
	printf("@INFO: number of hash groups: %d\n", mbk->numGroups);

	return mbk;
}

bool checkAndResetPass(multipassConfig_t* mbk, multipassConfig_t* dmbk)
{
	cudaMemcpy(mbk, dmbk, sizeof(multipassConfig_t), cudaMemcpyDeviceToHost);
	bool failedFlag = false;
	int* hostCompleteFlag = mbk->hostCompleteFlag;
	int* gpuFlags = mbk->gpuFlags;
	bool* dfailedFlag = mbk->dfailedFlag;
	int* dmyNumbers = mbk->dmyNumbers;
	int* myNumbers = mbk->myNumbers;
	int flagSize = mbk->flagSize;
	void* hhashTableBaseAddr = mbk->hhashTableBaseAddr;
	largeInt hhashTableBufferSize = mbk->hhashTableBufferSize;
	int numGroups = mbk->numGroups;
	int numThreads = mbk->numThreads;
	char* epochSuccessStatus = mbk->epochSuccessStatus;
	char* depochSuccessStatus = mbk->depochSuccessStatus;
	int epochNum = mbk->epochNum;
	int groupSize = mbk->groupSize;

	cudaMemcpy(epochSuccessStatus, depochSuccessStatus, epochNum * sizeof(char), cudaMemcpyDeviceToHost);
	for(int i = 0; i < epochNum; i ++)
	{
		if(epochSuccessStatus[i] == UNTESTED)
			epochSuccessStatus[i] = SUCCEED;
		else if(epochSuccessStatus[i] == FAILED)
			epochSuccessStatus[i] = UNTESTED;
	}
	cudaMemcpy(depochSuccessStatus, epochSuccessStatus, epochNum * sizeof(char), cudaMemcpyHostToDevice);


	memset((void*) hostCompleteFlag, 0, flagSize);
	cudaMemset(gpuFlags, 0, flagSize / 2);

	cudaMemcpy(&failedFlag, dfailedFlag, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemset(dfailedFlag, 0, sizeof(bool));


	cudaMemcpy((void*) mbk->hashTableOffset, mbk->dbuffer, mbk->totalNumPages * PAGE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemset(mbk->dbuffer, 0, mbk->totalNumPages * PAGE_SIZE);

	printf("totalnoPage * pagesize: %llu, hhashtbufferSize: %llu\n", (largeInt) mbk->totalNumPages * PAGE_SIZE, (largeInt) hhashTableBufferSize);
	mbk->hashTableOffset += mbk->totalNumPages * PAGE_SIZE;
	if((mbk->hashTableOffset + mbk->totalNumPages * PAGE_SIZE) > ((largeInt) hhashTableBaseAddr + hhashTableBufferSize) && failedFlag)
	{
		printf("Need more space on CPU memory for the hash table. Aborting...\n");
		exit(1);
	}
	cudaMemcpy(mbk->pages, mbk->hpages, mbk->totalNumPages * sizeof(page_t), cudaMemcpyHostToDevice);
	mbk->initialPageAssignedCounter = 0;


	printf("Before calling setGroupPointer, number of grids: %d\n", ((NUM_BUCKETS) + 1023) / 1024);
	setGroupsPointersDead<<<(((NUM_BUCKETS) + 1023) / 1024), 1024>>>(dmbk, NUM_BUCKETS);
	//setGroupsPointersDead<<<(((NUM_BUCKETS) + 256) / 255), 256>>>(mbk->groups, NUM_BUCKETS, GROUP_SIZE);
	cudaThreadSynchronize();

	cudaError_t errR = cudaGetLastError();
	printf("#######Error after setGroupPointer is: %s\n", cudaGetErrorString(errR));

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

	cudaMemcpy(dmbk, mbk, sizeof(multipassConfig_t), cudaMemcpyHostToDevice);

	return failedFlag;
}

void* getKey(hashBucket_t* bucket)
{
	return (void*) ((largeInt) bucket + sizeof(hashBucket_t));
}

void* getValue(hashBucket_t* bucket)
{
	int keySizeAligned = (bucket->keySize % ALIGNMET == 0)? bucket->keySize : bucket->keySize + (ALIGNMET - (bucket->keySize % ALIGNMET));
	return (void*) ((largeInt) bucket + sizeof(hashBucket_t) + keySizeAligned);
}


