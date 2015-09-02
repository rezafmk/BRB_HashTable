#include "hashGlobal.h"
#define NUM_BUCKETS 10000000

void hashtableInit(int numBuckets, hashtableConfig_t* hconfig)
{
	hconfig->numBuckets = numBuckets;
	int numGroups = (numBuckets + (GROUP_SIZE - 1)) / GROUP_SIZE;
	cudaMalloc((void**) &(hconfig->groups), numGroups * sizeof(bucketGroup_t));
	cudaMemset(hconfig->groups, 0, numGroups * sizeof(bucketGroup_t));
	//hconfig->groups = (bucketGroup_t*) malloc(numGroups * sizeof(bucketGroup_t));
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

__device__ hashBucket_t* containsKey(hashBucket_t* bucket, void* key, int keySize, pagingConfig_t* pconfig)
{
	while(bucket != NULL)
	{
		char* oldKey = (char*) ((largeInt) bucket + sizeof(hashBucket_t));
		bool success = true;
		//OPTIMIZE: do the comparisons 8-byte by 8-byte
		for(int i = 0; i < keySize; i ++)
		{
			if(oldKey[i] != ((char*) key)[i])
			{
				success = false;
				break;
			}
		}
		if(success)
			break;

		if(bucket->isNextDead == 0 && bucket->next != NULL)
			bucket = (hashBucket_t*) ((largeInt) bucket->next - pconfig->hashTableOffset + (largeInt) pconfig->dbuffer);
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

__device__ bool addToHashtable(void* key, int keySize, void* value, int valueSize, hashtableConfig_t* hconfig, pagingConfig_t* pconfig)
{
	bool success = true;
	unsigned hashValue = hashFunc((char*) key, keySize, hconfig->numBuckets);

	unsigned groupNo = hashValue / GROUP_SIZE;
	unsigned offsetWithinGroup = hashValue % GROUP_SIZE;

	bucketGroup_t* group = &(hconfig->groups[groupNo]);
	
	hashBucket_t* existingBucket;

	int keySizeAligned = (keySize % ALIGNMET == 0)? keySize : keySize + (ALIGNMET - (keySize % ALIGNMET));
	int valueSizeAligned = (valueSize % ALIGNMET == 0)? valueSize : valueSize + (ALIGNMET - (valueSize % ALIGNMET));

	unsigned oldLock = 1;

	do
	{
		oldLock = atomicExch((unsigned*) &(group->locks[offsetWithinGroup]), 1);

		if(oldLock == 0)
		{
			hashBucket_t* bucket = NULL;
			if(group->buckets[offsetWithinGroup] != NULL)
				bucket = (hashBucket_t*) ((largeInt) group->buckets[offsetWithinGroup] - pconfig->hashTableOffset + (largeInt) pconfig->dbuffer);
			//First see if the key already exists in one of the entries of this bucket
			//The returned bucket is the 'entry' in which the key exists
			if(group->isNextDead[offsetWithinGroup] != 1 && (existingBucket = containsKey(bucket, key, keySize, pconfig)) != NULL)
			{
				void* oldValue = (void*) ((largeInt) existingBucket + sizeof(hashBucket_t) + keySizeAligned);
				resolveSameKeyAddition(key, value, oldValue);
			}
			else
			{
				hashBucket_t* newBucket = (hashBucket_t*) multipassMalloc(sizeof(hashBucket_t) + keySizeAligned + valueSizeAligned, group, pconfig, groupNo);
				if(newBucket != NULL)
				{
					//TODO reduce the base offset if not null
					//newBucket->next = (bucket == NULL)? NULL : (hashBucket_t*) ((largeInt) bucket - (largeInt) pconfig->dbuffer);
					//group->failed = 1;
					newBucket->next = NULL;
					if(bucket != NULL)
						newBucket->next = (hashBucket_t*) ((largeInt) bucket - (largeInt) pconfig->dbuffer + pconfig->hashTableOffset);
					if(group->isNextDead[offsetWithinGroup] == 1)
						newBucket->isNextDead = 1;
					newBucket->keySize = (short) keySize;
					newBucket->valueSize = (short) valueSize;
						
					group->buckets[offsetWithinGroup] = (hashBucket_t*) ((largeInt) newBucket - (largeInt) pconfig->dbuffer + pconfig->hashTableOffset);
					group->isNextDead[offsetWithinGroup] = 0;

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

			atomicExch((unsigned*) &(group->locks[offsetWithinGroup]), 0);
		}
	} while(oldLock == 1);

	return success;
}

__global__ void setGroupsPointersDead(bucketGroup_t* groups, int numGroups)
{
	int index = TID;
	if(index < (numGroups * GROUP_SIZE))
	{
		int i = index / GROUP_SIZE;
		int j = index % GROUP_SIZE;
		groups[i].isNextDead[j] = 1;
	}
	
}



multipassConfig_t* initMultipassBookkeeping(int* hostCompleteFlag, 
						int* gpuFlags, 
						int flagSize,
						int groupSize,
						int numThreads,
						int epochNum,
						int numRecords)
{
	
	multipassConfig_t* mbk = (multipassConfig_t*) malloc(sizeof(multipassConfig_t));
	mbk->hostCompleteFlag = hostCompleteFlag;
	mbk->gpuFlags = gpuFlags;
	mbk->flagSize = flagSize;
	mbk->groupSize = groupSize;
	mbk->numThreads = numThreads;
	mbk->epochNum = epochNum;
	mbk->numRecords = numRecords;
	mbk->numGroups = (NUM_BUCKETS + (GROUP_SIZE - 1)) / GROUP_SIZE;

	mbk->myNumbers = (int*) malloc(2 * numThreads * sizeof(int));
	cudaMalloc((void**) &(mbk->dmyNumbers), 2 * numThreads * sizeof(int));
	cudaMemset((mbk->dmyNumbers), 0, 2 * numThreads * sizeof(int));

	mbk->availableGPUMemory = (1 << 30);
	mbk->hhashTableBufferSize = 3 * mbk->availableGPUMemory;
	mbk->hhashTableBaseAddr = malloc(mbk->hhashTableBufferSize);
	memset(mbk->hhashTableBaseAddr, 0, mbk->hhashTableBufferSize);

	cudaMalloc((void**) &(mbk->dfailedFlag), sizeof(bool));
	cudaMemset(mbk->dfailedFlag, 0, sizeof(bool));


	cudaMalloc((void**) &(mbk->depochSuccessStatus), epochNum * sizeof(char));
	cudaMemset(mbk->depochSuccessStatus, 0, epochNum * sizeof(char));
	mbk->epochSuccessStatus = (char*) malloc(epochNum * sizeof(char));


	size_t availableGPUMemory = (1 << 30);
	mbk->pconfig = (pagingConfig_t*) malloc(sizeof(pagingConfig_t));
	memset(mbk->pconfig, 0, sizeof(pagingConfig_t));
	// Calling initPaging
	initPaging(availableGPUMemory, mbk->pconfig);
	mbk->pconfig->hashTableOffset = (largeInt) mbk->hhashTableBaseAddr;

	mbk->hconfig = (hashtableConfig_t*) malloc(sizeof(hashtableConfig_t));
	hashtableInit(NUM_BUCKETS, mbk->hconfig);
	
	
	printf("@INFO: transferring config structs to GPU memory\n");
	cudaMalloc((void**) &(mbk->dpconfig), sizeof(pagingConfig_t));
	cudaMemcpy(mbk->dpconfig, mbk->pconfig, sizeof(pagingConfig_t), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(mbk->dhconfig), sizeof(hashtableConfig_t));
	cudaMemcpy(mbk->dhconfig, mbk->hconfig, sizeof(hashtableConfig_t), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(mbk->dstates), mbk->numRecords * sizeof(char));
	cudaMemset(mbk->dstates, 0, mbk->numRecords * sizeof(char));

	size_t total, free;
	cudaMemGetInfo(&free, &total);
	printf("total memory: %luMB, free: %luMB\n", total / (1 << 20), free / (1 << 20));


	printf("@INFO: number of page: %d\n", (int)(mbk->availableGPUMemory / PAGE_SIZE));
	printf("@INFO: number of hash groups: %d\n", mbk->numGroups);

	return mbk;
}

bool checkAndResetPass(multipassConfig_t* mbk)
{
	bool failedFlag = false;
	int* hostCompleteFlag = mbk->hostCompleteFlag;
	int* gpuFlags = mbk->gpuFlags;
	bool* dfailedFlag = mbk->dfailedFlag;
	pagingConfig_t* pconfig = mbk->pconfig;
	pagingConfig_t* dpconfig = mbk->dpconfig;
	hashtableConfig_t* hconfig = mbk->hconfig;
	int* dmyNumbers = mbk->dmyNumbers;
	int* myNumbers = mbk->myNumbers;
	int flagSize = mbk->flagSize;
	void* hhashTableBaseAddr = mbk->hhashTableBaseAddr;
	largeInt hhashTableBufferSize = mbk->hhashTableBufferSize;
	int numGroups = mbk->numGroups;
	int groupSize = mbk->groupSize;
	int numThreads = mbk->numThreads;
	char* epochSuccessStatus = mbk->epochSuccessStatus;
	char* depochSuccessStatus = mbk->depochSuccessStatus;
	int epochNum = mbk->epochNum;

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

	cudaMemcpy(pconfig, dpconfig, sizeof(pagingConfig_t), cudaMemcpyDeviceToHost);

	cudaMemcpy((void*) pconfig->hashTableOffset, pconfig->dbuffer, pconfig->totalNumPages * PAGE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemset(pconfig->dbuffer, 0, pconfig->totalNumPages * PAGE_SIZE);

	printf("totalnoPage * pagesize: %llu, hhashtbufferSize: %llu\n", (largeInt) pconfig->totalNumPages * PAGE_SIZE, (largeInt) hhashTableBufferSize);
	pconfig->hashTableOffset += pconfig->totalNumPages * PAGE_SIZE;
	if((pconfig->hashTableOffset + pconfig->totalNumPages * PAGE_SIZE) > ((largeInt) hhashTableBaseAddr + hhashTableBufferSize) && failedFlag)
	{
		printf("Need more space on CPU memory for the hash table. Aborting...\n");
		exit(1);
	}
	cudaMemcpy(pconfig->pages, pconfig->hpages, pconfig->totalNumPages * sizeof(page_t), cudaMemcpyHostToDevice);
	pconfig->initialPageAssignedCounter = 0;

	cudaMemcpy(dpconfig, pconfig, sizeof(pagingConfig_t), cudaMemcpyHostToDevice);

	setGroupsPointersDead<<<(((numGroups * groupSize) + 256) / 255), 256>>>(hconfig->groups, numGroups);
	cudaThreadSynchronize();

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


