#include "hashGlobal.h"
#define STATISTICS 1

void hashtableInit(unsigned numBuckets, multipassConfig_t* mbk, unsigned groupSize)
{
	mbk->numBuckets = numBuckets;
	mbk->groupSize = groupSize;
	int numGroups = (numBuckets + (groupSize - 1)) / groupSize;
	cudaMalloc((void**) &(mbk->groups), numGroups * sizeof(bucketGroup_t));
	cudaMalloc((void**) &(mbk->buckets), numBuckets * sizeof(hashBucket_t*));
	cudaMalloc((void**) &(mbk->dbuckets), numBuckets * sizeof(hashBucket_t*));
	cudaMalloc((void**) &(mbk->locks), numBuckets * sizeof(unsigned));
	cudaMalloc((void**) &(mbk->isNextDeads), numBuckets * sizeof(short));
	
	cudaMemset(mbk->groups, 0, numGroups * sizeof(bucketGroup_t));
	cudaMemset(mbk->dbuckets, 0, numBuckets * sizeof(hashBucket_t*));
	cudaMemset(mbk->buckets, 0, numBuckets * sizeof(hashBucket_t*));
	cudaMemset(mbk->locks, 0, numBuckets * sizeof(unsigned));
	cudaMemset(mbk->isNextDeads, 0, numBuckets * sizeof(short));
}


__device__ unsigned int hashFunc(char* str, int len, unsigned numBuckets)
{
        largeInt number = ((input_t*) str)->data1;
        return number % numBuckets;
}

__device__ bool addNewValueAtomically(void const* key, void* value, int valueSize, void* oldValue, bucketGroup_t* group, multipassConfig_t* mbk)
{
	//TODO: here make it compatible with the new structure of vlaue at the end fo the bucket...
	valueHolder_t* newValue = (valueHolder_t*) multipassMallocValue(sizeof(valueHolder_t) + valueSize, group, mbk);
	if(newValue != NULL)
	{
		newValue->valueSize = (largeInt) valueSize;
		setValue(newValue, value, valueSize);

		largeInt atomicReturnedValue, previousValue;
		largeInt toBeInsertedValue = ((largeInt) newValue - (largeInt) mbk->dbuffer + group->valueParentPage->hashTableOffset); 

		newValue->next = ((valueHolder_t*) oldValue)->next;

		// Appending the new value at the beginning of the linked list atomically
		do
		{

			previousValue = (largeInt) ((valueHolder_t*) oldValue)->next;
			newValue->next = ((valueHolder_t*) oldValue)->next;

			atomicReturnedValue = atomicCAS((unsigned long long int*) &(((valueHolder_t*) oldValue)->next), previousValue, toBeInsertedValue);

		} while(previousValue != atomicReturnedValue);

		return true;
	}
	return false;
}

__device__ hashBucket_t* containsKeyUntilEntry(hashBucket_t* bucket, void* key, int keySize, hashBucket_t* lastCheckedBucket, multipassConfig_t* mbk)
{
	// This should never be null... that's why I'm not checking for null
	while(bucket != lastCheckedBucket)
	{
		char* oldKey = (char*) ((largeInt) bucket + sizeof(hashBucket_t));
		bool success = true;

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

		if(success)
		{
			return bucket;
		}

		bucket = bucket->dnext;
	}
	return NULL;
}


__device__ hashBucket_t* containsKey(hashBucket_t* bucket, void* key, int keySize, multipassConfig_t* mbk)
{
	while(bucket != NULL)
	{
		char* oldKey = (char*) ((largeInt) bucket + sizeof(hashBucket_t));
		bool success = true;

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


		if(success)
			break;

		if(bucket->isNextDead == 0 && bucket->dnext != NULL)
			bucket = bucket->dnext;
		else
			bucket = NULL;
	}

	return bucket;
}

__device__ bool insert_multi_value(void* key, int keySize, void* value, int valueSize, multipassConfig_t* mbk)
{
	bool success = true;
	unsigned hashValue = hashFunc((char*) key, keySize, mbk->numBuckets);

	unsigned groupNo = hashValue / mbk->groupSize;

	bucketGroup_t* group = &(mbk->groups[groupNo]);
	if(group->overflownKey == 1 && group->overflownValue == 1)
		return false;
	
	hashBucket_t* existingBucket;
	hashBucket_t* lastCheckedBucket;

	int keySizeAligned = (keySize % ALIGNMET == 0)? keySize : keySize + (ALIGNMET - (keySize % ALIGNMET));
	int valueSizeAligned = (valueSize % ALIGNMET == 0)? valueSize : valueSize + (ALIGNMET - (valueSize % ALIGNMET));

	hashBucket_t* dbucket = mbk->dbuckets[hashValue];
	lastCheckedBucket = dbucket;
	

	// Scenario 1: if the key already exists, without acquiring the lock, just insert the new value atomically
	if(mbk->isNextDeads[hashValue] != 1 && (existingBucket = containsKey(dbucket, key, keySize, mbk)) != NULL)
	{
		void* oldValue = (void*) ((largeInt) existingBucket + sizeof(hashBucket_t) + keySizeAligned);
		if(!addNewValueAtomically(key, value, valueSizeAligned, oldValue, group, mbk))
		{
			group->needed = 1;
			page_t* temp = group->parentPage;
			while(temp != NULL)
			{
				temp->needed = 1;
				temp = temp->next;
			}
			success = false;
		}
#ifdef STATISTICS
		atomicInc((unsigned*) &(mbk->counter1), INT_MAX);
#endif
	}
	else 
	{
		unsigned oldLock = 1;

		do
		{
			oldLock = atomicExch((unsigned*) &(mbk->locks[hashValue]), 1);
			if(oldLock == 0)
			{
				dbucket = mbk->dbuckets[hashValue];

				if(mbk->isNextDeads[hashValue] != 1 && (existingBucket = containsKeyUntilEntry(dbucket, key, keySize, lastCheckedBucket, mbk)) != NULL)
				{
					// Scenario 2: the key is added a moment ago (just before we acquired the lock), let's unlock and inser the value atomically
					atomicExch((unsigned*) &(mbk->locks[hashValue]), 0);

					void* oldValue = (void*) ((largeInt) existingBucket + sizeof(hashBucket_t) + keySizeAligned);
					if(!addNewValueAtomically(key, value, valueSizeAligned, oldValue, group, mbk))
					{
						group->needed = 1;
						page_t* temp = group->parentPage;
						while(temp != NULL)
						{
							temp->needed = 1;
							temp = temp->next;
						}
						success = false;
					}
#ifdef STATISTICS
					atomicInc((unsigned*) &(mbk->counter2), INT_MAX);
#endif

				}
				else
				{
#ifdef STATISTICS
					atomicInc((unsigned*) &(mbk->counter3), INT_MAX);
#endif
					hashBucket_t* newBucket = (hashBucket_t*) multipassMalloc(sizeof(hashBucket_t) + keySizeAligned + sizeof(valueHolder_t) + valueSizeAligned, group, mbk);
					hashBucket_t* hbucket = mbk->buckets[hashValue];

					if(newBucket != NULL)
					{
						//TODO reduce the base offset if not null
						//newBucket->next = (bucket == NULL)? NULL : (hashBucket_t*) ((largeInt) bucket - (largeInt) mbk->dbuffer);
						newBucket->dnext = NULL;
						newBucket->next = NULL;
						if(dbucket != NULL)
						{
							newBucket->dnext = dbucket;
							newBucket->next = hbucket;
						}

						if(mbk->isNextDeads[hashValue] == 1)
							newBucket->isNextDead = 1;
						newBucket->keySize = (short) keySize;
						newBucket->valueSize = (short) valueSize;
						newBucket->dvalueHolder = (valueHolder_t*) ((largeInt) newBucket + sizeof(hashBucket_t) + keySizeAligned);
						newBucket->valueHolder = (valueHolder_t*) ((largeInt) newBucket->dvalueHolder - (largeInt) mbk->dbuffer + group->parentPage->hashTableOffset);;


						mbk->isNextDeads[hashValue] = 0;

						for(int i = 0; i < (keySizeAligned / ALIGNMET); i ++)
							*((largeInt*) ((largeInt) newBucket + sizeof(hashBucket_t) + i * ALIGNMET)) = *((largeInt*) ((largeInt) key + i * ALIGNMET));
						setValue(newBucket->dvalueHolder, value, valueSizeAligned);
						newBucket->dvalueHolder->next = NULL;
						newBucket->dvalueHolder->valueSize = (largeInt) valueSize;

						for(int i = 0; i < (valueSizeAligned / ALIGNMET); i ++)
							*((largeInt*) ((largeInt) newBucket + sizeof(hashBucket_t) + keySizeAligned + sizeof(valueHolder_t) + i * ALIGNMET)) = *((largeInt*) ((largeInt) value + i * ALIGNMET));
						((valueHolder_t*) ((largeInt) newBucket + sizeof(hashBucket_t) + keySizeAligned))->next = NULL;
						((valueHolder_t*) ((largeInt) newBucket + sizeof(hashBucket_t) + keySizeAligned))->valueSize = valueSize;

						mbk->dbuckets[hashValue] = newBucket;
						mbk->buckets[hashValue] = (hashBucket_t*) ((largeInt) newBucket - (largeInt) mbk->dbuffer + group->parentPage->hashTableOffset);

					}
					else
					{
						success = false;
					}

					atomicExch((unsigned*) &(mbk->locks[hashValue]), 0);

				}

			}
		} while(oldLock == 1);
	
		
	}

	return success;
}

__device__ bool insert_basic(void* key, int keySize, void* value, int valueSize, multipassConfig_t* mbk)
{
	bool success = true;
	unsigned hashValue = hashFunc((char*) key, keySize, mbk->numBuckets);

	unsigned groupNo = hashValue / mbk->groupSize;
	//unsigned groupNo = hashValue / GROUP_SIZE;

	bucketGroup_t* group = &(mbk->groups[groupNo]);
	
	int keySizeAligned = (keySize % ALIGNMET == 0)? keySize : keySize + (ALIGNMET - (keySize % ALIGNMET));
	int valueSizeAligned = (valueSize % ALIGNMET == 0)? valueSize : valueSize + (ALIGNMET - (valueSize % ALIGNMET));

	hashBucket_t* newBucket = (hashBucket_t*) multipassMalloc(sizeof(hashBucket_t) + keySizeAligned + sizeof(valueHolder_t) + valueSizeAligned, group, mbk);
	if(newBucket != NULL)
	{
		//TODO reduce the base offset if not null
		//newBucket->next = (bucket == NULL)? NULL : (hashBucket_t*) ((largeInt) bucket - (largeInt) mbk->dbuffer);
		newBucket->next = NULL;
		newBucket->keySize = (short) keySize;
		newBucket->valueSize = (short) valueSize;

		for(int i = 0; i < (keySizeAligned / ALIGNMET); i ++)
			*((largeInt*) ((largeInt) newBucket + sizeof(hashBucket_t) + i * ALIGNMET)) = *((largeInt*) ((largeInt) key + i * ALIGNMET));
		for(int i = 0; i < (valueSizeAligned / ALIGNMET); i ++)
			*((largeInt*) ((largeInt) newBucket + sizeof(hashBucket_t) + keySizeAligned + i * ALIGNMET)) = *((largeInt*) ((largeInt) value + i * ALIGNMET));

		largeInt atomicOldValue, oldValue;
		largeInt newValue = ((largeInt) newBucket - (largeInt) mbk->dbuffer + mbk->hashTableOffset);

		do
		{
			newBucket->isNextDead = 0;
			if(mbk->isNextDeads[hashValue] == 1)
				newBucket->isNextDead = 1;
			oldValue = (largeInt) mbk->buckets[hashValue];
			newBucket->next = (hashBucket_t*) oldValue;
			atomicOldValue = atomicCAS((unsigned long long int*) &(mbk->buckets[hashValue]), oldValue, newValue);
		} while(oldValue != atomicOldValue);
	}
	else
	{
		success = false;
	}

	return success;
}

__device__ hashBucket_t* lookup_basic(void* key, int keySize, multipassConfig_t* mbk)
{
	unsigned hashValue = hashFunc((char*) key, keySize, mbk->numBuckets);

	unsigned groupNo = hashValue / mbk->groupSize;
	//unsigned groupNo = hashValue / GROUP_SIZE;

	bucketGroup_t* group = &(mbk->groups[groupNo]);

	int isCPUResident = mbk->isNextDeads[hashValue];
	hashBucket_t* bucket = mbk->buckets[hashValue];

	while(bucket != NULL)
	{
		if(isCPUResident == 0)
		{
			bucket = (hashBucket_t*) ((largeInt) bucket - mbk->hashTableOffset + (largeInt) mbk->dbuffer);
		}

		
		char* oldKey = (char*) ((largeInt) bucket + sizeof(hashBucket_t));
		bool success = true;
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

		if(success)
			return bucket;
		
		isCPUResident = bucket->isNextDead;
		bucket = bucket->next;
	}

	return NULL;
}

__global__ void setGroupsPointersDead(multipassConfig_t* mbk, unsigned numBuckets)
{
	int index = TID;
	if(index < numBuckets)
	{
		int groupNo = index / mbk->groupSize;
		if(mbk->groups[groupNo].needed == 0)
			mbk->isNextDeads[index] = 1;
	}
	if(index < mbk->groupSize)
	{
		mbk->groups[index].overflownKey = 0;
		mbk->groups[index].overflownValue = 0;
	}

}



multipassConfig_t* initMultipassBookkeeping(	int numThreads,
						int numRecords,
						int pagePerGroup)
{
	
	multipassConfig_t* mbk = (multipassConfig_t*) malloc(sizeof(multipassConfig_t));
	mbk->numThreads = numThreads;
	mbk->numRecords = numRecords;


	mbk->availableGPUMemory = (1400 * (1 << 20));
	mbk->hhashTableBufferSize = MAX_NO_PASSES * mbk->availableGPUMemory;
	mbk->hhashTableBaseAddr = malloc(mbk->hhashTableBufferSize);
	memset(mbk->hhashTableBaseAddr, 0, mbk->hhashTableBufferSize);
	mbk->hashTableOffset = (largeInt) mbk->hhashTableBaseAddr;

	//This is how we decide the number of groups: based on the number of available pages, we make sure 
	//group size is calculated in a way that a given number of `pagePerGroup` pages are assigned to each group
	int availableNumPages = mbk->availableGPUMemory / PAGE_SIZE;
	mbk->groupSize = (pagePerGroup * NUM_BUCKETS) / availableNumPages;
	mbk->numGroups = (NUM_BUCKETS + (mbk->groupSize - 1)) / mbk->groupSize;
	//mbk->numGroups = (NUM_BUCKETS + (GROUP_SIZE - 1)) / GROUP_SIZE;
	printf("############# groupSize: %d, number of groups: %d\n", mbk->groupSize, mbk->numGroups);


	cudaMalloc((void**) &(mbk->dfailedFlag), sizeof(bool));
	cudaMemset(mbk->dfailedFlag, 0, sizeof(bool));


	// Calling initPaging
	initPaging(mbk->availableGPUMemory, mbk);

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
	cudaError_t errR = cudaGetLastError();
	printf("#######Error at the beginning of checkAndReset: %s\n", cudaGetErrorString(errR));

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

	cudaMemcpy(mbk->hpages, mbk->pages, mbk->totalNumPages * sizeof(page_t), cudaMemcpyDeviceToHost);

	
	cudaMemcpy(mbk->hfreeListId, mbk->freeListId, mbk->totalNumPages * sizeof(int), cudaMemcpyDeviceToHost);

	int freeListCounter = 0;
	int neededCounter = 0;
	int unneededCounter = 0;
	// Resetting the key page counter
	mbk->keyPageCounter = 0;
	for(int i = 0; i < mbk->totalNumPages; i ++)
	{
		if(mbk->hpages[i].needed == 0)
		{
			cudaMemcpy((void*) ((largeInt) mbk->hpages[i].hashTableOffset + mbk->hpages[i].id * PAGE_SIZE), (void*) ((largeInt) mbk->dbuffer + mbk->hpages[i].id * PAGE_SIZE), PAGE_SIZE, cudaMemcpyDeviceToHost);
			cudaMemset((void*) ((largeInt) mbk->dbuffer + mbk->hpages[i].id * PAGE_SIZE), 0, PAGE_SIZE);

			mbk->hpages[i].hashTableOffset += mbk->totalNumPages * PAGE_SIZE;
			mbk->hpages[i].next = NULL;
			mbk->hpages[i].used = 0;

			mbk->hfreeListId[freeListCounter ++] = mbk->hpages[i].id;
			unneededCounter ++;
		}
		else
		{
			mbk->hpages[i].needed = 0;
			//printf("Page %d is needed..\n", i);
			neededCounter ++;
			//mbk->keyPageCounter ++;
		}
	}

	printf("@INFO: number of needed pages: %d, and number of unneededpages: %d (number of groups: %d)\n", neededCounter, unneededCounter, NUM_BUCKETS / mbk->groupSize);

	cudaMemcpy(mbk->freeListId, mbk->hfreeListId, mbk->totalNumPages * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(mbk->pages, mbk->hpages, mbk->totalNumPages * sizeof(page_t), cudaMemcpyHostToDevice);
	mbk->totalNumFreePages = freeListCounter;
	

	printf("totalnoPage * pagesize: %llu, hhashtbufferSize: %llu\n", (largeInt) mbk->totalNumPages * PAGE_SIZE, (largeInt) hhashTableBufferSize);
	mbk->hashTableOffset += mbk->totalNumPages * PAGE_SIZE;
	if((mbk->hashTableOffset + mbk->totalNumPages * PAGE_SIZE) > ((largeInt) hhashTableBaseAddr + hhashTableBufferSize) && failedFlag)
	{
		printf("Need more space on CPU memory for the hash table. Aborting...\n");
		exit(1);
	}


	mbk->initialPageAssignedCounter = 0;



	errR = cudaGetLastError();
	printf("#######Error before setGroupPointer is: %s\n", cudaGetErrorString(errR));

	printf("Before calling setGroupPointer, number of grids: %d\n", ((NUM_BUCKETS) + 1023) / 1024);
	setGroupsPointersDead<<<(((NUM_BUCKETS) + 1023) / 1024), 1024>>>(dmbk, NUM_BUCKETS);
	//setGroupsPointersDead<<<(((NUM_BUCKETS) + 256) / 255), 256>>>(mbk->groups, NUM_BUCKETS, GROUP_SIZE);
	cudaThreadSynchronize();

	errR = cudaGetLastError();
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

void* getValueHolder(hashBucket_t* bucket)
{
	int keySizeAligned = (bucket->keySize % ALIGNMET == 0)? bucket->keySize : bucket->keySize + (ALIGNMET - (bucket->keySize % ALIGNMET));
	return (void*) ((largeInt) bucket + sizeof(hashBucket_t) + keySizeAligned);
}

void* getValue(valueHolder_t* valueHolder)
{
	return (void*) ((largeInt) valueHolder + sizeof(valueHolder_t));
}

__device__ inline void setValue(valueHolder_t* valueHoder, void* value, int valueSize)
{
	for(int i = 0; i < (valueSize / ALIGNMET); i ++)
		*((largeInt*) ((largeInt) valueHoder + sizeof(valueHolder_t) + i * ALIGNMET)) = *((largeInt*) ((largeInt) value + i * ALIGNMET));
}
