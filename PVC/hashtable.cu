#include "hashGlobal.h"

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

__device__ hashBucket_t* containsKey(hashBucket_t* bucket, void* key, int keySize)
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

		if(bucket->isNextDead == 0)
			bucket = bucket->next;
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
			if(group->isNextDead[offsetWithinGroup] != 1 && (existingBucket = containsKey(bucket, key, keySize)) != NULL)
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


