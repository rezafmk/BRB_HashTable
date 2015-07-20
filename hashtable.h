#ifndef __HASHTABLE_CU__
#define __HASHTABLE_CU__

#include <stdio.h>
#define PAGE_SIZE (1 << 12)
#define GROUP_SIZE (PAGE_SIZE / sizeof(hashBucket_t))

typedef largeInt long long int;

typedef struct
{
	char key[64];
	largeInt value;

} userBucket_t;

typedef struct
{
	userBucket_t myUserBucket;
	hashBucket_t* next;
	unsigned lock;

} hashBucket_t;

typedef struct
{
	hashbucket_t* buckets[GROUP_SIZE];
	unsigned locks[GROUP_SIZE];
	page_t* parentPage;
	unsigned pageLock;

} bucketGroup_t;

class hashtable
{
	int numBuckets;
	int bucketsPerGroup;
	bucketGroup_t* groups;

	public:

	void hashtable(int numBuckets, int bucketsPerGroup)
	{
		this.numBuckets = numBuckets;
		this.bucketsPerGroup = bucketsPerGroup;
		int numGroups = (numBuckets + (bucketsPerGroup - 1)) / bucketsPerGroup;
		groups = (bucketGroup_t*) malloc(numGroups * sizeof(bucketGroup_t));
	}

	__device__ void resolveSameKeyAddition(void* key, void* value, userBucket_t* existingBucket)
	{
		existingBucket->value ++;
	}

	__device__ bool add(void* key, int keySize, void* value, int valueSize)
	{
		bool success = true;
		unsigned hashValue = hashFunc(key, keySize);

		unsigned groupNo = hashValue / GROUP_SIZE;
		unsigned offsetWithinGroup = hashValue % GROUP_SIZE;

		hashGroup_t* group = &groups[groupNo];
		hashBucket_t* bucket = group->buckets[offsetWithinGroup];
	
		unsigned oldLock = 1;

		do
		{
			oldLock = atomicExch((unsigned*) &(group->locks[offsetWithinGroup]), 1);

			if(oldLock == 0)
			{
				//some lock acquired here
				if((bucket = containsKey(bucket, key)) != NULL)
				{
					resolveSameKeyAddition(key, value, &(bucket->myUserBucket));
				}
				else
				{
					hashBucket_t* newBucket = (hashBucket_t*) multipassMalloc(sizeof(hashBucket_t), group);
					if(newBucket != NULL)
					{
						//TODO use proper base
						newBucket->next = bucket - base;
						group->buckets[offsetWithinGroup] = newBucket;
							
						//This can be put in an efficient function written by user
						*((largeInt*) ((largetInt) &(newBucket->myUserBucket.key) + 0)) = *((largeInt*) ((largetInt) key + 0));
						*((largeInt*) ((largetInt) &(newBucket->myUserBucket.key) + 8)) = *((largeInt*) ((largetInt) key + 8));
						*((largeInt*) ((largetInt) &(newBucket->myUserBucket.key) + 16)) = *((largeInt*) ((largetInt) key + 16));
						*((largeInt*) ((largetInt) &(newBucket->myUserBucket.key) + 24)) = *((largeInt*) ((largetInt) key + 24));
						*((largeInt*) ((largetInt) &(newBucket->myUserBucket.key) + 32)) = *((largeInt*) ((largetInt) key + 32));
						*((largeInt*) ((largetInt) &(newBucket->myUserBucket.key) + 40)) = *((largeInt*) ((largetInt) key + 40));
						*((largeInt*) ((largetInt) &(newBucket->myUserBucket.key) + 48)) = *((largeInt*) ((largetInt) key + 48));
						*((largeInt*) ((largetInt) &(newBucket->myUserBucket.key) + 56)) = *((largeInt*) ((largetInt) key + 56));
						*((largeInt*) &(newBucket->myUserBucket.value)) = *((largeInt*) value);
					}
					else
					{
						success = false;
					}
				}
			}
		} while(oldLock == 1);
		
		atomicExch((unsigned*) &(group->locks[offsetWithinGroup]), 0);
		
		//release the lock
		return success;
	}
	
	
}


#endif
