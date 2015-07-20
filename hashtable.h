#ifndef __HASHTABLE_CU__
#define __HASHTABLE_CU__

#include <stdio.h>
#define PAGE_SIZE (1 << 12)
#define GROUP_SIZE (PAGE_SIZE / sizeof(hashBucket_t))
#define ALIGNMET 8

typedef largeInt long long int;

//We need to ask user to have a field named 'key'
typedef struct
{
	char key[64];
	largeInt value;

} userBucket_t;


//Key and value will be appended to the instance of hashBucket_t every time `multipassMalloc` is called in `add`
//NOTE: sizeof(hashBucket_t) should be aligned by ALIGNMET
typedef struct
{
	hashBucket_t* next;
	unsigned lock;
	short keySize
	short valueSize
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
				//First see if the key already exists in one of the entries of this bucket
				//The returned bucket is the 'entry' in which the key exists
				if((bucket = containsKey(bucket, key)) != NULL)
				{
					resolveSameKeyAddition(key, value, &(bucket->myUserBucket));
				}
				else
				{
					int keySizeAligned = (keySize % ALIGNMET == 0)? keySize : keySize + (ALIGNMET - (keySize % ALIGNMET));
					int valueSizeAligned = (valueSize % ALIGNMET == 0)? valueSize : valueSize + (ALIGNMET - (valueSize % ALIGNMET));
					hashBucket_t* newBucket = (hashBucket_t*) multipassMalloc(sizeof(hashBucket_t) + keySizeAligned + valueSizeAligned, group);

					if(newBucket != NULL)
					{
						//TODO use proper base
						newBucket->next = (bucket == NULL)? bucket : (hashBucket_t*)((largeInt) bucket - base);
						group->buckets[offsetWithinGroup] = newBucket;
						
						//TODO: this assumes that input key is aligned by ALIGNMENT, which is not a safe assumption
						for(int i = 0; i < (keySizeAligned / ALIGNMET); i ++)
							*((largeInt*) ((largetInt) newBucket + sizeof(hashBucket_t) + i * ALIGNMET)) = *((largeInt*) ((largetInt) key + i * ALIGNMET));
						for(int i = 0; i < (valueSizeAligned / ALIGNMET); i ++)
							*((largeInt*) ((largetInt) newBucket + sizeof(hashBucket_t) + keySizeAligned + i * ALIGNMET)) = *((largeInt*) ((largetInt) value + i * ALIGNMET));
						
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
