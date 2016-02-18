#ifndef __HASHGLOBAL_H__
#define __HASHGLOBAL_H__

#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

#define PAGE_SIZE (1 << 19)
#define NUM_BUCKETS 10000000
#define ALIGNMET 8
#define MAX_NO_PASSES 4

#define HOST_BUFFER_SIZE (1 << 31)

#define BLOCK_ID (gridDim.y * blockIdx.x + blockIdx.y)
#define THREAD_ID (threadIdx.x)
#define TID (BLOCK_ID * blockDim.x + THREAD_ID)

enum recordType { UNTESTED = 0, SUCCEED = 1, FAILED = 2};

typedef long long int largeInt;

typedef struct valueHolder_t
{
	struct valueHolder_t* next;
	struct valueHolder_t* dnext;
	largeInt valueSize;
} valueHolder_t;


//================ paging structures ================//

typedef struct page_t
{
	struct page_t* next;
	largeInt hashTableOffset;
	unsigned used;
	short id;
	short needed;
} page_t;

typedef struct
{
	page_t* pages;
	page_t* hpages;

	void* dbuffer;
	//void* hbuffer;
	largeInt hashTableOffset;
	int totalNumPages;

	int initialPageAssignedCounter;
	int initialPageAssignedCap;
} pagingConfig_t;



//================ hashing structures ================//

//Key and value will be appended to the instance of hashBucket_t every time `multipassMalloc` is called in `add`
//NOTE: sizeof(hashBucket_t) should be aligned by ALIGNMET
typedef struct hashBucket_t
{
	struct hashBucket_t* next;
	struct hashBucket_t* dnext;
	valueHolder_t* valueHolder;
	valueHolder_t* dvalueHolder;
	short isNextDead;
	unsigned short lock;
	short keySize;
	short valueSize;
} hashBucket_t;

typedef struct
{
	page_t* parentPage;
	page_t* valueParentPage;
	unsigned pageLock;
	unsigned needed;
	
} bucketGroup_t;


typedef struct
{
	bucketGroup_t* groups;
	hashBucket_t** buckets;
	unsigned* locks;
	short* isNextDeads;
	unsigned numBuckets;
	unsigned groupSize;
} hashtableConfig_t;

typedef struct
{
	int* hostCompleteFlag;
	int* gpuFlags;
	bool* dfailedFlag;
	int* myNumbers;
	int* dmyNumbers;
	void* hhashTableBaseAddr;
	largeInt hhashTableBufferSize;
	size_t availableGPUMemory;
	char* epochSuccessStatus;
	int* depochSizes;
	char* depochSuccessStatus;
	char* dstates;
	int* freeListId;
	int* hfreeListId;
	//pagingConig and hashConfig
	page_t* pages;
	page_t* hpages;
	void* dbuffer;
	void* hbuffer;

	bucketGroup_t* groups;
	hashBucket_t** buckets;
	hashBucket_t** dbuckets;
	unsigned* locks;
	short* isNextDeads;

	largeInt hashTableOffset;
	int totalNumPages;
	int totalNumFreePages;
	unsigned numBuckets;
	unsigned groupSize;

	int initialPageAssignedCounter;
	int initialPageAssignedCap;
	//===========================//

	unsigned numGroups;
	int flagSize;
	int numThreads;
	int epochNum;
	int numRecords;
} multipassConfig_t;




void initPaging(largeInt availableGPUMemory, multipassConfig_t* mbk);
__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup, multipassConfig_t* mbk);
__device__ void* multipassMallocValue(unsigned size, bucketGroup_t* myGroup, multipassConfig_t* mbk);
__device__ page_t* allocateNewPage(multipassConfig_t* mbk);


void hashtableInit(unsigned numBuckets, multipassConfig_t* mbk, unsigned groupSize);
__device__ unsigned int hashFunc(char* str, int len, unsigned numBuckets);
__device__ bool resolveSameKeyAddition(void const* key, void* value, void* oldValue, bucketGroup_t* group, multipassConfig_t* mbk);
__device__ hashBucket_t* containsKey(hashBucket_t* bucket, void* key, int keySize, multipassConfig_t* mbk);
//__device__ bool addToHashtable(void* key, int keySize, void* value, int valueSize, multipassConfig_t* mbk);
__device__ bool addToHashtable(void* key, int keySize, void* value, int valueSize, multipassConfig_t* mbk, int passno);
__device__ bool atomicAttemptIncRefCount(int* refCount);
__device__ int atomicDecRefCount(int* refCount);
__device__ bool atomicNegateRefCount(int* refCount);

multipassConfig_t* initMultipassBookkeeping(int* hostCompleteFlag, 
						int* gpuFlags, 
						int flagSize,
						int numThreads,
						int epochNum,
						int numRecords,
						int pagePerGroup);

__global__ void setGroupsPointersDead(multipassConfig_t* mbk, unsigned numBuckets);;
bool checkAndResetPass(multipassConfig_t* mbk, multipassConfig_t* dmbk);
void* getKey(hashBucket_t* bucket);
void* getValueHolder(hashBucket_t* bucket);
void* getValue(valueHolder_t* valueHolder);
__device__ void setValue(valueHolder_t* valueHoder, void* value, int valueSize);
#endif
