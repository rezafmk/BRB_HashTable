#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#define PAGE_SIZE (1 << 12)
#define GROUP_SIZE 64
#define ALIGNMET 8

#define PAGE_SIZE (1 << 12)
#define QUEUE_SIZE 200
#define HOST_BUFFER_SIZE (1 << 31)

typedef long long int largeInt;

//================ paging structures ================//
typedef struct
{
	//Alternatively, this can be page_t*
	int pageIds[QUEUE_SIZE]; 
	int front;
	int rear;
	int dirtyRear;
	int lock;
	
} pageQueue_t;

typedef struct page_t
{
	struct page_t* next;
	unsigned used;
	unsigned id;
} page_t;

typedef struct
{
	page_t* pages;
	page_t* hpages;

	void* dbuffer;
	void* hbuffer;
	int totalNumPages;

	//This will be a queue, holding pointers to pages that are available
	pageQueue_t* queue;
	pageQueue_t* dqueue;

	int initialPageAssignedCounter;
	int initialPageAssignedCap;
	int minimumQueueSize;
} pagingConfig_t;

//================ hashing structures ================//

//Key and value will be appended to the instance of hashBucket_t every time `multipassMalloc` is called in `add`
//NOTE: sizeof(hashBucket_t) should be aligned by ALIGNMET
typedef struct hashBucket_t
{
	struct hashBucket_t* next;
	unsigned lock;
	short keySize;
	short valueSize;
} hashBucket_t;

typedef struct
{
	hashBucket_t* buckets[GROUP_SIZE];
	unsigned locks[GROUP_SIZE];
	page_t* parentPage;
	unsigned pageLock;
	//unsigned int refCount;
	int failed;

} bucketGroup_t;

typedef struct
{
	bucketGroup_t* groups;
	int numBuckets;
} hashtableConfig_t;



void initPaging(largeInt availableGPUMemory, int minimumQueueSize, pagingConfig_t* pconfig);
void initQueue(pagingConfig_t* pconfig);
void pushCleanPage(page_t* page, pagingConfig_t* pconfig);
__device__ void pushDirtyPage(page_t* page, pagingConfig_t* pconfig);
__device__ page_t* popCleanPage(pagingConfig_t* pconfig);
page_t* peekDirtyPage(pagingConfig_t* pconfig);
__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup, pagingConfig_t* pconfig);
__device__ page_t* allocateNewPage(pagingConfig_t* pconfig);
__device__ void revokePage(page_t* page, pagingConfig_t* pconfig);
void pageRecycler(pagingConfig_t* pconfig, cudaStream_t* serviceStream);


void hashtableInit(int numBuckets, hashtableConfig_t* hconfig);
__device__ unsigned int hashFunc(char* str, int len, unsigned numBuckets);
__device__ void resolveSameKeyAddition(void const* key, void* value, void* oldValue);
__device__ hashBucket_t* containsKey(hashBucket_t* bucket, void* key, int keySize);
__device__ bool addToHashtable(void* key, int keySize, void* value, int valueSize, hashtableConfig_t* hconfig, pagingConfig_t* pconfig);
#endif
