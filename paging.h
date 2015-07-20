#ifndef __HASH_CU__
#define __HASH_CU__

#include <stdio.h>
#define PAGE_SIZE (1 << 12)
#define QUEUE_SIZE 200

typedef largeInt long long int;

typedef struct
{
	unsigned pageId;
	int status; //0: available, 1: still dirty
} pageQueueItem_t;

typedef struct
{
	//Alternatively, this can be page_t*
	int pageIds[QUEUE_SIZE]; 
	int front;
	int rear;
	int dirtyRear;
	int lock;
	
} pageQueue_t;

typedef struct
{
	page_t* next;
	unsigned used;
	unsigned id;
} page_t;

typedef struct
{
	page_t* pages;
	int totalNumPages;

	//This will be a queue, holding pointers to pages that are available
	pageQueue_t* queue;

	int initialPageAssignedCounter;
	int initialPageAssignedCap;
	int minimumQueueSize;
} pagingConfig_t;


__host__ void initPaging(int hashBuckets, largeInt availableGPUMemory, int minimumQueueSize, largeInt freeMemBaseAddr, pagingConfig_t* pconfig)
{

	initQueue(pconfig);
	pconfig->totalNumPages = availableGPUMemory / PAGE_SIZE;
	pconfig->initialPageAssignedCounter = 0;
	pconfig->initialPageAssignedCap = pconfig->totalNumPages - minimumQueueSize;

	//This has to be allocated GPU-side
	page_t* hdpages = (page_t*) malloc(pconfig->totalNumPages * sizeof(page_t));
	for(int i = 0; i < pconfig->totalNumPages; i ++)
	{
		hdpages[i].id = i;
		hdpages[i].next = NULL;
		hdpages[i].used = 0;
	}
	cudaMalloc((void**) &(pconfig->pages), pconfig->totalNumPages * sizeof(page_t));
	cudaMemcpy(pconfig->pages, hdpages, pconfig->totalNumPages * sizeof(page_t), cudaMemcpyHostToDevice);

	//This has to be allocated host-pinned
	cudaHostAlloc((void**) &(pconfig->queue), 100 * minimumQueueSize * sizeof(pageQueueItem_t), cudaHostAllocMapped);
	memset(pconfig->queue, 0, 100 * minimumQueueSize * sizeof(pageQueueItem_t));

	pconfig->minimumQueueSize = minimumQueueSize;
	//Adding last free pages to queue
	for(int i = pconfig->totalNumPages - minimumQueueSize; i < pconfig->totalNumPages; i ++)
	{
		pconfig->queue->pushCleanPage(&(pconfig.pages[i]));
	}
}

__host__ void initQueue(pagingConfig_t* pconfig)
{
	//This has to be allocated as host-pinned
	pconfig->queue = (pageQueue_t*) malloc(sizeof(pageQueue_t));
	pconfig->queue->front = 0;
	pconfig->queue->rear = 0;
	pconfig->queue->dirtyRear = 0;
	pconfig->queue->lock = 0;
}

//This is called only by CPU thread, so needs no synchronization (unless run by multiple threads)
__host__ void pushCleanPage(page_t* page, pagingConfig_t* pconfig)
{
	pconfig->queue->pagesIds[pconfig->queue->rear] = page->id;
	pconfig->queue->rear ++;
	pconfig->queue->rear %= QUEUE_SIZE;
	pconfig->queue->dirtyRear = pconfig->queue->rear;
}

//TODO: I assume the queue will not be full which is a faulty assumption.
__device__ void pushDirtyPage(page_t* page, pagingConfig_t* pconfig)
{
	//acquire lock
	pconfig->queue->pageIds[pconfig->queue->dirtyRear] = page->id;
	pconfig->queue->dirtyRear ++;
	pconfig->queue->dirtyRear %= QUEUE_SIZE;
	//release lock
}

//Run by GPU
__device__ page_t* popCleanPage(pagingConfig_t* pconfig)
{
	//acquire lock
	page_t* page = NULL;
	if(pconfig->queue->rear != pconfig->queue->front)
	{
		page = pconfig->pages[pconfig->queue->pageIds[pconfig->queue->front]];
		pconfig->queue->front ++;
		pconfig->queue->front %= QUEUE_SIZE;
	}
	//release lock
	return page;

}
//This is run only by one CPU, so can be simplified (in terms of synchronizatio)
__host__ page_t* virtualDirtyPeek(pagingConfig_t* pconfig)
{
	page_t* page = NULL;
	if(pconfig->queue->rear != pconfig->queue->dirtyRear)
	{
		page = pconfig->pages[pconfig->queue->pageIds[pconfig->queue->rear]];
	}
	return page;
}

__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup)
{
	page_t* parentPage = myGroup->parentPage;

	unsigned oldUsed = atomicAdd(&(parentPage->used), size);
	if(oldUsed < PAGE_SIZE)
	{
		return (void*) (parentPage.id * PAGE_SIZE + oldUsed);
	}

	//acquire some lock
	unsigned oldLock = 1;
	do
	{
		oldLock = atomicExch(&(myGroup->pageLock), 1);

		if(oldLock == 0)
		{
			page_t* newPage = allocateNewPage();

			//If no more page exists and no page is used yet (for this bucketgroup), don't do anything
			if(newPage == NULL)
			{
				revokePage(parentPage);
				//releaseLock
				atomicExch(&(myGroup->pageLock), 0);
				return NULL;
			}

			newPage->next = parentPage;
			myGroup->parentPage = newPage;
		}

	} while(oldLock == 1);

	//releaseLock
	atomicExch(&(myGroup->pageLock), 0);
	oldUsed = atomicAdd(&(newPage->used), size);

	return (void*) (oldUsed + newPage.id * PAGE_SIZE);
}

page_t* allocateNewPage()
{
	int pageIdToAllocate = atomicInc(&initialPageAssignedCounter, INT_MAX)
		if(initialPageAssignedCounter < initialPageAssignedCap)
		{
			return &pages[pageIdToAllocate];
		}
		else
		{
			//atomiPop will pop an item only if `minimmQuerySize` free entry is available, NULL otherwise.
			return queue->atomicPop(minimumQueueSize);
		}
}

//Freeing the chain pages
__device__ void revokePage(page_t* page)
{
	//If parent page is -1, then we have nothing to do
	while(page != NULL)
	{
		queue->pushDirtyPage(page);
		page = page->next;
	}
}

//Executed by a separate CPU thread
__host__ void pageReuser()
{
	while(true)
	{
		page_t* page;
		if((page = virtualDirtyPeek()) != NULL)
		{
			//cudaMemcpyAsync(hashtable_base_CPU + page->id * PAGE_SIZE, hashtable_base_GPU + page->id * PAGE_SIZE, PAGE_SIZE, cudaMemcpyDeviceToHost, serviceStream);
			//cudaMemsetAsync(hashtable_base_GPU + page->id * PAGE_SIZE, 0, PAGE_SIZE, serviceStream);
			page->used = 0;
			page->next= NULL;

			//Advancing rear..
			queue->rear ++;
			queue->rear %= QUEUE_SIZE;

		}
	}
}



#endif
