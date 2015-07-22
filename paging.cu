#include "global.h"

void initPaging(largeInt availableGPUMemory, int minimumQueueSize, pagingConfig_t* pconfig)
{

	initQueue(pconfig);
	printf("@INFO: done doing initQueue\n");
	pconfig->totalNumPages = availableGPUMemory / PAGE_SIZE;
	pconfig->initialPageAssignedCounter = 0;
	pconfig->initialPageAssignedCap = pconfig->totalNumPages - minimumQueueSize;

	cudaMalloc((void**) &(pconfig->dbuffer), pconfig->totalNumPages * PAGE_SIZE);
	cudaMemset(pconfig->dbuffer, 0, pconfig->totalNumPages * PAGE_SIZE);
	printf("@INFO: done allocating base buffer in GPU memory\n");

	pconfig->hbuffer = malloc((unsigned) HOST_BUFFER_SIZE);
	memset(pconfig->hbuffer, 0, (unsigned) HOST_BUFFER_SIZE);

	//This has to be allocated GPU-side
	pconfig->hpages = (page_t*) malloc(pconfig->totalNumPages * sizeof(page_t));
	for(int i = 0; i < pconfig->totalNumPages; i ++)
	{
		pconfig->hpages[i].id = i;
		pconfig->hpages[i].next = NULL;
		pconfig->hpages[i].used = 0;
	}
	printf("@INFO: done initializing pages meta data\n");
	cudaMalloc((void**) &(pconfig->pages), pconfig->totalNumPages * sizeof(page_t));
	cudaMemcpy(pconfig->pages, pconfig->hpages, pconfig->totalNumPages * sizeof(page_t), cudaMemcpyHostToDevice);

	pconfig->minimumQueueSize = minimumQueueSize;
	printf("@INFO: Adding some clean pages to queue\n");
	//Adding last free pages to queue
	for(int i = pconfig->totalNumPages - minimumQueueSize; i < pconfig->totalNumPages; i ++)
	{
		pushCleanPage(&(pconfig->hpages[i]), pconfig);
	}
	printf("@INFO: done doing initPaging\n");
}

void initQueue(pagingConfig_t* pconfig)
{
	//This has to be allocated as host-pinned
	cudaHostAlloc((void**) &(pconfig->queue), sizeof(pageQueue_t), cudaHostAllocMapped);
	memset(pconfig->queue, 0, sizeof(pageQueue_t));
	cudaHostGetDevicePointer((void**) &(pconfig->dqueue), pconfig->queue, 0);

	pconfig->queue->front = 0;
	pconfig->queue->rear = 0;
	pconfig->queue->dirtyRear = 0;
	pconfig->queue->lock = 0;
}

//This is called only by CPU thread, so needs no synchronization (unless run by multiple threads)
void pushCleanPage(page_t* page, pagingConfig_t* pconfig)
{
	pconfig->queue->pageIds[pconfig->queue->rear] = page->id;
	pconfig->queue->rear ++;
	pconfig->queue->rear %= QUEUE_SIZE;
	pconfig->queue->dirtyRear = pconfig->queue->rear;
}

//TODO: I assume the queue will not be full which is a faulty assumption.
__device__ void pushDirtyPage(page_t* page, pagingConfig_t* pconfig)
{
	unsigned oldLock = 1;
	do
	{
		oldLock = atomicExch(&(pconfig->dqueue->lock), 1);
		if(oldLock == 0)
		{
			//FIXME: this assumes we never have a full queue
			pconfig->dqueue->pageIds[pconfig->dqueue->dirtyRear] = page->id;
			pconfig->dqueue->dirtyRear ++;
			pconfig->dqueue->dirtyRear %= QUEUE_SIZE;

			//Unlocking
			atomicExch(&(pconfig->dqueue->lock), 0);
		}
	} while(oldLock == 1);
}

//Run by GPU
__device__ page_t* popCleanPage(pagingConfig_t* pconfig)
{
	page_t* page = NULL;
	unsigned oldLock = 1;
	do
	{
		oldLock = atomicExch(&(pconfig->dqueue->lock), 1);
		if(oldLock == 0)
		{
			if(pconfig->dqueue->rear != pconfig->dqueue->front)
			{
				page = &(pconfig->pages[pconfig->dqueue->pageIds[pconfig->dqueue->front]]);
				pconfig->dqueue->front ++;
				pconfig->dqueue->front %= QUEUE_SIZE;
			}
			//Unlocking
			atomicExch(&(pconfig->dqueue->lock), 0);
		}
	} while(oldLock == 1);

	return page;

}


__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup, pagingConfig_t* pconfig)
{
	page_t* parentPage = myGroup->parentPage;

	unsigned oldUsed = 0;
	if(parentPage != NULL)
	{
		oldUsed = atomicAdd(&(parentPage->used), size);
		if(oldUsed < PAGE_SIZE)
		{
			return (void*) ((largeInt) pconfig->dbuffer + parentPage->id * PAGE_SIZE + oldUsed);
		}
	}

	page_t* newPage = NULL;
	//acquire some lock
	unsigned oldLock = 1;
	do
	{
		oldLock = atomicExch(&(myGroup->pageLock), 1);

		if(oldLock == 0)
		{
			newPage = allocateNewPage(pconfig);

			//If no more page exists and no page is used yet (for this bucketgroup), don't do anything
			if(newPage == NULL)
			{
				//revokePage(parentPage, pconfig); //TODO uncomment
				//releaseLock
				atomicExch(&(myGroup->pageLock), 0);
				return NULL;
			}

			newPage->next = parentPage;
			myGroup->parentPage = newPage;

			//Unlocking
			atomicExch(&(myGroup->pageLock), 0);
		}

	} while(oldLock == 1);

	oldUsed = atomicAdd(&(newPage->used), size);

	return (void*) ((largeInt) pconfig->dbuffer + oldUsed + newPage->id * PAGE_SIZE);
}

__device__ page_t* allocateNewPage(pagingConfig_t* pconfig)
{
	int pageIdToAllocate = atomicInc((unsigned*) &(pconfig->initialPageAssignedCounter), INT_MAX);
	if(pageIdToAllocate < pconfig->initialPageAssignedCap)
	{
		return &(pconfig->pages[pageIdToAllocate]);
	}
	else
	{
		return NULL; //TODO remove this line. Is just put here to simplify debugging.
		//atomiPop will pop an item only if `minimmQuerySize` free entry is available, NULL otherwise.
		return popCleanPage(pconfig);
	}
}

//Freeing the chain pages
__device__ void revokePage(page_t* page, pagingConfig_t* pconfig)
{
	//If parent page is -1, then we have nothing to do
	while(page != NULL)
	{
		pushDirtyPage(page, pconfig);
		page = page->next;
	}
}



//This is run only by one CPU, so it can be simplified (in terms of synchronization)
page_t* peekDirtyPage(pagingConfig_t* pconfig)
{
	page_t* page = NULL;
	if(pconfig->queue->rear != pconfig->queue->dirtyRear)
	{
		page = &(pconfig->pages[pconfig->queue->pageIds[pconfig->queue->rear]]);
	}
	return page;
}

//Executed by a separate CPU thread
void pageRecycler(pagingConfig_t* pconfig, cudaStream_t* serviceStream)
{
	while(true)
	{
		page_t* page;
		if((page = peekDirtyPage(pconfig)) != NULL)
		{
			cudaMemcpyAsync((void*) ((largeInt) pconfig->hbuffer + page->id * PAGE_SIZE), (void*) ((largeInt) pconfig->dbuffer + page->id * PAGE_SIZE), PAGE_SIZE, cudaMemcpyDeviceToHost, *serviceStream);
			cudaMemsetAsync((void*) ((largeInt) pconfig->dbuffer + page->id * PAGE_SIZE), 0, PAGE_SIZE, *serviceStream);
        		while(cudaSuccess != cudaStreamQuery(*serviceStream));

			page->used = 0;
			page->next= NULL;

			//[Atomically] advancing rear..
			int tempRear = pconfig->queue->rear;
			tempRear ++;
			tempRear %= QUEUE_SIZE;
			pconfig->queue->rear = tempRear;

		}
	}
}



