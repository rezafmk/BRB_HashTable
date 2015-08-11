#include "hashGlobal.h"

void initPaging(largeInt availableGPUMemory, pagingConfig_t* pconfig)
{

	initQueue(pconfig);
	printf("@INFO: done doing initQueue\n");
	pconfig->totalNumPages = availableGPUMemory / PAGE_SIZE;
	printf("@INFO: total number of pages: %d [each %dKB]\n", pconfig->totalNumPages, (PAGE_SIZE / (1 << 10)));
	pconfig->initialPageAssignedCounter = 0;
	pconfig->initialPageAssignedCap = pconfig->totalNumPages;

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

	printf("@INFO: done doing initPaging\n");
}

void initQueue(pagingConfig_t* pconfig)
{
	
	printf("@INFO: inside initQueue\n");
	//This has to be allocated as host-pinned
	for(int i = 0; i < NUM_QUEUES; i ++)
	{
		cudaHostAlloc((void**) &(pconfig->queue[i]), sizeof(pageQueue_t), cudaHostAllocMapped);
		memset(pconfig->queue[i], 0, sizeof(pageQueue_t));
		cudaHostGetDevicePointer((void**) &(pconfig->dqueue[i]), pconfig->queue[i], 0);
	}
	printf("@INFO: inside initQueue: after loop\n");
	//cudaMalloc((void**) &(pconfig->dqueue), NUM_QUEUES * sizeof(void *));
	//cudaMemcpy(pconfig->dqueue, pconfig->hdqueue, NUM_QUEUES * sizeof(void *), cudaMemcpyHostToDevice);
}


//TODO: I assume the queue will not be full which is a faulty assumption.
__device__ void pushDirtyPage(page_t* page, pagingConfig_t* pconfig, int queueId)
{
	unsigned* dirtyRearAddress = (unsigned*) &(pconfig->dqueue[queueId]->dirtyRear);
	unsigned* frontAddress = (unsigned*) &(pconfig->dqueue[queueId]->front);
	unsigned oldDirtyRear = *dirtyRearAddress;
	unsigned assume;
	bool success;
	do
	{
		success = false;
		assume = oldDirtyRear;

		// Only push if there's a slot available
		if(((oldDirtyRear + 1) % QUEUE_SIZE) != (*frontAddress % QUEUE_SIZE))
		{
			oldDirtyRear = atomicCAS(dirtyRearAddress, assume, oldDirtyRear + 1);
			success = true;
		}
	} while(assume != oldDirtyRear);

	if(success)
	{
		pconfig->dqueue[queueId]->pageIds[oldDirtyRear % QUEUE_SIZE] = page->id;
	}
		
}


//Run by GPU
__device__ page_t* popCleanPage(pagingConfig_t* pconfig, int queueId)
{
	unsigned* frontAddress = (unsigned*) &(pconfig->dqueue[queueId]->front);
	unsigned* rearAddress = (unsigned*) &(pconfig->dqueue[queueId]->rear);
	unsigned oldFront = *frontAddress;
	unsigned assume;

	if(*rearAddress == oldFront)
		return NULL;

	do
	{
		assume = oldFront;
		if(*rearAddress != oldFront)
		{
			oldFront = atomicCAS(frontAddress, assume, oldFront + 1);	
		}
		else
		{
			return NULL;
		}
		
	} while(assume != oldFront);

	printf("providing one page\n");
	page_t* page = &(pconfig->pages[pconfig->dqueue[queueId]->pageIds[oldFront % QUEUE_SIZE]]);
	page->used = 0;
	page->next = NULL; //OPT: This can be removed..
	return page;
}



//TODO: currently we don't mark a bucket group to not ask for more memory if it previously revoked its pages
__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup, pagingConfig_t* pconfig)
{
	page_t* parentPage = myGroup->parentPage;

	unsigned oldUsed = 0;
	if(parentPage != NULL)
	{
		oldUsed = atomicAdd(&(parentPage->used), size);
		if((oldUsed + size) < PAGE_SIZE)
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
			//Re-testing if the parent page has room (because the partenPage might have changed)
			parentPage = myGroup->parentPage;
			if(parentPage != NULL)
			{
				oldUsed = atomicAdd(&(parentPage->used), size);
				if((oldUsed + size) < PAGE_SIZE)
				{
					//Unlocking
					atomicExch(&(myGroup->pageLock), 0);
					return (void*) ((largeInt) pconfig->dbuffer + parentPage->id * PAGE_SIZE + oldUsed);
				}
			}
			
			newPage = allocateNewPage(pconfig);

			//If no more page exists and no page is used yet (for this bucketgroup), don't do anything
			if(newPage == NULL)
			{
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

	//This assumes that the newPage is not already full, which is to be tested.
	oldUsed = atomicAdd(&(newPage->used), size);

	//if((oldUsed + size) < PAGE_SIZE)
		return (void*) ((largeInt) pconfig->dbuffer + oldUsed + newPage->id * PAGE_SIZE);
	//else
		//return NULL;
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
		//atomiPop will pop an item only if `minimmQuerySize` free entry is available, NULL otherwise.
		return popCleanPage(pconfig, threadIdx.x % NUM_QUEUES);
	}
}

//Freeing the chain pages
__device__ void revokePage(page_t* page, pagingConfig_t* pconfig)
{
	//If parent page is -1, then we have nothing to do
	while(page != NULL)
	{
		pushDirtyPage(page, pconfig, threadIdx.x % NUM_QUEUES);
		page = page->next;
	}
}



//This is run only by one CPU, so it can be simplified (in terms of synchronization)
page_t* peekDirtyPage(pagingConfig_t* pconfig, int queueId)
{
	page_t* page = NULL;
	if(pconfig->queue[queueId]->rear != pconfig->queue[queueId]->dirtyRear)
	{
		printf("found a dirty page in queue %d\n", queueId);
		page = &(pconfig->hpages[pconfig->queue[queueId]->pageIds[pconfig->queue[queueId]->rear]]);
	}
	return page;
}

//Executed by a separate CPU thread
void pageRecycler(pagingConfig_t* pconfig, cudaStream_t* serviceStream, cudaStream_t* execStream)
{
	int i = 0;
	int queueId = 0;
	while(cudaErrorNotReady == cudaStreamQuery(*execStream))
	{
		page_t* page;
		if((page = peekDirtyPage(pconfig, queueId)) != NULL)
		{
			cudaMemcpyAsync((void*) ((largeInt) pconfig->hbuffer + page->id * PAGE_SIZE), (void*) ((largeInt) pconfig->dbuffer + page->id * PAGE_SIZE), PAGE_SIZE, cudaMemcpyDeviceToHost, *serviceStream);
			cudaMemsetAsync((void*) ((largeInt) pconfig->dbuffer + page->id * PAGE_SIZE), 0, PAGE_SIZE, *serviceStream);
        		while(cudaSuccess != cudaStreamQuery(*serviceStream));

			int queueSize = pconfig->queue[queueId]->dirtyRear - pconfig->queue[queueId]->front;
			printf("one page recycled: %d [queue size: %d]\n", i ++, queueSize);

			//TODO: The following is not gonna be done on the actual page meta data on GPU memory. It is only on hpages (thus pointless)
			page->used = 0;
			page->next= NULL;

			//[Atomically] advancing rear..
			int tempRear = pconfig->queue[queueId]->rear;
			tempRear ++;
			//tempRear %= QUEUE_SIZE;
			pconfig->queue[queueId]->rear = tempRear;

		}
		queueId ++;
		queueId %= NUM_QUEUES;
	}
}



