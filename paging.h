#ifndef __HASH_CU__
#define __HASH_CU__

#include <stdio.h>
#define PAGE_SIZE (1 << 12)
#define QUEUE_SIZE 20

typedef largeInt long long int;

typedef struct
{
	unsigned pageId;
	int status; //0: available, 1: still dirty
} pageQueueItem_t;

typedef struct
{
	//Alternatively, this can be page_t*
	page_t* pages[QUEUE_SIZE]; 
	int front;
	int rear;
	int dirtyRear;
	int lock;
	
} pageQueue_t;

typedef struct
{
	largeInt paddr;
	page_t* next;
	unsigned used;
} page_t;




class multipassPaging 
{
	//Each page_t is the meta data kept for each physical page
	page_t* pages;
	int totalNumPages;

	//This will be a queue, holding pointers to pages that are available
	pageQueue_t* queue;

	int initialPageAssignedCounter;
	int initialPageAssignedCap;
	int minimumQueueSize;

	public:

	__host__ void multipassPaging(int hashBuckets, largeInt availableGPUMemory, int minimumQueueSize, largeInt freeMemBaseAddr)
	{
		
		initQueue();
		totalNumPages = availableGPUMemory / PAGE_SIZE;
		initialPageAssignedCounter = 0;
		initialPageAssignedCap = totalNumPages - minimumQueueSize;

		//This has to be allocated GPU-side
		pages = (page_t*) malloc(totalNumPages * sizeof(page_t));
		for(int i = 0; i < totalNumPages; i ++)
		{
			pages[i].paddr = freeMemBaseAddr + i * PAGE_SIZE;
			pages[i].next = NULL;
			pages[i].used = 0;
		}

		//This has to be allocated host-pinned
		pageQueue = (pageQueueItem_t*) malloc(10 * minimumQueueSize * sizeof(pageQueueItem_t));
		memset(pageQueue, 0, 10 * minimumQueueSize * sizeof(pageQueueItem_t));

		this.minimumQueueSize = minimumQueueSize;
		//Adding last free pages to queue
		for(int i = totalNumPages - minimumQueueSize; i < totalNumPages; i ++)
		{
			queue.pushCleanPage(&pages[i]);
		}
	}

	__host__ void initQueue()
	{
		//This has to be allocated as host-pinned
		queue = (pageQueue_t*) malloc(sizeof(pageQueue_t));
		queue->front = 0;
		queue->rear = 0;
		queue->dirtyRear = 0;
		queue->lock = 0;
	}

	//This is called only by CPU thread, so needs no synchronization (unless run by multiple threads)
	__host__ void pushCleanPage(page_t* page)
	{
		queue->pages[queue->rear] = page;
		queue->rear ++;
		queue->rear %= QUEUE_SIZE;
		queue->dirtyRear = queue->rear;
	}

	//TODO: I assume the queue will not be full which is a faulty assumption.
	__device__ void pushDirtyPage(page_t* page)
	{
		//acquire lock
		queue->pages[queue->dirtyRear] = page;
		queue->dirtyRear ++;
		queue->dirtyRear %= QUEUE_SIZE;
		//release lock
	}

	//Run by GPU
	__device__ page_t* popCleanPage()
	{
		//acquire lock
		page_t* page = NULL;
		if(queue->rear != queue->front)
		{
			page = queue->pages[queue->front];
			queue->front ++;
			queue->front %= QUEUE_SIZE;
		}
		//release lock
		return page;

	}

	//This is run only by one CPU, so can be simplified (in terms of synchronizatio)
	__host__ page_t* virtualDirtyPeek()
	{
		page_t* page = NULL;
		if(queue->rear != queue->dirtyRear)
		{
			page = queue->pages[queue->rear];
		}
		return page;
	}

	__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup)
	{
		page_t* parentPage = myGroup->parentPage;

		unsigned oldUsed = atomicAdd(&(parentPage->used), size);
		if(oldUsed < PAGE_SIZE)
		{
			return (void*) (parentPage.paddr + oldUsed);
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

		return (void*) (oldUsed + newPage.paddr);
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
			return queue.atomicPop(minimumQueueSize);
		}
	}
	
	//Freeing the chain pages
	__device__ void revokePage(page_t* page)
	{
		//If parent page is -1, then we have nothing to do
		while(page != NULL)
		{
			queue.pushDirtyPage(page);
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
				//cudaMemcpyAsync(hashtable_base_CPU + page->paddr, hashtable_base_GPU + page->paddr, PAGE_SIZE, cudaMemcpyDeviceToHost, serviceStream);
				//cudaMemsetAsync(hashtable_base_GPU + page->paddr, 0, PAGE_SIZE, serviceStream);
				page->used = 0;
				page->next= NULL;

				//Advancing rear..
				queue->rear ++;
				queue->rear %= QUEUE_SIZE;

			}
		}
	}
	
}


#endif
