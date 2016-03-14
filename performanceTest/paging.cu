#include "hashGlobal.h"

void initPaging(largeInt availableGPUMemory, multipassConfig_t* mbk)
{

	mbk->totalNumPages = availableGPUMemory / PAGE_SIZE;
	printf("@INFO: total number of pages: %d [each %dKB]\n", mbk->totalNumPages, (PAGE_SIZE / (1 << 10)));
	mbk->initialPageAssignedCounter = 0;
	mbk->totalNumFreePages = mbk->totalNumPages;
	mbk->hfreeListId = (int*) malloc(mbk->totalNumPages * sizeof(int));
	for(int i = 0; i < mbk->totalNumPages; i ++)
		mbk->hfreeListId[i] = i;

	cudaMalloc((void**) &(mbk->freeListId), mbk->totalNumPages * sizeof(int));
	cudaMemcpy(mbk->freeListId, mbk->hfreeListId, mbk->totalNumPages * sizeof(int), cudaMemcpyHostToDevice);
	

	cudaMalloc((void**) &(mbk->dbuffer), mbk->totalNumPages * PAGE_SIZE);
	cudaMemset(mbk->dbuffer, 0, mbk->totalNumPages * PAGE_SIZE);
	printf("@INFO: done allocating base buffer in GPU memory\n");

	//This has to be allocated GPU-side
	mbk->hpages = (page_t*) malloc(mbk->totalNumPages * sizeof(page_t));
	for(int i = 0; i < mbk->totalNumPages; i ++)
	{
		mbk->hpages[i].id = i;
		mbk->hpages[i].next = NULL;
		mbk->hpages[i].used = 0;
		mbk->hpages[i].needed = 0;
		mbk->hpages[i].hashTableOffset = mbk->hashTableOffset;
	}
	printf("@INFO: done initializing pages meta data\n");
	cudaMalloc((void**) &(mbk->pages), mbk->totalNumPages * sizeof(page_t));
	cudaMemcpy(mbk->pages, mbk->hpages, mbk->totalNumPages * sizeof(page_t), cudaMemcpyHostToDevice);

	printf("@INFO: done doing initPaging\n");
}



//TODO: currently we don't mark a bucket group to not ask for more memory if it previously revoked its pages
__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup, multipassConfig_t* mbk)
{
	page_t* parentPage = myGroup->parentPage;

	unsigned oldUsed = 0;
	if(parentPage != NULL)
	{
		oldUsed = atomicAdd(&(parentPage->used), size);
		if((oldUsed + size) < PAGE_SIZE)
		{
			return (void*) ((largeInt) mbk->dbuffer + parentPage->id * PAGE_SIZE + oldUsed);
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
					return (void*) ((largeInt) mbk->dbuffer + parentPage->id * PAGE_SIZE + oldUsed);
				}
			}
			
			newPage = allocateNewPage(mbk);

			//If no more page exists and no page is used yet (for this bucketgroup), don't do anything
			if(newPage == NULL)
			{
				//releaseLock
				atomicExch(&(myGroup->pageLock), 0);
				myGroup->overflownKey = 1;
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

	if((oldUsed + size) < PAGE_SIZE)
		return (void*) ((largeInt) mbk->dbuffer + oldUsed + newPage->id * PAGE_SIZE);
	else
	{
		return NULL;
	}
}

__device__ void* multipassMallocValue(unsigned size, bucketGroup_t* myGroup, multipassConfig_t* mbk)
{
	page_t* parentPage = myGroup->valueParentPage;

	unsigned oldUsed = 0;
	if(parentPage != NULL)
	{
		oldUsed = atomicAdd(&(parentPage->used), size);
		if((oldUsed + size) < PAGE_SIZE)
		{
			return (void*) ((largeInt) mbk->dbuffer + parentPage->id * PAGE_SIZE + oldUsed);
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
			parentPage = myGroup->valueParentPage;
			if(parentPage != NULL)
			{
				oldUsed = atomicAdd(&(parentPage->used), size);
				if((oldUsed + size) < PAGE_SIZE)
				{
					//Unlocking
					atomicExch(&(myGroup->pageLock), 0);
					return (void*) ((largeInt) mbk->dbuffer + parentPage->id * PAGE_SIZE + oldUsed);
				}
			}
			
			newPage = allocateNewPage(mbk);

			//If no more page exists and no page is used yet (for this bucketgroup), don't do anything
			if(newPage == NULL)
			{
				//releaseLock
				atomicExch(&(myGroup->pageLock), 0);
				myGroup->overflownValue = 1;
				return NULL;
			}

			newPage->next = parentPage;
			myGroup->valueParentPage = newPage;

			//Unlocking
			atomicExch(&(myGroup->pageLock), 0);
		}

	} while(oldLock == 1);

	//This assumes that the newPage is not already full, which is to be tested.
	oldUsed = atomicAdd(&(newPage->used), size);

	if((oldUsed + size) < PAGE_SIZE)
		return (void*) ((largeInt) mbk->dbuffer + oldUsed + newPage->id * PAGE_SIZE);
	else
	{
		return NULL;
	}
}

__device__ page_t* allocateNewPage(multipassConfig_t* mbk)
{
	int indexToAllocate = atomicInc((unsigned*) &(mbk->initialPageAssignedCounter), INT_MAX);
	if(indexToAllocate < mbk->totalNumFreePages)
	{
		return &(mbk->pages[mbk->freeListId[indexToAllocate]]);
	}
	return NULL;
}


