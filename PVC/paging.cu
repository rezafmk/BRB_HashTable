#include "hashGlobal.h"

void initPaging(largeInt availableGPUMemory, pagingConfig_t* pconfig)
{

	pconfig->totalNumPages = availableGPUMemory / PAGE_SIZE;
	printf("@INFO: total number of pages: %d [each %dKB]\n", pconfig->totalNumPages, (PAGE_SIZE / (1 << 10)));
	pconfig->initialPageAssignedCounter = 0;
	pconfig->initialPageAssignedCap = pconfig->totalNumPages;

	cudaMalloc((void**) &(pconfig->dbuffer), pconfig->totalNumPages * PAGE_SIZE);
	cudaMemset(pconfig->dbuffer, 0, pconfig->totalNumPages * PAGE_SIZE);
	printf("@INFO: done allocating base buffer in GPU memory\n");

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



//TODO: currently we don't mark a bucket group to not ask for more memory if it previously revoked its pages
__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup, pagingConfig_t* pconfig, int groupNo)
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
			
			newPage = allocateNewPage(pconfig, groupNo);

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

	if((oldUsed + size) < PAGE_SIZE)
		return (void*) ((largeInt) pconfig->dbuffer + oldUsed + newPage->id * PAGE_SIZE);
	else
	{
		return NULL;
	}
}

__device__ page_t* allocateNewPage(pagingConfig_t* pconfig, int groupNo)
{
	int pageIdToAllocate = atomicInc((unsigned*) &(pconfig->initialPageAssignedCounter), INT_MAX);
	if(pageIdToAllocate < pconfig->totalNumPages)
	{
		return &(pconfig->pages[pageIdToAllocate]);
	}
	return NULL;
}


