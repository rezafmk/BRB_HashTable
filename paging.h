
__host__ void initPaging(int hashBuckets, largeInt availableGPUMemory, int minimumQueueSize, largeInt freeMemBaseAddr, pagingConfig_t* pconfig);
__host__ void initQueue(pagingConfig_t* pconfig);
__host__ void pushCleanPage(page_t* page, pagingConfig_t* pconfig);
__device__ void pushDirtyPage(page_t* page, pagingConfig_t* pconfig);
__device__ page_t* popCleanPage(pagingConfig_t* pconfig);
__host__ page_t* peekDirtyPage(pagingConfig_t* pconfig);
__device__ void* multipassMalloc(unsigned size, bucketGroup_t* myGroup, pagingConfig_t* pconfig);
page_t* allocateNewPage(pagingConfig_t* pconfig);
__device__ void revokePage(page_t* page);
__host__ void pageRecycler(pagingConfig_t* pconfig, cudaStream_t* serviceStream);
