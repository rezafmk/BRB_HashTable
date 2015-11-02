#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <ctype.h>
#include <fcntl.h>
#include <cuda.h>
#include <pthread.h>
#include <linux/wait.h>
#include <linux/sched.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

#define MAXFIELDSIZE (2 * 1024) 
#define O_RDONLY             00
#define PATTERNSIZE 4
#define WARPSIZE 32


#define MAXBLOCKS 16
#define BLOCKSIZE 512
#define NUMTHREADS (MAXBLOCKS * BLOCKSIZE)

//#define COAL 1
#define EPOCHNUM 1

#define BLOCK_ID (gridDim.y * blockIdx.x + blockIdx.y)
#define THREAD_ID (threadIdx.x)
#define TID (BLOCK_ID * blockDim.x + THREAD_ID)
#define TID2 (BLOCK_ID * BLOCKSIZE + (threadIdx.x % BLOCKSIZE))

#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )
#define ALIGN_ADDRESS(x,size) {int remaining = x % size; x += (size - remaining);}

#define WORDSIZELIMIT 16

//================ Data assembly fine tuning definitions =======================//
#define COPYSIZE 8
#define COPYTHREADS 4
#define COALESCEITEMSIZE COPYSIZE
//=============================================================================//


typedef long long unsigned int ptr_t;
typedef long long int largeInt;
typedef long long int copytype_t;

extern __shared__ char sbuf[];

typedef struct
{
	char word[24];
	int length;
	int counter;
	int numCollisions;
	int valid;
} hashEntry;


//extern "C" __device__ int addToHashTable(char* word, int length, hashEntry* hashTable, volatile unsigned int* volatile  locks);

typedef struct
{
	int strides[16];
	int strideSize;
} strides_t;

typedef struct
{
	ptr_t firstAddr;
	ptr_t lastAddr;
	int itemCount;
} firstLastAddr_t;


typedef struct
{
	cudaStream_t* streamPtr;
        int volatile * volatile flags;
        ptr_t* textAddrs[3];
	char* textData[3];
        char* fdata;
        int myBlock;
	int threadBlockSize;
	int textItemSize;
	unsigned int textItems;
	char* textHostBuffer[3];
	strides_t* stridesSpace1[3];
	int* stridesNoSpace1[3];
	unsigned int iterations;
	firstLastAddr_t* firstLastAddrsSpace1[3];
	long long unsigned int sourceSpaceSize1;
	int* gpuFlags;
	cudaStream_t* execStream;


} dataPackage;

//spaceGroup_t can potentially include all of the spaces. The number of elements in it is defined
//by compiler: the number of memory accesses in each iteration of the loop within kernel
typedef struct
{
	firstLastAddr_t* firstLastAddrs[2];
	strides_t* strides[2];
	char* sourceSpaces[2];
	char* destSpaces[2];
	char* gpuSpaces[2];
	int counter;
} spaceGroup_t;

typedef struct
{
	unsigned int myBlock;
	unsigned int dataSize;
	unsigned int iterations;
	unsigned int loopStart;
	unsigned int loopEnd;
	ptr_t* addresses[2];
	char* hostBuffer[2];
	char* srcSpace;
	unsigned int spaceDataItemSizes[2];

} copyPackage;

typedef struct
{
	int tid;
	unsigned int myBlock;
	strides_t* stridesSpace[4];
	firstLastAddr_t* firstLastAddrsSpace[4];
	unsigned warpStart;
	unsigned warpEnd;
	unsigned int spaceDataItemSizes[4];
	char* hostBuffer[4];
        ptr_t* textAddrs;
	char* srcSpace;
	unsigned int iterations;
	long long unsigned int sourceSpaceSize1;

} copyPackagePattern;

typedef struct
{
	char* src;
	char* dest;
	unsigned size;
	int myId;
	int numThreads;
	
} simpleCopyPackage;

typedef struct
{
	unsigned address;
	unsigned size;
} addr_size_t;

#endif
