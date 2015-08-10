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
#define WORDSIZELIMIT 16
#define MAXREAD 2040109465
#define CHUNKSIZE 400 
#define NUMTHREADS (MAXBLOCKS * BLOCKSIZE)

//===================== GPU-based impl detials ====================//
#define MAXBLOCKS 16
#define BLOCKSIZE 256
#define BLOCK_ID (gridDim.y * blockIdx.x + blockIdx.y)
#define THREAD_ID (threadIdx.x)
#define TID (BLOCK_ID * blockDim.x + THREAD_ID)
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )
#define ALIGN_ADDRESS(x,size) {int remaining = x % size; x += (size - remaining);}
//=================================================================//


typedef long long unsigned int ptr_t;
typedef long long int largeInt;
typedef long long int copytype_t;

extern __shared__ char sbuf[];

typedef struct
{
	char url[128];
	int length;
	int counter;
	int numCollisions;
	int valid;
} hashEntry;


#endif
