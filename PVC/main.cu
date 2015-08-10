#include "global.h"
#include "kernel.cu"

#define TEXTITEMSIZE 1
#define DATAITEMSIZE 1

#define EPOCHCHUNK 16

#define HASHWALKSTEP 133
#define NUMHASHROWS 10000000
#define NUMTHREADS (MAXBLOCKS * BLOCKSIZE)

__device__ inline void addToHashTable(char* word, int wordSize, hashEntry* hashTable, int numBuckets, int hIndex, volatile unsigned int* volatile locks)
{
	int oldValue = 1;
	while(oldValue == 1)
	{
		hashEntry* possibleMatch = &hashTable[hIndex];

		if(possibleMatch->valid == 1)
		{
			if(wordSize == possibleMatch->length && isEqual(word, possibleMatch->word, wordSize))
			{
				possibleMatch->counter ++;
				//atomicExch((unsigned int*) &locks[hIndex], 0);
				oldValue = 0;
			}
			else
			{
				//atomicExch((unsigned int*) &locks[hIndex], 0);
				hIndex = (hIndex+ HASHWALKSTEP) % numBuckets;
				//hIndex = hashFunc(word, wordSize, NUMHASHROWS, hashCounter);
			}
		}
		else
		{
			while(oldValue == 1)
			{
				oldValue = atomicCAS((unsigned int*) &locks[hIndex], 0, 1);
				//oldValue = 0;

				if(oldValue == 0)
				{
					hashEntry* possibleMatch = &hashTable[hIndex];

					if(possibleMatch->valid == 1)
					{
						if(wordSize == possibleMatch->length && isEqual(word, possibleMatch->word, wordSize))
						{
							possibleMatch->counter ++;
							atomicExch((unsigned int*) &locks[hIndex], 0);
						}
						else
						{
							atomicExch((unsigned int*) &locks[hIndex], 0);
							hIndex = (hIndex+ HASHWALKSTEP) % numBuckets;
							//hIndex = hashFunc(word, wordSize, NUMHASHROWS, hashCounter);
							oldValue = 1;
							//hashCounter ++;
						}
					}
					else
					{
						possibleMatch->counter = 1;
						possibleMatch->length = wordSize;
						for(int j = 0; j < wordSize; j ++)
							possibleMatch->word[j] = word[j];
						possibleMatch->valid = 1;

						atomicExch((unsigned int*) &locks[hIndex], 0);
					}

				}
			}	
		}
	}
	

	return;
}


__global__ void wordCountKernelMultipass(
				char* data, 
				largeInt* startIndexes, 
				hashEntry* hashTable, 
				unsigned int* locks,
				ptr_t* textAddrs,
				char* textData,
				int volatile* completeFlag,
				int* gpuFlags,
				strides_t* stridesSpace1,
				firstLastAddr_t* firstLastAddrsSpace1,
				int iterations,
				int epochNum, 
				int* myNumbers,
				int numThreads)
{
	int index = TID2;
	bool prediction = (threadIdx.x < BLOCKSIZE);

	ptr_t start = startIndexes[index];
	ptr_t end = startIndexes[index + 1];

	//int textAddrCounter = (blockIdx.x * BLOCKSIZE + (threadIdx.x / 32) * WARPSIZE) * iterations + (threadIdx.x % 32);
	//int textDataCounter = ((blockIdx.x * BLOCKSIZE + ((threadIdx.x - (blockDim.x / 2)) / WARPSIZE) * WARPSIZE) * iterations) * DATAITEMSIZE + (threadIdx.x % 32) * COPYSIZE;
	int genericCounter;
	
	int flagGPU[3];
	flagGPU[0] = 1;
	flagGPU[1] = 1;
	flagGPU[2] = 1;

	int flagCPU[3];
	flagCPU[0] = 1;
	flagCPU[1] = 1;
	flagCPU[2] = 1;

	char myWord[WORDSIZELIMIT];
	int myWordCounter = 0;
	
	int startWord = start;
	bool inWord = true;


	__shared__ int addrDis1[PATTERNSIZE * BLOCKSIZE];
	__shared__ bool validatedArray[BLOCKSIZE / WARPSIZE];

	int s = 0;

	ptr_t previousAddrSpace1;
	ptr_t firstAddrSpace1;
	int i = start;

	for(int j = 0; i < end; j ++)
	{
		if(prediction && j < epochNum)	
		{

			genericCounter = (blockIdx.x * BLOCKSIZE + (threadIdx.x / 32) * WARPSIZE) * iterations + (threadIdx.x % 32);
			if(threadIdx.x == 0)
				for(int m = 0; m < BLOCKSIZE / WARPSIZE; m ++)
					validatedArray[m] = true;
			__threadfence_block();

			previousAddrSpace1 = 0;
			bool firstTimeSpace1 = true;
			int addrCounterSpace1 = 0;

			int ihistory = i; 
			int loopCounter = 0;
			for(; (loopCounter < iterations) && (i < end); loopCounter ++, i ++)
			{
				ptr_t addr = (ptr_t) &data[i];
				if(addrCounterSpace1 < PATTERNSIZE)
					addrDis1[(threadIdx.x % BLOCKSIZE) * PATTERNSIZE + addrCounterSpace1] = (int) (addr - previousAddrSpace1);

				previousAddrSpace1 = addr;
				addrCounterSpace1 ++;
				if(firstTimeSpace1)
				{
					addrCounterSpace1 --;
					firstTimeSpace1 = false;
					firstAddrSpace1 = previousAddrSpace1;
				}
			}

			int strideSizeSpace1 = findPatternKernel(&addrDis1[(threadIdx.x % BLOCKSIZE) * PATTERNSIZE], PATTERNSIZE);
			bool validated = true;
			int strideCounterSpace1 = 0;
			previousAddrSpace1 = firstAddrSpace1;
			int dataCountSpace1 = 0;

			i = ihistory;
			loopCounter = 0;
			for(; (loopCounter < iterations) && (i < end); loopCounter ++, i ++)
			{
				ptr_t addr = (ptr_t) &data[i];
				dataCountSpace1 ++;
				if(addr != previousAddrSpace1)
					validated = false;
				previousAddrSpace1 += addrDis1[strideCounterSpace1 % strideSizeSpace1];
				strideCounterSpace1 ++;
			}
			(stridesSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].strideSize = strideSizeSpace1;

			if(!validated)
				validatedArray[(threadIdx.x / 32)] = false;
			__threadfence_block();
			(firstLastAddrsSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].firstAddr = firstAddrSpace1;
			(firstLastAddrsSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].lastAddr = previousAddrSpace1;
			(firstLastAddrsSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].itemCount = dataCountSpace1;

			if(validatedArray[(threadIdx.x / 32)])
			{
				bool equalPattern = true;
				for(int j = 0; j < 32; j ++)
				{
					//can be aggressively optimized!!
					for(int m = 0; m < strideSizeSpace1; m ++)
						if(addrDis1[(threadIdx.x % BLOCKSIZE) * PATTERNSIZE + m] != addrDis1[(((threadIdx.x % BLOCKSIZE) / WARPSIZE) * WARPSIZE + j) * PATTERNSIZE + m])
							equalPattern = false;
				}

				if(equalPattern)
				{
					if((threadIdx.x % 32) != 0)
					{
						for(int m = 0; m < strideSizeSpace1; m ++)
							addrDis1[(threadIdx.x % BLOCKSIZE) * PATTERNSIZE + m] = -1;
					}
				}

				for(int m = 0; m < strideSizeSpace1; m ++)
					(stridesSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].strides[m] = addrDis1[(threadIdx.x % BLOCKSIZE) * PATTERNSIZE + m];
				(stridesSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].strideSize = strideSizeSpace1;
			}
			else
			{
				for(int m = 0; m < (stridesSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].strideSize; m ++)
					(stridesSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].strides[m] = -1;

				i = ihistory;
				loopCounter = 0;
				for(; (loopCounter < iterations) && (i < end); loopCounter ++, i ++)
				{
					(textAddrs + (s * (iterations * (blockDim.x / 2) * gridDim.x)))[genericCounter] = (ptr_t) &data[i];
					genericCounter += 32;
				}
			}

		}

		if(prediction)
			asm volatile("bar.sync %0, %1;" ::"r"(4), "r"(blockDim.x / 2)); 


		if(threadIdx.x == 0 && j < epochNum)
		{
			flagGPU[s] *= -1;
			completeFlag[blockIdx.x * 12 + s * 2] = flagGPU[s];
			__threadfence_system();
		}
	

		if(prediction && j < epochNum)
			s = (s + 1) % 3;


		if(!prediction && threadIdx.x == BLOCKSIZE && j > 1)
		{
			flagCPU[s] *= -1;
			volatile int value = 0;
			do
			{
				asm volatile("ld.global.cg.u32 %0, [%1];" :"=r"(value) :"l"(&gpuFlags[blockIdx.x * 6 + s]));
				
			} while(value != flagCPU[s]);
		}

		if(!prediction)
			asm volatile("bar.sync %0, %1;" ::"r"(5), "r"(blockDim.x / 2));

		if(!prediction && j > 1)
		{
			genericCounter = ((blockIdx.x * BLOCKSIZE + ((threadIdx.x - (blockDim.x / 2)) / WARPSIZE) * WARPSIZE) * iterations) * DATAITEMSIZE + (threadIdx.x % 32) * COPYSIZE;
			int step = 0;

			int loopCounter = 0;
			for(; (loopCounter < iterations) && (i < end); loopCounter ++, i ++)
			{
				char c = (textData + (s * iterations * DATAITEMSIZE * (blockDim.x / 2) * gridDim.x))[genericCounter + step];

				step ++;
				genericCounter += (step / COALESCEITEMSIZE) * (WARPSIZE * COALESCEITEMSIZE);
				step %= COALESCEITEMSIZE;

				myNumbers[index] ++;

				if((c < 'a' || c > 'z') && inWord)
				{
					myWordCounter = 0;
					inWord = false;
					int length = i - startWord;
					if(length > 5 && length <= WORDSIZELIMIT)
					{
						unsigned hIndex = hashFunc(myWord, length, NUMHASHROWS, 0);
						addToHashTable(myWord, length, hashTable, NUMHASHROWS, hIndex, locks);
					}
				}
				else if((c >= 'a' && c <= 'z') && !inWord)
				{
					startWord = i;
					inWord = true;
				}

				if(inWord)
					myWord[myWordCounter ++] = c;
			}

			s = (s + 1) % 3;
		
		}

		__syncthreads();
	}
}

int countToNextWord(char* start)
{
	int counter = 0;
	while(start[counter] != ' ')
		counter ++;
	while(start[counter] < 'a' || start[counter] > 'z')
		counter ++;

	return counter;
}

struct timeval global_start[MAXBLOCKS];//, global_end;


void startGlobalTimer(int tid)
{
	gettimeofday(&global_start[tid], NULL);
}

void endGlobalTimer(int tid, char* message)
{
	struct timeval end;
	time_t sec, ms, diff;

	gettimeofday(&end, NULL);
	sec = end.tv_sec - global_start[tid].tv_sec;
	ms = end.tv_usec - global_start[tid].tv_usec;
	diff = sec * 1000000 + ms;

	printf("[%d] %10s:\t\t%0.1fms\n", tid, message, (double)((double)diff/1000.0));
	fflush(stdout);
}

void* copyMethodPattern(void* arg)
{
	copyPackagePattern* pkg = (copyPackagePattern*) arg;

	char* fdata = pkg->srcSpace;
	int myBlock = pkg->myBlock;
	int epochDuration = pkg->epochDuration;
	//strides_t* stridesSpace[1];
	//stridesSpace[0] = pkg->stridesSpace[0];
	//long long unsigned int sourceSpaceSize1 = pkg->sourceSpaceSize1;

	firstLastAddr_t* firstLastAddrsSpace[1];
	firstLastAddrsSpace[0] = pkg->firstLastAddrsSpace[0];

	unsigned warpStart = pkg->warpStart;
	unsigned warpEnd = pkg->warpEnd;

	unsigned spaceDataItemSizes[1];
	spaceDataItemSizes[0] = pkg->spaceDataItemSizes[0];

	char* hostBuffer[1];
	hostBuffer[0] = pkg->hostBuffer[0];

	//for(int k = 0; k < (BLOCKSIZE / WARPSIZE); k ++)
	struct timeval partial_start, partial_end;//, exec_start, exec_end;
	time_t sec;
	time_t ms;
	time_t diff;

	int copies = 0;

	gettimeofday(&partial_start, NULL);

	for(int k = warpStart; k < warpEnd; k ++)
	{
		//strides_t* warpStrides = &(stridesSpace[0][myBlock * BLOCKSIZE + k * WARPSIZE]);
		firstLastAddr_t* warpFirstLastAddrs = &(firstLastAddrsSpace[0][myBlock * BLOCKSIZE + k * WARPSIZE]);
		unsigned int curAddrs;
		//int strideCounter = 0;
		unsigned int offset;

		for(int i = 0; i < WARPSIZE; i ++)
		{
			curAddrs = warpFirstLastAddrs[i].firstAddr;

			//1
			offset = ((myBlock * BLOCKSIZE + k * WARPSIZE) * epochDuration) * spaceDataItemSizes[0] + i * COPYSIZE;

			//offset = ((myBlock * BLOCKSIZE + k * WARPSIZE + i ) * epochDuration) * spaceDataItemSizes[0];
			//offset = ((myBlock * BLOCKSIZE + k * WARPSIZE + i) * epochDuration) * spaceDataItemSizes[0];
			//if(warpFirstLastAddrs[i].itemCount >= epochDuration)
				//printf("########## itemCount: %d epochDuration: %d\n", warpFirstLastAddrs[i].itemCount, epochDuration);
			//assert(warpFirstLastAddrs[i].itemCount <= epochDuration);
			copytype_t* tempSpace = (copytype_t*) &hostBuffer[0][offset];

			//for(int j = 0; j < warpFirstLastAddrs[i].itemCount / COPYSIZE; j ++)
			for(int j = 0; j < (warpFirstLastAddrs[i].itemCount + 7) / COPYSIZE; j ++)
			{
				//hostBuffer[0][offset + j] = fdata[curAddrs + j];
				//copies ++;
				//__builtin_prefetch(&tempSpace[(j + 1) * WARPSIZE], 1, 3); 
				//for(int n = 0; n < spaceDataItemSizes[0] / COPYSIZE; n ++)
				//@@*((copytype_t*) &hostBuffer[0][offset + j * WARPSIZE * COPYSIZE]) = *((copytype_t*) &fdata[(curAddrs + j * COPYSIZE)]);
				//*((copytype_t*) &hostBuffer[0][offset + j * WARPSIZE * COPYSIZE]) = *((copytype_t*) &fdata[(curAddrs + j * COPYSIZE)]);

				//tempSpace[j] = *((copytype_t*) &fdata[(curAddrs + j * COPYSIZE)]);

				//1
				tempSpace[j * WARPSIZE] = *((copytype_t*) &fdata[(curAddrs + j * COPYSIZE)]);

				//curAddrs += warpStrides[0].strides[strideCounter % warpStrides[0].strideSize] * COPYSIZE;
				//offset += spaceDataItemSizes[0] * COPYSIZE;
				//strideCounter ++;
			}

		}
	}

	gettimeofday(&partial_end, NULL);
	sec = partial_end.tv_sec - partial_start.tv_sec;
	ms = partial_end.tv_usec - partial_start.tv_usec;
	diff = sec * 1000000 + ms;

	//printf("\n%10s:\t\t%0.1fms\n", "@@@ my memcpy", (double)((double)diff/1000.0));



	return NULL;
}

void* pipelineData(void* argument)
{

        dataPackage* threadData = (dataPackage*) argument;
	//endGlobalTimer(threadData->myBlock, "@@ Thread creation");

        cudaStream_t* streamPtr = threadData->streamPtr;
        int volatile * volatile flags = threadData->flags;
	int* gpuFlags = threadData->gpuFlags;
	cudaStream_t* execStream = threadData->execStream;

        //ptr_t* textAddrs = threadData->textAddrs;

	//ptr_t* addresses[1];
	//addresses[0] = textAddrs;
	
	char* textData[3];
	textData[0] = threadData->textData[0];
	textData[1] = threadData->textData[1];
	textData[2] = threadData->textData[2];

	char* gpuSpaces[1][3];
	gpuSpaces[0][0] = textData[0];
	gpuSpaces[0][1] = textData[1];
	gpuSpaces[0][2] = textData[2];

        char* fdata = threadData->fdata;
        int myBlock = threadData->myBlock;

        int threadBlockSize = threadData->threadBlockSize;

	unsigned int epochDuration = threadData->epochDuration;

	int textItemSize = threadData->textItemSize;

	char* textHostBuffer[3];
	textHostBuffer[0] = threadData->textHostBuffer[0];
	textHostBuffer[1] = threadData->textHostBuffer[1];
	textHostBuffer[2] = threadData->textHostBuffer[2];

	long long unsigned int sourceSpaceSize1 = threadData->sourceSpaceSize1;

	strides_t* stridesSpace1[3];
	stridesSpace1[0] = threadData->stridesSpace1[0];
	stridesSpace1[1] = threadData->stridesSpace1[1];
	stridesSpace1[2] = threadData->stridesSpace1[2];

	strides_t* stridesSpace[1][3];
	stridesSpace[0][0] = stridesSpace1[0];
	stridesSpace[0][1] = stridesSpace1[1];
	stridesSpace[0][2] = stridesSpace1[2];

	//int* stridesNoSpace1 = threadData->stridesNoSpace1;
	//int* stridesNoSpace2 = threadData->stridesNoSpace2;

	//int* strideNoSpace[2];
	//strideNoSpace[0] = stridesNoSpace1;
	//strideNoSpace[1] = stridesNoSpace2;

	firstLastAddr_t* firstLastAddrsSpace1[3];
	firstLastAddrsSpace1[0] = threadData->firstLastAddrsSpace1[0];
	firstLastAddrsSpace1[1] = threadData->firstLastAddrsSpace1[1];
	firstLastAddrsSpace1[2] = threadData->firstLastAddrsSpace1[2];

	firstLastAddr_t* firstLastAddrsSpace[1][3];
	firstLastAddrsSpace[0][0] = firstLastAddrsSpace1[0];
	firstLastAddrsSpace[0][1] = firstLastAddrsSpace1[1];
	firstLastAddrsSpace[0][2] = firstLastAddrsSpace1[2];

	//int dataItemsPerEpoch[2];
	//dataItemsPerEpoch[0] = creditItems;
	//dataItemsPerEpoch[1] = merchantItems;

	int spaceDataItemSizes[1];
	spaceDataItemSizes[0] = textItemSize;

	char* hostBuffer[1][3];// = (char**) malloc(2 * sizeof(char**));
	hostBuffer[0][0] = textHostBuffer[0];
	hostBuffer[0][1] = textHostBuffer[1];
	hostBuffer[0][2] = textHostBuffer[2];

	int flagGPU[3];
	flagGPU[0] = -1;
	flagGPU[1] = -1;
	flagGPU[2] = -1;
	int flagCPU[3];
	flagCPU[0] = 1;
	flagCPU[1] = 1;
	flagCPU[2] = 1;

	int notDone = 0;

	//mava
	//printf("About entering the while %d\n", myBlock);
	int s = 0;
        //while(notDone < 2)
	while(cudaSuccess != cudaStreamQuery(*execStream))
	{
		if(flags[myBlock * 12 + s * 2] == flagGPU[s])
		{
			//printf("##########inside if, s is %d\n", s);
			if(notDone == 0)
				endGlobalTimer(myBlock, "@@ prediction");
			else
				endGlobalTimer(myBlock, "@@ computation");

			startGlobalTimer(myBlock); //data assembly

			pthread_t copyThreads[COPYTHREADS - 1];
			copyPackagePattern pkg[COPYTHREADS];
			
#if 1
			for(int h = 0; h < COPYTHREADS; h ++)
			{
				unsigned int warpChunk = (BLOCKSIZE / WARPSIZE) / COPYTHREADS;
				assert(warpChunk > 0);
				unsigned int warpStart = warpChunk * h;
				unsigned int warpEnd = warpStart + warpChunk;

				pkg[h].tid = h;
				pkg[h].myBlock = myBlock;
				pkg[h].epochDuration = epochDuration;
				pkg[h].warpStart = warpStart;
				pkg[h].warpEnd = warpEnd;
				pkg[h].spaceDataItemSizes[0] = spaceDataItemSizes[0];
				pkg[h].hostBuffer[0] = hostBuffer[0][s];
				pkg[h].srcSpace = fdata;
				pkg[h].stridesSpace[0] = stridesSpace[0][s];
				pkg[h].firstLastAddrsSpace[0] = firstLastAddrsSpace[0][s];
				pkg[h].sourceSpaceSize1 = sourceSpaceSize1;

				if(h < (COPYTHREADS - 1))
					int rc = pthread_create(&copyThreads[h], NULL, copyMethodPattern, (void*) &pkg[h]);
				else
					copyMethodPattern(&pkg[h]);
			}

			for(int h = 0; h < COPYTHREADS - 1; h ++)
				pthread_join(copyThreads[h], NULL);
#endif

			endGlobalTimer(myBlock, "@@ Assemble data into pinned buffer");
			
			notDone ++;

			startGlobalTimer(myBlock); //copy from  pinned buffer to GPU memory
			cudaMemcpyAsync(&gpuSpaces[0][s][myBlock * threadBlockSize * spaceDataItemSizes[0] * epochDuration], &(hostBuffer[0][s][myBlock * threadBlockSize * epochDuration * spaceDataItemSizes[0]]), epochDuration * threadBlockSize * spaceDataItemSizes[0], cudaMemcpyHostToDevice, *streamPtr);

			//while(cudaSuccess != cudaStreamQuery(*streamPtr));
			endGlobalTimer(myBlock, "@@ Copy from pinned buffer to GPU memory");

			flagCPU[s] *= -1;
			flags[myBlock * 12 + s * 2 + 1] = flagCPU[s];
			flagGPU[s] *= -1;

			asm volatile ("" : : : "memory");
			
			//FIXME: This next cudaMemcpyAsync, in an extreme case might take a while to send the signal to GPU, In such case, it may skip one signal@@
			cudaMemcpyAsync(&gpuFlags[myBlock * 6 + s], (int*) &flags[myBlock * 12 + s * 2 + 1], sizeof(int), cudaMemcpyHostToDevice, *streamPtr);

			//s = (s == 0)? 1 : 0;
			s = (s + 1) % 3;
			startGlobalTimer(myBlock); //computation
		}
	}

	return NULL;
}


int main(int argc, char** argv)
{
	cudaError_t errR;
	cudaThreadExit();

	int fd;
	char *fdata;
	struct stat finfo;
	char *fname;

	if (argc < 2)
	{
		printf("USAGE: %s <inputfilename>\n", argv[0]);
		exit(1);
	}

	fname = argv[1];
	fd = open(fname, O_RDONLY);
	fstat(fd, &finfo);
	printf("Allocating %lluMB for the input file.\n", ((long long unsigned int)finfo.st_size) / (1 << 20));
	fdata = (char *) malloc(finfo.st_size);
	size_t readed = read (fd, fdata, finfo.st_size);
	size_t fileSize = (size_t) finfo.st_size;
	if(readed != fileSize)
		printf("Not all of the file is read\n");


	//================= startIndexes =====================//
	size_t threadChunkSize = fileSize / NUMTHREADS;
	largeInt* startIndexes = (largeInt*) malloc((NUMTHREADS + 1) * sizeof(largeInt));
	//largeInt* startIndexes2 = (largeInt*) malloc((NUMTHREADS + 1) * sizeof(largeInt));

#if 1
	for(int i = 0; i < NUMTHREADS; i ++)
		startIndexes[i] = i * threadChunkSize;

	startIndexes[NUMTHREADS] = (ptr_t) fileSize;

	for(int i = 1; i < NUMTHREADS; i ++)
	{
		int disToNextWhiteSpace = countToNextWord(&fdata[startIndexes[i]]);
		startIndexes[i] += (disToNextWhiteSpace - 1);
	}
#endif

	largeInt* dstartIndexes;
	//largeInt* dstartIndexes2;
	cudaMalloc((void**) &dstartIndexes, (NUMTHREADS + 1) * sizeof(largeInt));
	cudaMemcpy(dstartIndexes, startIndexes, (NUMTHREADS + 1) * sizeof(largeInt), cudaMemcpyHostToDevice);
	//======================================================//
	
	//======================================================//
	int chunkSize = EPOCHCHUNK * (1 << 20);

	int epochNum = (int) (fileSize / chunkSize);
	if(fileSize % chunkSize)
		epochNum ++;

	printf("Number of epochs: %d\n", epochNum);

	//======================================================//

	//=================== Max num of iterations ============//
	int maxIterations = 0;
	for(int i = 0; i < NUMTHREADS; i ++)
		if((startIndexes[i + 1] - startIndexes[i]) > maxIterations)
			maxIterations = startIndexes[i + 1] - startIndexes[i];

	maxIterations ++;
	if(epochNum > 1)
		maxIterations /= (epochNum);
	//======================================================//

	//============== Hash table and locks===================//
	size_t hashTableSize = NUMHASHROWS * sizeof(hashEntry);
	
	hashEntry* dhashTable;
	cudaMalloc((void**) &dhashTable, hashTableSize);
	cudaMemset(dhashTable, 0, hashTableSize);

	unsigned int* dcolumnLocks;
	cudaMalloc((void**) &dcolumnLocks, NUMHASHROWS * sizeof(int));
	cudaMemset(dcolumnLocks, 0, NUMHASHROWS * sizeof(int));
	//======================================================//

	errR = cudaGetLastError();
	printf("Error after allocating memory spaces is: %s\n", cudaGetErrorString(errR));
	if(errR != cudaSuccess)
		exit(1);
	
	dim3 block(BLOCKSIZE, 1, 1);
	dim3 block2((BLOCKSIZE * 2), 1, 1);
	dim3 grid(MAXBLOCKS, 1, 1);
	int numThreads = BLOCKSIZE * grid.x * grid.y;

#if 1

	printf("EpochNum: %d\n", epochNum);

	int iterations = maxIterations;
	if(iterations % 8 != 0)
		iterations += (8 - (iterations % 8));

	//========= merchantAddrHostBuffer ===========//
	unsigned int textAddrsHostBufferSize = sizeof(ptr_t) * iterations * numThreads * 3;
	ptr_t* tempTextAddrHostBuffer;
	tempTextAddrHostBuffer = (ptr_t*) malloc(textAddrsHostBufferSize + MEMORY_ALIGNMENT);
	ptr_t* hostTextAddrHostBuffer;
	hostTextAddrHostBuffer = (ptr_t*) ALIGN_UP(tempTextAddrHostBuffer, MEMORY_ALIGNMENT);
	memset((void*) hostTextAddrHostBuffer, 0, textAddrsHostBufferSize);
	cudaHostRegister((void*) hostTextAddrHostBuffer, textAddrsHostBufferSize, CU_MEMHOSTALLOC_DEVICEMAP);
	ptr_t* textAddrs;
	cudaHostGetDevicePointer((void **)&textAddrs, (void *)hostTextAddrHostBuffer, 0);
	//============================================//	

	//========= creditDataHostBuffer ===========//
	int textHostBufferSize = TEXTITEMSIZE * iterations * numThreads * 3;
	char* tempTextHostBuffer;
	tempTextHostBuffer = (char*) malloc(textHostBufferSize + MEMORY_ALIGNMENT);
	char* hostTextHostBuffer;
	hostTextHostBuffer = (char*) ALIGN_UP(tempTextHostBuffer, MEMORY_ALIGNMENT);
	memset((void*) hostTextHostBuffer, 0, textHostBufferSize);
	cudaHostRegister((void*) hostTextHostBuffer, textHostBufferSize, CU_MEMHOSTALLOC_DEVICEMAP);
	//============================================//	


	//================= completeFlag ===============//
	int flagSize = grid.x * grid.y * 12 * sizeof(int);
	int volatile * volatile tempCompleteFlag = (int*) malloc(flagSize + MEMORY_ALIGNMENT);
	int volatile * volatile hostCompleteFlag = (int*) ALIGN_UP(tempCompleteFlag, MEMORY_ALIGNMENT);
	memset((void*) hostCompleteFlag, 0, flagSize);
	cudaHostRegister((void*) hostCompleteFlag, flagSize, CU_MEMHOSTALLOC_DEVICEMAP);
	int volatile * volatile flags;
	cudaHostGetDevicePointer((void **)&flags, (void *)hostCompleteFlag, 0);

	int* gpuFlags;
	cudaMalloc((void**) &gpuFlags, flagSize / 2);
	cudaMemset(gpuFlags, 0, flagSize / 2);
	//============================================//

	//================= strides ===============//
	int stridesSize = numThreads * sizeof(strides_t) * 3;//16 * sizeof(int) + 2 * sizeof(long long int);
	strides_t* tempStridesSpace1;

	tempStridesSpace1 = (strides_t*) malloc(stridesSize + MEMORY_ALIGNMENT);

	strides_t* hostStridesSpace1;
	hostStridesSpace1 = (strides_t*) ALIGN_UP(tempStridesSpace1, MEMORY_ALIGNMENT);

	memset((void*) hostStridesSpace1, 0, stridesSize);

	cudaHostRegister((void*) hostStridesSpace1, stridesSize, CU_MEMHOSTALLOC_DEVICEMAP);

	strides_t* stridesSpace1;

	cudaHostGetDevicePointer((void **)&stridesSpace1, (void *)hostStridesSpace1, 0);
	//============================================//

	//================= firstLastAddrs ===============//
	int fistLastAddrSize = numThreads * sizeof(firstLastAddr_t) * 3;//16 * sizeof(int) + 2 * sizeof(long long int);
	firstLastAddr_t* tempFirstLastSpace1;
	tempFirstLastSpace1 = (firstLastAddr_t*) malloc(fistLastAddrSize + MEMORY_ALIGNMENT);

	firstLastAddr_t* hostFirstLastSpace1;
	hostFirstLastSpace1 = (firstLastAddr_t*) ALIGN_UP(tempFirstLastSpace1, MEMORY_ALIGNMENT);

	memset((void*) hostFirstLastSpace1, 0, fistLastAddrSize);

	cudaHostRegister((void*) hostFirstLastSpace1, fistLastAddrSize, CU_MEMHOSTALLOC_DEVICEMAP);

	firstLastAddr_t* firstLastAddrsSpace1;

	cudaHostGetDevicePointer((void **)&firstLastAddrsSpace1, (void *)hostFirstLastSpace1, 0);
	//============================================//


	char* textData;
	cudaMalloc((void**) &textData, TEXTITEMSIZE * iterations * numThreads * 3);

	char* phony = (char*) 0x0;
	cudaStream_t execStream;
	cudaStreamCreate(&execStream);
	cudaStream_t copyStream;
	cudaStreamCreate(&copyStream);
	
	int* dmyNumbers;
	cudaMalloc((void**) &dmyNumbers, numThreads * sizeof(int));
	cudaMemset(dmyNumbers, 0, numThreads * sizeof(int));

	struct timeval partial_start, partial_end;//, exec_start, exec_end;
	time_t sec;
	time_t ms;
	time_t diff;

	int* dfinalChar;
	cudaMalloc((void**) &dfinalChar, numThreads * sizeof(int));
	cudaMemset(dfinalChar, 0, numThreads * sizeof(int));


	errR = cudaGetLastError();
	printf("#######Error before calling the kernel is: %s\n", cudaGetErrorString(errR));

	gettimeofday(&partial_start, NULL);

	printf("Calling the kernel...\n");

	//FIXME: one thing that should be thought about is that when iterations is not the same for all threads. What to do then????
	//int sharedMemSize = BLOCKSIZE * PATTERNSIZE * sizeof(int) * 1;

	wordCountKernelMultipass<<<grid, block2, 0, execStream>>>(
			phony, 
			dstartIndexes, 
			dhashTable, 
			dcolumnLocks,
			textAddrs,
			textData,
			flags,
			gpuFlags,
			stridesSpace1,
			firstLastAddrsSpace1,
			iterations,
			epochNum,
			dmyNumbers,
			numThreads
			);

	
	pthread_t threads[MAXBLOCKS];
	dataPackage* argument[MAXBLOCKS];

	for(int m = 0; m < MAXBLOCKS; m ++)
	{
		startGlobalTimer(m);  //prediction
		argument[m] = (dataPackage*) malloc(sizeof(dataPackage));
		argument[m]->streamPtr = &copyStream;
		argument[m]->execStream = &execStream;
		argument[m]->flags = hostCompleteFlag;
		argument[m]->gpuFlags = gpuFlags;
		argument[m]->fdata = fdata;
		argument[m]->myBlock = m;
		argument[m]->threadBlockSize = BLOCKSIZE;
		argument[m]->textItems = iterations;
		argument[m]->textHostBuffer[0] = hostTextHostBuffer;
		argument[m]->textHostBuffer[1] = hostTextHostBuffer + iterations * numThreads * DATAITEMSIZE;
		argument[m]->textHostBuffer[2] = hostTextHostBuffer + iterations * numThreads * DATAITEMSIZE * 2;
		argument[m]->textAddrs[0] = hostTextAddrHostBuffer;
		argument[m]->textAddrs[1] = hostTextAddrHostBuffer + iterations * numThreads;
		argument[m]->textAddrs[2] = hostTextAddrHostBuffer +  iterations * numThreads * 2;
		argument[m]->textData[0] = textData;
		argument[m]->textData[1] = textData + iterations * numThreads * DATAITEMSIZE;
		argument[m]->textData[2] = textData +  iterations * numThreads * DATAITEMSIZE * 2;
		argument[m]->stridesSpace1[0] = hostStridesSpace1;
		argument[m]->stridesSpace1[1] = hostStridesSpace1 + numThreads;
		argument[m]->stridesSpace1[2] = hostStridesSpace1 + numThreads * 2;
		argument[m]->epochDuration = iterations;
		argument[m]->firstLastAddrsSpace1[0] = hostFirstLastSpace1;
		argument[m]->firstLastAddrsSpace1[1] = hostFirstLastSpace1 + numThreads;
		argument[m]->firstLastAddrsSpace1[2] = hostFirstLastSpace1 + numThreads * 2;
		argument[m]->textItemSize = TEXTITEMSIZE;
		argument[m]->sourceSpaceSize1 = fileSize;

		if(m < (MAXBLOCKS - 1))
			pthread_create(&threads[m], NULL, pipelineData, (void*) argument[m]);
		else
			pipelineData(argument[m]);
		//if(rc)
		//{
			//printf("ERROR: could not create the thread %d\n", m);
			//exit(1);
		//}
	}

	//cudaThreadSynchronize();
	while(cudaSuccess != cudaStreamQuery(execStream))
		usleep(300);	


	errR = cudaGetLastError();
	printf("#######Error after calling the kernel is: %s\n", cudaGetErrorString(errR));

	cudaThreadSynchronize();

	for(int m = 0; m < MAXBLOCKS; m ++)
		endGlobalTimer(m, "@@ computation");

	gettimeofday(&partial_end, NULL);
	sec = partial_end.tv_sec - partial_start.tv_sec;
	ms = partial_end.tv_usec - partial_start.tv_usec;
	diff = sec * 1000000 + ms;

	printf("\n%10s:\t\t%0.1fms\n", "Multipass wordcount", (double)((double)diff/1000.0));


	hashEntry* hhashTable = (hashEntry*) malloc(hashTableSize);
	cudaMemcpy(hhashTable, dhashTable, hashTableSize, cudaMemcpyDeviceToHost);
	
	
	unsigned showCounter = 0;

	for(int i = 0; i < hashTableSize / sizeof(hashEntry); i ++)
	{
		if(hhashTable[i].valid == 1)
		{
			if(showCounter < 20)
			{
				printf("%20s", hhashTable[i].word);
				printf("\t count: %d\n", hhashTable[i].counter);
			}
			showCounter ++;
		}
	}
	printf("Number of words: %u\n", showCounter);


	
	int* myNumbers = (int*) malloc(numThreads * sizeof(int));
	cudaMemcpy(myNumbers, dmyNumbers, numThreads * sizeof(int), cudaMemcpyDeviceToHost);

	largeInt totalCount = 0;
	for(int i = 0; i < numThreads; i ++)
		totalCount += myNumbers[i];

	printf("Total number: %lld\n", totalCount);

#endif

	return 0;
}
