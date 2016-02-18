#include "global.h"
#include "hashGlobal.h"
#include "kernel.cu"

#define GB 1073741824

#define WORD_MAX_SIZE 32

#define MAXREAD 2040109465

#define EPOCHCHUNK 15
#define DISPLAY_RESULTS
#define NUM_RESULTS_TO_SHOW 20
#define ESTIMATED_RECORD_SIZE 1024


#define NUMTHREADS (MAXBLOCKS * BLOCKSIZE)

__device__ inline void map(unsigned recordSize, 
		multipassConfig_t* mbk, char* states, unsigned* stateCounter, int* myNumbers, int epochNumber, 
		char* textData, int iCounter, unsigned epochDataSizePerThread, int index);

__device__ inline void emit(void* key, unsigned keySize, void* value, unsigned valueSize,
		 multipassConfig_t* mbk, char* states, unsigned* stateCounter, int* myNumbers, int epochNumber, int index);

__device__ inline char data_in_char(int i, int iCounter, char* textData, unsigned epochDataSizePerThread)
{
	
	i += iCounter;
	unsigned genericCounter = ((blockIdx.x * BLOCKSIZE + ((threadIdx.x - (blockDim.x / 2)) / WARPSIZE) * WARPSIZE) * epochDataSizePerThread) + (threadIdx.x % 32) * COPYSIZE;
	genericCounter += (i / COALESCEITEMSIZE) * (WARPSIZE * COALESCEITEMSIZE);
	int step = i % COALESCEITEMSIZE;
	return textData[genericCounter + step];
}

__device__ inline char data_in_int(int i, int iCounter, char* textData, unsigned epochDataSizePerThread)
{
	i *= sizeof(int);
	i += iCounter;
	unsigned genericCounter = ((blockIdx.x * BLOCKSIZE + ((threadIdx.x - (blockDim.x / 2)) / WARPSIZE) * WARPSIZE) * epochDataSizePerThread) + (threadIdx.x % 32) * COPYSIZE;
	genericCounter += (i / COALESCEITEMSIZE) * (WARPSIZE * COALESCEITEMSIZE);
	int step = i % COALESCEITEMSIZE;
	return *((int*) &textData[genericCounter + step]);
}

#if 0
__device__ inline int data_in_int(int i, unsigned genericCounter)
{
	i *= sizeof(int);
	genericCounter += (i / COALESCEITEMSIZE) * (WARPSIZE * COALESCEITEMSIZE);
	//genericCounter += (i >> 3) * (WARPSIZE * COALESCEITEMSIZE);
	//int step = i & (COALESCEITEMSIZE - 1);
	int step = i % COALESCEITEMSIZE;
	return *((int*) (textData + (s * epochDataSizePerThread * (blockDim.x / 2) * gridDim.x) + genericCounter + step));
}
#endif

__global__ void MapReduceKernelMultipass(
				char* data,
				int numRecords,
				int epochIterations,
				unsigned* recordIndices,
				unsigned* recordSizes,
				ptr_t* deviceAddrs,
				char* textData,
				int volatile* completeFlag,
				int* gpuFlags,
				firstLastAddr_t* firstLastAddrsSpace1,
				int epochNum,
				int numThreads,
				multipassConfig_t* mbk,
				char* states,
				unsigned numStates,
				unsigned epochDataSizePerThread
				)
{
	int index = TID2;
	int* myNumbers = mbk->dmyNumbers;
	bool prediction = (threadIdx.x < BLOCKSIZE);

	__shared__ bool reloop;
	reloop = true;
	__syncthreads();
	int chunkSize = numRecords / numThreads;
	chunkSize = (numRecords % numThreads == 0)? chunkSize : chunkSize + 1;
	int start = index * chunkSize;
	int end = start + chunkSize;
	end = (end > numRecords)? numRecords : end;

	unsigned genericCounter;
	
	int flagGPU[3];
	flagGPU[0] = 1;
	flagGPU[1] = 1;
	flagGPU[2] = 1;

	int flagCPU[3];
	flagCPU[0] = 1;
	flagCPU[1] = 1;
	flagCPU[2] = 1;


	// We don't check if we go out of bound in the stateCounter, basically because we allocate a very
	// high number of states per threads to never overflow. But not a bad idea to think about having a boundary.
	unsigned stateCounter = (numStates / numThreads) * index;
	int s = 0;

	int i = start;

	for(int j = 0; reloop; j ++)
	{
		__syncthreads();
		reloop = false;
#if 0
		if((prediction && j < epochNum && mbk->depochSuccessStatus[j] == (char) 1) || (!prediction && j > 1 && mbk->depochSuccessStatus[j - 2] == (char) 1))
		{
			i += numRecords;
			continue;
		}
#endif

		//lala
		if(prediction && j < epochNum)
		{
			genericCounter = (blockIdx.x * BLOCKSIZE + (threadIdx.x / 32) * WARPSIZE) * epochIterations + (threadIdx.x % 32);

			unsigned predictedDataSize = recordSizes[i];
			int dataCountSpace1 = 0;
			for(; (predictedDataSize < epochDataSizePerThread) && (i < end); i ++)
			{
				addr_size_t temp;
				temp.address = (unsigned) recordIndices[i];
				temp.size = recordSizes[i];
				(deviceAddrs + (s * (epochIterations * (blockDim.x / 2) * gridDim.x)))[genericCounter] = *((ptr_t*) &temp);
				predictedDataSize += recordSizes[i + 1];
				genericCounter += 32;
				dataCountSpace1 ++;
			}

			(firstLastAddrsSpace1 + (s * (blockDim.x / 2) * gridDim.x))[index].itemCount = dataCountSpace1;
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
			genericCounter = ((blockIdx.x * BLOCKSIZE + ((threadIdx.x - (blockDim.x / 2)) / WARPSIZE) * WARPSIZE) * epochDataSizePerThread) + (threadIdx.x % 32) * COPYSIZE;
			unsigned usedDataSize = recordSizes[i];
			int iCounter = 0;
			for(; (usedDataSize < epochDataSizePerThread) && (i < end); i ++)
			{
				char* mapData = (char*) ((largeInt) textData + (s * epochDataSizePerThread * (blockDim.x / 2) * gridDim.x));
				map(recordSizes[i],
					mbk, states, &stateCounter, myNumbers, j,
					mapData, iCounter, epochDataSizePerThread, index);
				usedDataSize += recordSizes[i + 1];
				iCounter += recordSizes[i];
				//The following will make sure each record starts at a new 8-byte aligned location
				if(iCounter % 8 != 0)
					iCounter += (8 - (iCounter % 8));
			}

			s = (s + 1) % 3;
		}

		__syncthreads();
		//If at least one thread has to do the loop again, all should do it as well to avoid sync-lock
		if(i < end)
			reloop = true;
		__syncthreads();
		
	}

}

__device__ inline void map(unsigned recordSize, 
		multipassConfig_t* mbk, char* states, unsigned* stateCounter, int* myNumbers, int epochNumber, 
		char* textData, int iCounter, unsigned epochDataSizePerThread, int index)
{
	largeInt tempWord[WORD_MAX_SIZE/sizeof(largeInt)];
	char* word = (char*) &tempWord[0];
	bool inWord = false;
	unsigned length = 0;
	for(unsigned i = 0; i < recordSize; i ++)
	{

		char c = data_in_char(i, iCounter, textData, epochDataSizePerThread);
#if 1
		if((c < 'a' || c > 'z') && inWord)
		{
			inWord = false;
			if(length >= 4 && length <= WORD_MAX_SIZE)
			{
				//myNumbers[threadIdx.x] = 20;
				largeInt myValue = 1;
				emit((void*) word, length, (void*) &myValue, sizeof(largeInt), 
					mbk, states, stateCounter, myNumbers, epochNumber, index);
			}
		}
		else if((c >= 'a' && c <= 'z') && !inWord)
		{
			inWord = true;
			word[0] = c;
			length = 1;
		}
		else if(inWord)
		{
			word[length] = c;
			length ++;
		}
#endif
	}

	if(inWord)
	{
		if(length >= 5 && length <= WORD_MAX_SIZE)
		{
			largeInt myValue = 1;
			emit((void*) word, length, (void*) &myValue, sizeof(largeInt), 
					mbk, states, stateCounter, myNumbers, epochNumber, index);
		}
	}
}

#if 1
__device__ inline void emit(void* key, unsigned keySize, void* value, unsigned valueSize,
		 multipassConfig_t* mbk, char* states, unsigned* stateCounter, int* myNumbers, int epochNumber, int index)
{

	if(states[*stateCounter] == (char) 0)
	{
		if(addToHashtable(key, keySize, value, valueSize, mbk) == true)
		{
			myNumbers[index * 2] += 1;
			states[*stateCounter] = SUCCEED;
		}
		else
		{
			myNumbers[index * 2 + 1] += 1;
			*(mbk->dfailedFlag) = true; //TODO: provide this
			mbk->depochSuccessStatus[epochNumber - 2] = FAILED;
		}
	}
	*stateCounter += 1;
}
#endif

int countLines(char* input, size_t fileSize)
{

        int numLines = 0;
        for(int i = 0; i < fileSize; i ++)
        {
                if(input[i] == '\n')
                        numLines ++;
        }

        return numLines;
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
#if 0
	struct timeval end;
	time_t sec, ms, diff;

	gettimeofday(&end, NULL);
	sec = end.tv_sec - global_start[tid].tv_sec;
	ms = end.tv_usec - global_start[tid].tv_usec;
	diff = sec * 1000000 + ms;
#endif

	//printf("[%d] %10s:\t\t%0.1fms\n", tid, message, (double)((double)diff/1000.0));
	//fflush(stdout);
}

void* copyMethodPattern(void* arg)
{
	copyPackagePattern* pkg = (copyPackagePattern*) arg;

	char* fdata = pkg->srcSpace;
	int myBlock = pkg->myBlock;
	int maxRecordPerThreadPerEpoch = pkg->maxRecordPerThreadPerEpoch; // this is numRecords
	addr_size_t* hostAddrs = (addr_size_t*) pkg->hostAddrs;
	//strides_t* stridesSpace[1];
	//stridesSpace[0] = pkg->stridesSpace[0];
	//long long unsigned int sourceSpaceSize1 = pkg->sourceSpaceSize1;

	firstLastAddr_t* firstLastAddrsSpace[1];
	firstLastAddrsSpace[0] = pkg->firstLastAddrsSpace[0];

	unsigned warpStart = pkg->warpStart;
	unsigned warpEnd = pkg->warpEnd;

	unsigned spaceDataItemSizes[1];
	spaceDataItemSizes[0] = pkg->spaceDataItemSizes[0];

	char* hostDataBuffer[1];
	hostDataBuffer[0] = pkg->hostBuffer[0];


	for(int k = warpStart; k < warpEnd; k ++)
	{
		firstLastAddr_t* warpFirstLastAddrs = &(firstLastAddrsSpace[0][myBlock * BLOCKSIZE + k * WARPSIZE]);

		for(int i = 0; i < WARPSIZE; i ++)
		{
			unsigned itemCount = warpFirstLastAddrs[i].itemCount;
			//1
			unsigned addressIndex = (myBlock * BLOCKSIZE + k * WARPSIZE) * maxRecordPerThreadPerEpoch + i;

			unsigned dataOffset = (myBlock * BLOCKSIZE + k * WARPSIZE) * spaceDataItemSizes[0] + i * COPYSIZE; //spaceDataItemSizes is epochDataSizePerThread
			copytype_t* tempDataSpace = (copytype_t*) &hostDataBuffer[0][dataOffset];

			//TODO this has to use strides to know what address to load next.
			unsigned storedOffset = 0;
			for(int j = 0; j < itemCount; j ++)
			{
				unsigned address = hostAddrs[addressIndex + j * WARPSIZE].address;
				unsigned size = hostAddrs[addressIndex + j * WARPSIZE].size;
				for(int m = 0; m < (size + (COPYSIZE - 1)) / COPYSIZE; m ++)
				{
					copytype_t t = *((copytype_t*) &fdata[(address + m * COPYSIZE)]);
					tempDataSpace[storedOffset + m * WARPSIZE] = t;
				}
				storedOffset += ((size + (COPYSIZE - 1)) / COPYSIZE) * WARPSIZE;

			}

		}
	}


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

	unsigned int maxRecordPerThreadPerEpoch = threadData->maxRecordPerThreadPerEpoch;

	int textItemSize = threadData->textItemSize;

	char* textHostBuffer[3];
	textHostBuffer[0] = threadData->textHostBuffer[0];
	textHostBuffer[1] = threadData->textHostBuffer[1];
	textHostBuffer[2] = threadData->textHostBuffer[2];

	ptr_t* hostAddrs[3];
	hostAddrs[0] = threadData->hostAddrs[0];
	hostAddrs[1] = threadData->hostAddrs[1];
	hostAddrs[2] = threadData->hostAddrs[2];


	long long unsigned int sourceSpaceSize1 = threadData->sourceSpaceSize1;

	strides_t* stridesSpace1[3];
	stridesSpace1[0] = threadData->stridesSpace1[0];
	stridesSpace1[1] = threadData->stridesSpace1[1];
	stridesSpace1[2] = threadData->stridesSpace1[2];

	strides_t* stridesSpace[1][3];
	stridesSpace[0][0] = stridesSpace1[0];
	stridesSpace[0][1] = stridesSpace1[1];
	stridesSpace[0][2] = stridesSpace1[2];

	firstLastAddr_t* firstLastAddrsSpace1[3];
	firstLastAddrsSpace1[0] = threadData->firstLastAddrsSpace1[0];
	firstLastAddrsSpace1[1] = threadData->firstLastAddrsSpace1[1];
	firstLastAddrsSpace1[2] = threadData->firstLastAddrsSpace1[2];

	firstLastAddr_t* firstLastAddrsSpace[1][3];
	firstLastAddrsSpace[0][0] = firstLastAddrsSpace1[0];
	firstLastAddrsSpace[0][1] = firstLastAddrsSpace1[1];
	firstLastAddrsSpace[0][2] = firstLastAddrsSpace1[2];

	int spaceDataItemSizes[1];
	spaceDataItemSizes[0] = textItemSize;

	char* hostBuffer[1][3];
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
#if 1

			pthread_t copyThreads[COPYTHREADS - 1];
			copyPackagePattern pkg[COPYTHREADS];
			
			for(int h = 0; h < COPYTHREADS; h ++)
			{
				unsigned int warpChunk = (BLOCKSIZE / WARPSIZE) / COPYTHREADS;
				assert(warpChunk > 0);
				unsigned int warpStart = warpChunk * h;
				unsigned int warpEnd = warpStart + warpChunk;

				pkg[h].tid = h;
				pkg[h].myBlock = myBlock;
				pkg[h].maxRecordPerThreadPerEpoch = maxRecordPerThreadPerEpoch;
				pkg[h].warpStart = warpStart;
				pkg[h].warpEnd = warpEnd;
				pkg[h].spaceDataItemSizes[0] = spaceDataItemSizes[0];
				pkg[h].hostBuffer[0] = hostBuffer[0][s];
				pkg[h].srcSpace = fdata;
				pkg[h].stridesSpace[0] = stridesSpace[0][s];
				pkg[h].firstLastAddrsSpace[0] = firstLastAddrsSpace[0][s];
				pkg[h].sourceSpaceSize1 = sourceSpaceSize1;
				pkg[h].hostAddrs = hostAddrs[s];

				if(h < (COPYTHREADS - 1))
					int rc = pthread_create(&copyThreads[h], NULL, copyMethodPattern, (void*) &pkg[h]);
				else
					copyMethodPattern(&pkg[h]);
			}

			for(int h = 0; h < COPYTHREADS - 1; h ++)
				pthread_join(copyThreads[h], NULL);

			endGlobalTimer(myBlock, "@@ Assemble data into pinned buffer");
			
			notDone ++;

			startGlobalTimer(myBlock); //copy from  pinned buffer to GPU memory
			cudaMemcpyAsync(&gpuSpaces[0][s][myBlock * threadBlockSize * spaceDataItemSizes[0]], &(hostBuffer[0][s][myBlock * threadBlockSize * spaceDataItemSizes[0]]), threadBlockSize * spaceDataItemSizes[0], cudaMemcpyHostToDevice, *streamPtr);

			//while(cudaSuccess != cudaStreamQuery(*streamPtr));
			endGlobalTimer(myBlock, "@@ Copy from pinned buffer to GPU memory");
#endif

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

void partitioner(char* data, unsigned size, unsigned* numRecords, unsigned** recordIndices, unsigned** recordSizes)
{
	unsigned estimatedRecordSize = ESTIMATED_RECORD_SIZE;
	*recordIndices = (unsigned*) malloc(((size + estimatedRecordSize) / estimatedRecordSize) * sizeof(unsigned));
	*recordSizes = (unsigned*) malloc(((size + estimatedRecordSize) / estimatedRecordSize) * sizeof(unsigned));

	unsigned recordCounter = 0;
	unsigned sizeCounter = 0;

	while(sizeCounter < size)
	{
		if((sizeCounter + estimatedRecordSize) > size)
		{
			(*recordIndices)[recordCounter] = sizeCounter;
			(*recordSizes)[recordCounter] = size - sizeCounter; 
			recordCounter ++;
			break;
		}

		int i = 0;
		while((sizeCounter + estimatedRecordSize + i) < size && 
				data[sizeCounter + estimatedRecordSize + i] <= 'z' && 
				data[sizeCounter + estimatedRecordSize + i] >= 'a')
		{
			i ++;
		}

		(*recordIndices)[recordCounter] = sizeCounter;
		(*recordSizes)[recordCounter] = estimatedRecordSize + i;
		recordCounter ++;

		sizeCounter += estimatedRecordSize + i;

	}
	
	*numRecords = recordCounter;
}


int main(int argc, char** argv)
{
	cudaError_t errR;
	cudaThreadExit();

	int fd;
        char *fdata;
        struct stat finfo;
        char *fname;

        if(argc < 2)
        {
                printf("USAGE: %s <inputFile>\n", argv[0]);
                exit(1);
        }

        fname = argv[1];

        fd = open(fname, O_RDONLY);
        fstat(fd, &finfo);

        size_t fileSize = (size_t) finfo.st_size;
        printf("Allocating %luMB for the input file.\n", (fileSize) / (1 << 20));
        fdata = (char *) malloc(fileSize);

        largeInt maxReadSize = MAXREAD;
        largeInt successfulRead = 0;
        largeInt toRead = 0;

        if(fileSize > maxReadSize)
        {
                largeInt offset = 0;
                while(offset < fileSize)
                {
                        toRead = (maxReadSize < (fileSize - offset))? maxReadSize : (fileSize - offset);
                        successfulRead += pread(fd, fdata + offset, toRead, offset);
                        printf("read: %lldMB\n", toRead / (1 << 20));
                        offset += toRead;
                }
        }
        else
        {
                successfulRead = read (fd, fdata, fileSize);
        }

        if(successfulRead != fileSize)
        {
                printf("Aborting: not all of the input file was read. Read: %lluMB, total: %luMB\n", successfulRead, fileSize);
                return 1;
        }


	dim3 block(BLOCKSIZE, 1, 1);
	dim3 block2((BLOCKSIZE * 2), 1, 1);
	dim3 grid(MAXBLOCKS, 1, 1);
	int numThreads = BLOCKSIZE * grid.x * grid.y;

	unsigned* recordIndices;
	unsigned* recordSizes;
	unsigned numRecords;

	partitioner(fdata, fileSize, &numRecords, &recordIndices, &recordSizes);

	unsigned* drecordIndices;
	unsigned* drecordSizes;
	cudaMalloc((void**) &drecordIndices, numRecords * sizeof(unsigned));
	cudaMalloc((void**) &drecordSizes, numRecords * sizeof(unsigned));
	cudaMemcpy(drecordIndices, recordIndices, numRecords * sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(drecordSizes, recordSizes, numRecords * sizeof(unsigned), cudaMemcpyHostToDevice);

	printf("Number of records [a map is launched ro process each record]: %d\n", numRecords);

	//======================== Determining the size of epoch chunks ==============================//
	int chunkSize = EPOCHCHUNK * (1 << 20);
	int epochDataSizePerThread = chunkSize / numThreads;
	// Padding epochDataSizePerThread to make sure no one overflows
	epochDataSizePerThread += ESTIMATED_RECORD_SIZE;

	if(epochDataSizePerThread % 8 != 0)
		epochDataSizePerThread += (8 - (epochDataSizePerThread % 8));

	int epochNum = (int) (fileSize / chunkSize);
	if(fileSize % chunkSize)
		epochNum ++;
	printf("Estimated number of epochs: %d\n", epochNum);

	unsigned maxRecordPerThreadPerEpoch = (((numRecords / numThreads) + 1) / epochNum) + 1;
	if(maxRecordPerThreadPerEpoch % 8 != 0)
		maxRecordPerThreadPerEpoch += (8 - (maxRecordPerThreadPerEpoch % 8));

	//============================================================================================//


	//========= AddrHostBuffer ===========//
	unsigned int addrsHostBufferSize = sizeof(ptr_t) * maxRecordPerThreadPerEpoch * numThreads * 3;
	ptr_t* tempAddrHostBuffer;
	tempAddrHostBuffer = (ptr_t*) malloc(addrsHostBufferSize + MEMORY_ALIGNMENT);
	ptr_t* hostAddrHostBuffer;
	hostAddrHostBuffer = (ptr_t*) ALIGN_UP(tempAddrHostBuffer, MEMORY_ALIGNMENT);
	memset((void*) hostAddrHostBuffer, 0, addrsHostBufferSize);
	cudaHostRegister((void*) hostAddrHostBuffer, addrsHostBufferSize, CU_MEMHOSTALLOC_DEVICEMAP);
	ptr_t* deviceAddrs;
	cudaHostGetDevicePointer((void **)&deviceAddrs, (void *)hostAddrHostBuffer, 0);
	//============================================//	

	//========= DataHostBuffer ===========//
	int textHostBufferSize = epochDataSizePerThread * numThreads * 3;
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
	int stridesSize = numThreads * sizeof(strides_t) * 3;
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
	cudaMalloc((void**) &textData, epochDataSizePerThread * numThreads * 3);

	char* phony = (char*) 0x0;
	cudaStream_t execStream;
	cudaStreamCreate(&execStream);
	cudaStream_t copyStream;
	cudaStreamCreate(&copyStream);
	
	//============ initializing the hash table and page table ==================//
	int pagePerGroup = 50;
	int numStates = numRecords * (ESTIMATED_RECORD_SIZE / 4); //conservatively, assumeing 4-byte per words
	multipassConfig_t* mbk = initMultipassBookkeeping(	(int*) hostCompleteFlag, 
								gpuFlags, 
								flagSize,
								numThreads,
								epochNum,
								numStates,
								pagePerGroup);
	multipassConfig_t* dmbk;
	cudaMalloc((void**) &dmbk, sizeof(multipassConfig_t));
	cudaMemcpy(dmbk, mbk, sizeof(multipassConfig_t), cudaMemcpyHostToDevice);

	//==========================================================================//

	struct timeval partial_start, partial_end, bookkeeping_start, bookkeeping_end, passtime_start, passtime_end;
	time_t sec;
	time_t ms;
	time_t diff;

	errR = cudaGetLastError();
	printf("#######Error before calling the kernel is: %s\n", cudaGetErrorString(errR));

	gettimeofday(&partial_start, NULL);
	int passNo = 1;
	bool failedFlag = false;
	do
	{
		printf("====================== starting pass %d ======================\n", passNo);
		gettimeofday(&passtime_start, NULL);

		MapReduceKernelMultipass<<<grid, block2, 0, execStream>>>(
				phony,
				numRecords,
				maxRecordPerThreadPerEpoch,
				drecordIndices,
				drecordSizes,
				deviceAddrs,
				textData,
				flags,
				gpuFlags,
				firstLastAddrsSpace1,
				epochNum,
				numThreads,
				dmbk,
				mbk->dstates,
				numStates,
				epochDataSizePerThread
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
			argument[m]->textItems = maxRecordPerThreadPerEpoch;
			argument[m]->textHostBuffer[0] = hostTextHostBuffer;
			argument[m]->textHostBuffer[1] = hostTextHostBuffer + epochDataSizePerThread * numThreads;
			argument[m]->textHostBuffer[2] = hostTextHostBuffer + epochDataSizePerThread * numThreads * 2;
			argument[m]->hostAddrs[0] = hostAddrHostBuffer;
			argument[m]->hostAddrs[1] = hostAddrHostBuffer + maxRecordPerThreadPerEpoch * numThreads;
			argument[m]->hostAddrs[2] = hostAddrHostBuffer +  maxRecordPerThreadPerEpoch * numThreads * 2;
			argument[m]->textData[0] = textData;
			argument[m]->textData[1] = textData + epochDataSizePerThread * numThreads;
			argument[m]->textData[2] = textData +  epochDataSizePerThread * numThreads * 2;
			argument[m]->maxRecordPerThreadPerEpoch = maxRecordPerThreadPerEpoch;
			argument[m]->firstLastAddrsSpace1[0] = hostFirstLastSpace1;
			argument[m]->firstLastAddrsSpace1[1] = hostFirstLastSpace1 + numThreads;
			argument[m]->firstLastAddrsSpace1[2] = hostFirstLastSpace1 + numThreads * 2;
			argument[m]->textItemSize = epochDataSizePerThread;
			argument[m]->sourceSpaceSize1 = fileSize;

			pthread_create(&threads[m], NULL, pipelineData, (void*) argument[m]);
		}


		//while(cudaSuccess != cudaStreamQuery(execStream))
		while(cudaErrorNotReady == cudaStreamQuery(execStream))
			usleep(300);	


		errR = cudaGetLastError();
		printf("Error after calling the kernel is: %s\n", cudaGetErrorString(errR));

		cudaThreadSynchronize();

		gettimeofday(&passtime_end, NULL);
		sec = passtime_end.tv_sec - passtime_start.tv_sec;
		ms = passtime_end.tv_usec - passtime_start.tv_usec;
		diff = sec * 1000000 + ms;
		printf("\n%10s:\t\t%0.1fms\n", "Pass time ", (double)((double)diff/1000.0));

		for(int m = 0; m < MAXBLOCKS; m ++)
			endGlobalTimer(m, "@@ computation");


		//======================= Some reseting ===========================
		

		gettimeofday(&bookkeeping_start, NULL);
		// Doing the bookkeeping
		failedFlag = checkAndResetPass(mbk, dmbk);
		
		
		gettimeofday(&bookkeeping_end, NULL);
		sec = bookkeeping_end.tv_sec - bookkeeping_start.tv_sec;
		ms = bookkeeping_end.tv_usec - bookkeeping_start.tv_usec;
		diff = sec * 1000000 + ms;
		printf("\n%10s:\t\t%0.1fms\n", "Pass bookkeeping", (double)((double)diff/1000.0));


		passNo ++;

	} while(failedFlag);

	gettimeofday(&partial_end, NULL);
	sec = partial_end.tv_sec - partial_start.tv_sec;
	ms = partial_end.tv_usec - partial_start.tv_usec;
	diff = sec * 1000000 + ms;
	printf("\n%10s:\t\t%0.1fms\n", "Total time", (double)((double)diff/1000.0));

	hashBucket_t** buckets = (hashBucket_t**) malloc(NUM_BUCKETS * sizeof(hashBucket_t*));
	cudaMemcpy(buckets, mbk->buckets, NUM_BUCKETS * sizeof(hashBucket_t*), cudaMemcpyDeviceToHost);

#ifdef DISPLAY_RESULTS
	int topScores[NUM_RESULTS_TO_SHOW];
	int topScoreIds[NUM_RESULTS_TO_SHOW];
	int topScoreTab[NUM_RESULTS_TO_SHOW];
	memset(topScores, 0, NUM_RESULTS_TO_SHOW * sizeof(int));
	memset(topScoreIds, 0, NUM_RESULTS_TO_SHOW * sizeof(int));
	memset(topScoreTab, 0, NUM_RESULTS_TO_SHOW * sizeof(int));

	int tabCount = 0;
	for(int i = 0; i < NUM_BUCKETS; i ++)
	{
		hashBucket_t* bucket = buckets[i];

		while(bucket != NULL)
		{
			int* value = (int*) getValue(bucket);
			for(int j = 0; j < NUM_RESULTS_TO_SHOW; j ++)
			{
				if(*value > topScores[j])
				{
					for(int m = (NUM_RESULTS_TO_SHOW - 1); m >= j && m > 0; m --)
					{
						topScores[m] = topScores[m - 1]; //what if m is 0?
						topScoreIds[m] = topScoreIds[m - 1];
						topScoreTab[m] = topScoreTab[m - 1];
					}
					topScores[j] = *value;
					topScoreIds[j] = i;
					topScoreTab[j] = tabCount;
					break;
				}
			}
			
			bucket = bucket->next;

			tabCount ++;
		}
		tabCount = 0;

	}
	printf("Top %d scores:\n", NUM_RESULTS_TO_SHOW);
	for(int i = 0; i < NUM_RESULTS_TO_SHOW; i ++)
	{
		hashBucket_t* bucket = buckets[topScoreIds[i]];
		for(int j = 0; j < topScoreTab[i]; j ++)
			bucket = bucket->next;

		printf("ID: %d, ", topScoreIds[i]);
		char* word = (char*) getKey(bucket);
		int* value = (int*) getValue(bucket);
		for(int j = 0; j < bucket->keySize; j ++)
			printf("%c", word[j]);
		printf(": %d\n", *value);
	}
#endif

	int totalDepth = 0;
	int totalValidBuckets = 0;
	int totalEmpty = 0;
	int maximumDepth = 0;
	for(int i = 0; i < NUM_BUCKETS; i ++)
	{
		hashBucket_t* bucket = buckets[i];
		if(bucket == NULL)
			totalEmpty ++;
		else
			totalValidBuckets ++;

		int localMaxDepth = 0;
		while(bucket != NULL)
		{
			totalDepth ++;
			localMaxDepth ++;
			bucket = bucket->next;
		}
		if(localMaxDepth > maximumDepth)
			maximumDepth = localMaxDepth;
	}

	float emptyPercentage = ((float) totalEmpty / (float) NUM_BUCKETS) * (float) 100;
	float averageDepth = (float) totalDepth / (float) totalValidBuckets;
	printf("Empty percentage: %0.1f\n", emptyPercentage);
	printf("Average depth: %0.1f\n", averageDepth);
	printf("Max depth: %d\n", maximumDepth);
	printf("Total valid buckets: %d\n", totalValidBuckets);

	return 0;
}
