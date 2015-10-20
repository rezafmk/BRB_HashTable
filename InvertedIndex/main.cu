#include "global.h"
#include "hashGlobal.h"
#include "kernel.cu"
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#define GB 1073741824

#define RECORD_SIZE 1

#define MAXREAD 2040109465

#define EPOCHCHUNK 30
#define NUM_RESULTS_TO_SHOW 20

#define NUMTHREADS (MAXBLOCKS * BLOCKSIZE)
#define DISPLAY_RESULTS
#define URL_SIZE 128

#define START           0x00 
#define IN_TAG          0x01 
#define IN_ATAG         0x02 
#define FOUND_HREF      0x03 
#define START_LINK      0x04


typedef struct flist{
   char *data;
   char *name;
   long long int fd; 
   long long int size;
} filelist_t;

typedef struct
{
	largeInt startOffset;
	largeInt endOffset;
	char name[72];
	largeInt nameSize;
} fileName_t;



long long unsigned int totalSize = 0;
long long unsigned int curOffset = 0;
int count = 0;
int count2 = 0;

void getFilesSize(char* path)
{
	DIR* dir;
	struct dirent *ent;

	if((dir=opendir(path)) != NULL)
	{
		while (( ent = readdir(dir)) != NULL)
		{
			if(ent->d_type == DT_REG)
			{
				char* newPath2 = (char*) malloc(strlen(path) + strlen(ent->d_name) + 2);
				strcpy(newPath2, path);
				strcat(newPath2, "/");
				strcat(newPath2, ent->d_name);

				struct stat finfo;
				int fd = open(newPath2, O_RDONLY);
				fstat(fd, &finfo);
				totalSize += (finfo.st_size + 1);

				close(fd);
				free(newPath2);

				count ++;
			}
				//printf("%s\n", ent->d_name);
				
			if(ent->d_type == DT_DIR && strcmp(ent->d_name, ".") != 0  && strcmp(ent->d_name, "..") != 0)
			{
				//printf("%s\n", ent->d_name);
				char* newPath = (char*) malloc(strlen(path) + strlen(ent->d_name) + 2);
				strcpy(newPath, path);
				strcat(newPath, "/");
				strcat(newPath, ent->d_name);
				getFilesSize(newPath);
				free(newPath);

			}
			//else if(ent->d_type == DT_REG && strcmp(ent->d_name, ".") != 0  && strcmp(ent->d_name, "..") != 0)
			//{
				//printf("%s\n", ent->d_name);
			//}
			
		}

		closedir(dir);
	}
}

void getDataFiles(char* path, char* fdata, unsigned* offsets, fileName_t* fileNames)
{
	DIR* dir;
	struct dirent *ent;

	if((dir=opendir(path)) != NULL)
	{
		while (( ent = readdir(dir)) != NULL)
		{
			if(ent->d_type == DT_REG)
			{
				char* newPath2 = (char*) malloc(strlen(path) + strlen(ent->d_name) + 2);
				strcpy(newPath2, path);
				strcat(newPath2, "/");
				strcat(newPath2, ent->d_name);


				struct stat finfo;
				int fd = open(newPath2, O_RDONLY);
				fstat(fd, &finfo);


				fileNames[count2].startOffset = curOffset;
				fileNames[count2].endOffset = curOffset + finfo.st_size;
				for(int i = 0; i < strlen(ent->d_name); i ++)
					fileNames[count2].name[i] = ent->d_name[i];
				
				fileNames[count2].nameSize = strlen(ent->d_name);

				offsets[count2] = curOffset;
				count2 ++;
				read(fd, fdata + curOffset, finfo.st_size);
				//There should be another data structure in which we store the name of the file. And somehow correspond the conent..
				curOffset += finfo.st_size;
				fdata[curOffset] = '\0';
				curOffset ++;

				close(fd);
				free(newPath2);
			}
				//printf("%s\n", ent->d_name);
				
			if(ent->d_type == DT_DIR && strcmp(ent->d_name, ".") != 0  && strcmp(ent->d_name, "..") != 0)
			{
				//printf("%s\n", ent->d_name);
				char* newPath = (char*) malloc(strlen(path) + strlen(ent->d_name) + 2);
				strcpy(newPath, path);
				strcat(newPath, "/");
				strcat(newPath, ent->d_name);
				getDataFiles(newPath, fdata, offsets, fileNames);
				free(newPath);

			}
			//else if(ent->d_type == DT_REG && strcmp(ent->d_name, ".") != 0  && strcmp(ent->d_name, "..") != 0)
			//{
				//printf("%s\n", ent->d_name);
			//}
			
		}

		closedir(dir);
	}
}



__global__ void invertedIndexKernelMultipass(
				char* data, 
				fileName_t* fileNames,
				unsigned numFiles,
				unsigned numRecords,
				ptr_t* textAddrs,
				char* textData,
				int volatile* completeFlag,
				int* gpuFlags,
				strides_t* stridesSpace1,
				firstLastAddr_t* firstLastAddrsSpace1,
				int iterations,
				int epochNum, 
				int numThreads,
				multipassConfig_t* mbk,
				char* states,
				int passNo,
				char* urlBuffer,
				unsigned urlBufferSizePerThread,
				int* urlSizes,
				int numUrlPerThread
				)
{
	int index = TID2;
	bool prediction = (threadIdx.x < BLOCKSIZE);
	int* myNumbers = mbk->dmyNumbers;
	bool* failedFlag = mbk->dfailedFlag;
	char* epochSuccessStatus = mbk->depochSuccessStatus;

	unsigned recordChunkSize = (fileNames[numFiles - 1].endOffset - fileNames[0].startOffset) / numThreads;
	if(recordChunkSize % numThreads != 0)
		recordChunkSize ++;

	unsigned start = index * recordChunkSize;
	unsigned end = start + recordChunkSize;
	end = (end > fileNames[numFiles - 1].endOffset)? fileNames[numFiles - 1].endOffset : end;

	int fileInUse = 0;
	for(int i = 0; i < numFiles; i ++)
	{
		if(start < fileNames[i].endOffset)
		{
			fileInUse = i;
			break;
		}
	}

	__syncthreads();

	int genericCounter;
	
	int flagGPU[3];
	flagGPU[0] = 1;
	flagGPU[1] = 1;
	flagGPU[2] = 1;

	int flagCPU[3];
	flagCPU[0] = 1;
	flagCPU[1] = 1;
	flagCPU[2] = 1;

	__shared__ int addrDis1[PATTERNSIZE * BLOCKSIZE];
	__shared__ bool validatedArray[BLOCKSIZE / WARPSIZE];

	int s = 0;

	ptr_t previousAddrSpace1;
	ptr_t firstAddrSpace1;
	unsigned storedUrlOffset = urlBufferSizePerThread * index;
	unsigned numStoredUrls = numUrlPerThread * index;

	int stateCounter = (numRecords / numThreads) * index;
	int state = START;
	unsigned i = start;
	for(unsigned j = 0; i < end; j ++)
	{
		if((prediction && j < epochNum && epochSuccessStatus[j] == (char) 1) || (!prediction && j > 1 && epochSuccessStatus[j - 2] == (char) 1))
		{
			i += iterations;
			continue;
		}

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
				for(int k = 0; k < 32; k ++)
				{
					//can be aggressively optimized!!
					for(int m = 0; m < strideSizeSpace1; m ++)
						if(addrDis1[(threadIdx.x % BLOCKSIZE) * PATTERNSIZE + m] != addrDis1[(((threadIdx.x % BLOCKSIZE) / WARPSIZE) * WARPSIZE + k) * PATTERNSIZE + m])
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

		//Not all threads get to this point, some of them return early
		if(!prediction)
			asm volatile("bar.sync %0, %1;" ::"r"(5), "r"(blockDim.x / 2));

		if(!prediction && j > 1)
		{
			largeInt URL_largeInt[URL_SIZE / sizeof(largeInt)];
			char* URL = (char*) URL_largeInt;
			long long int temp;
			char* href = (char*) &temp;
			href[0] = 'h';
			href[1] = 'r';
			href[2] = 'e';
			href[3] = 'f';
			href[4] = '\0';

			genericCounter = ((blockIdx.x * BLOCKSIZE + ((threadIdx.x - (blockDim.x / 2)) / WARPSIZE) * WARPSIZE) * iterations) + (threadIdx.x % 32) * COPYSIZE;
			int step = 0;

			int loopCounter = 0;
			largeInt localEndOffset = fileNames[fileInUse].endOffset;
			for(; (loopCounter < iterations) && (i < end); loopCounter ++, i ++)
			{
				char c = (textData + (s * iterations * (blockDim.x / 2) * gridDim.x))[genericCounter + step];

				step ++;
				genericCounter += (step / COALESCEITEMSIZE) * (WARPSIZE * COALESCEITEMSIZE);
				step %= COALESCEITEMSIZE;

				

				switch(state)
				{
					case START:
						if (c == '<')
						{
							state = IN_TAG;
						}
						break;
					case IN_TAG:
						if (c == 'a')
						{
							state = IN_ATAG;
						}
						else if (c == ' ')
						{
							state = IN_TAG;
						}
						else state = START;
						break;
					case IN_ATAG:
						if (c == 'h')
						{
							int x;
							for (x = 1; x < 4 && i < end; x++)
							{
								c = (textData + (s * iterations * (blockDim.x / 2) * gridDim.x))[genericCounter + step];
								step ++;
								genericCounter += (step / COALESCEITEMSIZE) * (WARPSIZE * COALESCEITEMSIZE);
								step %= COALESCEITEMSIZE;
								i ++;
								loopCounter ++;

								if (href[x] != c)
								{
									state = START;
									break;
								}
							}
							state = FOUND_HREF;
						}
						else if (c == ' ') state = IN_ATAG;
						else state = START;
						break;
					case FOUND_HREF:
						if (c == ' ') state = FOUND_HREF;
						else if (c == '=') state = FOUND_HREF;
						else if (c == '\"' || c == '\'')
						{
							state = START_LINK;
						}
						else{ 
							state = START;
						
						}
						break;
					case START_LINK:

						while(i >= localEndOffset)
						{
							fileInUse ++;
							localEndOffset = fileNames[fileInUse].endOffset;
						}

						int linkSize = 0;
						while(linkSize < URL_SIZE && c != '\"' && c != '\'')
						{
							//urlBuffer[storedUrlOffset + linkSize] = c;
							//linkSize ++;
							
							URL[linkSize ++] = c;

							c = (textData + (s * iterations * (blockDim.x / 2) * gridDim.x))[genericCounter + step];
							step ++;
							genericCounter += (step / COALESCEITEMSIZE) * (WARPSIZE * COALESCEITEMSIZE);
							step %= COALESCEITEMSIZE;
							i ++;
							loopCounter ++;
						}

						//urlSizes[numStoredUrls] = linkSize;
						//storedUrlOffset += (linkSize % 8 == 0)? linkSize : (linkSize + (8 - linkSize % 8));
						
						
#if 1

						if(states[stateCounter] == (char) 0)
						{
							//if(true)
							if(addToHashtable((void*) URL, linkSize, (void*) fileNames[fileInUse].name, fileNames[fileInUse].nameSize, mbk, passNo) == true)
							{
								myNumbers[index * 2] ++;
								states[stateCounter] = SUCCEED;
							}
							else
							{
								myNumbers[index * 2 + 1] ++;
								*failedFlag = true;
								epochSuccessStatus[j - 2] = FAILED;
							}
							stateCounter ++;
						}
#endif

						state = START;
						break;
				}

			}

			s = (s + 1) % 3;
		}

		__syncthreads();
	}
}

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
	struct timeval end;
	time_t sec, ms, diff;

	gettimeofday(&end, NULL);
	sec = end.tv_sec - global_start[tid].tv_sec;
	ms = end.tv_usec - global_start[tid].tv_usec;
	diff = sec * 1000000 + ms;

	//printf("[%d] %10s:\t\t%0.1fms\n", tid, message, (double)((double)diff/1000.0));
	//fflush(stdout);
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
			offset = ((myBlock * BLOCKSIZE + k * WARPSIZE) * epochDuration) + i * COPYSIZE;

			copytype_t* tempSpace = (copytype_t*) &hostBuffer[0][offset];

			//TODO this has to use strides to know what address to load next.
			for(int j = 0; j < warpFirstLastAddrs[i].itemCount / COPYSIZE; j ++)
			{
				tempSpace[j * WARPSIZE] = *((copytype_t*) &fdata[(curAddrs + j * COPYSIZE)]);
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
	while(cudaErrorNotReady == cudaStreamQuery(*execStream))
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

        if (argc != 2)
	{
		printf("usage: %s <dir>\n", argv[0]);
		exit(-1);
	}

	getFilesSize(argv[1]);
	printf("Count files is %d\n", count);
	printf("Tottal size is %lluMB\n", totalSize / (1 << 20));
	printf("Allocating %ldMB for fileNames\n", (count * sizeof(fileName_t)) / (1 << 20));
	fileName_t* fileNames = (fileName_t*) malloc(count * sizeof(fileName_t));
	
	unsigned int* offsets = (unsigned*) malloc((count + 1) * sizeof(unsigned));
	memset(offsets, 0, count * sizeof(unsigned));

	char* fdata = (char*) malloc(totalSize);

	getDataFiles(argv[1], fdata, offsets, fileNames);
	offsets[count] = totalSize;
	fileName_t* dfileNames;
	cudaMalloc((void**) &dfileNames, count * sizeof(fileName_t));
	cudaMemcpy(dfileNames, fileNames, count * sizeof(fileName_t), cudaMemcpyHostToDevice);

	largeInt fileSize = totalSize;

	dim3 block(BLOCKSIZE, 1, 1);
	dim3 block2((BLOCKSIZE * 2), 1, 1);
	dim3 grid(MAXBLOCKS, 1, 1);
	int numThreads = BLOCKSIZE * grid.x * grid.y;

	unsigned numRecords = fileSize / URL_SIZE;

	//======================================================//
	int chunkSize = EPOCHCHUNK * (1 << 20);

	//TODO: make epochNum unnecessary
	int epochNum = (int) (fileSize / chunkSize);
	if(fileSize % chunkSize)
		epochNum ++;

	printf("Number of epochs: %d\n", epochNum);
	//======================================================//

	//=================== Max num of iterations ============//
	int maxIterations = fileSize / numThreads;

	maxIterations ++;
	if(epochNum > 1)
		maxIterations /= (epochNum);
	//======================================================//


	int iterations = maxIterations;
	if(iterations % 8 != 0)
		iterations += (8 - (iterations % 8));

	//========= URLAddrHostBuffer ===========//
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

	//========= URLHostBuffer ===========//
	int textHostBufferSize = iterations * numThreads * 3;
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
	cudaMalloc((void**) &textData, iterations * numThreads * 3);

	char* phony = (char*) 0x0;
	cudaStream_t execStream;
	cudaStreamCreate(&execStream);
	cudaStream_t copyStream;
	cudaStreamCreate(&copyStream);

	unsigned urlBufferSizePerThread = 10 * (1 << 10);
	char* durlBuffer;
	cudaMalloc((void**) &durlBuffer, urlBufferSizePerThread * numThreads);
	cudaMemset(durlBuffer, 0, urlBufferSizePerThread * numThreads);

	unsigned numUrlPerThread = 1000;
	int* durlSizes;
	cudaMalloc((void**) &durlSizes, numUrlPerThread * numThreads * sizeof(int));
	cudaMemset(durlSizes, 0, numUrlPerThread * numThreads * sizeof(int));
	
	
	//============ initializing the hash table and page table ==================//
	int pagePerGroup = 50;
	multipassConfig_t* mbk = initMultipassBookkeeping(	(int*) hostCompleteFlag, 
								gpuFlags, 
								flagSize,
								numThreads,
								epochNum,
								numRecords,
								pagePerGroup);
	multipassConfig_t* dmbk;
	cudaMalloc((void**) &dmbk, sizeof(multipassConfig_t));
	cudaMemcpy(dmbk, mbk, sizeof(multipassConfig_t), cudaMemcpyHostToDevice);

	//==========================================================================//

	struct timeval partial_start, partial_end, bookkeeping_start, bookkeeping_end, passtime_start, passtime_end;
	time_t sec;
	time_t ms;
	time_t diff;


	gettimeofday(&partial_start, NULL);
	int passNo = 1;
	bool failedFlag = false;
	do
	{
		printf("====================== starting pass %d ======================\n", passNo);
		errR = cudaGetLastError();
		printf("#######Error before calling the kernel is: %s\n", cudaGetErrorString(errR));
		gettimeofday(&passtime_start, NULL);

		printf("iteration is %d\n", iterations);
		invertedIndexKernelMultipass<<<grid, block2, 0, execStream>>>(
				phony, 
				dfileNames,
				count,
				numRecords, //TODO fill this
				textAddrs,
				textData,
				flags,
				gpuFlags,
				stridesSpace1,
				firstLastAddrsSpace1,
				iterations,
				epochNum,
				numThreads,
				dmbk,
				mbk->dstates,
				passNo,
				durlBuffer,
				urlBufferSizePerThread,
				durlSizes,
				numUrlPerThread
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
			argument[m]->textHostBuffer[1] = hostTextHostBuffer + iterations * numThreads;
			argument[m]->textHostBuffer[2] = hostTextHostBuffer + iterations * numThreads * 2;
			argument[m]->textAddrs[0] = hostTextAddrHostBuffer;
			argument[m]->textAddrs[1] = hostTextAddrHostBuffer + iterations * numThreads;
			argument[m]->textAddrs[2] = hostTextAddrHostBuffer +  iterations * numThreads * 2;
			argument[m]->textData[0] = textData;
			argument[m]->textData[1] = textData + iterations * numThreads;
			argument[m]->textData[2] = textData +  iterations * numThreads * 2;
			argument[m]->stridesSpace1[0] = hostStridesSpace1;
			argument[m]->stridesSpace1[1] = hostStridesSpace1 + numThreads;
			argument[m]->stridesSpace1[2] = hostStridesSpace1 + numThreads * 2;
			argument[m]->epochDuration = iterations;
			argument[m]->firstLastAddrsSpace1[0] = hostFirstLastSpace1;
			argument[m]->firstLastAddrsSpace1[1] = hostFirstLastSpace1 + numThreads;
			argument[m]->firstLastAddrsSpace1[2] = hostFirstLastSpace1 + numThreads * 2;
			argument[m]->textItemSize = RECORD_SIZE;
			argument[m]->sourceSpaceSize1 = fileSize;

			pthread_create(&threads[m], NULL, pipelineData, (void*) argument[m]);
		}


		//while(cudaSuccess != cudaStreamQuery(execStream))
		while(cudaErrorNotReady == cudaStreamQuery(execStream))
			usleep(300);	


		errR = cudaGetLastError();
		printf("Error after calling the kernel is: %s\n", cudaGetErrorString(errR));
		if(errR != cudaSuccess)
			exit(1);

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
	printf("Some results: \n");
	int tabCount = 0;
	for(int i = 0; i < 1000; i ++)
	{
		hashBucket_t* bucket = buckets[i];

		while(bucket != NULL)
		{
			char* dna = (char*) getKey(bucket);
			valueHolder_t* valueHolder = bucket->valueHolder;
			for(int j = 0; j < bucket->keySize; j ++)
				printf("%c", dna[j]);
			printf(": ");
			while(valueHolder != NULL)
			{
				printf("[");
				char* value = (char*) getValue(valueHolder);
				unsigned valueSize = (unsigned) (valueHolder->valueSize);
				for(int j = 0; j < valueSize; j ++)
					printf("%c", value[j]);
				
				//printf(" DocID: %lld", value->documentId);
				//void* getValue(value_t* valueHolder)
				valueHolder = valueHolder->next;
				printf("] ");
			}
			printf("\n");

			bucket = bucket->next;

			tabCount ++;
		}
		tabCount = 0;

	}
	printf("\n");
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

	return 0;
}
