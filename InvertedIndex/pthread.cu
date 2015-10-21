#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <tcmalloc.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <dirent.h>

typedef long long int largeInt;

#define RECORD_SIZE 128
#define RECORD_SIZE_ALIGNED 128
#define IDSIZE 31
#define READSIZE 48
#define READSIZE_ALIGNED 48

#define NUM_BUCKETS 10000000
#define ALIGNMET 8
#define NUMTHREADS 8
#define GB 1073741824

#define DISPLAY_RESULTS
#define NUM_RESULTS_TO_SHOW 20

#define URL_SIZE 128

#define START           0x00 
#define IN_TAG          0x01 
#define IN_ATAG         0x02 
#define FOUND_HREF      0x03 
#define START_LINK      0x04


typedef struct valueHolder_t
{
	char* value;
	largeInt valueSize;
	struct valueHolder_t* next;
	
} valueHolder_t;


typedef struct hashEntry_t
{
	struct hashEntry_t* next;
	char key[URL_SIZE];	
	valueHolder_t* valueHolder;
} hashEntry_t;

typedef struct
{
	hashEntry_t* entry;
	unsigned lock;
} hashBucket_t;

typedef struct
{
	largeInt startOffset;
	largeInt endOffset;
	char name[72];
	largeInt nameSize;
} fileName_t;




typedef struct
{
	char* records;
	hashBucket_t* hashTable;
	int* status;
	fileName_t* fileNames;
	unsigned numRecords;
	int numThreads;
	int index;
	int numFiles;

} dataPackage_t;


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




unsigned int hashFunc(char* str, int len, unsigned numBuckets)
{
	unsigned hash = 2166136261;
	unsigned FNVMultiple = 16777619;

        for(int i = 0; i < len; i ++)
        {
                char c = str[i];

                hash += (int) c;
                hash = hash * FNVMultiple;  /* multiply by the magic number */
                hash += len;
                hash -= (int) c;
        }

        return hash % numBuckets;
}

hashEntry_t* containsKey(hashEntry_t* entry, char* key, int keySize)
{
	while(entry != NULL)
	{
		char* oldKey = (char*) &(entry->key);
		bool success = true;
		//OPTIMIZE: do the comparisons 8-byte by 8-byte
		for(int i = 0; i < keySize; i ++)
		{
			if(oldKey[i] != key[i])
			{
				success = false;
				break;
			}
		}
		if(success)
			break;

		entry = entry->next;
	}

	return entry;

}

// Allocation of hash table
	
bool addToHashtable(hashBucket_t* hashTable, char* key, int keySize, char* value, int valueSize)
{
	bool success = true;
	unsigned hashValue = hashFunc((char*) key, keySize, NUM_BUCKETS);

	
	hashBucket_t* bucket = &hashTable[hashValue];
	hashEntry_t* entry = bucket->entry;


	int keySizeAligned = (keySize % ALIGNMET == 0)? keySize : keySize + (ALIGNMET - (keySize % ALIGNMET));
	int valueSizeAligned = (valueSize % ALIGNMET == 0)? valueSize : valueSize + (ALIGNMET - (valueSize % ALIGNMET));

	unsigned oldLock = 1;

	do
	{
		oldLock = __sync_lock_test_and_set(&(bucket->lock), (unsigned) 1);
		
	} while(oldLock == 1);
	
	hashEntry_t* existingEntry;
	//First see if the key already exists in one of the entries of this bucket
	//The returned bucket is the 'entry' in which the key exists
	if((existingEntry = containsKey(entry, key, keySize)) != NULL)
	{
		valueHolder_t* valueHolder = (valueHolder_t*) tc_malloc(sizeof(valueHolder_t) + valueSize);
		valueHolder->value = (char*) ((largeInt) valueHolder + sizeof(valueHolder_t));
		valueHolder->valueSize = valueSize;
		for(int j = 0; j < valueSize; j ++)
			valueHolder->value[j] = value[j];

		valueHolder->next = entry->valueHolder;
		entry->valueHolder = valueHolder;
	}
	else
	{
		hashEntry_t* newEntry = (hashEntry_t*) tc_malloc(sizeof(hashEntry_t));

		if(newEntry != NULL)
		{
			newEntry->next = entry;
			
			for(int j = 0; j < keySize; j ++)
				((char*) &(newEntry->key))[j] = key[j];

			valueHolder_t* valueHolder = (valueHolder_t*) tc_malloc(sizeof(valueHolder_t) + valueSize);
			valueHolder->value = (char*) ((largeInt) valueHolder + sizeof(valueHolder_t));
			valueHolder->valueSize = valueSize;
			for(int j = 0; j < valueSize; j ++)
				valueHolder->value[j] = value[j];
			
			newEntry->valueHolder = valueHolder;

			bucket->entry = newEntry;

		}
		else
		{
			printf("Failed to malloc\n");
			success = false;
		}
	}

	bucket->lock = 0;

	return success;
}


void* kernel(void* arg)//char* records, int numRecords, int* recordSizes, int numThreads, pagingConfig_t* pconfig, hashtableConfig_t* hconfig, int* status)
{
	dataPackage_t* argument = (dataPackage_t*) arg;
	char* records = argument->records;
	unsigned numRecords = argument->numRecords;
	int numFiles = argument->numFiles;
	int numThreads = argument->numThreads;
	hashBucket_t* hashTable = argument->hashTable;
	int* status = argument->status;
	int index = argument->index;
	fileName_t* fileNames = argument->fileNames;

	int chunkSize = numRecords / numThreads;
	
	int start = chunkSize * index;
	int end = start + chunkSize;
	end --;

	int fileInUse = 0;
	for(int i = 0; i < numFiles; i ++)
	{
		if(start < fileNames[i].endOffset)
		{
			fileInUse = i;
			break;
		}
	}

	
	int state = START;
	char URL[URL_SIZE];
	long long int temp;
	char href[6];

	href[0] = 'h';
	href[1] = 'r';
	href[2] = 'e';
	href[3] = 'f';
	href[4] = '\0';

	for(unsigned i = start; i < end; i += 1)
	{
		char c = records[i];

		if(i >= fileNames[fileInUse].endOffset)
		{
			fileInUse ++;
			//printf("changed to %d\n", fileInUse);
		}


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
						i ++;
						c = records[i];

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

				int linkSize = 0;
				while(linkSize < URL_SIZE && c != '\"' && c != '\'')
				{
					URL[linkSize ++] = c;

					i ++;
					c = records[i];
				}

#if 1
				//if(true)
				if(addToHashtable(hashTable, URL, linkSize, fileNames[fileInUse].name, fileNames[fileInUse].nameSize) == true)
					status[index * 2] ++;
				else
					status[index * 2 + 1] ++;
#endif

				state = START;
				break;
		}


	}

	printf("Thread %d ended\n", index);
	return NULL;
}


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("USAGE: %s <inputfilename>\n", argv[0]);
		exit(1);
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
	largeInt fileSize = totalSize;

	unsigned numRecords = (unsigned) ((largeInt) fileSize / (largeInt) RECORD_SIZE);

	int numThreads = NUMTHREADS;
	numRecords = (numRecords % numThreads == 0)? numRecords : (numRecords + (numThreads - (numRecords % numThreads)));
	printf("@INFO: Number of records: %u (%u per thread)\n", numRecords, numRecords / numThreads);
	
	char* records = fdata;

	printf("@INFO: done initializing the input data\n");
	hashBucket_t* hashTable = (hashBucket_t*) malloc(NUM_BUCKETS * sizeof(hashBucket_t));
	int* status = (int*) malloc(numThreads * 2 * sizeof(int));


	//==========================================================================//
	
	struct timeval partial_start, partial_end;//, exec_start, exec_end;
        time_t sec;
        time_t ms;
        time_t diff;


	//====================== Calling the kernel ================================//


	//Spawn the pthread functions

	gettimeofday(&partial_start, NULL);
	
	
	pthread_t thread[NUMTHREADS];
	dataPackage_t argument[NUMTHREADS];

	for(int i = 0; i < NUMTHREADS; i ++)
	{
		argument[i].records = records;
		argument[i].numRecords = totalSize;
		argument[i].fileNames = fileNames;
		argument[i].numFiles = count;
		argument[i].hashTable = hashTable;
		argument[i].status = status;
		argument[i].index = i;
		argument[i].numThreads = NUMTHREADS;

		pthread_create(&thread[i], NULL, kernel, &argument[i]);
	}


	for(int i = 0; i < NUMTHREADS; i ++)
		pthread_join(thread[i], NULL);


	//Join threads here

	gettimeofday(&partial_end, NULL);
        sec = partial_end.tv_sec - partial_start.tv_sec;
        ms = partial_end.tv_usec - partial_start.tv_usec;
        diff = sec * 1000000 + ms;

        printf("\n%10s:\t\t%0.0f\n", "Total time", (double)((double)diff/1000.0));



	int totalSuccess = 0, totalFailed = 0;
	for(int i = 0; i < numThreads; i ++)
	{
		totalSuccess += status[i * 2];
		totalFailed += status[i * 2 + 1];
	}

	printf("Total success: %d\n", totalSuccess);
	printf("Total failed: %d\n", totalFailed);

	int totalDepth = 0;
	int totalValidBuckets = 0;
	int totalEmpty = 0;
	int maximumDepth = 0;
	for(int i = 0; i < NUM_BUCKETS; i ++)
	{
		hashEntry_t* bucket = hashTable[i].entry;
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
