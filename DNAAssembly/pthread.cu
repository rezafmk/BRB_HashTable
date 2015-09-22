#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <tcmalloc.h>
#include <fcntl.h>
#include <sys/stat.h>

typedef long long int largeInt;

#define RECORD_SIZE 128
#define RECORD_SIZE_ALIGNED 128
#define IDSIZE 31
#define READSIZE 48
#define READSIZE_ALIGNED 48

#define IDLOCATION 0
#define READLOCATION 32
#define QSLOCATION 135

#define NUM_BUCKETS 10000000
#define ALIGNMET 8
#define NUMTHREADS 8
#define GB 1073741824

#define USERAIDLOCATION 6
#define USERBIDLOCATION 35
#define USERARATELOCATION 33
#define USERBRATELOCATION 62

#define DISPLAY_RESULTS
#define NUM_RESULTS_TO_SHOW 20


typedef struct
{
	unsigned pad;
	bool lunique;
	bool runique;
	char lextension;
	char rextension;
} value_t;


typedef struct hashEntry_t
{
	struct hashEntry_t* next;
	char key[READSIZE];	
	value_t value;
} hashEntry_t;

typedef struct
{
	hashEntry_t* entry;
	unsigned lock;
} hashBucket_t;



typedef struct
{
	char* records;
	hashBucket_t* hashTable;
	int* status;
	unsigned numRecords;
	int numThreads;
	int index;
	int numUsers;

} dataPackage_t;


unsigned int hashFunc(char* str, int len, unsigned numBuckets)
{

	int numOuterLoopIterations = len / 16;
	if(len % 16 != 0)
		numOuterLoopIterations ++;
	
	unsigned finalValue = 0;

	for(int j = 0; j < numOuterLoopIterations; j ++)
	{
		unsigned hashValue = 0;
		int startLen = 16 * j;
		int endLen = startLen + 16;
		endLen = (endLen < len)? endLen : len;

		char temp[4];
		temp[0] = (char) 0;
		temp[1] = (char) 0;
		temp[2] = (char) 0;
		temp[3] = (char) 0;

		for(int i = startLen; i < endLen; i ++)
		{
			int charCounter = (i % 16) / 4;

			if(charCounter == 0)
				charCounter = 3;
			else if(charCounter == 1)
				charCounter = 2;
			else if(charCounter == 2)
				charCounter = 1;
			else if(charCounter == 3)
				charCounter = 0;

			if(i % 4 == 0)
			{
				if(str[i] == 'C')
					temp[charCounter] = temp[charCounter] | (1 << 6);
				else if(str[i] == 'G')
					temp[charCounter] = temp[charCounter] | (1 << 7);
				else if(str[i] == 'T')
				{
					temp[charCounter] = temp[charCounter] | (1 << 7);
					temp[charCounter] = temp[charCounter] | (1 << 6);
				}

			}
			else if(i % 4 == 1)
			{
				if(str[i] == 'C')
					temp[charCounter] = temp[charCounter] | (1 << 4);
				else if(str[i] == 'G')
					temp[charCounter] = temp[charCounter] | (1 << 5);
				else if(str[i] == 'T')
				{
					temp[charCounter] = temp[charCounter] | (1 << 5);
					temp[charCounter] = temp[charCounter] | (1 << 4);
				}

			}
			else if(i % 4 == 2)
			{
				if(str[i] == 'C')
					temp[charCounter] = temp[charCounter] | (1 << 2);
				else if(str[i] == 'G')
					temp[charCounter] = temp[charCounter] | (1 << 3);
				else if(str[i] == 'T')
				{
					temp[charCounter] = temp[charCounter] | (1 << 3);
					temp[charCounter] = temp[charCounter] | (1 << 2);
				}


			}
			else
			{
				if(str[i] == 'C')
					temp[charCounter] = temp[charCounter] | (1 << 0);
				else if(str[i] == 'G')
					temp[charCounter] = temp[charCounter] | (1 << 1);
				else if(str[i] == 'T')
				{
					temp[charCounter] = temp[charCounter] | (1 << 1);
					temp[charCounter] = temp[charCounter] | (1 << 0);
				}

			}
		}

		hashValue = *((unsigned int*) &temp[0]);
		finalValue += hashValue;
	}

        return finalValue % numBuckets;
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
	
bool addToHashtable(hashBucket_t* hashTable, char* key, int keySize, value_t* value, int valueSize)
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
		value_t* mainValue = (value_t*) &(existingEntry->value);
		value_t* newValue = (value_t*) value;
		
		if(mainValue->rextension != newValue->rextension)
			mainValue->runique = false;
		if(mainValue->lextension != newValue->lextension)
			mainValue->lunique = false;

	}
	else
	{
		hashEntry_t* newEntry = (hashEntry_t*) tc_malloc(sizeof(hashEntry_t));

		if(newEntry != NULL)
		{
			newEntry->next = entry;
			
			for(int j = 0; j < keySize; j ++)
				((char*) &(newEntry->key))[j] = key[j];

			newEntry->value.lunique = value->lunique;
			newEntry->value.runique = value->runique;
			newEntry->value.lextension = value->lextension;
			newEntry->value.rextension = value->rextension;

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
	int numRecords = argument->numRecords;
	int numThreads = argument->numThreads;
	hashBucket_t* hashTable = argument->hashTable;
	int* status = argument->status;
	int index = argument->index;
	int numUsers = argument->numUsers;

	int chunkSize = numRecords / numThreads;
	
	int start = chunkSize * index;
	int end = start + chunkSize;
	end --;

	printf("Thread %d started numUsers: %d\n", index, numUsers);
	
	for(unsigned i = start; i < end; i += 1)
	{

		char* myrecord = &records[i * RECORD_SIZE + READLOCATION];
		char record[READSIZE];
		for(int j = 0; j < READSIZE; j ++)
		{
			char c = myrecord[j];
			record[j] = c;
		}

		
		value_t value;
		value.runique = true;
		value.lunique = true;

		if(addToHashtable(hashTable, (char*) record, READSIZE, &value, sizeof(value_t)) == true)
			status[index * 2] ++;
		else
			status[index * 2 + 1] ++;
	}

	printf("Thread %d ended\n", index);
	return NULL;
}


int main(int argc, char** argv)
{
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
	int numUsers = 15000;
	fd = open(fname, O_RDONLY);
	fstat(fd, &finfo);
	printf("Allocating %lluMB for the input file.\n", ((long long unsigned int)finfo.st_size) / (1 << 20));
	fdata = (char *) malloc(finfo.st_size);
	size_t fileSize = (size_t) finfo.st_size;

	largeInt maxReadSize = GB;
	largeInt readed = 0;
	largeInt toRead = 0;

	if(fileSize > maxReadSize)
        {
                largeInt offset = 0;
                while(offset < fileSize)
                {
                        toRead = (maxReadSize < (fileSize - offset))? maxReadSize : (fileSize - offset);
                        readed += pread(fd, fdata + offset, toRead, offset);
                        printf("writing %lliMB\n", toRead / (1 << 20));
                        //pwrite(fdw, fdata + offset, toRead, offset);
                        offset += toRead;
                }
        }
        else
                readed = read (fd, fdata, fileSize);


	if(readed != fileSize)
		printf("Not all of the file is read\n");


	unsigned numRecords = (unsigned) ((largeInt) fileSize / (largeInt) RECORD_SIZE);

	int numThreads = NUMTHREADS;
	numRecords = (numRecords % numThreads == 0)? numRecords : (numRecords + (numThreads - (numRecords % numThreads)));
	printf("@INFO: Number of records: %u (%u per thread)\n", numRecords, numRecords / numThreads);
	
	

	printf("@INFO: Allocating %lldMB for input data\n", (largeInt) (numRecords * RECORD_SIZE) / (largeInt) (1 << 20));
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
		argument[i].numRecords = numRecords;
		argument[i].hashTable = hashTable;
		argument[i].status = status;
		argument[i].index = i;
		argument[i].numThreads = NUMTHREADS;
		argument[i].numUsers = numUsers;

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
