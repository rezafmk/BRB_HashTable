#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <tcmalloc.h>
#include <fcntl.h>
#include <sys/stat.h>

typedef long long int largeInt;

#define RECORD_LENGTH 64
#define NUM_BUCKETS 1000000
#define ALIGNMET 8
#define NUMTHREADS 8
#define RECORD_SIZE 64

typedef struct hashEntry_t
{
	struct hashEntry_t* next;
	int value;	
	char key[128];
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
	int numRecords;
	int numThreads;
	int index;

} dataPackage_t;


unsigned hashFunc(char* str, int len, int numBuckets)
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
		char* oldKey = (char*) entry->key;
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
	
bool addToHashtable(hashBucket_t* hashTable, char* key, int keySize, int value, int valueSize)
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
		existingEntry->value ++;
	}
	else
	{
		hashEntry_t* newEntry = (hashEntry_t*) tc_malloc(sizeof(hashEntry_t));

		if(newEntry != NULL)
		{
			newEntry->next = entry;
			for(int j = 0; j < keySize; j ++)
				newEntry->key[j] = key[j];

			newEntry->value = value;
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

	int chunkSize = numRecords / numThreads;
	
	int start = chunkSize * index;
	int end = start + chunkSize;
	
	
	for(int i = start; i < end; i += 1)
	{
		char* record = &records[i * RECORD_LENGTH];
		int recordSize = 0;
		char URL[RECORD_SIZE];
		for(int j = 0; j < RECORD_SIZE; j ++)
		{
			char c = record[j];
			URL[j] = c;
			if(c != ' ' && c != '\n')
				recordSize ++;
		}
		recordSize = (recordSize % 8 == 0)? recordSize : (recordSize + (8 - (recordSize % 8)));
		largeInt value = 1;
		if(addToHashtable(hashTable, URL, recordSize, value, 4) == true)
			status[index * 2] ++;
		else
			status[index * 2 + 1] ++;
	}
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
	fd = open(fname, O_RDONLY);
	fstat(fd, &finfo);
	printf("Allocating %lluMB for the input file.\n", ((long long unsigned int)finfo.st_size) / (1 << 20));
	fdata = (char *) malloc(finfo.st_size);
	size_t readed = read (fd, fdata, finfo.st_size);
	size_t fileSize = (size_t) finfo.st_size;
	if(readed != fileSize)
		printf("Not all of the file is read\n");

	int numRecords = fileSize / RECORD_SIZE;

	int numThreads = NUMTHREADS;
	numRecords = (numRecords % numThreads == 0)? numRecords : (numRecords + (numThreads - (numRecords % numThreads)));
	printf("@INFO: Number of records: %d (%d per thread)\n", numRecords, numRecords / numThreads);
	
	

	printf("@INFO: Allocating %dMB for input data\n", (numRecords * RECORD_LENGTH) / (1 << 20));
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

	return 0;
}
