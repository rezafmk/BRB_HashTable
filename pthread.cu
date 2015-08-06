#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

typedef long long int largeInt;

#define RECORD_LENGTH 64
#define NUM_BUCKETS 1000000

typedef struct
{
	pagingConfig_t* pconfig;
	cudaStream_t* serviceStream;
} dataPackage_t;

typedef struct
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

hashBucket_t* containsKey(hashBucket_t* bucket, char* key, int keySize)
{
	while(bucket != NULL)
	{
		char* oldKey = (char*) bucket->key;
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

		bucket = bucket->next;
	}

	return bucket;

}

// Allocation of hash table
	
bool addToHashtable(hashBucket_t* hashTable, char* key, int keySize, int value, int valueSize)
{
	bool success = true;
	unsigned hashValue = hashFunc((char*) key, keySize, hconfig->numBuckets);

	
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
		hashEntry_t* newEntry = (hashEntry_t*) malloc(sizeof(hashEntry_t));

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
	char* recrods = argument->records;
	int numRecords = argument->numRecords;
	int* recordSizes  = argument->recordSizes;
	int numThreads = argument->numThreads;
	hashEntry_t* hashTable = agument->hashTable;
	int index = argument->index;
	
	
	for(int i = index; i < numRecords; i += numThreads)
	{
		char* record = &records[i * RECORD_LENGTH];
		int recordSize = recordSizes[i];
		recordSize = (recordSize % 8 == 0)? recordSize : (recordSize + (8 - (recordSize % 8)));
		largeInt value = 1;
		if(addToHashtable((void*) record, recordSize, (void*) &value, sizeof(largeInt), hconfig, pconfig) == true)
			status[index * 2] ++;
		else
			status[index * 2 + 1] ++;
	}
}


int main(int argc, char** argv)
{
	cudaError_t errR;
	int numRecords = 4500000;
	if(argc == 2)
	{
		numRecords = atoi(argv[1]);
	}	

	dim3 grid(8, 1, 1);
	dim3 block(512, 1, 1);
	int numThreads = grid.x * block.x;
	numRecords = (numRecords % numThreads == 0)? numRecords : (numRecords + (numThreads - (numRecords % numThreads)));
	printf("@INFO: Number of records: %d (%d per thread)\n", numRecords, numRecords / numThreads);
	
	

	printf("@INFO: Allocating %dMB for input data\n", (numRecords * RECORD_LENGTH) / (1 << 20));
	char* records = (char*) malloc(numRecords * RECORD_LENGTH);
	int* recordSizes = (int*) malloc(numRecords * sizeof(int));

	srand(time(NULL));

	for(int i = 0; i < numRecords; i ++)
	{
		recordSizes[i] = rand() % (RECORD_LENGTH - 8);
		if(recordSizes[i] < 14)
			recordSizes[i] = 14;
	}

	for(int i = 0; i < numRecords; i ++)
	{
		records[i * RECORD_LENGTH + 0] = 'w';
		records[i * RECORD_LENGTH + 1] = 'w';
		records[i * RECORD_LENGTH + 2] = 'w';
		records[i * RECORD_LENGTH + 3] = '.';

		int j = 4;
		for(; j < recordSizes[i] - 4; j ++)
			records[i * RECORD_LENGTH + j] = rand() % 26 + 97;

		records[i * RECORD_LENGTH + j + 0] = '.';
		records[i * RECORD_LENGTH + j + 1] = 'c';
		records[i * RECORD_LENGTH + j + 2] = 'o';
		records[i * RECORD_LENGTH + j + 3] = 'm';
	}

	printf("Some records:\n");
	for(int i = 0; i < 20; i ++)
	{
		for(int j = 0; j < recordSizes[i]; j ++)
		{
			printf("%c", records[i * RECORD_LENGTH + j]);
		}
		printf("\n");
	}

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

	
	pthread_t thread;
	dataPackage_t argument;

	argument.pconfig = pconfig;
	argument.serviceStream = &serviceStream;

	pthread_create(&thread, NULL, recyclePages, &argument);


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
