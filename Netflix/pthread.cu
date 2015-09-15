#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <tcmalloc.h>
#include <fcntl.h>
#include <sys/stat.h>

typedef long long int largeInt;

#define RECORD_LENGTH 56
#define NUM_BUCKETS 10000000
#define ALIGNMET 8
#define NUMTHREADS 8
#define RECORD_SIZE 56
#define GB 1073741824

#define USERAIDLOCATION 0
#define USERBIDLOCATION 6
#define USERARATELOCATION 12
#define USERBRATELOCATION 14
#define DISPLAY_RESULTS
#define NUM_RESULTS_TO_SHOW 20


typedef struct
{
	largeInt numUsers;
	int userAId;
	int userBId;
} userIds;

typedef struct hashEntry_t
{
	struct hashEntry_t* next;
	int value;	
	userIds key;
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

int myAtoi(char* p)
{
	int k = 0;
	while (*p >= '0' && *p <= '9') 
	{
		k = (k << 3) + (k << 1) + (*p) - '0';
		p ++;
	}

	return k;
}
int getUserAID(char* record)
{
	return myAtoi(&record[USERAIDLOCATION]);
}

int getUserBID(char* record)
{
	return myAtoi(&record[USERBIDLOCATION]);
}

int getUserARate(char* record)
{
	return myAtoi(&record[USERARATELOCATION]);
}
int getUserBRate(char* record)
{
	return myAtoi(&record[USERBRATELOCATION]);
}


unsigned hashFunc(char* str, int len, int numBuckets)
{
	userIds* ids = (userIds*) str;
	unsigned hash = ids->userAId * ids->numUsers + ids->userBId;

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
		existingEntry->value += value;
	}
	else
	{
		hashEntry_t* newEntry = (hashEntry_t*) tc_malloc(sizeof(hashEntry_t));

		if(newEntry != NULL)
		{
			newEntry->next = entry;
			
			for(int j = 0; j < keySize; j ++)
				((char*) &(newEntry->key))[j] = key[j];

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
	int numUsers = argument->numUsers;

	int chunkSize = numRecords / numThreads;
	
	int start = chunkSize * index;
	int end = start + chunkSize;
	end --;

	printf("Thread %d started\n", index);
	
	for(unsigned i = start; i < end; i += 1)
	{

		char* myrecord = &records[i * RECORD_LENGTH];
		int recordSize = 0;
		char record[RECORD_SIZE];
		for(int j = 0; j < RECORD_SIZE; j ++)
		{
			char c = myrecord[j];
			record[j] = c;
		}

		userIds key;
		key.userAId = getUserAID(record);
		key.userBId = getUserBID(record);
		if(key.userBId > key.userAId)
		{
			int swap = key.userBId;
			key.userBId = key.userAId;
			key.userAId = swap;
		}
		key.numUsers = numUsers;

		int rateA = getUserARate(record);
		int rateB = getUserBRate(record);
		int localScore = 0;
		if(rateA == rateB)
			localScore = 10;
		else if(abs(rateA - rateB) == 1)
			localScore = 2;
		else
			localScore = -5;



		if(addToHashtable(hashTable, (char*) &key, sizeof(userIds), localScore, 4) == true)
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
	
	

	printf("@INFO: Allocating %lldMB for input data\n", (largeInt) (numRecords * RECORD_LENGTH) / (largeInt) (1 << 20));
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
		argument[i].numUsers = NUMTHREADS;

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
		hashEntry_t* bucket = hashTable[i].entry;


		while(bucket != NULL)
		{
			userIds* ids = &(bucket->key);
			int* value = &(bucket->value);
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
		hashEntry_t* bucket = hashTable[topScoreIds[i]].entry;
		for(int j = 0; j < topScoreTab[i]; j ++)
			bucket = bucket->next;

		userIds* ids = &(bucket->key);
		int* value = &(bucket->value);
		printf("IDs: %d and %d: %d\n", ids->userAId, ids->userBId, *value);
	}
#endif

	return 0;
}
