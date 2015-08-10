#include "global.h"

__device__ int isEqual(char* first, char* second, int len)
{
	for(int i = 0; i < len; i ++)
		if(first[i] != second[i])
			return 0;
	return 1;
	
}

__device__ unsigned int hashFunc(char* str, int len, unsigned numBuckets, int counter)
{                       
        unsigned hash = 2166136261;
        unsigned FNVMultiple = 16777619;

	for(int i = 0; i < len; i ++)
        {
		hash += counter;
                hash = hash ^ (str[i]);     /* xor  the low 8 bits */
		hash += (int) str[i];
                hash = hash * FNVMultiple;  /* multiply by the magic number */
		hash += len;
		hash *= (len + counter);
        }

        return hash % numBuckets;
}

__device__ unsigned int hashFunc(char* str, int len, unsigned numBuckets)
{                       
        unsigned hash = 2166136261;
        unsigned FNVMultiple = 16777619;

	for(int i = 0; i < len; i ++)
        {
                hash = hash ^ (str[i]);     /* xor  the low 8 bits */
		hash += (int) str[i];
                hash = hash * FNVMultiple;  /* multiply by the magic number */
        }

        return hash % numBuckets;
}

__device__ int findPatternKernel(int* array, int size)
{
        int curPatternSize = 1;
        int counter = 0;

        for(int i = 1; i < size; i ++)
        {
                if(array[i] != array[counter])
                {
                        curPatternSize ++;
                        i = curPatternSize - 1;
                        counter = 0;
                }
                else
                {
                        counter ++;
                        counter %= curPatternSize;
                }
        }

	return curPatternSize;
}


