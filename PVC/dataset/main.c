#include <stdio.h>
#include <stdlib.h>
#include <time.h>


char** generateNames(int count, int minSize, int maxSize)
{
	char** names = (char**) malloc(count * sizeof(char*));

	int i;
	for(i = 0; i < count; i ++)
	{
		int size = rand() % maxSize;
		if(size < minSize)
			size = minSize;
		names[i] = (char*) malloc(size * sizeof(char));
		int j;
		for(j = 0; j < size; j ++)
			names[i][j] = 'a' + rand() % 26;
	}

	return names;

}


int main(int argc, char **argv)
{
	if (argc != 5)
	{
		printf("usage: %s file recordNum uniqueNameNum type\n\ttype:\n\t\tcount\n\t\trank\n", argv[0]);
		return -1;
	}
		
	char *fileName = argv[1];
	int recNum = atoi(argv[2]);
	int uniqueNameNum = atoi(argv[3]);
	char *type = argv[4];

	srand(time(0));

	if (strcmp(type, "count") == 0)
	{
		FILE *fp = fopen(fileName, "w+");

		char** names = generateNames(uniqueNameNum, 12, 47);
		
		int i;
		for (i = 0; i < recNum; i++)
		{
			int randomNumber = rand() % uniqueNameNum;
			fprintf(fp, "http://www.%s.com", names[randomNumber]);
			int urlSize = 16 + strlen(names[randomNumber]);
			int paddingSize = 64 - urlSize;
			int j;
			for(j = 0; j < paddingSize; j ++)
				fprintf(fp, " ", ' ');
			fprintf(fp, "\n");
		

		}

		fclose(fp);
	}
	else if (strcmp(type, "rank") == 0)
	{
		FILE *fp = fopen(fileName, "w+");
		int i;
		for (i = 0; i < recNum; i++)
			fprintf(fp, "http://www.abcdefg.com/%d.html\t%d\n", rand()%1024, rand());
		fclose(fp);
	}
	else
	{
		printf("usage: %s file recordNum type\n", argv[0]);
		return -1;
	}

	return 0;
}
