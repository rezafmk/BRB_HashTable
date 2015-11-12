#!/usr/bin/python
import sys
import getopt
import random
import os
import string

minChars = 3
maxChars = 32
numUniquePatents = 5000
numRecords = 1000
#================= uasge ===================
def usage():
	print "Necessary input arguments:"+\
		"\n\t-n\t\tNumber of output records"+\
		"\n\t-u\t\tNumber of unique patents"+\
		"\n\t-h\t\tShow this usage"
		
			

#============== file_length =================
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def file_size(fname):
	fsize = os.path.getsize(fname)
	return fsize

def random_number(low, high):
	num = random.randrange(low, high, 1)
	return num;

#=========== get a random number ===========
def random_word(wordSize):
	w = ''.join(random.choice(string.ascii_lowercase) for _ in range(wordSize))
	return w
	
#============================= Main of the program ===========================
try:
	opts, args = getopt.getopt(sys.argv[1:], "n:u:h", [])

except getopt.GetoptError:
		usage()
		sys.exit(2)


outputname = "output.txt"

	
for opt, arg in opts:
	if opt == "-n":
		numRecords = int(arg)
	elif opt == "-u":
		numUniquePatents = int(arg)
	elif opt == "-h":
		usage()
		sys.exit(0)

try:
	outputf = open(outputname, 'w')
except IOError:
	print "Cannont open the output file to write!"
	sys.exit(2)

print "Number of unique patent numbers: " + str(numUniquePatents)
print "Number of total records: " + str(numRecords)

listOfPatents = []
for x in xrange(numUniquePatents):
	listOfPatents.append(str(random_number(100000, (100000 + numUniquePatents * 10))))
	
count = 0;
for x in xrange(numRecords):
	count += 1
	patent = listOfPatents[random_number(0, numUniquePatents)]
	citingPatent = listOfPatents[random_number(0, numUniquePatents)]
	if(patent != citingPatent):
		outputf.write(patent + ', ' + citingPatent + '\n')
	else:
		x -= 1



