#!/usr/bin/python
import sys
import getopt
import random
import os
import string

minChars = 3
maxChars = 32
numUniqueLocations = 5000
numRecords = 1000
#================= uasge ===================
def usage():
	print "Necessary input arguments:"+\
		"\n\t-n\t\tNumber of output records"+\
		"\n\t-u\t\tNumber of unique locations"+\
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

def getTillSecondTab(line):
	tabStop = 0
	for i in range(len(line)):
		if(line[i] == '\t'):
			tabStop += 1
			if(tabStop == 2):
				return line[:i+1]

	
#============================= Main of the program ===========================
try:
	opts, args = getopt.getopt(sys.argv[1:], "n:u:h", [])

except getopt.GetoptError:
		usage()
		sys.exit(2)


outputname = "output.txt"
inputname = "input.csv"

	
for opt, arg in opts:
	if opt == "-n":
		numRecords = int(arg)
	elif opt == "-u":
		numUniqueLocations = int(arg)
	elif opt == "-h":
		usage()
		sys.exit(0)

try:
	outputf = open(outputname, 'w')
except IOError:
	print "Cannont open the output file to write!"
	sys.exit(2)

try:
	inputf = open(inputname, 'r')
except IOError:
	print "Cannont open the input file to read!"
	sys.exit(2)


print "Number of unique locations: " + str(numUniqueLocations)
print "Number of total records: " + str(numRecords)

listOfLocations = []
for x in xrange(numUniqueLocations):
	location = str(random_number(0, 1000)) + " " + str(random_number(0, 1000)) + " "
	listOfLocations.append(location)
	
	
for x in xrange(numRecords):
	line = inputf.readline()
	newLine = getTillSecondTab(line)

	newLocation = listOfLocations[random_number(0, numUniqueLocations)]
	toWriteLine = newLine + newLocation
	outputf.write(toWriteLine + "\n")
	#if(patent != citingPatent):
		#outputf.write(patent + ', ' + citingPatent + '\n')
	#else:
		#x -= 1



