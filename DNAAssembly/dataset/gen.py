#!/usr/bin/python
import sys
import getopt
import random
import os
import string

#================= uasge ===================
def usage():
	print "Necessary input arguments:"+\
		"\n\t-o\t\tOutput file name"+\
		"\n\t-n\t\tNumber of records"+\
		"\n\t-u\t\tTotal number of unique strands"+\
		"\n\t-h\t\tShow this usage"
		
		
			
#=========== get a random number ===========
def random_number(low, high):
	num = random.randrange(low, high, 1)
	return num;


#======== create a random card number (16digit unique) ============
def create_a_random_string_number(size):
	temp = ""
	for x in xrange(size):
		temp += str(random_number(0, 9))
	return temp

def create_a_random_string(size):
	temp = ""
	for x in xrange(size):
		temp += random.choice(string.letters)
	return temp

def create_a_random_strand(size):
	temp = ""
	for x in xrange(size):
		temp += random.choice("ATCG")	
	return temp
	
	
#============================= Main of the program ===========================
try:
	opts, args = getopt.getopt(sys.argv[1:], "o:n:u:h", [])

except getopt.GetoptError:
		usage()
		sys.exit(2)


if any("-o" in s for s in opts):
	if any("-u" in s for s in opts):
		if any("-n" in s for s in opts):
			print "arguments OK"
		else:
			usage()
			sys.exit(2)
	else:
		usage();
		sys.exit(2)
else:
	usage();
	sys.exit(2)
	

outputname = "output.txt"
numRecords = int(1000000)
numUniques = int(100000)
	
for opt, arg in opts:
	if opt == "-o":
		outputname = arg
	elif opt == "-n":
		numRecords = int(arg)
	elif opt == "-u":
		numUniques = int(arg)
	elif opt == "-h":
		usage()

try:
	outputf = open(outputname, 'w')
except IOError:
	print "Cannont open the output file to write!"
	sys.exit(2)

uniqueList = []
for x in xrange(numUniques):
	line1 = create_a_random_string(31)
	line2 = create_a_random_strand(48)
	line3 = "+"
	line4 = create_a_random_string(44)
	uniqueList.append(line1 + "\n" + line2 + "\n" + line3 + "\n" + line4 + "\n")

toWrite = ""
for x in xrange(numRecords):
	randNum = random_number(0, numUniques)
	toWrite += uniqueList[randNum]
outputf.write(toWrite)

