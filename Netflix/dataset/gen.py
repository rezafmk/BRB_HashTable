#!/usr/bin/python
import sys
import getopt
import random
import os

summarySize = 7
transactionIDSize = 4
emailSize = 50
customerEmailSize = 255
merchantNameSize = 50
firstNameSize = 60
lastNameSize = 60
phoneSize = 15
addressSize = 60
citySize = 50
stateSize = 2
zipSize = 10
countrySize = 60
creditCardTypeSize = 1
creditCardNumberSize = 16
IPSize = 15
authMessageSize = 255
settleMessegeSize = 255
authMoneySize = 4
settleMoneySize = 4
creditMoneySize = 4

#================= uasge ===================
def usage():
	print "Necessary input arguments:"+\
		"\n\t-o\t\tOutput file name"+\
		"\n\t-n\t\tNumber of records"+\
		"\n\t-u\t\tTotal number of users"+\
		"\n\t-m\t\tTotal number of movies"+\
		"\n\t-l\t\tMinimum number of random jumps between movies"+\
		"\n\t-x\t\tMaximum number of random jumps between movies"+\
		"\nOptional arguments:"+\
		"\n\t-a\t\tA list size"+\
		"\n\t-b\t\tB list size"+\
		"\n\t-x\t\tMaximum number of random jumps between movies"+\
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


#=============== add to time some seconds =========================
def addToTime(currentTime, secondsToAdd):
	hr = int(currentTime[0:2])
	mi = int(currentTime[2:4])
	se = int(currentTime[4:6])
	se += secondsToAdd
	if(se > 59):
		se = 0
		mi += 1
		if(mi > 59):
			mi = 0
			hr += 1
			if(hr > 23):
				hr = 0
	return str(hr).zfill(2) + str(mi).zfill(2) + str(se).zfill(2)


#=================== create a random date ========================
def random_date():
	sec = random_number(0, 59)
	mi = random_number(0, 59)
	hr = random_number(0, 23)
	day = random_number(1, 30)
	mon = random_number(1, 12)
	year = random_number(2000, 2013)

	return str(hr).zfill(2) + ':' + str(mi).zfill(2) + ':' + str(sec).zfill(2) + ' ' + str(day).zfill(2) + '/' + str(mon).zfill(2) + '/' + str(year)
	
	
#============================= Main of the program ===========================
try:
	opts, args = getopt.getopt(sys.argv[1:], "o:n:u:m:l:x:a:b:h", [])

except getopt.GetoptError:
		usage()
		sys.exit(2)


if any("-o" in s for s in opts):
	if any("-u" in s for s in opts):
		if any("-n" in s for s in opts):
			if any("-m" in s for s in opts):
				if any("-l" in s for s in opts):
					if any("-x" in s for s in opts):
						print "arguments OK"
					else:
						usage()
						sys.exit(2)
				else:
					usage()
					sys.exit(2)
			else:
				usage()
				sys.exit(2)
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
numUsers = int(10000)
numRecords = int(1000000)
numMovies = int(1000)
minJumps = int(10)
maxJumps = int(800)
numAlist = int(150)
numBlist = int(300)

	
for opt, arg in opts:
	if opt == "-o":
		outputname = arg
	elif opt == "-n":
		numRecords = int(arg)
	elif opt == "-u":
		numUsers = int(arg)
	elif opt == "-m":
		numMovies = int(arg)
	elif opt == "-l":
		minJumps = int(arg)
	elif opt == "-x":
		maxJumps = int(arg)
	elif opt == "-a":
		numAlist = int(arg)
	elif opt == "-b":
		numBlist = int(arg)
	elif opt == "-h":
		usage()

try:
	outputf = open(outputname, 'w')
except IOError:
	print "Cannont open the output file to write!"
	sys.exit(2)

try:
	indexf = open("index.txt", 'w')
except IOError:
	print "Cannont open the index file to write!"
	sys.exit(2)


#The format is like this:
#Summary(7), transactionID(4), date(8), time(6), settleNow(1), trConfirmationEmail(50), merchantName(50), cardHoderLastName(60),
#cardHolderFirstName(60), customerEmail(255), customerPhone(15), billAddress(60), billCity(50), billState(2), billZip(10),
#billCountry(60), creditCardType(1), creditNumber(16), expirationMonth(2), expirationYear(4), customerIPAddress(15), authCode(1),
#authMessage(255), settleMessage(255), authAmount(4), settleAmount(4), creditAmount(4)


#===================== create the A and B lists of famous movies =======================#
#alist = []
#print "A list movies:"
#for x in xrange(numAlist):
	#newAlistMovie = random_number(0, numMovies);
	#alist.append(newAlistMovie)
#
#
#blist = []
#for x in xrange(numBlist):
	#blist.append(random_number(0, numMovies))


n = 0
threshold = 1000
h = 1
index = 0
toWrite = ""
for x in xrange(numRecords):
	movieId = random_number(0, numMovies)
	userA = random_number(0, numUsers)
	userB = random_number(0, numUsers)
	date = random_date()
	while(userB == userA):
		userB = random_number(0, numUsers)
	rateA = random_number(0, 5)
	rateB = random_number(0, 5)
	toWrite += str(movieId).zfill(5) + ' ' + str(userA).zfill(6) + ' ' + date + ' ' + str(rateA) + ' ' + str(userB).zfill(6) + ' ' + date + ' ' + str(rateB) +'\n'
outputf.write(toWrite)

