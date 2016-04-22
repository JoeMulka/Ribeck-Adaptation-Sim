# This program will work in conjunction with the adaptation sim.
# It will take the name of a folder as a command line argument, and then read in all the adaptation sim data that
# is in that folder.  It will then be able to graph it.

import os
import sys

source_directory = ""

if len(sys.argv)!=2:
    print "please pass the name of the directory containing the data files as a command line argument"
    sys.exit
else:
    source_directory = sys.argv[1]

file_list = os.listdir(source_directory) # a list of all the files in the given folder

fitness_list = []
fix_list =[]
for file in file_list:  #this separates the list into fitness files and fixation files
    if "fitness" in file:
        fitness_list.append(file)
    elif "fixation" in file:
        fix_list.append(file)


print fitness_list
print fix_list
