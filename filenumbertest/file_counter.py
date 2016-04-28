
# a quick tool to count the number of fitness and fixation data files are in a folder
# used to ensure that we're actually getting the number of runs we think we are from the HPCC

import os

file_list = os.listdir("C:\Users\msute_000\Documents\GitHub\Ribeck-Git-Repository\\filenumbertest")

fitness_list = []
fix_list =[]
for file in file_list:  #this separates the list into fitness files and fixation files
    if "fitness" in file:
        fitness_list.append(file)
    elif "fixation" in file:
        fix_list.append(file)

print "fitness files: " + str(len(fitness_list))
print "fixation files: " + str(len(fix_list))