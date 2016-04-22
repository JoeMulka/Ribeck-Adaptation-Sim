# This program will work in conjunction with the adaptation sim.
# It will take the name of a folder as a command line argument, and then read in all the adaptation sim data that
# is in that folder.  It will then be able to graph it.

import os
import sys
import csv
import matplotlib
from matplotlib import pyplot
import numpy

_plot = matplotlib.pyplot.plot
_xticks = matplotlib.pyplot.xticks
_yticks = matplotlib.pyplot.yticks
_xlabel = matplotlib.pyplot.xlabel
_ylabel = matplotlib.pyplot.ylabel


_title = matplotlib.pyplot.title
_savefig = matplotlib.pyplot.savefig
_polyfit = numpy.polyfit
_poly1d = numpy.poly1d
_std=numpy.std

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




fitness_trajectories = []
column_counter=0
# Import the fitness data
os.chdir(source_directory) #make the source folder the working directory
for fit_file in fitness_list:
    fitness_trajectories.append([])
    with open(fit_file,'rb') as csvfile:
        fitreader = csv.reader(csvfile,delimiter =',',quotechar =' ', quoting=csv.QUOTE_MINIMAL)
        for row in fitreader:
            fitness_trajectories[column_counter].append(row[0])
    column_counter +=1


fixation_trajectories = []
column_counter=0
# Import the fixation data
for fix_file in fix_list:
    fixation_trajectories.append([])
    with open(fix_file,'rb') as csvfile:
        fixreader = csv.reader(csvfile,delimiter =',',quotechar =' ', quoting=csv.QUOTE_MINIMAL)
        for row in fixreader:
            fixation_trajectories[column_counter].append(row[0])
    column_counter +=1
column_counter=0



# Change the characters in the lists to doubles
for i in xrange(len(fitness_trajectories)):
    fitness_trajectories[i] = map(float,fitness_trajectories[i] )
for i in xrange(len(fixation_trajectories)):
    fixation_trajectories[i] = map(float,fixation_trajectories[i] )

#
#
# Do some computations with the data
#
#

# Puts generation number in [i]axis and run number within the [i] axis of the inner lists
fitness_trajectories = numpy.transpose(fitness_trajectories)
fixation_trajectories = numpy.transpose(fixation_trajectories)

num_gens = len(fixation_trajectories)

mean_fitness_trajectory =[]
mean_fixation_trajectory =[]
#compute lines of best fit, standard deviations, and lag
for i in xrange(num_gens):
    mean_fitness_trajectory.append(numpy.mean(fitness_trajectories[i]))
for i in xrange(num_gens):
    mean_fixation_trajectory.append(numpy.mean(fixation_trajectories[i]))

lag = next((i for i, v in enumerate(mean_fixation_trajectory) if v >0.5), 0)#lag position based on
                                        #where half of runs have fixed one mutation
fit_degree = 1 #degree of polynomial of best fit

x_vals = range (len(mean_fitness_trajectory))
log_fitness = numpy.log(mean_fitness_trajectory)


log_fitness_coeffs=_polyfit(x_vals[lag:(len(x_vals)-1)],log_fitness[lag:(len(log_fitness)-1)],fit_degree)
fixation_coeffs=_polyfit(range(len(mean_fixation_trajectory)),mean_fixation_trajectory,fit_degree)

log_fitness_polynomial = _poly1d(log_fitness_coeffs)
fixation_polynomial = _poly1d(fixation_coeffs)

log_fitness_y_vals=log_fitness_polynomial(x_vals)
fixation_y_vals = fixation_polynomial(x_vals)

slope_log_fit=log_fitness_coeffs[0]
slope_fixation=fixation_coeffs[0]


fitness_stand_devs=numpy.zeros(len(fitness_trajectories))
fixation_stand_devs=numpy.zeros(len(fixation_trajectories))
for i in xrange(num_gens):
    fitness_stand_devs[i]=_std(fitness_trajectories[i])
    fixation_stand_devs[i]=_std(fixation_trajectories[i])





#
#
# Plotting
#
#
plotmin = 0 #can be adjusted to show lag phase or to ignore lag phase
plotmax = len(fixation_trajectories) #can be adjusted along with plot min to zoom in
matplotlib.pyplot.clf()

_plot(fitness_trajectories[plotmin:plotmax],'k'), _plot(mean_fitness_trajectory[plotmin:plotmax], 'b', linewidth=4)

_xticks(size=14), _yticks(size=14), _xlabel("generations", size=16), _ylabel("fitness", size=16)
# show()

#_plot(numpy.log(fitness_trajectories[plotmin:plotmax]),'k'), _plot(numpy.log(mean_fitness_trajectory[plotmin:plotmax]), 'b', linewidth=3)
_title("Raw Fitness")
_savefig("raw_fit.png")

matplotlib.pyplot.clf()

_plot(numpy.log(fitness_trajectories[plotmin:plotmax]),'k'), _plot(numpy.log(mean_fitness_trajectory[plotmin:plotmax]), 'b', linewidth=3)
_plot(log_fitness_y_vals[plotmin:plotmax],'r--',linewidth=2)
_xticks(size=14), _yticks(size=14), _xlabel("generations", size=16), _ylabel("log(fitness)", size=16)
_title("Log Fitness")
_savefig('test_fit.png')

matplotlib.pyplot.clf()
# show()
#adaptation_rate=slope_log_fit
#print ("adaptation rate: " + `adaptation_rate`)

_plot(fixation_trajectories[plotmin:plotmax],'k'), _plot(mean_fixation_trajectory[plotmin:plotmax], 'b', linewidth=3)
_plot(fixation_y_vals[plotmin:plotmax],'r--',linewidth=2)
_xticks(size=14), _yticks(size=14), _xlabel("generations", size=16), _ylabel("fixations", size=16)
_title("Fixations")
_savefig('test_fix.png')
#show()
#fixation_rate=slope_fixation
#print ("fixation rate: " + `fixation_rate`)