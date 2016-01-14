from __future__ import print_function
import sys
import time
import numpy as np, scipy.special, random, bisect, time
import cProfile, pstats, StringIO
#from matplotlib import pyplot
from IPython.display import clear_output
import csv
import os
import os.path
import adaptation_sim_functions
import datetime

_exp = np.random.exponential
_bin = np.random.binomial
_poisson = np.random.poisson
_gamma = scipy.special.gammainc
_polyfit = np.polyfit
_poly1d = np.poly1d
_std=np.std
_func = adaptation_sim_functions

#Variable Initialization
pop_size = 3.3E7
mut_rate = 1E-2
alpha = 100 #describes mean of distribution from which beneficial effect sizes are drawn from higher alpha means smaller beneficial mutations
g=0 #epistasis parameter
num_gens = 100
num_runs =1
mutation_tracker_toggle = False #turns the mutation tracker on or off
is_binary = False #Which model of reproduction is being used
can_overwrite=True #sets whether or not you are allowed to overwrite existing files
output_directory = "C:\Users\Lenski Lab\Documents\Noah's Adaptation Sim\\"



def adaptation(current_run):
    
    mean_fitness=1
    w = [1.0]      #genotype_fitnesses
    p = [pop_size] #genotype_frequencies
    m = [0]        #genotype_mutations
    fitnesses = [1.0]
    fixations = [0]
    pop_sizes = [pop_size] #A list of the population sizes after each generation
    dot_product = pop_size #The dot product of the frequency and fitness lists, the "total fitness" used in calculating the mean
    new_pop_size = pop_size
    master_mut_list_background=[] # a parallel list to master mut list that holds the mutation effect
                            #sizes including the background
    master_mut_list=[] #effect sizes of each mutation, index of list is ID of mutation
    mutation_tracker=[[]]#list of lists, index is the genotype associated with it, list at that
                    #index is the ID's of the mutations that it has
            #does not yet work with multiple runs
    
    for i in xrange(num_gens):
        p, new_pop_size, dot_product = _func.reproduction(p, w, dot_product,is_binary,pop_size) #REPRODUCTION
        p, w, m, added_fitness,master_mut_list,master_mut_list_background,mutation_tracker = _func.mutation(p, w, m, new_pop_size,
                                                                                                            master_mut_list,master_mut_list_background,mutation_tracker,mutation_tracker_toggle,
                                                                                                            mean_fitness,mut_rate,alpha,g)       #MUTATION
        p, w, m,mutation_tracker = _func.tidyup_genotypes(p, w, m,mutation_tracker)                            #CLEANUP
    
        dot_product += added_fitness #since the dot product was initially calculated in the Reproduction function, this adds in
        #the part of the dot product from the extra fitness of the new new mutants.  This is just the additional fitness from the
        #mutations, not the total fitness of the mutants, which avoids double counting them in both Reproduction and Mutation
        mean_fitness = dot_product/new_pop_size
    
        fitnesses.append(mean_fitness)
        fixations.append(min(m)) 
        pop_sizes.append(new_pop_size)

        if i % 1000 == 0: # prints every 1000 generations
            formatted_output= "current run: " + `current_run` + "  generation: " + `i` \
            + "  mean fitness: " + '%.4f'% mean_fitness + "  fixations: " + `fixations[-1]`#had i+1 earlier, check accuracy of i with actual gen number
        
            print (formatted_output,end = '\r')
            a=5

    if i+1==num_gens and current_run==num_runs:  #ensures the last bit of output doesn't get accidentally cleared
        formatted_output= "current run: " + `current_run` + "  generation: " + `i+1` \
            + "  mean fitness: " + '%.4f'% mean_fitness + "  fixations: " + `fixations[-1]`
        print (formatted_output)

    return fitnesses, fixations, pop_sizes,master_mut_list,master_mut_list_background,mutation_tracker
    
    

#MAIN

#Main runs the simulation several times and gathers the results

start_time = datetime.datetime.now()
#RUN ADAPTATION MULTIPLE TIMES
fitness_trajectories = [] #A list of lists.  Index is run number, values are fitness-over-generation lists
fixation_trajectories = [] #similar to above
pop_size_trajectories = [] #similar to above
fixed_muts_trajectory = [] #a long list of all the mutation sizes that have fixed
fixed_mut_back_trajectory = [] #a long list of all the mutation sizes that have fixed, taking the background into effect

test_fitnesses=[] #testing mutation tracker algorithm

for i in xrange(num_runs):
    new_fitness_traj, new_fixation_traj, new_pop_size_traj,new_master_mut,new_master_mut_background,new_mut_tracker = adaptation(i+1)
    fitness_trajectories.append(new_fitness_traj)
    fixation_trajectories.append(new_fixation_traj)
    pop_size_trajectories.append(new_pop_size_traj)
    if ((len(new_mut_tracker)!=0) & mutation_tracker_toggle == True): #calculates more accurately the total number of fixations at the end of each run
        fixed_muts = set(new_mut_tracker[0]) #intersection_update is a function of the set class, so fixed muts needs to be casted to a set
        for i in new_mut_tracker[1:]: #The intersection of the first and second mutation list in new_mut_tracker is taken, then 
            fixed_muts.intersection_update(i) #fixed muts is updated to this intersection.  This intersection is then intersected
        fixed_muts = list(fixed_muts) #with the next list.  When all lists have been intersected, the resulting intersection of all lists is turned back into a list type element
        fixed_mut_sizes = [new_master_mut[x] for x in fixed_muts] #for histograms regarding what kinds of mutations end up fixing
        fixed_mut_back_sizes = [new_master_mut_background[x] for x in fixed_muts]
        fixed_muts_trajectory=fixed_muts_trajectory+fixed_mut_sizes #concatenates the lists of fixed mutation effect sizes for each run
        fixed_mut_back_trajectory=fixed_mut_back_trajectory+fixed_mut_back_sizes #concatenates the lists of fixed mutation effect sizes plus background for each run
    fixed_muts=[]
    fixed_mut_sizes=[]
    fixed_mut_back_sizes=[]
    #This loop runs the adaptation sim and records the fitness, fixation, and population size data



    
fitness_trajectories = np.transpose(fitness_trajectories)
fixation_trajectories = np.transpose(fixation_trajectories)
pop_size_trajectories = np.transpose(pop_size_trajectories)
#puts generation number in [i]axis and run number within the [i] axis of the inner lists

#CALCULATE AVERAGE TRAJECTORIES
mean_fitness_trajectory = [] #indices are generation number, values are mean fitness of all runs
mean_fixation_trajectory = [] #similar to above
mean_pop_size_trajectory = [] #similar to above

for i in xrange(num_gens+1):
    mean_fitness_trajectory.append(np.mean(fitness_trajectories[i]))
    mean_fixation_trajectory.append(np.mean(fixation_trajectories[i]))
    mean_pop_size_trajectory.append(np.mean(pop_size_trajectories[i]))
    
    print("averaging: "+`i`+"/"+`num_gens`,end='\r') #Shows the user how much time is spent averaging/which generation is being
    #averaged
    
clear_output()
print("averaging: "+`num_gens`+"/"+`num_gens`+  "  Averaging Complete at mut_rate: "+ `mut_rate`) #displays when averaging is done
time_elapsed = str(datetime.datetime.now() - start_time)
print("time elapsed: " + time_elapsed)
    
    
#CSV WRITER



pop_exponent = int(np.log10(pop_size))
pop_string = pop_size/(10**pop_exponent)
generic_file_name="_"+`pop_string`+"e"+`pop_exponent`+"_"+`alpha`+"_"+`mut_rate`  #this allows us to
#potentially loop over multiple files if we need to


error_message="Warning, a file you are trying to write to already exists.  If you would like to" +'\n' \
               + "overwrite, then set the \'can_overwrite\' tag to \'True\'" + '\n' \
               +"File was: "


#All the data
file_name = output_directory   + "fitness" + generic_file_name
file_type=".csv"
file_path  = adaptation_sim_functions.name_fixer(file_name, file_type,1) #Checks if the file already exists, and appends a number if it does
with open(file_path, 'wb') as csvfile:
    outwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for i in xrange(num_gens):
        outwriter.writerow(fitness_trajectories[i].tolist())

        
file_name= output_directory +"fixation"+generic_file_name
file_type=".csv"
file_path  = adaptation_sim_functions.name_fixer(file_name,file_type,1)
with open(file_path,'wb') as csvfile:
    outwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for i in xrange(num_gens):
        outwriter.writerow(fixation_trajectories[i].tolist())

    
#mutation tracker
#concatenates the fixed mutation effect sizes of all the runs
if (mutation_tracker_toggle==True):
    file_name =output_directory +"mutation_tracker"+generic_file_name
    file_type=".csv"
    file_path  = adaptation_sim_functions.name_fixer(file_name,file_type,1)
    with open(file_path,'wb') as csvfile:
        outwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        outwriter.writerow(["Fixed Mutations","Fixed Mutations w/ Background"])
        for i in xrange(len(fixed_muts_trajectory)):
            outwriter.writerow([fixed_muts_trajectory[i],fixed_mut_back_trajectory[i]])


"""file_name = output_directory + "_"+"pop_size"+generic_file_name#this could get the axe
if (((os.path.isfile(file_name)==False) or (can_overwrite==True))):
    with open(file_name,'wb') as csvfile:
        outwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for i in xrange(num_gens):
            outwriter.writerow(pop_size_trajectories[i].tolist())
else:
    print (error_message+file_name) 
"""    

