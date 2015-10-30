__author__ = 'Lenski Lab'
# This file contains functions for use with the adaptation sim
import numpy as np, scipy.special, random, bisect, time
from itertools import izip
_bin = np.random.binomial
_exp = np.random.exponential
_gamma = scipy.special.gammainc


def tidyup_genotypes(genotype_frequencies, genotype_fitnesses, genotype_mutations, mutation_tracker):
    # zipping the lists together is part of the python "list comprehension" process, which builds a new list from
    # elements of the zipped lists based on the criteria after "if".  This proved to be the fastest way to remove
    # extinct genotypes from our tracking list.

    tidy_fitness_list = [x for x, y in izip(genotype_fitnesses, genotype_frequencies) if y > 0]
    tidy_mutation_list = [x for x, y in izip(genotype_mutations, genotype_frequencies) if y > 0]
    tidy_mutation_tracker = [x for x, y in izip(mutation_tracker, genotype_frequencies) if y > 0]
    tidy_freq_list = [x for x in genotype_frequencies if x != 0]

    return tidy_freq_list, tidy_fitness_list, tidy_mutation_list, tidy_mutation_tracker

def reproduction(genotype_frequencies, genotype_fitnesses, dot_product, is_binary,pop_size):


    running_dot_prod = 0
    running_pop_size = 0

    if is_binary == True:
        norm = pop_size/(2*dot_product) #binary fission
    else:
        norm = 1/dot_product  #<-- part of poisson reproduction
    for i in xrange(len(genotype_frequencies)):
        if is_binary == True:
            num_new_individuals = _bin(2*genotype_frequencies[i], genotype_fitnesses[i]*norm) #binary fission
        else:
            num_new_individuals = _bin(pop_size, genotype_frequencies[i]*genotype_fitnesses[i]*norm) #<-- poisson reproduction
        genotype_frequencies[i] = num_new_individuals
        running_pop_size += num_new_individuals #sums the new population size
        running_dot_prod += num_new_individuals*genotype_fitnesses[i] #iteratively calculates the dot product of the
        #frequency list and and the fitness list for use in later calculations.  Doing this process in this loop saved some time
        #as we had to loop through the entire genotype list here anyway to update it each generation

    return genotype_frequencies, running_pop_size, running_dot_prod

def mutation(genotype_frequencies, genotype_fitnesses, genotype_mutations, new_pop_size,
             master_mut_list, master_mut_list_background,mutation_tracker,mutation_tracker_toggle,
             mean_fitness,mut_rate,alpha,g):

    total_mutants = _bin(new_pop_size, (1 - np.exp(-mut_rate)))  #Calculates the total number of mutants expected this generation

    multiplicity=[] # an array whose indices correspond to number of mutations minus one, and whose entries are the number of
    #mutated individuals with that number of mutations this generation
    r = total_mutants
    k=2
    while r > 0:
        #r is the number of individuals with at least k mutations
        #rate inside binomial is the prob of an individual having at least k mutations given that it has at least k-1 mutations
        new_r = _bin(r, _gamma(k, mut_rate)/_gamma(k-1, mut_rate))
        multiplicity.append(r - new_r)
        r = new_r
        k+=1

    cum_sum=0
    thresholds=[] #This list is used in conjunction with the mutated_individuals list to pick the genotypes which will mutate.
    #It sums the amount of individuals in each genotype, effectively finding which ID numbers are in which genotype
    for i in xrange(len(genotype_frequencies)):
        if total_mutants > 0:
            cum_sum += genotype_frequencies[i]
            thresholds.append(cum_sum)

    mutated_individuals = random.sample(xrange(1,new_pop_size+1), total_mutants) #Think of every individual in the population having
    #a unique ID number.  This function creates a list of randomly selected ID numbers, the ID of each individual that will mutate
    #this generation.  While seemingly unorthodox, this was the fastest way to randomly pick individuals (without replacement)
    #to mutate.  This whole process is basically a weighted random sampling of the genotype_frequencies list.

    multiplicity_counter=1 #how many mutations this particular genotype acquired this gen
    this_mult_mutated=0 #how many indivduals have been mutated at this multiplicity count, for example if we have already dealt
    #with 4 double mutants, this_mult_mutated will equal 4
    added_fitness = 0 #for use in calculating new mean fitness
    for j in xrange(total_mutants):
        while this_mult_mutated == multiplicity[multiplicity_counter-1]:
            multiplicity_counter+=1
            this_mult_mutated=0
            #This loop checks if the required number of individuals at this multiplicity count have been dealt with, and increments
            #the multiplicity counter to the next multiplicity level with a nonzero number of mutants if the required number of
            #mutants has been met for the current multiplicity level.
        this_geno = bisect.bisect_left(thresholds,mutated_individuals[j]) #picks the genotype of a random individual from the
        #entire population.
        genotype_frequencies[this_geno] -=1 #subtracts the individual that mutated from its original genotype, as it now has
        #a new genotype
        genotype_mutations.append(multiplicity_counter + genotype_mutations[this_geno]) #records how many mutations the new
        #genotype has, which is equal to the amount its previous genotype had plus the amount of mutations it accrued this generation
        mutant_fitness = genotype_fitnesses[this_geno] #the fitness of the new mutant BEFORE the mutation benefit is applied

        for k in xrange(1, multiplicity_counter+1):
            s = _exp(1.0/(alpha*(mutant_fitness**g))) #the percent magnitude of the fitness gain from the mutation
            if mutation_tracker_toggle == True:
                master_mut_list.append(s) #adds the effect size of this mutation to the master mutation tracker
                master_mut_list_background.append((genotype_fitnesses[this_geno]/mean_fitness)-1+s)
            added_fitness += mutant_fitness*s #records the magnitude of the fitness
            mutant_fitness *= (1 + s) #applies the beneficial effect to the mutant
            #This loop applies the mutations and loops through the multiplicity number, so a double mutant will have two
            #mutations applied, for example

        genotype_fitnesses.append(mutant_fitness) #sets the updated fitness of the new mutant
        if mutation_tracker_toggle == True:
            mutation_tracker.append(mutation_tracker[this_geno][:] \
            + [x + len(master_mut_list)-multiplicity_counter for x in xrange(multiplicity_counter)]) #Adds the ID numbers of the
        #mutations that this genotype now has to the mutation tracker

        this_mult_mutated+=1
    genotype_frequencies.extend((1,)*total_mutants) #Adds the new genotypes to the genotype list.  All the new mutants have
    #a frequency of 1, thus this just adds entries of 1 to the end of the list

    return genotype_frequencies, genotype_fitnesses, genotype_mutations, added_fitness,master_mut_list,master_mut_list_background,mutation_tracker

