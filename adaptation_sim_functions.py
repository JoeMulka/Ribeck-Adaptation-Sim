__author__ = "Lenski Lab"
# This file contains functions for use with the adaptation sim
import numpy as np, scipy.special, random, bisect, time, os

_bin = np.random.binomial
_exp = np.random.exponential
_gamma = scipy.special.gammainc


def tidyup_genotypes(
    genotype_frequencies, genotype_fitnesses, genotype_mutations, mutation_tracker
):
    # zipping the lists together is part of the python "list comprehension" process, which builds a new list from
    # elements of the zipped lists based on the criteria after "if".  This proved to be the fastest way to remove
    # extinct genotypes from our tracking list.

    tidy_fitness_list = [
        x for x, y in zip(genotype_fitnesses, genotype_frequencies) if y > 0
    ]
    tidy_mutation_list = [
        x for x, y in zip(genotype_mutations, genotype_frequencies) if y > 0
    ]
    tidy_mutation_tracker = [
        x for x, y in zip(mutation_tracker, genotype_frequencies) if y > 0
    ]
    tidy_freq_list = [x for x in genotype_frequencies if x != 0]

    return tidy_freq_list, tidy_fitness_list, tidy_mutation_list, tidy_mutation_tracker


def reproduction(
    genotype_frequencies, genotype_fitnesses, dot_product, is_binary, pop_size
):
    running_dot_prod = 0
    running_pop_size = 0

    if is_binary == True:
        norm = pop_size / (2 * dot_product)  # binary fission
    else:
        # Part of poisson reproduction
        norm = 1 / dot_product
    for i in range(len(genotype_frequencies)):
        if is_binary == True:
            num_new_individuals = _bin(
                2 * genotype_frequencies[i], genotype_fitnesses[i] * norm
            )  # binary fission
        else:
            # Poisson reproduction
            num_new_individuals = _bin(
                pop_size, genotype_frequencies[i] * genotype_fitnesses[i] * norm
            )
        genotype_frequencies[i] = num_new_individuals
        # sums the new population size
        running_pop_size += num_new_individuals

        # Iteratively calculates the dot product of the frequency list and and the fitness list
        # for use in later calculations.  Doing this process in this loop saved some time
        # as we had to loop through the entire genotype list here anyway to update it each generation
        running_dot_prod += num_new_individuals * genotype_fitnesses[i]

    return genotype_frequencies, running_pop_size, running_dot_prod


def mutation(
    genotype_frequencies,
    genotype_fitnesses,
    genotype_mutations,
    new_pop_size,
    master_mut_list,
    master_mut_list_background,
    mutation_tracker,
    mutation_tracker_toggle,
    mean_fitness,
    mut_rate,
    alpha,
    g,
):
    # Calculates the total number of mutants this generation
    total_mutants = _bin(new_pop_size, (1 - np.exp(-mut_rate)))

    # An array whose indices correspond to number of mutations minus one, and whose entries are the number of
    # mutated individuals with that number of mutations this generation
    # Example: [100,2] means that 100 individuals have 1 mutation, and 2 individuals have 2 mutations this generation
    multiplicity = []
    r = total_mutants
    k = 2
    while r > 0:
        # r is the number of individuals with at least k mutations
        # rate inside binomial is the prob of an individual having at least k mutations given that it has at least k-1 mutations
        new_r = _bin(r, _gamma(k, mut_rate) / _gamma(k - 1, mut_rate))
        multiplicity.append(r - new_r)
        r = new_r
        k += 1

    cum_sum = 0  # Cumulative sum

    # This list is used in conjunction with the mutated_individuals list to pick the genotypes which will mutate.
    # It sums the amount of individuals in each genotype, effectively finding which ID numbers are in which genotype
    thresholds = []

    for i in range(len(genotype_frequencies)):
        if total_mutants > 0:
            cum_sum += genotype_frequencies[i]
            thresholds.append(cum_sum)

    # Think of every individual in the population having a unique ID number.  This function creates a list of
    # randomly selected ID numbers, the ID of each individual that will mutate this generation.
    # While seemingly unorthodox, this was the fastest way to randomly pick individuals (without replacement)
    # to mutate.  This whole process is basically a weighted random sampling of the genotype_frequencies list.
    mutated_individuals = random.sample(range(1, new_pop_size + 1), total_mutants)

    # How many mutations this particular genotype acquired this generation
    multiplicity_counter = 1
    # How many indivduals have been mutated at this multiplicity count, for example if we have already dealt with 4 double mutants, this_mult_mutated will equal 4
    this_mult_mutated = 0

    added_fitness = 0  # for use in calculating new mean fitness
    for j in range(total_mutants):
        while this_mult_mutated == multiplicity[multiplicity_counter - 1]:
            # This loop checks if the required number of individuals at this multiplicity count have been dealt with, and increments
            # the multiplicity counter to the next multiplicity level with a nonzero number of mutants if the required number of
            # mutants has been met for the current multiplicity level.
            multiplicity_counter += 1
            this_mult_mutated = 0

        # Picks the genotype of a random individual from the entire population.
        this_geno = bisect.bisect_left(thresholds, mutated_individuals[j])

        # Subtracts the individual that mutated from its original genotype, as it now has a new genotype
        genotype_frequencies[this_geno] -= 1

        # Records how many mutations the new genotype has, which is equal to the amount its previous genotype had plus the amount of mutations it accrued this generation
        genotype_mutations.append(multiplicity_counter + genotype_mutations[this_geno])
        # The fitness of the new mutant BEFORE the mutation benefit is applied
        mutant_fitness = genotype_fitnesses[this_geno]

        for k in range(1, multiplicity_counter + 1):
            # This loop applies the mutations and loops through the multiplicity number, so a double mutant will have two
            # mutations applied, for example

            # the percent magnitude of the fitness gain from the mutation
            s = _exp(1.0 / (alpha * (mutant_fitness**g)))
            if mutation_tracker_toggle == True:
                # adds the effect size of this mutation to the master mutation tracker
                master_mut_list.append(s)
                master_mut_list_background.append(
                    (genotype_fitnesses[this_geno] / mean_fitness) - 1 + s
                )
            added_fitness += mutant_fitness * s  # records the magnitude of the fitness
            mutant_fitness *= 1 + s  # applies the beneficial effect to the mutant

        # sets the updated fitness of the new mutant
        genotype_fitnesses.append(mutant_fitness)
        # The mutation tracker is a list of lists, where each list is the ID numbers of the mutations that a genotype has.
        # It is memory intensive to track the ID of each mutation at high mutation rate, so this is an optional feature
        # The IDs of each mutation form a sort of "list of genes" for each genotype.  Multiplying the fitnesses of each
        # of these mutations together gives the fitness of the genotype.
        if mutation_tracker_toggle == True:
            # Adds the ID numbers of the mutations that this genotype now has to the mutation tracker
            # The new mutant's list of mutation IDs contains all the IDs that the parent genotype had, plus the new mutation ID
            # In high mutation rate simulations, there can me multiple mutations on an individual at the same time, so we add
            # multiple mutation IDs to the list
            # The mutations have already been added to the master_mut_list, which records fitness value of all mutations for all genotypes
            # so for a single mutant the next id number is len(master_mut_list) - 1
            # [:] takes a shallow copy of a list
            # Example:  an individual with parent having mutation tracker [1], aquiring one mutation this generation,
            # which is the 4th mutation to occur in the entire experiment, should get the new mutation tracker [1,4]
            # Example, an individual with parent having mutation tracker [1], aquiring two mutations this generation,
            # which are the 4th and 5th mutations to occur in the entire experiment, should get the new mutation tracker [1,4,5].
            mutation_tracker.append(
                mutation_tracker[this_geno][:]
                + [
                    len(master_mut_list) - multiplicity_counter + x
                    for x in range(multiplicity_counter)
                ]
            )
        this_mult_mutated += 1

    # Adds the new genotypes to the genotype list.  All the new mutants have
    # a frequency of 1, thus this just adds entries of 1 to the end of the list
    genotype_frequencies.extend((1,) * total_mutants)

    return (
        genotype_frequencies,
        genotype_fitnesses,
        genotype_mutations,
        added_fitness,
        master_mut_list,
        master_mut_list_background,
        mutation_tracker,
    )


def name_fixer(target_file_name, file_type, attempt_number):
    append_number = attempt_number + 1
    if os.path.isfile(target_file_name + file_type) == False:
        return target_file_name + file_type
    if os.path.isfile(f"{target_file_name}({append_number}){file_type}") == False:
        return f"{target_file_name}_{append_number}{file_type}"
    else:
        return name_fixer(target_file_name, file_type, attempt_number + 1)
