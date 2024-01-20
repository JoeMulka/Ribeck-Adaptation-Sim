import numpy as np, scipy.special, random, bisect, time
from IPython.display import clear_output
import csv
import adaptation_sim_functions
import datetime
import argparse
import os
import pandas as pd

_exp = np.random.exponential
_bin = np.random.binomial
_poisson = np.random.poisson
_gamma = scipy.special.gammainc
_polyfit = np.polyfit
_poly1d = np.poly1d
_std = np.std
_func = adaptation_sim_functions


def adaptation(
    pop_size,
    mut_rate,
    alpha,
    g,
    num_gens,
    mutation_tracker_toggle,
    is_binary,
    progress_update_interval,
):
    mean_fitness = 1
    w = [1.0]  # genotype_fitnesses
    p = [pop_size]  # genotype_frequencies
    m = [0]  # genotype_mutations
    fitnesses = [1.0]
    fixations = [0]
    pop_sizes = [pop_size]  # A list of the population sizes after each generation
    # The dot product of the frequency and fitness lists, the "total fitness" used in calculating the mean
    dot_product = pop_size
    new_pop_size = pop_size
    # a parallel list to master mut list that holds the mutation effect sizes including the background
    master_mut_list_background = []
    # effect sizes of each mutation, index of list is ID of mutation
    master_mut_list = []
    # list of lists, index is the genotype associated with it, list at that index is the ID's of the mutations that it has
    # does not yet work with multiple runs
    mutation_tracker = [[]]

    for i in range(num_gens):
        # REPRODUCTION
        p, new_pop_size, dot_product = _func.reproduction(
            p, w, dot_product, is_binary, pop_size
        )
        (
            p,
            w,
            m,
            added_fitness,
            master_mut_list,
            master_mut_list_background,
            mutation_tracker,
        ) = _func.mutation(
            p,
            w,
            m,
            new_pop_size,
            master_mut_list,
            master_mut_list_background,
            mutation_tracker,
            mutation_tracker_toggle,
            mean_fitness,
            mut_rate,
            alpha,
            g,
        )  # MUTATION
        p, w, m, mutation_tracker = _func.tidyup_genotypes(
            p, w, m, mutation_tracker
        )  # CLEANUP

        # Since the dot product was initially calculated in the Reproduction function, this adds in
        # the part of the dot product from the extra fitness of the new new mutants.  This is just the additional fitness from the
        # mutations, not the total fitness of the mutants, which avoids double counting them in both Reproduction and Mutation
        dot_product += added_fitness
        mean_fitness = dot_product / new_pop_size

        fitnesses.append(mean_fitness)
        fixations.append(min(m))
        pop_sizes.append(new_pop_size)

        # prints every progress_update_interval generations, for instance every 1000 generations
        # had i+1 earlier, check accuracy of i with actual gen number
        if i % progress_update_interval == 0:
            formatted_output = f"generation: {i}\nmean fitness: {mean_fitness:.4f}\nfixations: {fixations[-1]}\n\n"
            print(formatted_output)

    # ensures the last bit of output doesn't get accidentally cleared
    if i + 1 == num_gens:
        formatted_output = f"generation: {i}\nmean fitness: {mean_fitness:.4f}\nfixations: {fixations[-1]}\n\n"
        print(formatted_output)

    return (
        fitnesses,
        fixations,
        pop_sizes,
        master_mut_list,
        master_mut_list_background,
        mutation_tracker,
    )


# MAIN
if __name__ == "__main__":
    # Main runs the simulation several times and gathers the results

    # Variable Initialization
    # pop_size = 3.4e7
    # mut_rate = 1e-7
    # alpha = 100
    # epistasis parameter
    # g = 0
    # num_gens = 50
    # turns the mutation tracker on or off
    # mutation_tracker_toggle = False
    # Which model of reproduction is being used
    # is_binary = False
    # sets whether or not you are allowed to overwrite existing files
    # can_overwrite = True
    # output_directory = "/home/joe/UbuntuDev/Ribeck-Git-Repository/output"
    # sets how often the program prints out the generation number and mean fitness
    # progress_update_interval = 10

    parser = argparse.ArgumentParser(description="Run adaptation simulation")
    parser.add_argument(
        "-p",
        "--pop_size",
        type=float,
        default=3.4e7,
        help="population size",
    )
    parser.add_argument(
        "-m",
        "--mut_rate",
        type=float,
        default=1e-7,
        help="mutation rate",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=100,
        help="Describes mean of distribution from which beneficial effect sizes are drawn from higher alpha means smaller beneficial mutations",
    )
    parser.add_argument("-g", "--g", type=float, default=0, help="epistasis parameter")
    parser.add_argument(
        "-n",
        "--num_gens",
        type=int,
        default=50,
        help="number of generations",
    )
    parser.add_argument(
        "-t",
        "--mutation_tracker_toggle",
        type=bool,
        default=False,
        help="turns mutation tracker on or off",
    )
    parser.add_argument(
        "-b",
        "--is_binary",
        type=bool,
        default=False,
        help="sets whether or not you are using binary fission",
    )
    parser.add_argument(
        "-v",
        "--can_overwrite",
        type=bool,
        default=True,
        help="sets whether or not you are allowed to overwrite existing files",
    )
    parser.add_argument(
        "-o", "--output_directory", type=str, default=".", help="output directory"
    )
    parser.add_argument(
        "-u",
        "--progress_update_interval",
        type=int,
        default=10,
        help="sets how often the program prints out the generation number and mean fitness",
    )
    parser.add_argument(
        "--pop_size_trajectories",
        type=bool,
        default=False,
        help="sets whether to write population size trajectories to file",
    )

    args = parser.parse_args()

    start_time = datetime.datetime.now()
    # A list of lists.  Index is run number, values are fitness-over-generation lists
    fitness_trajectories = []
    fixation_trajectories = []
    pop_size_trajectories = []
    fixed_muts_trajectory = []  # a long list of all the mutation sizes that have fixed
    # a long list of all the mutation sizes that have fixed, taking the background into effect
    fixed_mut_back_trajectory = []

    test_fitnesses = []  # testing mutation tracker algorithm

    # Run the adaptation sim once
    (
        new_fitness_traj,
        new_fixation_traj,
        new_pop_size_traj,
        new_master_mut,
        new_master_mut_background,
        new_mut_tracker,
    ) = adaptation(
        args.pop_size,
        args.mut_rate,
        args.alpha,
        args.g,
        args.num_gens,
        args.mutation_tracker_toggle,
        args.is_binary,
        args.progress_update_interval,
    )
    fitness_trajectories.append(new_fitness_traj)
    fixation_trajectories.append(new_fixation_traj)
    pop_size_trajectories.append(new_pop_size_traj)
    # calculates more accurately the total number of fixations at the end of each run
    if (len(new_mut_tracker) != 0) & args.mutation_tracker_toggle == True:
        # intersection_update is a function of the set class, so fixed muts needs to be casted to a set
        fixed_muts = set(new_mut_tracker[0])
        # The intersection of the first and second mutation list in new_mut_tracker is taken, then
        # fixed muts is updated to this intersection.  This intersection is then intersected with the next list.
        # When all lists have been intersected, the resulting intersection of all lists is turned back into a list type element
        for i in new_mut_tracker[1:]:
            fixed_muts.intersection_update(i)

        fixed_muts = list(fixed_muts)
        # for histograms regarding what kinds of mutations end up fixing
        fixed_mut_sizes = [new_master_mut[x] for x in fixed_muts]
        fixed_mut_back_sizes = [new_master_mut_background[x] for x in fixed_muts]
        # concatenates the lists of fixed mutation effect sizes for each run
        fixed_muts_trajectory = fixed_muts_trajectory + fixed_mut_sizes
        # concatenates the lists of fixed mutation effect sizes plus background for each run
        fixed_mut_back_trajectory = fixed_mut_back_trajectory + fixed_mut_back_sizes
    fixed_muts = []
    fixed_mut_sizes = []
    fixed_mut_back_sizes = []

    # Puts generation number in [i] axis and run number within the [i] axis of the inner lists
    fitness_trajectories = np.transpose(fitness_trajectories)
    fixation_trajectories = np.transpose(fixation_trajectories)
    pop_size_trajectories = np.transpose(pop_size_trajectories)

    # print(f"averaging: "+`num_gens`+"/"+`num_gens`+  "  Averaging Complete at mut_rate: "+ `mut_rate`) #displays when averaging is done
    clear_output()
    time_elapsed = str(datetime.datetime.now() - start_time)
    print(f"time elapsed: {time_elapsed}")

    #
    # CSV WRITER
    #

    pop_exponent = int(np.log10(args.pop_size))
    pop_string = args.pop_size / (10**pop_exponent)
    # The part of the file name that describes the parameters used
    generic_file_name = f"_{pop_string}e{pop_exponent}_{args.alpha}_{args.mut_rate}"

    error_message = (
        "Warning, a file you are trying to write to already exists.  If you would like to "
        + "overwrite, then set the 'can_overwrite' tag to 'True'\n"
        + "File was: {}"
    )

    # All the data
    file_name = os.path.join(args.output_directory, "fitness" + generic_file_name)
    file_type = ".csv"
    # Checks if the file already exists, and appends a number if it does
    file_path = adaptation_sim_functions.name_fixer(file_name, file_type, 1)

    with open(file_path, "w") as csvfile:
        # outwriter = csv.writer(
        #    csvfile, delimiter=",", quotechar=" ", quoting=csv.QUOTE_MINIMAL
        # )
        np.savetxt(file_path, fitness_trajectories, delimiter=",")
        # for i in range(args.num_gens):
        #    print("writing generation: " + str(i))

        # outwriter.writerow(fitness_trajectories[i].tolist())

    file_name = os.path.join(args.output_directory, "fixation" + generic_file_name)
    file_type = ".csv"
    file_path = adaptation_sim_functions.name_fixer(file_name, file_type, 1)
    with open(file_path, "wb") as csvfile:
        outwriter = csv.writer(
            csvfile, delimiter=",", quotechar=" ", quoting=csv.QUOTE_MINIMAL
        )
        for i in range(args.num_gens):
            outwriter.writerow(fixation_trajectories[i].tolist())

    # mutation tracker
    # concatenates the fixed mutation effect sizes of all the runs
    if args.mutation_tracker_toggle == True:
        file_name = os.path.join(
            args.output_directory, "mutation_tracker" + generic_file_name
        )
        file_type = ".csv"
        file_path = adaptation_sim_functions.name_fixer(file_name, file_type, 1)
        with open(file_path, "wb") as csvfile:
            outwriter = csv.writer(
                csvfile, delimiter=",", quotechar=" ", quoting=csv.QUOTE_MINIMAL
            )
            outwriter.writerow(["Fixed Mutations", "Fixed Mutations w/ Background"])
            for i in range(len(fixed_muts_trajectory)):
                outwriter.writerow(
                    [fixed_muts_trajectory[i], fixed_mut_back_trajectory[i]]
                )

    if args.pop_size_trajectories == True:
        file_name = os.path.join(args.output_directory, "pop_size" + generic_file_name)
        file_type = ".csv"
        file_path = adaptation_sim_functions.name_fixer(file_name, file_type, 1)

        with open(file_path, "wb") as csvfile:
            outwriter = csv.writer(
                csvfile, delimiter=",", quotechar=" ", quoting=csv.QUOTE_MINIMAL
            )
            for i in range(args.num_gens):
                outwriter.writerow(pop_size_trajectories[i].tolist())
