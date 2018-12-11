import numpy
from GA import *
import sys 
import copy
import matplotlib.pyplot as plt

# Number of the weights we are looking to optimize.
# 5 weights for each matrix

if len(sys.argv) != 2:  #it's 1 by default if no parameters are given
    print len(sys.argv)
    print "Please, pass the population size you want to work with."
    sys.exit()


num_weights = 10

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
#number of elements in the population
sol_per_pop = int(sys.argv[1])
#number of parents to consider for next population
num_parents_mating = sol_per_pop/2

# Defining the population size.
pop_size = [sol_per_pop,num_weights] # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

coverage_matrix_ACC = numpy.zeros(shape=(5, 11))
coverage_matrix_EC = numpy.zeros(shape=(5, 11))
#Creating the initial population.
#change this function in order to make it collect values from the 2 matrices


greatest_iterations_amount = 0
IT = []

colors = ["red", "yellow", "blue", "green", "black", "brown", "cyan", "magenta", "blueviolet", "coral"]

#FROM HERE DOWN, I'VE JUST RE_USED THE CODE FROMN single_run.py AND I WILL RUN IT 10 TIMES SENDING THE VALUES TO THE PLOT THE WILL BE CREATED AT THE END
for i in range(10):
    new_population = create_initial_population(pop_size[0], pop_size[1], [ACC, EC], coverage_matrix_ACC, coverage_matrix_EC)
    # sys.exit()
    initial_population = copy.copy(new_population)

    # num_generations = 10
    COUNTER_THRESHOLD = 20
    previous_best = 0

    # THRESHOLD = 2080
    THRESHOLD = 2100
    current_best = 0
    counter = 0
    threshold_counter = 0
    best_so_far = 0

    #VALUES FOR THE PLOT
    FS = []
    


    # for generation in range(num_generations):
    while current_best < THRESHOLD:
        # print("Generation : ", generation)
        fitness = cal_pop_fitness(equation_inputs, new_population)
        # print "FITNESS: "
        # print fitness

        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(new_population, fitness, 
                                        num_parents_mating)
        # print "****************PARENTS****************"
        # print parents

        # Generating next generation using crossover.
        
        #FIX CROSSOVER OFFSPRING SIZE?
        offspring_crossover = crossover(parents,
                                        offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        # print "OFFSPRING"
        # print offspring_crossover
        # Adding some variations to the offsrping using mutation.
        offspring_mutation = mutation(offspring_crossover)
        # print "MUTATION"
        # print offspring_mutation, "\n\n"
        # Creating the new population based on the parents and offspring.
        # print new_population
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation #offspring_mutation

    # calculating the fitness score to evaluate stopping the search
        fitness = cal_pop_fitness(equation_inputs, new_population)
        best_match_idx = numpy.where(fitness == numpy.max(fitness))
        current_best = fitness[best_match_idx[0][0]]

        if best_so_far < current_best:
            best_so_far = current_best

        # print "Current best:", current_best

        #this is just to initialize the initial value and have it available at the end of the program
        if counter == 0:
            initial_score = current_best
        counter += 1

        #update the value that is used to check how much the fitness value will have raised after COUNTER_THRESHOLD executions
        if threshold_counter == 0:
            previous_best = current_best

        threshold_counter += 1

        #when the threshold is reached i check if the values has increased enough (2%) compared to COUNTER_THRESHOLD executions before
        if threshold_counter == COUNTER_THRESHOLD:
            if current_best < (previous_best*1.02):
                print "No significant improvements in the last", COUNTER_THRESHOLD, "populations"
                break
            threshold_counter = 0
        
        FS.append(current_best)
        if greatest_iterations_amount < len(FS):
            greatest_iterations_amount = len(FS)

    # print "Initial population:"
    # print initial_population
    # print "Initial score:"
    # print initial_score


    # print "Stopped at iteration #", counter
    # print new_population

    # # Getting the best solution after iterating finishing all generations.
    # #At first, the fitness is calculated for each solution in the final generation.
    # fitness = cal_pop_fitness(equation_inputs, new_population)
    # # # Then return the index of that solution corresponding to the best fitness.
    # best_match_idx = numpy.where(fitness == numpy.max(fitness))

    # print "New population"
    # print new_population
    print "Best solution at index ", best_match_idx[0][0], ": " 
    print new_population[best_match_idx[0][0]]
    print "Best solution fitness : "
    print fitness[best_match_idx[0][0]]
    print counter, " iterations"
    IT.append(counter)


    X_FS = [j for j in range(len(FS))]
    plt.plot(X_FS, FS, color = colors[i])

X_FS = [i for i in range(greatest_iterations_amount)]
FS = [THRESHOLD for i in range(greatest_iterations_amount)]
plt.plot(X_FS, FS, color="red", linestyle="dashed")

print "#########**********Iterations amount**********#########"
print IT


plt.xlabel('Iterations')
plt.ylabel('Fitness score')
plt.ylim(1500, 2150)
plt.title('Population size: ' + str(sol_per_pop))
plt.grid(True)
plt.show()