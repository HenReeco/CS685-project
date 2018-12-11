import numpy
import random
import sys

numpy.set_printoptions(precision=3, suppress=True)
numpy.seterr(over='raise')

# ACCELEROMETER VALUES
x = numpy.arange(0, 101, 10)

xp = numpy.array([0, 10, 20, 50, 100])
fp = numpy.array([0, 15, 22, 31, 42])
ACCEL_EC = numpy.interp(x, xp, fp).tolist()
ACCEL_EC = [int(round(k)) for k in ACCEL_EC]
xp = numpy.array([0, 10, 20, 50, 100])
fp = numpy.array([0, 81, 82, 83, 83])
ACCEL_ACC = numpy.interp(x, xp, fp)
ACCEL_ACC = [int(round(k)) for k in ACCEL_ACC]
# MICROPHONE VALUES
MIC_EC = [int(round(k*3.4)) for k in ACCEL_EC]

# x = numpy.arange(0, 101, 10)
xp = numpy.array([0, 10, 30, 50, 80])
fp = numpy.array([0, 67, 73, 75, 94])
MIC_ACC = numpy.interp(x, xp, fp)
MIC_ACC = [int(round(k)) for k in MIC_ACC]
#GYROSCOPE VALUES
GYRO_EC = [int(round(k*11)) for k in ACCEL_EC]
GYRO_ACC = [int(round(k)) for k in ACCEL_ACC]
GYRO_ACC[0] = 0


# Inputs of the equation.
#0 accel, 1 gyro, 2 HR, 3 mic, 4 orient
equation_inputs = numpy.array([1,0.3,0.5,0.5,0.3])
# equation_inputs = [1,0.5,0.8,0.6,0.3]
ACC = [ACCEL_ACC, GYRO_ACC, [0, 10, 15, 35, 45, 55, 70, 85, 95, 98, 100], MIC_ACC, [0, 10, 15, 35, 45, 55, 70, 85, 95, 98, 100]]
EC = [ACCEL_EC, GYRO_EC, GYRO_EC, MIC_EC, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]]

# for i in ACC:
#     print i
# for i in EC:
#     print i

for idx, x in enumerate(ACC):
    for jdx, elem in enumerate(x):
        ACC[idx][jdx] *= 10


def create_initial_population(pop_size, entry_len, matrices, coverage_matrix_ACC, coverage_matrix_EC):
    
    resultACC = []
    resultEC = []

    counter = 0

    for k in range(0, pop_size):
        # for i in range(0, 2):
        for j in range (0,5):
            flag = True
            while(flag == True):
                y = random.randint(0, 10)
                # if (i == 0) and (coverage_matrix_ACC[j][y] < 2):
                if (coverage_matrix_ACC[j][y] < 2):
                    # print "ACC"
                    flag = False
                    counter += 1
                    coverage_matrix_ACC[j][y] = 1

            resultACC.append(matrices[0][j][y])
            resultEC.append(matrices[1][j][y])
    resultACC = numpy.asarray(resultACC)
    resultACC = resultACC.transpose()
    resultACC = resultACC.reshape(-1, 5)

    resultEC = numpy.asarray(resultEC)
    resultEC = resultEC.transpose()
    resultEC = resultEC.reshape(-1, 5)
    
    # print type(resultEC)
    result = numpy.append(resultACC, resultEC, axis = 1)

    return result

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.

    # for each population set, the first 5 elements come from the accuracy matrix, while
    # the remaining 5 are from the energy consumption matrix
    result = []
    for elem in pop:
        acc_fitness = [numpy.prod(x) for x in zip(elem[0:5:1], equation_inputs)]
        # beta = random.uniform(0.8, 1.2)
        ec_fitness = [-numpy.prod(x) for x in zip(elem[5:11:1], equation_inputs)]
        # ec_fitness = [x for x in ec_fitness]
        for i in range(len(acc_fitness)):
            acc_fitness[i] = acc_fitness[i]
            ec_fitness[i] = ec_fitness[i]
        fitness = numpy.asarray([acc_fitness, ec_fitness])
        fitness = fitness.flatten()
        temp_result = numpy.sum(fitness)
        result.append(temp_result)
        
        # print acc_fitness, ec_fitness, result
        # sys.exit()
    return result

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    # print "Population:"
    # print pop
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness)) 
        max_fitness_idx = max_fitness_idx[0][0]
        
        # print "Max indexes: ", max_fitness_idx
        parents[parent_num, :] = pop[max_fitness_idx, :]
        # print "\n\n", parents 
        for index, elem in enumerate(parents):
            parents[index] = numpy.trunc(parents[index])
            for j, val in enumerate(parents[index]):
                if val < 0:
                    parents[index, j] = 0
        # print parents 
        fitness[max_fitness_idx] = -200
    
    return parents

def crossover(parents, offspring_size):

    # offspring = numpy.empty(offspring_size)
    offspring = numpy.zeros(shape=(offspring_size))
    # The point at which crossover takes place between two parents. Usually it is at the center.
    # Make crossover_point random?
    crossover_point = numpy.uint8(offspring_size[1]/4)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:5] = parents[parent2_idx, crossover_point:5]
        #the two other halves
        offspring[k, 5:] = parents[parent1_idx, 5:]
        offspring[k, 5:] = parents[parent2_idx, 5:]
        #here i will reflect the changes to the other part of the chromosome
        for idx in range(0,5):
            flag = True
            for jdx, elem_j in enumerate(ACC[idx][:]):
                if offspring[k, idx] <= elem_j and flag == True:
                    # might want to perform a calculation on the value from EC
                    offspring[k, 5+idx] = EC[idx][jdx]
                    flag = False    #continue instead?
                if jdx == 10 and flag == False:
                    offspring[k, 9] = EC[idx][10]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    # print offspring_crossover
    for idx in range(offspring_crossover.shape[0]):    
        # Do mutation under random probability:
        if random.randint(0,100) <= 20:
            # print offspring_crossover[idx]
            # The random value to be added to the gene.
            random_value = random.randint(-50,50)
            #find a gene to change only in the first part of the chromosome (ACCURACY), i will reflect the changes to the
            #other part (ENERGY CONSUMPTION) a little below
            gene_idx = random.randint(0, 4)
            if (offspring_crossover[idx, gene_idx] + random_value > 1000):
                offspring_crossover[idx, gene_idx] = 1000
            elif (offspring_crossover[idx, gene_idx] + random_value < 0):
                offspring_crossover[idx, gene_idx] = 0
            else:
                offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value

            #here i will reflect the changes to the other part of the chromosome
            flag = True
            # for jdx, elem_j in enumerate(ACC[idx][:]):
            for jdx in range(0, 11):
                if offspring_crossover[idx, gene_idx] <= ACC[gene_idx][jdx] and flag == True:
                    # might want to perform a calculation on the value from EC
                    offspring_crossover[idx, 5+gene_idx] = EC[gene_idx][jdx]
                    flag = False
                if jdx == 10 and flag == True:
                    offspring_crossover[idx, 5+gene_idx] = EC[gene_idx][jdx]
    return offspring_crossover

