'''
    ME674 Soft Computing in Engineering
    Programming Assignment 2
    Binary Coded Genetic Algorithm

    Name = Mayank Pathania
    Roll No. = 204103314
    Specialization = Machine Design

    Roulette Wheel Reproduction Scheme
    Two Point Crossover
    Bitwise mutation
'''

import random
import math


# Binary Codded GA class
class BGA:
    def __init__(self, p_size, p_cross, p_mutation, func, chromosomes, gene_bits, gene_min, gene_max, itr):
        self.__population_size = p_size
        self.__probability_crossover = p_cross
        self.__probability_mutation = p_mutation
        self.__function = func
        self.__chromosomes = chromosomes
        self.__bits = gene_bits
        self.__min = gene_min
        self.__max_itr = itr
        self.__factor = [(gene_max[i] - gene_min[i])/(2**self.__bits[i] - 1) for i in range(0, self.__chromosomes)]


    # self.find_max() sets the self.__objective_function = self.__function
    # Then calls self.__solve() to find maxima
    # returns the solution returned by self.__solve()
    def find_max(self):
        self.__objective_function = self.__function
        return self.__solve()


    # self.find_min() sets the self.__objective_function = self.__set_objective
    # Then calls self.__solve() to find minima
    # returns the solution return by self.__solve()
    def find_min(self, n = 1):
        self.__objective_function = self.__set_objective(n)
        return self.__solve()


    # self.__set_objective() transforms the self.__function()
    # so that BGA can find minima of the problem
    def __set_objective(self, n):
        def wrapper(x):
            return 1/(n + self.__function(x))
        return wrapper


    # self.__decode() takes a binary string and return decimal value of chromosomes in sting
    # the length of chromosomes is decided from self.__bits
    def __decode(self, solution):
        x = [0 for x in range(0, self.__chromosomes)]
        start = 0
        for i in range(0, self.__chromosomes):
            for j in range(start, start + self.__bits[i]):
                x[i] = x[i]*2 + solution[j]
            start += self.__bits[i]
        return x


    # self.__real_string() takes binary string as a solution
    # returns the real value of the chromosomes
    # self.__decode() is used to decode the string 
    # then the chromosomes are converted to real values depending upon self.__factor and self.__min
    def __real_string(self, solution):
        decoded = self.__decode(solution)
        x = [0 for x in range(0, self.__chromosomes)]
        for i in range(0, self.__chromosomes):
            x[i] = self.__min[i] + self.__factor[i]*decoded[i]
        return x


    # self.__fitness() return array containing fitness value of each solution and corresponding index
    # fitness value is calculated by self.__objective_function()
    # returns [fitness_value for solution in self.__population]
    def __fitness(self, population):
        fitness = [0 for solution in range(0, len(population))]
        for i in range(0, len(fitness)):
            fitness[i] = self.__objective_function(self.__real_string(population[i]))
        return fitness


    # self.__roulette_wheel() performs the reproduction phase by roulette wheel reproduction scheme
    # it calculates fitness values from self.__fitness() and then selects solutions 
    # from population based on their fitness values
    def __roulette_wheel(self, population):
        fitness = self.__fitness(population)
        total = sum(fitness)

        # Calculation probability
        for i in range(0, len(fitness)): fitness[i] =  fitness[i]/total

        # Calculating commutative probability
        for i in range(1, len(fitness)): fitness[i] += fitness[i - 1]

        # Creating the mating pool
        mating_pool = [[0 for x in population[0]] for y in population]
        for i in range(0, self.__population_size):
            random_number = random.uniform(0, 1)
            for j in range(0, self.__population_size):
                if random_number < fitness[j]:
                    mating_pool[i] = population[j]
                    break


    # self.__crossover() performs crossover by two points crossover
    # selects two parents at random and swaps the randomly selected portion of parents 
    def __crossover(self, mating_pool):
        parent = [x for x in range(0, self.__population_size)]
        random.shuffle(parent)

        # Performing crossover
        for i in range(0, self.__population_size, 2):
            if random.uniform(0, 1) < self.__probability_crossover and i + 1 < self.__population_size:
                P1 = mating_pool[parent[i]]                          # Chromosome of parent 1
                P2 = mating_pool[parent[i + 1]]                      # Chromosome of parent 2

                start = random.randrange(0, len(P1))
                end = random.randrange(0, len(P1))

                while end == start: end = random.randrange(0, len(P1))
                if end < start: start, end = end, start

                (P1, P2) = (P1[:start] + P2[start:end] + P1[end:], P2[:start] + P1[start:end] + P2[end:])
                mating_pool[parent[i]] = P1
                mating_pool[parent[i + 1]] = P2


    # self.__mutation() performs bitwise mutation
    # each bit of each solution is compared with a random number to check if it should be changed
    def __mutation(self, children):
        for i in range(0, self.__population_size):
            for j in range(0, len(children[i])):
                if random.random() < self.__probability_mutation:
                    children[i][j] = not children[i][j]


    # self.__solve() solves the problem
    # returns tuple (binary solution, real valued solution) of the problem
    def __solve(self):
        total_bits = sum(self.__bits)
        self.__population = [[(random.uniform(-1, 1) >= 0) for gene in range(0, total_bits)] for solution in range(0, self.__population_size)]
        iteration = 0
        prev_avg = 1e6
        with open("result_iterations.dat", 'w') as file:
            print("Generation\tavg_fitness\tmax_fitness\tmin_fitness\tfunction_value")
            file.writelines("Generation\tavg_fitness\tmax_fitness\tmin_fitness\tfunction_value\n")
            while iteration < self.__max_itr:
                new_population = [[bit for bit in solution] for solution in self.__population]
                self.__roulette_wheel(new_population)
                self.__crossover(new_population)
                self.__mutation(new_population)

                total_population = [[bit for bit in solution] for solution in self.__population]
                for i in new_population:
                    total_population.append(i)
                fitness = self.__fitness(total_population)
                for i in range(0, len(fitness)):
                    fitness[i] = [fitness[i], i]
                fitness.sort(reverse = True)
                
                for i in range(0, self.__population_size):
                    self.__population[i] = total_population[fitness[i][1]]

                max_fitness = fitness[0][0]
                min_fitness = fitness[self.__population_size - 1][0]
                avg_fitness = sum([x[0] for x in fitness[:self.__population_size]])/self.__population_size

                max_fit_real = self.__real_string(self.__population[0])
                iteration += 1

                if iteration%100 == 0:
                    print("{:06}\t\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(iteration, avg_fitness, max_fitness, min_fitness, self.__function(max_fit_real)))
                file.write("{:06}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(iteration, avg_fitness, max_fitness, min_fitness, self.__function(max_fit_real)))

                # Checking Termination conditions
                if iteration > 10 and abs(avg_fitness - prev_avg) < 1e-8:
                    break
                else:
                    prev_avg = avg_fitness

        print("Generations Completed ===>\t", iteration)
        with open("result_last_iteration.dat", "w") as file:
            for i in self.__population:
                for j in i:
                    file.write(str(int(j)) + " ")
                file.write("\t\t{:.8f}\t{:.8f}\n".format(self.__real_string(i)[0], self.__real_string(i)[1]))
        return (self.__population[0], self.__real_string(self.__population[0]))


def main():
    print("This file only contains BGA class.")
    print("Run main.py")


if __name__ == '__main__':
    main()