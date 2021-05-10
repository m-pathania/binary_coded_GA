'''
    ME674 Soft Computing in Engineering
    Programming Assignment 2
    Binary Coded Genetic Algorithm

    Name = Mayank Pathania
    Roll No. = 204103314
    Specialization = Machine Design
'''

import random
import math

# importing function to be optimized from file
import problem

# importing BGA Class
from BGA import BGA


def main():
    # Input Data
    population_size = int(input("Population Size :\t"))
    probability_crossover = float(input("Crossover Probability :\t"))
    probability_mutation = float(input("Mutation Probability :\t"))
    max_generations = int(input("Maximum Generations :\t"))

    # Reading the problem from problem.py
    variables = problem.variables
    variable_bits = problem.variable_bits
    variable_min = problem.variable_min
    variable_max = problem.variable_max
    function = problem.function

    # Solving Problem by BGA
    my_model = BGA(population_size, probability_crossover, probability_mutation, function, variables, variable_bits, variable_min, variable_max, max_generations)
    (solution_binary, solution_real) = my_model.find_min()

    for i in range(0, len(solution_binary)):
        solution_binary[i] = int(solution_binary[i])

    # Saving Results to file
    with open("result_final.dat", "w") as file:
        file.write("Binary Solution ===>\n")
        print("\n\nBinary Solution ===>\n", end = "")
        start = 0
        for i in variable_bits:
            file.write("\t")
            print("\t", end = "")
            for j in range(start, start + i):
                file.write("{:d} ".format(solution_binary[j]))
                print("{:d} ".format(solution_binary[j]), end = "")
            file.write("\n")
            print()
            start += i
        file.write("\n\nReal Solution ===>\n")
        print("\n\nReal Solution ===>\n")
        for i in range(0, variables):
            file.write("\t")
            file.write("{:.6f}".format(solution_real[i]))
            file.write("\n")
            print("\t",end = "")
            print("{:.6f}".format(solution_real[i]))
        file.write("\n\nOptimum Function Value = \t{:.6f}\n".format(function(solution_real)))
        print("\n\nOptimum Function Value = \t{:.6f}\n".format(function(solution_real)))



if __name__ == "__main__":
    main()
