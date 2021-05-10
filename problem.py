'''
    ME674 Soft Computing in Engineering
    Programming Assignment 2
    Binary Coded Genetic Algorithm

    Name = Mayank Pathania
    Roll No. = 204103314
    Specialization = Machine Design
'''


# Function to be optimized
def function(x):
    return x[0] + x[1] - 2*(x[0]**2) - x[1]**2 + x[0]*x[1]

# Data related to the problem
variables = 2
variable_bits = [19, 19]
variable_min = [0, 0]
variable_max = [0.5, 0.5]



def main():
    print("This file only contains function to be optmimized.")
    print("Run main.py")


if __name__ == "__main__":
    main()
