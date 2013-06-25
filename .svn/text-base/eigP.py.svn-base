import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def eigP(k, eps, A):
    '''Compute the eigenvalue decomposition of the symmetric
    matrix A
    '''
    num_rows = A.shape[0]
    eigenvectors = []
    eigenvalues = []
    counts = []
    for _ in range(k):
        b = np.random.randn(num_rows) # Creates a random vector, distributed normally (N(0,1))
        b = b.T # Make it into a column vector
        counter = 0

        while True:
            b_old = b
            dot_prod = np.dot(A, b)
            b = np.divide(dot_prod, np.linalg.norm(dot_prod))
            b = orthogonalize(b, eigenvectors)

            e_plus =  np.linalg.norm(b - b_old)
            e_minus = np.linalg.norm(b + b_old)

            counter += 1
            if e_plus < eps or e_minus < eps:
                break


        eigval = find_eigenvalue(b, A)

        eigenvectors.append(b)
        eigenvalues.append(eigval)
        counts.append(counter)

    generate_output(eigenvalues, eigenvectors)
    


    return zip(eigenvectors, eigenvalues, counts)


def find_eigenvalue(eigvec, A):
    '''Finds the eigenvalue corresopnding with eigvec'''
    return np.divide(np.inner(np.dot(A, eigvec), eigvec), np.inner(eigvec,eigvec))


def orthogonalize(b, eigenvectors):
    '''Orthogonalizes the vector b with the list of eigenvectors'''
    for i in range(len(eigenvectors)):
        b = b - np.dot(np.inner(eigenvectors[i], b), eigenvectors[i])
    return b

def normalize(vector):
    '''Normalizes the vector'''
    return np.divide(vector, np.linalg.norm(vector))


def get_matrix(file):
    '''Returns a numpy matrix object from the ASCII file'''
    array = np.genfromtxt(file)
    return array

def generate_output(eigenvalues, eigenvectors):
    '''Writes output to ASCII text files'''
    eigvecs_file = open("eigenvectors.txt", 'w')
    eigvals_file = open("eigenvalues.txt", 'w')

    eigvals_file.writelines([str(val) + '\n' for val in eigenvalues])

    eig_array = np.array(eigenvectors).T
    for row in eig_array:
        eigvecs_file.writelines([str(val) + " " for val in row])
        eigvecs_file.write('\n')

def test_on_random_matrices1():
    '''Tests rate of convergence for single eigenvector'''
    ## Need to get randommly generated symmetric matrices
    dims = [(i, i) for i in xrange(5, 100, 1)]
    eps_list = [10**-i for i in xrange(1, 6)]
    counts = []
    for shape in dims:
        for eps in eps_list:
            matrix = np.random.uniform(-100,100,size=shape)
            matrix = (matrix + matrix.T) / 2 # Make symmetric
            count = get_iteration_count(eps, matrix)
            counts.append((count, shape, eps))
    generate_plots(counts)

def dimension_plot():
    '''Plots the number of iterations with size of matrix'''
    dims = [10+10*i for i in range(10)]
    eps = .001
    avg_counts = []
    for dim in dims:
        counts = []
        for i in range(100):
            matrix = np.random.uniform(-100,100,size=(dim,dim))
            matrix = (matrix + matrix.T) / 2 # Make symmetric
            (vec, val, count) = eigP(1, eps, matrix)[0]
            counts.append(count)
        avg_count = np.median(counts)
        avg_counts.append(avg_count)
    plt.plot(dims, avg_counts)
    plt.xlabel("Size of matrix")
    plt.ylabel("Number of iterations before convergence")
    plt.show()

def epsilon_plot():
    '''Plots the number of iterations with epsilon bound'''
    eps = [.1, .05, .01, .005, .001, .0005, .0001, .00005, .00001]
    dim = 50
    avg_counts = []
    for ep in eps:
        counts = []
        for i in range(100):
            print i
            matrix = np.random.uniform(-100,100,size=(dim,dim))
            matrix = (matrix + matrix.T) / 2 # Symmetric
            (vec, val, count) = eigP(1, ep, matrix)[0]
            counts.append(count)
        avg_count = np.median(counts)
        avg_counts.append(avg_count)
    plt.plot(map(math.log, eps), avg_counts)
    plt.xlabel("Log of epsilon bound")
    plt.ylabel("Number of iterations before convergence")
    plt.show()





def generate_plots(counts):
    '''Takes as input (count, shape, eps) where
    shape is a tuple of ints, and eps is a float.
    '''
    # Plot iteration count over size of matrix
    dims = [dim for (count, (dim, _), eps) in counts if eps == .001]
    cs = [count for (count, _, eps) in counts if eps == .001]
    plt.plot(dims, cs)
    plt.xlabel('Size of matrix')
    plt.ylabel('Number of iterations before convergence')
    plt.show()

    # Plot iteration count over epsilons
    eps = [ep for (_, (dim,_), ep) in counts if dim == 50]
    cs = [count for (count, (dim, _), _) in counts if dim == 50]
    plt.plot(map(math.log, eps), cs)
    plt.xlabel('Log of epsilon')
    plt.ylabel('Number of iterations before convergence')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Invalid number of arguments. Proper use:"
        print "> python eigP.py <k> <epsilon> <matrix_file_path>"
        print "k = number of eigenvectors"
        print "epsilon = bound for power method"
        print "matrix_file_path is a path to an ASCII file encoding a symmetric matrix"
        raise Exception("Invalid input!")
    k = int(sys.argv[1])
    eps = float(sys.argv[2])
    matrix_filepath = sys.argv[3]
    matrix_file = open(matrix_filepath, 'r')
    matrix = get_matrix(matrix_file)
    res = eigP(k, eps, matrix)

    eigvecs = [vec for (vec, val, c) in res]
    eigvals = [val for (vec, val, c) in res]
    print "Eigenvectors: "
    print eigvecs
    print "Eigenvalues: "
    print eigvals




